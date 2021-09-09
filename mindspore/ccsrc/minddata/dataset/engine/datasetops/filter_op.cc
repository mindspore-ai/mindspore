/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "minddata/dataset/engine/datasetops/filter_op.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"
#include "minddata/dataset/core/tensor.h"

#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
FilterOp::FilterOp(const std::vector<std::string> &in_col_names, int32_t num_workers, int32_t op_queue_size,
                   std::shared_ptr<TensorOp> predicate_func)
    : ParallelOp(num_workers, op_queue_size), predicate_func_(std::move(predicate_func)), in_columns_(in_col_names) {
  worker_queues_.Init(num_workers, op_queue_size);
}
Status FilterOp::LaunchThreadsAndInitOp() {
  // The operator class just starts off threads by calling the tree_ function.
  if (tree_ == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "[Internal ERROR] Pipeline init failed, Execution tree not set.");
  }
  filter_queues_.Init(num_workers_, oc_queue_size_);
  RETURN_IF_NOT_OK(filter_queues_.Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(worker_queues_.Register(tree_->AllTasks()));

  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&FilterOp::WorkerEntry, this, std::placeholders::_1), Name(), id()));
  RETURN_IF_NOT_OK(
    tree_->AllTasks()->CreateAsyncTask("FilterCollector", std::bind(&FilterOp::Collector, this), nullptr, id()));

  return Status::OK();
}

Status FilterOp::operator()() {
  // Synchronize with TaskManager.
  Status rc = LaunchThreadsAndInitOp();
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(rc);

  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  TensorRow new_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  int64_t cnt = 0;
  while (child_iterator_->EofHandled() == false) {
    while (new_row.empty() == false) {
      RETURN_IF_NOT_OK(worker_queues_[cnt % num_workers_]->EmplaceBack(new_row));
      cnt++;
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
    }

    RETURN_IF_NOT_OK(worker_queues_[cnt++ % num_workers_]->EmplaceBack(std::move(TensorRow(TensorRow::kFlagEOE))));
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  }
  RETURN_IF_NOT_OK(worker_queues_[cnt++ % num_workers_]->EmplaceBack(std::move(TensorRow(TensorRow::kFlagEOF))));
  // EOF received, send quit signal to all workers
  for (int32_t ind = 0; ind < num_workers_; ind++) {
    RETURN_IF_NOT_OK(worker_queues_[cnt++ % num_workers_]->EmplaceBack(std::move(TensorRow(TensorRow::kFlagQuit))));
  }

  return Status::OK();
}

Status FilterOp::EofReceived(int32_t) { return Status::OK(); }

Status FilterOp::EoeReceived(int32_t) { return Status::OK(); }

// Validating if each of the input_columns exists in the column_name_id_map_.
Status FilterOp::ValidateInColumns(const std::vector<std::string> &input_columns) {
  for (const auto &inCol : input_columns) {
    bool found = column_name_id_map_.find(inCol) != column_name_id_map_.end() ? true : false;
    if (!found) {
      std::string err_msg = "Invalid parameter, column name: " + inCol + " does not exist in the dataset columns.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  return Status::OK();
}

// A print method typically used for debugging.
void FilterOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nInput column names:";
    for (size_t i = 0; i < in_columns_.size(); i++) {
      out << " " << in_columns_[i];
    }
    out << "\n\n";
  }
}

Status FilterOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  TensorRow new_row;
  RETURN_IF_NOT_OK(worker_queues_[worker_id]->PopFront(&new_row));

  while (!new_row.quit()) {
    // Getting a TensorRow to work on.
    if (new_row.eoe()) {
      RETURN_IF_NOT_OK(filter_queues_[worker_id]->EmplaceBack(std::make_pair(new_row, filterCtrl::kFilterEoe)));
    } else if (new_row.eof()) {
      RETURN_IF_NOT_OK(filter_queues_[worker_id]->EmplaceBack(std::make_pair(new_row, filterCtrl::kFilterEof)));
    } else {
      RETURN_IF_NOT_OK(ValidateInColumns(in_columns_));

      bool result = false;
      RETURN_IF_NOT_OK(WorkerCompute(new_row, &result));

      if (result)
        RETURN_IF_NOT_OK(
          filter_queues_[worker_id]->EmplaceBack(std::make_pair(std::move(new_row), filterCtrl::kFilterFull)));
      else
        RETURN_IF_NOT_OK(
          filter_queues_[worker_id]->EmplaceBack(std::make_pair(std::move(new_row), filterCtrl::kFilterEmpty)));
    }
    RETURN_IF_NOT_OK(worker_queues_[worker_id]->PopFront(&new_row));
  }
  return Status::OK();
}

Status FilterOp::WorkerCompute(const TensorRow &in_row, bool *out_predicate) {
  TensorRow to_process;
  if (in_columns_.empty() == true) {
    MS_LOG(INFO) << "Input columns in filter operator is empty, will apply to the all column in the current table.";
    to_process = in_row;
  } else {
    (void)std::transform(
      in_columns_.begin(), in_columns_.end(), std::back_inserter(to_process),
      [&in_row, this](const auto &it) -> std::shared_ptr<Tensor> { return in_row[column_name_id_map_[it]]; });
  }
  RETURN_IF_NOT_OK(InvokePredicateFunc(to_process, out_predicate));
  return Status::OK();
}

// if the filtered TensorRow is written directly to out_connector_,
// the thread fetching data will block in a queue.
// Collector thread will reorder the TensorRow in order until EOF is received
// for example in two work queues:
// int filter_queues_:
// queue1:  TR(data1 kFilterEmpty)    TR(eoe)                                  TR(data4)   TR(eof)
// queue2:  TR(data2)                                TR(data3 kFilterEmpty)  TR(eoe)
// after reorder in out_connector_:
// queue1:  TR(data2)    TR(data4)        TR(eof)
// queue2:  TR(eoe)        TR(eoe)
Status FilterOp::Collector() {
  TaskManager::FindMe()->Post();
  bool collector_stop = false;
  uint64_t task_id_cnt = 0;
  uint64_t out_id_cnt = 0;
  std::pair<TensorRow, filterCtrl> in_pair;
  while (collector_stop == false) {
    uint32_t w_id = task_id_cnt % num_workers_;
    RETURN_IF_NOT_OK(filter_queues_[w_id]->PopFront(&in_pair));
    if (in_pair.second == filterCtrl::kFilterFull || in_pair.second == filterCtrl::kFilterPartial ||
        in_pair.second == filterCtrl::kFilterEoe) {
      uint32_t out_task_id = out_id_cnt % num_workers_;
      if (in_pair.second == filterCtrl::kFilterEoe) {
        UpdateRepeatAndEpochCounter();
        RETURN_IF_NOT_OK(out_connector_->SendEOE(static_cast<int>(out_task_id)));
      } else {
        RETURN_IF_NOT_OK(out_connector_->Add(std::move(in_pair.first), static_cast<int>(out_task_id)));
      }
      out_id_cnt++;
      task_id_cnt++;
    } else if (in_pair.second == filterCtrl::kFilterEof) {
      uint32_t out_task_id = out_id_cnt % num_workers_;
      RETURN_IF_NOT_OK(out_connector_->SendEOF(static_cast<int>(out_task_id)));
      collector_stop = true;
    } else {  // kFilterEmpty
      task_id_cnt++;
    }
  }
  return Status::OK();
}

Status FilterOp::CheckInput(const TensorRow &input) const {
  for (auto &item : input) {
    if (item == nullptr) {
      RETURN_STATUS_UNEXPECTED("Invalid data, input tensor is null.");
    }
  }
  return Status::OK();
}

Status FilterOp::InvokePredicateFunc(const TensorRow &input, bool *out_predicate) {
  RETURN_IF_NOT_OK(CheckInput(input));

  TensorRow output;
  RETURN_IF_NOT_OK(predicate_func_->Compute(input, &output));
  RETURN_IF_NOT_OK(output.at(0)->GetItemAt(out_predicate, {}));

  return Status(StatusCode::kSuccess, "FilterOp predicate func call succeed");
}
int32_t FilterOp::NumConsumers() const { return 1; }

}  // namespace dataset
}  // namespace mindspore
