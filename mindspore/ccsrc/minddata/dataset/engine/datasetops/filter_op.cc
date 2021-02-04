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
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {

Status FilterOp::Builder::SanityCheck() {
  std::string err;
  err += builder_op_connector_size_ <= 0 ? "Invalid parameter, connector_size must be greater than 0, but got " +
                                             std::to_string(builder_op_connector_size_) + ".\n"
                                         : "";
  err += builder_num_workers_ <= 0 ? "Invalid parameter, num_parallel_workers must be greater than 0, but got " +
                                       std::to_string(builder_num_workers_) + ".\n"
                                   : "";
  return err.empty() ? Status::OK() : Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, common::SafeCStr(err));
}

FilterOp::Builder::Builder() {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status FilterOp::Builder::Build(std::shared_ptr<FilterOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<FilterOp>(std::move(build_in_col_names_), builder_num_workers_, builder_op_connector_size_,
                                    builder_predicate_func_);
  return Status::OK();
}

FilterOp::FilterOp(const std::vector<std::string> &in_col_names, int32_t num_workers, int32_t op_queue_size,
                   std::shared_ptr<TensorOp> predicate_func)
    : ParallelOp(num_workers, op_queue_size), predicate_func_(std::move(predicate_func)), in_columns_(in_col_names) {}

Status FilterOp::operator()() {
  // The operator class just starts off threads by calling the tree_ function.
  if (tree_ == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "Pipeline init failed, Execution tree not set.");
  }
  filter_queues_.Init(num_workers_, oc_queue_size_);
  RETURN_IF_NOT_OK(filter_queues_.Register(tree_->AllTasks()));
  Status rc =
    tree_->LaunchWorkers(num_workers_, std::bind(&FilterOp::WorkerEntry, this, std::placeholders::_1), Name(), id());
  // Synchronize with TaskManager.
  TaskManager::FindMe()->Post();
  RETURN_IF_NOT_OK(rc);
  RETURN_IF_NOT_OK(Collector());
  return Status::OK();
}

Status FilterOp::EofReceived(int32_t) { return Status::OK(); }

Status FilterOp::EoeReceived(int32_t) { return Status::OK(); }

// Validating if each of the input_columns exists in the DataBuffer.
Status FilterOp::ValidateInColumns(const std::vector<std::string> *input_columns) {
  for (const auto &inCol : *input_columns) {
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
  // Handshake with TaskManager that thread creation is successful.
  TaskManager::FindMe()->Post();
  std::unique_ptr<DataBuffer> in_buffer;
  bool worker_stop = false;
  while (worker_stop == false) {
    // Getting a databuffer to work on.
    RETURN_IF_NOT_OK(child_[0]->GetNextBuffer(&in_buffer, worker_id));
    if (in_buffer->eoe()) {
      filter_queues_[worker_id]->EmplaceBack(std::make_pair(std::move(in_buffer), filterCtrl::kFilterEoe));
      continue;
    } else if (in_buffer->eof()) {
      filter_queues_[worker_id]->EmplaceBack(std::make_pair(std::move(in_buffer), filterCtrl::kFilterEof));
      worker_stop = true;
      continue;
    }

    RETURN_IF_NOT_OK(CheckColumns(in_buffer.get(), &in_columns_));

    // if the databuffer was all filtered, it is marked as kFilterEmpty.
    // if the databuffer was partially filtered, it is marked as kFilterPartial.
    // if the databuffer was not filtered, it is marked as kFilterFull.
    int32_t num_rows = in_buffer->NumRows();
    std::unique_ptr<TensorQTable> new_tensor_table;
    RETURN_IF_NOT_OK(WorkerCompute(in_buffer.get(), &new_tensor_table));

    if (new_tensor_table->empty()) {
      RETURN_IF_NOT_OK(
        filter_queues_[worker_id]->EmplaceBack(std::make_pair(std::move(in_buffer), filterCtrl::kFilterEmpty)));
    } else if (new_tensor_table->size() == num_rows) {
      in_buffer->set_tensor_table(std::move(new_tensor_table));
      RETURN_IF_NOT_OK(
        filter_queues_[worker_id]->EmplaceBack(std::make_pair(std::move(in_buffer), filterCtrl::kFilterFull)));
    } else {  // kFilterPartial
      in_buffer->set_tensor_table(std::move(new_tensor_table));
      RETURN_IF_NOT_OK(
        filter_queues_[worker_id]->EmplaceBack(std::make_pair(std::move(in_buffer), filterCtrl::kFilterPartial)));
    }
  }
  return Status::OK();
}

Status FilterOp::WorkerCompute(DataBuffer *in_buffer, std::unique_ptr<TensorQTable> *out) {
  *out = std::make_unique<TensorQTable>();
  int32_t num_rows = in_buffer->NumRows();
  for (int32_t i = 0; i < num_rows; i++) {
    TensorRow to_process;
    TensorRow cur_row;
    RETURN_IF_NOT_OK(in_buffer->PopRow(&cur_row));
    if (in_columns_.empty() == true) {
      MS_LOG(INFO) << "Input columns in filter operator is empty, will apply to the all column in the current table.";
      to_process = cur_row;
    } else {
      (void)std::transform(
        in_columns_.begin(), in_columns_.end(), std::back_inserter(to_process),
        [&cur_row, this](const auto &it) -> std::shared_ptr<Tensor> { return cur_row[column_name_id_map_[it]]; });
    }
    bool predicate = true;
    RETURN_IF_NOT_OK(InvokePredicateFunc(to_process, &predicate));
    if (predicate) {
      (*out)->push_back(std::move(cur_row));
    }
  }
  return Status::OK();
}

// if the filtered DataBuffer is written directly to out_connector_,
// the thread fetching data will block in a queue.
// Collector function will reorder the DataBuffer in order.
// for example in two work queues:
// int filter_queues_:
// queue1:  DB(data1 kFilterEmpty)    DB(eoe)                                  DB(data4)   DB(eof)
// queue2:  DB(data2)                                DB(data3 kFilterEmpty)  DB(eoe)
// after reorder in out_connector_:
// queue1:  DB(data2)    DB(data4)        DB(eof)
// queue2:  DB(eoe)        DB(eoe)
Status FilterOp::Collector() {
  bool collector_stop = false;
  uint64_t task_id_cnt = 0;
  uint64_t out_id_cnt = 0;
  std::pair<std::unique_ptr<DataBuffer>, filterCtrl> in_pair;
  while (collector_stop == false) {
    uint32_t w_id = task_id_cnt % num_workers_;
    RETURN_IF_NOT_OK(filter_queues_[w_id]->PopFront(&in_pair));
    if (in_pair.second == filterCtrl::kFilterFull || in_pair.second == filterCtrl::kFilterPartial ||
        in_pair.second == filterCtrl::kFilterEoe) {
      if (in_pair.second == filterCtrl::kFilterEoe) UpdateRepeatAndEpochCounter();
      uint32_t out_task_id = out_id_cnt % num_workers_;
      RETURN_IF_NOT_OK(out_connector_->Add(static_cast<int>(out_task_id), std::move(in_pair.first)));
      out_id_cnt++;
      task_id_cnt++;
    } else if (in_pair.second == filterCtrl::kFilterEof) {
      uint32_t out_task_id = out_id_cnt % num_workers_;
      RETURN_IF_NOT_OK(out_connector_->Add(static_cast<int>(out_task_id), std::move(in_pair.first)));
      collector_stop = true;
    } else {  // kFilterEmpty
      task_id_cnt++;
    }
  }
  return Status::OK();
}

// Private function for checking the column legality.
Status FilterOp::CheckColumns(const DataBuffer *in_buf, const std::vector<std::string> *input_columns) {
  int32_t num_rows = in_buf->NumRows();
  int32_t num_cols = in_buf->NumCols();
  if (num_rows == 0 || num_cols == 0) {
    RETURN_STATUS_UNEXPECTED("FilterOp is getting an empty DataBuffer.");
  }
  // Check if there is invalid column name in the inColumns.
  RETURN_IF_NOT_OK(ValidateInColumns(input_columns));
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

}  // namespace dataset
}  // namespace mindspore
