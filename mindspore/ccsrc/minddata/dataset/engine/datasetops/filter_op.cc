/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
  worker_in_queues_.Init(num_workers, op_queue_size);
}

Status FilterOp::operator()() {
  RETURN_IF_NOT_OK(RegisterAndLaunchThreads());
  // Synchronize with TaskManager.
  TaskManager::FindMe()->Post();

  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  TensorRow new_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  int64_t cnt = 0;
  while (child_iterator_->EofHandled() == false) {
    while (new_row.empty() == false) {
      RETURN_IF_NOT_OK(worker_in_queues_[static_cast<const int>(cnt % num_workers_)]->EmplaceBack(new_row));
      cnt++;
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
    }

    RETURN_IF_NOT_OK(worker_in_queues_[static_cast<const int>(cnt++ % num_workers_)]->EmplaceBack(
      std::move(TensorRow(TensorRow::kFlagEOE))));
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  }
  RETURN_IF_NOT_OK(worker_in_queues_[static_cast<const int>(cnt++ % num_workers_)]->EmplaceBack(
    std::move(TensorRow(TensorRow::kFlagEOF))));
  // EOF received, send quit signal to all workers
  for (int32_t ind = 0; ind < num_workers_; ind++) {
    RETURN_IF_NOT_OK(worker_in_queues_[static_cast<const int>(cnt++ % num_workers_)]->EmplaceBack(
      std::move(TensorRow(TensorRow::kFlagQuit))));
  }

  return Status::OK();
}

Status FilterOp::EofReceived(int32_t) { return Status::OK(); }

Status FilterOp::EoeReceived(int32_t) { return Status::OK(); }

// Validating if each of the input_columns exists in the column_name_id_map_.
Status FilterOp::ValidateInColumns(const std::vector<std::string> &input_columns) const {
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
  RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->PopFront(&new_row));

  while (!new_row.quit()) {
    // Getting a TensorRow to work on.
    if (new_row.eoe()) {
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(new_row));
    } else if (new_row.eof()) {
      RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(new_row));
    } else {
      RETURN_IF_NOT_OK(ValidateInColumns(in_columns_));

      bool result = false;
      RETURN_IF_NOT_OK(WorkerCompute(new_row, &result));

      if (result) {
        RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(new_row));
      } else {
        RETURN_IF_NOT_OK(worker_out_queues_[worker_id]->EmplaceBack(TensorRow(TensorRow::TensorRowFlags::kFlagSkip)));
      }
    }
    RETURN_IF_NOT_OK(worker_in_queues_[worker_id]->PopFront(&new_row));
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

Status FilterOp::CheckInput(const TensorRow &input) const {
  for (auto &item : input) {
    if (item == nullptr) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] input tensor is null.");
    }
  }
  return Status::OK();
}

Status FilterOp::InvokePredicateFunc(const TensorRow &input, bool *out_predicate) {
  RETURN_UNEXPECTED_IF_NULL(out_predicate);
  RETURN_IF_NOT_OK(CheckInput(input));

  TensorRow output;
  RETURN_IF_NOT_OK(predicate_func_->Compute(input, &output));
  RETURN_IF_NOT_OK(output.at(0)->GetItemAt(out_predicate, {}));

  return Status(StatusCode::kSuccess, "FilterOp predicate func call succeed");
}
}  // namespace dataset
}  // namespace mindspore
