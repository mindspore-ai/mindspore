/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/dataset_iterator.h"
#include <algorithm>
#include <unordered_map>
#include <utility>
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#ifndef ENABLE_SECURITY
#include "minddata/dataset/engine/perf/profiling.h"
#endif

namespace mindspore {
namespace dataset {
// Fetches one row of data from the iterator as a column map.
Status DatasetIterator::GetNextAsMap(TensorMap *out_map) {
  if (out_map == nullptr) {
    RETURN_STATUS_UNEXPECTED("Null output map in iterator!");
  }

  out_map->clear();

  TensorRow curr_row;
  MS_LOG(INFO) << "get next as map start.";
  RETURN_IF_NOT_OK(FetchNextTensorRow(&curr_row));
  MS_LOG(INFO) << "fetchNextTensor success.";

  // Return empty map if there's no data
  if (curr_row.empty()) {
    return Status::OK();
  }

  // The column name mapping is needed to be able to produce the tensor map output.
  // The column name mapping comes from the source operator that is producing the data into the iterator.
  // To avoid having to fetch this for every time, we'll take a local copy of the column name id mapping
  // and save in the iterator.  We only have to do this once.  All subsequent iterations use the same mapping.
  if (col_name_id_map_.empty()) {
    // Determine the column name map by calling the derived class method to retrieve the column
    // name map
    col_name_id_map_ = this->GetColumnNameMap();
  }

  // Populate the out map from the row and return it
  for (const auto &colMap : col_name_id_map_) {
    std::string column_name = colMap.first;
    // Need to filter meta column start with kDftMetaColumnPrefix
    size_t pos = column_name.find(kDftMetaColumnPrefix);
    if (pos != std::string::npos && pos == 0) {
      continue;
    }
    (*out_map)[colMap.first] = std::move(curr_row[colMap.second]);
  }
  if (out_map->size() == 0) {
    std::string err_msg = "No effective column found, maybe all columns are meta column and will be filtered. ";
    err_msg += "If you want to output meta column please rename column name to a new one which is not start with ";
    err_msg += "\"" + std::string(kDftMetaColumnPrefix) + "\"";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}
// Constructor of the DatasetIterator
DatasetIterator::DatasetIterator(std::shared_ptr<ExecutionTree> exe_tree)
    : root_(exe_tree->root()),
#ifndef ENABLE_SECURITY
      tracing_(nullptr),
#endif
      cur_batch_num_(0),
      cur_connector_size_(0),
      cur_connector_capacity_(0),
      eof_handled_(false) {
  std::shared_ptr<Tracing> node;
#ifndef ENABLE_SECURITY
  Status s = GlobalContext::profiling_manager()->GetTracingNode(kDatasetIteratorTracingName, &node);
  if (s.IsOk()) {
    tracing_ = std::dynamic_pointer_cast<DatasetIteratorTracing>(node);
  }
#endif
}

DatasetIterator::~DatasetIterator() = default;

// Fetches one row of data from the iterator.  Overrides the base class.  This one fetches
// from the tree root node directly.
Status DatasetIterator::FetchNextTensorRow(TensorRow *out_row) {
  if (out_row == nullptr) {
    RETURN_STATUS_UNEXPECTED("Null output row in iterator!");
  }
  // clear the old tensor row
  out_row->clear();
#ifndef ENABLE_SECURITY
  bool is_profiling_enable = GlobalContext::profiling_manager()->IsProfilingEnable(root_->Tree());
#endif
  // Once eof is handled, always return empty row.  Class must be destroyed and recreated if you
  // want to iterate again.
  if (eof_handled_) {
    std::string err = "EOF buffer encountered. Users try to fetch data beyond the specified number of epochs.";
    RETURN_STATUS_UNEXPECTED(err);
  }
#ifndef ENABLE_SECURITY
  if (tracing_ != nullptr) {
    cur_connector_size_ = root_->ConnectorSize();
    cur_connector_capacity_ = root_->ConnectorCapacity();
  }
#endif
  RETURN_IF_NOT_OK(root_->GetNextRow(out_row));

  // Since GetNextRow was used rather than GetNextInput(), it means we need to manually
  // handle eoe and eof messages here.
  //
  // An eoe row means we have iterated an epoch.
  // The next row in the pipeline might be an EOF or a TensorRow for next epoch
  if (out_row->eoe()) {
    MS_LOG(INFO) << "End of data iteration.  cur_batch_num_: " << cur_batch_num_;
#ifndef ENABLE_SECURITY
    if (is_profiling_enable) {
      root_->Tree()->SetEpochEnd();
      GlobalContext::profiling_manager()->RecordEndOfEpoch(cur_batch_num_);
    }
#endif
    return Status::OK();
  }

  // An eof buffer means it is the end of execution and all operators are shutting down.
  // Because there is no more data to return to the caller, this will change `eof_handled_` state and
  // returns status unexpected error.
  if (out_row->eof()) {
    eof_handled_ = true;
    root_->Tree()->SetFinished();
    std::string err = "EOF buffer encountered. Users try to fetch data beyond the specified number of epochs.";
    RETURN_STATUS_UNEXPECTED(err);
  }
#ifndef ENABLE_SECURITY
  if (tracing_ != nullptr) {
    cur_batch_num_++;
    tracing_->Record(static_cast<int32_t>(CONNECTOR_DEPTH), cur_connector_capacity_, cur_batch_num_,
                     cur_connector_size_, ProfilingTime::GetCurMilliSecond());
  }
#endif
  return Status::OK();
}

// Getter
std::unordered_map<std::string, int32_t> DatasetIterator::GetColumnNameMap() const {
  return root_->column_name_id_map();
}

// Constructor of the ChildIterator
ChildIterator::ChildIterator(DatasetOp *current_op, int32_t worker_id, int32_t child_idx)
    : current_op_(current_op), child_idx_(child_idx), worker_id_(worker_id), end_epoch_(false), eof_handled_(false) {}

ChildIterator::~ChildIterator() { current_op_ = nullptr; }

// Fetches one row of data from the iterator.  Overrides the base class.  This one fetches
// only from the child/worker id as given from the constructor.
Status ChildIterator::FetchNextTensorRow(TensorRow *out_row) {
  RETURN_UNEXPECTED_IF_NULL(out_row);
  // clear the old tensor row
  out_row->clear();

  // Once eof is handled, always return empty row.  Class must be destroyed and recreated if you
  // want to iterate again.
  if (eof_handled_) {
    std::string err = "EOF buffer encountered. Users try to fetch data beyond the specified number of epochs.";
    RETURN_STATUS_UNEXPECTED(err);
  }

  RETURN_IF_NOT_OK(current_op_->child(child_idx_)->GetNextRow(out_row));
  // If an eoe is picked up here, we simply return an empty vector and it's up to the
  // caller to decide what it wants to do next.TensorRow
  if (out_row->eoe()) {
    MS_LOG(DEBUG) << "(" << current_op_->NameWithID() << ", " << child_idx_ << ")"
                  << "Child iterator picked up EOE.";
    end_epoch_ = true;
    return Status::OK();
  } else {
    end_epoch_ = false;
  }

  if (out_row->eof()) {
    MS_LOG(DEBUG) << "(" << current_op_->NameWithID() << ", " << child_idx_ << ")"
                  << "Child iterator picked up EOF.";
    eof_handled_ = true;
    *out_row = TensorRow(TensorRow::kFlagEOF);
  }
  return Status::OK();
}

// drain till the next eoe
Status ChildIterator::Drain() {
  if (end_epoch_ == true) {
    // Calling drain against a child that is already at it's eoe state will not result in any action.
    // This allows you to do:
    // - fetch until empty row
    // - drain (will not actually drain because you are already at the end of the iteration)
    // However, the next time after that, it will perform it's normal draining activities.
    end_epoch_ = false;
    MS_LOG(DEBUG) << "No operation drain, already at end of epoch.";
    return Status::OK();
  }
  MS_LOG(DEBUG) << "Child draining buffers until eoe.";
  TensorRow row;
  // else we drain until eoe or eof, eof here is for sanity check
  while (!row.eoe() && !row.eof()) {
    RETURN_IF_NOT_OK(current_op_->child(child_idx_)->GetNextRow(&row));
  }
  if (row.eof()) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "Child iterator picked up EOF in drain.");
  }
  return Status::OK();
}

// Getter
std::unordered_map<std::string, int32_t> ChildIterator::GetColumnNameMap() const {
  return current_op_->child(child_idx_)->column_name_id_map();
}
}  // namespace dataset
}  // namespace mindspore
