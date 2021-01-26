/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <utility>
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/perf/profiling.h"

namespace mindspore {
namespace dataset {
// Constructor of the IteratorBase
IteratorBase::IteratorBase() : curr_buffer_(nullptr), eof_handled_(false) {}

IteratorBase::~IteratorBase() = default;

// Fetches one row of data from the iterator as a column map.
Status IteratorBase::GetNextAsMap(TensorMap *out_map) {
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
  for (const auto colMap : col_name_id_map_) {
    (*out_map)[colMap.first] = std::move(curr_row[colMap.second]);
  }

  return Status::OK();
}

// Fetches one row of data from the iterator.
// The base class version simply performs error handling and returns empty row. Actual
// functionality exists in the derived versions of this function.
Status IteratorBase::FetchNextTensorRow(TensorRow *out_row) {
  if (out_row == nullptr) {
    RETURN_STATUS_UNEXPECTED("Null output row in iterator!");
  }

  // clear the old tensor row
  out_row->clear();

  return Status::OK();
}

Status IteratorBase::GetNextAsOrderedPair(std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> *vec) {
  CHECK_FAIL_RETURN_UNEXPECTED(vec != nullptr && vec->empty(), "vec is null or non-empty.");

  TensorRow curr_row;

  RETURN_IF_NOT_OK(FetchNextTensorRow(&curr_row));
  RETURN_OK_IF_TRUE(curr_row.empty());

  size_t num_cols = curr_row.size();  // num_cols is non-empty.
  if (col_name_id_map_.empty()) col_name_id_map_ = this->GetColumnNameMap();
  // order the column names according to their ids
  if (column_order_.empty()) {
    const int32_t invalid_col_id = -1;
    column_order_.resize(num_cols, {std::string(), invalid_col_id});
    for (const auto &itr : col_name_id_map_) {
      int32_t ind = itr.second;
      CHECK_FAIL_RETURN_UNEXPECTED(ind < num_cols && ind >= 0, "column id out of bounds.");
      column_order_[ind] = std::make_pair(itr.first, ind);
    }
    // error check, make sure the ids in col_name_id_map are continuous and starts from 0
    for (const auto &col : column_order_) {
      CHECK_FAIL_RETURN_UNEXPECTED(col.second != invalid_col_id, "column ids are not continuous.");
    }
  }

  vec->reserve(num_cols);

  for (const auto &col : column_order_) {
    vec->emplace_back(std::make_pair(col.first, curr_row[col.second]));
  }

  return Status::OK();
}

// Constructor of the DatasetIterator
DatasetIterator::DatasetIterator(std::shared_ptr<ExecutionTree> exe_tree)
    : IteratorBase(),
      root_(exe_tree->root()),
      tracing_(nullptr),
      cur_batch_num_(0),
      cur_connector_size_(0),
      cur_connector_capacity_(0) {
  std::shared_ptr<Tracing> node;
  Status s = exe_tree->GetProfilingManager()->GetTracingNode(kDatasetIteratorTracingName, &node);
  if (s.IsOk()) {
    tracing_ = std::dynamic_pointer_cast<DatasetIteratorTracing>(node);
  }
}

DatasetIterator::~DatasetIterator() = default;

// Fetches one row of data from the iterator.  Overrides the base class.  This one fetches
// from the tree root node directly.
Status DatasetIterator::FetchNextTensorRow(TensorRow *out_row) {
  // Common code init and error checking in the base class.
  RETURN_IF_NOT_OK(IteratorBase::FetchNextTensorRow(out_row));

  bool isProfilingEnable = root_->Tree()->GetProfilingManager()->IsProfilingEnable();

  // Once eof is handled, always return empty row.  Class must be destroyed and recreated if you
  // want to iterate again.
  if (eof_handled_) {
    std::string err = "EOF buffer encountered. Users try to fetch data beyond the specified number of epochs.";
    RETURN_STATUS_UNEXPECTED(err);
  }

  // Check if we need to get a new DataBuffer to iterate.
  if (curr_buffer_ == nullptr || curr_buffer_->NumRows() == 0) {
    if (tracing_ != nullptr) {
      cur_connector_size_ = root_->ConnectorSize();
      cur_connector_capacity_ = root_->ConnectorCapacity();
    }
    RETURN_IF_NOT_OK(root_->GetNextBuffer(&curr_buffer_));

    // Since GetNextBuffer was used rather than GetNextInput(), it means we need to manually
    // handle eoe and eof messages here.
    //
    // An eoe buffer means we have iterated an epoch.
    // The next buffer in the pipeline might be an EOF or a databuffer for next epoch
    if (curr_buffer_->eoe()) {
      MS_LOG(INFO) << "End of data iteration.";
      curr_buffer_.reset();  // explicitly free the eoe buffer
      if (isProfilingEnable) {
        root_->Tree()->SetEpochEnd();
      }
      return Status::OK();
    }

    // An eof buffer means it is the end of execution and all operators are shutting down.
    // Because there is no more data to return to the caller, this will change `eof_handled_` state and
    // returns status unexpected error.
    if (curr_buffer_->eof()) {
      eof_handled_ = true;
      curr_buffer_.reset();  // explicitly free the eof buffer
      root_->Tree()->SetFinished();
      std::string err = "EOF buffer encountered. Users try to fetch data beyond the specified number of epochs.";
      RETURN_STATUS_UNEXPECTED(err);
    }
  }

  // If we got this far, now it's time to pop that next row for return to caller
  RETURN_IF_NOT_OK(curr_buffer_->PopRow(out_row));
  if (tracing_ != nullptr) {
    cur_batch_num_++;
    tracing_->Record(CONNECTOR_DEPTH, cur_connector_capacity_, cur_batch_num_, cur_connector_size_,
                     ProfilingTime::GetCurMilliSecond());
  }
  return Status::OK();
}

Status DatasetIterator::GetOutputShapes(std::vector<TensorShape> *out_shapes) {
  if (out_shapes == nullptr) {
    RETURN_STATUS_UNEXPECTED("Null output shape argument");
  }
  if (device_queue_row_.empty()) {
    RETURN_IF_NOT_OK(FetchNextTensorRow(&device_queue_row_));
  }
  for (const auto ts : device_queue_row_) {
    out_shapes->push_back(ts->shape());
  }

  return Status::OK();
}

Status DatasetIterator::GetOutputTypes(std::vector<DataType> *out_types) {
  if (out_types == nullptr) {
    RETURN_STATUS_UNEXPECTED("Null output type argument");
  }
  if (device_queue_row_.empty()) {
    RETURN_IF_NOT_OK(FetchNextTensorRow(&device_queue_row_));
  }
  for (const auto ts : device_queue_row_) {
    out_types->push_back(ts->type());
  }
  return Status::OK();
}

// Getter
std::unordered_map<std::string, int32_t> DatasetIterator::GetColumnNameMap() const {
  return root_->column_name_id_map();
}

// Constructor of the ChildIterator
ChildIterator::ChildIterator(DatasetOp *current_op, int32_t worker_id, int32_t child_idx)
    : IteratorBase(), current_op_(current_op), child_idx_(child_idx), worker_id_(worker_id), end_epoch_(false) {}

ChildIterator::~ChildIterator() { current_op_ = nullptr; }

// Fetches one row of data from the iterator.  Overrides the base class.  This one fetches
// only from the child/worker id as given from the constructor.
Status ChildIterator::FetchNextTensorRow(TensorRow *out_row) {
  // Common code init and error checking in the base class.
  RETURN_IF_NOT_OK(IteratorBase::FetchNextTensorRow(out_row));

  // Once eof is handled, always return empty row.  Class must be destroyed and recreated if you
  // want to iterate again.
  if (eof_handled_) {
    std::string err = "EOF buffer encountered. Users try to fetch data beyond the specified number of epochs.";
    RETURN_STATUS_UNEXPECTED(err);
  }

  // Check if we need to get a new DataBuffer to iterate.
  if (curr_buffer_ == nullptr || curr_buffer_->NumRows() == 0) {
    // GetNextInput() depends on current_op's EoeReceived. So, EOE buffer might be already be handled and
    // this child iterator might not see EOE buffer.
    RETURN_IF_NOT_OK(current_op_->GetNextInput(&curr_buffer_, worker_id_, child_idx_));

    // If an eoe is picked up here, we simply return an empty vector and it's up to the
    // caller to decide what it wants to do next.
    if (curr_buffer_->eoe()) {
      MS_LOG(DEBUG) << "Child iterator picked up EOE.";
      end_epoch_ = true;
      return Status::OK();
    } else {
      end_epoch_ = false;
    }

    if (curr_buffer_->eof()) {
      MS_LOG(DEBUG) << "Child iterator picked up EOF.";
      eof_handled_ = true;
      return Status::OK();
    }
  }

  // If we got this far, now it's time to pop that next row for return to caller
  RETURN_IF_NOT_OK(curr_buffer_->PopRow(out_row));

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
  // else we drain until eoe or eof, eof here is for sanity check
  while (!curr_buffer_->eoe() && !curr_buffer_->eof()) {
    RETURN_IF_NOT_OK(current_op_->GetNextInput(&curr_buffer_, worker_id_, child_idx_));
  }
  if (curr_buffer_->eof()) {
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
