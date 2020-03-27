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
#include "dataset/engine/dataset_iterator.h"
#include <utility>
#include "dataset/core/data_type.h"
#include "dataset/core/tensor.h"
#include "dataset/core/tensor_shape.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/util/status.h"
#include "dataset/engine/datasetops/dataset_op.h"

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
  RETURN_IF_NOT_OK(FetchNextTensorRow(&curr_row));

  // Return empty map if there's no data
  if (curr_row.empty()) {
    return Status::OK();
  }

  // Populate the out map from the row and return it
  for (auto colMap : col_name_id_map_) {
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

// Constructor of the DatasetIterator
DatasetIterator::DatasetIterator(std::shared_ptr<ExecutionTree> exe_tree) : IteratorBase(), root_(exe_tree->root()) {}

DatasetIterator::~DatasetIterator() = default;

// Fetches one row of data from the iterator.  Overrides the base class.  This one fetches
// from the tree root node directly.
Status DatasetIterator::FetchNextTensorRow(TensorRow *out_row) {
  // Common code init and error checking in the base class.
  RETURN_IF_NOT_OK(IteratorBase::FetchNextTensorRow(out_row));

  // Once eof is handled, always return empty row.  Class must be destroyed and recreated if you
  // want to iterate again.
  if (eof_handled_) {
    return Status::OK();
  }

  // Check if we need to get a new DataBuffer to iterate.
  if (curr_buffer_ == nullptr || curr_buffer_->NumRows() == 0) {
    col_name_id_map_.clear();
    RETURN_IF_NOT_OK(root_->GetNextBuffer(&curr_buffer_));

    // Since GetNextBuffer was used rather than GetNextInput(), it means we need to manually
    // handle eoe and eof messages here.
    //
    // An eoe buffer means we have iterated fully to the end of the tree.
    // An eoe buffer will be immediately followed by an eof buffer, which signals the shutdown of
    // all operators.
    if (curr_buffer_->eoe()) {
      MS_LOG(INFO) << "End of data iteration. Fetch eof and then return empty row.";

      // Before returning the last empty vector, fetch the eof buffer which should be the last
      // buffer, and then free it.
      RETURN_IF_NOT_OK(root_->GetNextBuffer(&curr_buffer_));

      if (!curr_buffer_->eof()) {
        RETURN_STATUS_UNEXPECTED("Non-eof after getting eoe in iterator!");
      }
      eof_handled_ = true;
      curr_buffer_.reset();  // explicitly free the eof buffer

      return Status::OK();
    }

    if (curr_buffer_->eof()) {
      // An eof by itself, without being preceded by an eoe, is possible if a repeat operator
      // exists below us in the stack. Repeat operator eats eoe's but eventually allows the
      // flow of an eof up the pipeline by itself.
      eof_handled_ = true;
      curr_buffer_.reset();  // explicitly free the eof buffer
      return Status::OK();
    }

    col_name_id_map_ = curr_buffer_->column_name_map();
  }

  // If we got this far, now it's time to pop that next row for return to caller
  RETURN_IF_NOT_OK(curr_buffer_->PopRow(out_row));

  return Status::OK();
}

Status DatasetIterator::GetOutputShapes(std::vector<TensorShape> *out_shapes) {
  if (out_shapes == nullptr) {
    RETURN_STATUS_UNEXPECTED("Null output shape argument");
  }
  if (device_queue_row_.empty()) {
    RETURN_IF_NOT_OK(FetchNextTensorRow(&device_queue_row_));
  }
  for (auto ts : device_queue_row_) {
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
  for (auto ts : device_queue_row_) {
    out_types->push_back(ts->type());
  }
  return Status::OK();
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
    return Status::OK();
  }

  // Check if we need to get a new DataBuffer to iterate.
  if (curr_buffer_ == nullptr || curr_buffer_->NumRows() == 0) {
    col_name_id_map_.clear();
    RETURN_IF_NOT_OK(current_op_->GetNextInput(&curr_buffer_, worker_id_, child_idx_));

    // Unlike the DatasetIterator, this child iterator does not quit after eoe.
    // Instead, if an eoe is picked up here, we simply return an empty vector and it's up to the
    // caller to decide what it wants to do next.
    if (curr_buffer_->eoe()) {
      MS_LOG(INFO) << "Child iterator picked up EOE.";
      end_epoch_ = true;
      return Status::OK();
    }

    if (curr_buffer_->eof()) {
      MS_LOG(INFO) << "Child iterator picked up EOF.";
      eof_handled_ = true;
      return Status::OK();
    }

    col_name_id_map_ = curr_buffer_->column_name_map();
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
    MS_LOG(INFO) << "No operation drain, already at end of epoch.";
    return Status::OK();
  }
  MS_LOG(INFO) << "Child draining buffers until eoe.";
  // else we drain until eoe or eof, eof here is for sanity check
  while (!curr_buffer_->eoe() && !curr_buffer_->eof()) {
    RETURN_IF_NOT_OK(current_op_->GetNextInput(&curr_buffer_, worker_id_, child_idx_));
  }
  if (curr_buffer_->eof()) {
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, "Child iterator picked up EOF in drain.");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
