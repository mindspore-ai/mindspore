/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/datasetops/barrier_op.h"
#include <iomanip>
#include <utility>
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/global_context.h"

namespace mindspore {
namespace dataset {
BarrierOp::Builder::Builder() {
  // Some arguments to the BarrierOp constructor have a default argument that is taken
  // from the client config.
  // The user may choose to change these values for the construction of the BarrierOp by
  // using the various builder set methods.

  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_rows_per_buffer_ = cfg->rows_per_buffer();
  builder_op_connector_size_ = cfg->op_connector_size();
}

Status BarrierOp::Builder::SanityCheck() const { return Status::OK(); }

Status BarrierOp::Builder::Build(std::shared_ptr<BarrierOp> *ptr) {
  RETURN_IF_NOT_OK(SanityCheck());
  *ptr = std::make_shared<BarrierOp>(builder_rows_per_buffer_, builder_op_connector_size_, builder_condition_name_,
                                     builder_condition_func_);
  return Status::OK();
}

// Construct BarrierOp here, local variables initialized in operator due to tree construction restrictions
BarrierOp::BarrierOp(int32_t rows_per_buffer, int32_t op_connector_size, const std::string &condition_name,
                     py::function condition_func)
    : PipelineOp(op_connector_size),
      rows_per_buffer_(rows_per_buffer),
      buffer_id_(0),
      clean_up_(false),
      eof_(false),
      condition_name_(condition_name),
      condition_function_(condition_func) {}

// destructor
BarrierOp::~BarrierOp() {}

// Entry point for Barrier, called by launch()
Status BarrierOp::operator()() {
  // The children_num_ parameter needs to be put here
  // Synchronize with TaskManager once the thread is created.
  TaskManager::FindMe()->Post();

  // create child iterator, right now this barrier is a pipeline operator
  const int32_t worker_id = 0;
  const int32_t child_idx = 0;
  child_iterator_ = std::make_unique<ChildIterator>(this, worker_id, child_idx);

  // Loop until eof is true
  while (!eof_) {
    // Create new table to put the new tensor rows
    std::unique_ptr<TensorQTable> curr_table = std::make_unique<TensorQTable>();
    RETURN_IF_NOT_OK(prepare(curr_table.get()));

    // If an eof got picked up during the above prepare, then we're done
    if (eof_) {
      break;
    }

    // we have to output new buffer with possibly different buffer size, possibly one row
    while (!clean_up_) {
      // 1. If a previous loop iteration sent the current table out, then create a new one.

      if (curr_table == nullptr) {
        curr_table = std::make_unique<TensorQTable>();
      }

      // 2 fill the table.  Note: clean_up mode might get turned on if epoch is finished
      RETURN_IF_NOT_OK(fillBuffer(curr_table.get()));

      // 3 create and update buffer and send it to the out connector
      if (!curr_table->empty()) {
        std::unique_ptr<DataBuffer> curr_buffer = std::make_unique<DataBuffer>(buffer_id_, DataBuffer::kDeBFlagNone);
        curr_buffer->set_tensor_table(std::move(curr_table));
        MS_LOG(DEBUG) << "Barrier operator finished one buffer, pushing, rows " << curr_buffer->NumRows() << ", cols "
                      << curr_buffer->NumCols() << ", map " << column_name_id_map_.size() << ".";
        RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(curr_buffer)));
        buffer_id_++;
      }
    }

    // 4 handle drain state.
    if (clean_up_) {
      MS_LOG(DEBUG) << "Barrier operator sending epoch ending signal.";
      // Send the eoe up.
      RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE))));
    }
  }
  // 5 handle eof
  // propagate eof here.
  MS_LOG(INFO) << "Barrier operator got EOF, propagating.";
  RETURN_IF_NOT_OK(out_connector_->Add(0, std::move(std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF))));
  return Status::OK();
}

// Handles preprocessing of the main loop, used when starting new epoch
Status BarrierOp::prepare(TensorQTable *const table) {
  MS_LOG(DEBUG) << "Barrier operator prepares for new epoch.";
  clean_up_ = false;
  buffer_id_ = 0;
  if (table == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "BarrierOp prepare phase requires a tensor table.");
  }
  // fill initial row
  TensorRow new_row = {};
  // use iterator to get next row and invoke pyfunc wait
  RETURN_IF_NOT_OK(getNextTensorRow(&new_row));

  // If the first row fetching resulted in eof, then we are done.
  if (eof_) {
    return Status::OK();
  }
  if (new_row.empty()) {
    // This epoch is empty
    return Status::OK();
  }
  // Pack this first row into our tensor table
  // first row we also have to check if we should block
  RETURN_IF_NOT_OK(blockCond());

  table->push_back(std::move(new_row));

  // the update code below shouldn't do anything bad if the column name already exists.
  return Status::OK();
}

// fillBuffer always expects a new table to fill
Status BarrierOp::fillBuffer(TensorQTable *const table) {
  if (table == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "BarrierOp fillBuffer null table pointer.");
  }
  TensorRow new_row = {};
  while (table->size() < static_cast<size_t>(rows_per_buffer_)) {
    RETURN_IF_NOT_OK(getNextTensorRow(&new_row));
    // Early exit the loop if we got empty row from any of our child iterations
    if (new_row.empty()) {
      return Status::OK();
    }
    // else we got a row so pack it into the tensor table.
    RETURN_IF_NOT_OK(blockCond());

    table->push_back(std::move(new_row));
  }
  return Status::OK();
}

// function executes a py_func and blocks until condition becomes true.
Status BarrierOp::blockCond() {
  {
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    // we have condition name, however the flexibility is in python today
    try {
      // Invoke python function
      py::object ret_py_obj = condition_function_();
      // Process the return value
      if (!py::isinstance<py::bool_>(ret_py_obj)) {
        return Status(StatusCode::kMDPyFuncException,
                      "Invalid parameter, condition wait function should return true/false.");
      }
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kMDPyFuncException, e.what());
    }
  }
  return Status::OK();
}

// fetches next Barrier buffer row
Status BarrierOp::getNextTensorRow(TensorRow *new_row) {
  // iterate over all iterators and generate a row
  RETURN_IF_NOT_OK((child_iterator_)->FetchNextTensorRow(new_row));
  // add each new row to iterator, check if row is empty, if row from iterator is empty return empty row
  if (new_row->empty()) {
    // If we did not get a row from any of the children, then it's the end of an epoch and we can move
    // to drain state.
    MS_LOG(INFO) << "Barrier operator child iterator produced empty row.";
    clean_up_ = true;
    // If we picked up an eof here, then we are completely done.
    if ((child_iterator_)->eof_handled()) {
      MS_LOG(INFO) << "Barrier operator iterator got EOF.";
      eof_ = true;
    }
    return Status::OK();
  }
  return Status::OK();
}

// A function that prints info about the Operator
void BarrierOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nCondition: " << condition_name_ << "\n\n";
  }
}

// overwrite function and handle eof
Status BarrierOp::EofReceived(int32_t) {
  MS_LOG(DEBUG) << "Barrier operator EOF received, do nothing now.";
  return Status::OK();
}

// overwrite function and handle eoe
Status BarrierOp::EoeReceived(int32_t) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
