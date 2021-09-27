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
#include "minddata/dataset/engine/datasetops/barrier_op.h"

#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/engine/db_connector.h"
#include "minddata/dataset/core/config_manager.h"

namespace mindspore {
namespace dataset {
// Construct BarrierOp here, local variables initialized in operator due to tree construction restrictions
BarrierOp::BarrierOp(int32_t op_connector_size, const std::string &condition_name, py::function condition_func)
    : PipelineOp(op_connector_size),
      clean_up_(false),
      eof_(false),
      condition_name_(condition_name),
      condition_function_(std::move(condition_func)) {}

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
    RETURN_IF_NOT_OK(prepare());
    // read the first row
    TensorRow new_row;
    RETURN_IF_NOT_OK(getNextTensorRow(&new_row));

    // If an eof got picked up during the above prepare, then we're done
    if (eof_) {
      break;
    }

    while (!clean_up_) {
      // 2 Block
      RETURN_IF_NOT_OK(blockCond());

      MS_LOG(DEBUG) << "Barrier operator finished one row, pushing, cols " << new_row.size() << ", map "
                    << column_name_id_map_.size() << ".";
      RETURN_IF_NOT_OK(out_connector_->Add(std::move(new_row)));
      RETURN_IF_NOT_OK(getNextTensorRow(&new_row));
    }

    if (clean_up_) {
      MS_LOG(DEBUG) << "Barrier operator sending epoch ending signal.";
      // 3 Send the eoe up.
      RETURN_IF_NOT_OK(out_connector_->SendEOE());
    }
  }
  // 4 handle eof
  // propagate eof here.
  MS_LOG(INFO) << "Barrier operator got EOF, propagating.";
  RETURN_IF_NOT_OK(out_connector_->SendEOF());
  return Status::OK();
}

// Handles preprocessing of the main loop, used when starting new epoch
Status BarrierOp::prepare() {
  MS_LOG(DEBUG) << "Barrier operator prepares for new epoch.";
  clean_up_ = false;
  // the update code below shouldn't do anything bad if the column name already exists.
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

// fetches next Barrier row
Status BarrierOp::getNextTensorRow(TensorRow *new_row) {
  RETURN_UNEXPECTED_IF_NULL(new_row);
  // iterate over all iterators and generate a row
  RETURN_IF_NOT_OK((child_iterator_)->FetchNextTensorRow(new_row));
  // add each new row to iterator, check if row is empty, if row from iterator is empty return empty row
  if (new_row->empty()) {
    // If we did not get a row from any of the children, then it's the end of an epoch and we can move
    // to drain state.
    MS_LOG(INFO) << "Barrier operator child iterator produced empty row.";
    clean_up_ = true;
    // If we picked up an eof here, then we are completely done.
    if ((child_iterator_)->EofHandled()) {
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
