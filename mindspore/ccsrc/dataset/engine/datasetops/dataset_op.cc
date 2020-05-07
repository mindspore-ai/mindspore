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
#include "dataset/engine/datasetops/dataset_op.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <string>

#include "dataset/engine/execution_tree.h"
#include "dataset/engine/datasetops/device_queue_op.h"
#include "dataset/engine/data_buffer.h"
#include "dataset/engine/db_connector.h"

#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
// Constructor
DatasetOp::DatasetOp(int32_t op_connector_size)
    : oc_queue_size_(op_connector_size),
      operator_id_(kInvalidOperatorId),
      tree_(nullptr),
      state_(OpState::kDeOpIdle),
      op_ctrl_flags_(kDeOpNone),
      first_fetch_(true) {
  // The operator starts out with an invalid operator id.  The only way to
  // get it out of invalid state is to assign the operator to an execution tree.
}

// Adds a operator to become our child.
Status DatasetOp::AddChild(std::shared_ptr<DatasetOp> child) {
  if (std::dynamic_pointer_cast<DeviceQueueOp>(child) != nullptr) {
    std::string err_msg("DeviceQueueOp cannot be added as a child, DeviceQueueOp must be a root node");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (operator_id_ == kInvalidOperatorId) {
    std::string err_msg(
      "Cannot add child node.  Tree node connections can only"
      "be made if the node belongs to a tree.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // disallow relationships with other trees
  if (tree_ != child->tree_) {
    std::string err_msg(
      "Cannot add child node.  Tree node connections can only be made if both nodes belong to the same tree.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  child_.push_back(child);
  child->AddParent(this);
  return Status::OK();
}

// Adds a parent operator to this operator
void DatasetOp::AddParent(const DatasetOp *parent) { parent_.push_back(parent); }

// Getter function to get a shared pointer to our childAdds a operator to become our child.
std::shared_ptr<DatasetOp> DatasetOp::child(int32_t child_index) const {
  DS_ASSERT(child_index < static_cast<int>(child_.size()));
  // Return a shared pointer
  return child_[child_index];
}

// Creates the connector within this operator
void DatasetOp::CreateConnector(int32_t num_producers, int32_t num_consumers) {
  MS_LOG(INFO) << "Creating connector in tree operator: " << operator_id_ << ". Producer: " << num_producers
               << ". Consumer: " << num_consumers << ".";
  if (oc_queue_size_ > 0) {
    out_connector_ = std::make_unique<DbConnector>(num_producers,  // The number of producers
                                                   num_consumers,  // Only one consumer (the training App)
                                                   oc_queue_size_);
  } else {
    // Some op's may choose not to have an output connector
    MS_LOG(INFO) << "Bypassed connector creation for tree operator: " << operator_id_ << ".";
    out_connector_ = nullptr;
  }
}

// A print method typically used for debugging.  showAll of true will recursively descend to child prints
void DatasetOp::Print(std::ostream &out, bool show_all) const {
  // When show_all is false, we display a 1 liner piece of text for the op.
  // When show_all is true, we display more detailed output for the op.
  // Derived printers should show their own header info, then call base class printer, followed by
  // derived-specific items.
  // For now, the base class doesn't have any summary info to show so it's a no-op in that case.
  if (show_all) {
    // The detailed display will show common base class info of the op.  Allow the derived class to print
    // it's own id and name though as the first line.
    out << "\nNumber of children     : " << child_.size();
    for (size_t i = 0; i < child_.size(); i++) {
      out << "\n  Child[" << i << "] id: " << child_[i]->id();
    }
    out << "\nNumber of parents      : " << parent_.size();
    for (size_t i = 0; i < parent_.size(); i++) {
      out << "\n  Parent[" << i << "] id: " << parent_[i]->id();
    }
    out << "\nConnector queue size   : " << oc_queue_size_ << "\nOperator control flags : 0x" << std::hex
        << std::setw(8) << std::setfill('0') << op_ctrl_flags_ << std::dec << std::setfill(' ');
  }
}

// Gets the next buffer from the given child
Status DatasetOp::GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe) {
#if defined(_WIN32) || defined(_WIN64)
  RETURN_IF_NOT_OK(out_connector_->PopWithRetry(static_cast<int>(worker_id), p_buffer, retry_if_eoe));
#else
  std::unique_ptr<DataBuffer> next_buff;
  // pop is a blocked call and will throw an interruption if the whole group shuts down.
  RETURN_IF_NOT_OK(out_connector_->PopWithRetry(static_cast<int>(worker_id), &next_buff, retry_if_eoe));

  *p_buffer = std::move(next_buff);
#endif
  return Status::OK();
}

// Gets the next buffer from the given child .  This function also has built-in eoe and eof
// message handling so that child classes don't have to manually code pass-through logic when
// those messages are received.
Status DatasetOp::GetNextInput(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, int32_t child_index) {
  if (child_.size() == 0) {
    return this->GetNextBuffer(p_buffer, worker_id);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(child_index < child_.size(), "Child index too big : " + std::to_string(child_index));
  std::shared_ptr<DatasetOp> child = child_[child_index];
  std::unique_ptr<DataBuffer> buf;
  RETURN_IF_NOT_OK(child->GetNextBuffer(&buf, worker_id));
  // Loop until non EOE is received
  while (buf->eoe()) {
    RETURN_IF_NOT_OK(EoeReceived(worker_id));
    if (state_ == OpState::kDeOpIdle) {
      *p_buffer = std::move(buf);
      return Status::OK();
    }
    RETURN_IF_NOT_OK(child->GetNextBuffer(&buf, worker_id));
  }
  // Check if the last buf is next eof
  if (buf->eof()) {
    RETURN_IF_NOT_OK(EofReceived(worker_id));
  }
  *p_buffer = std::move(buf);
  return Status::OK();
}

// Performs handling for when an eoe message is received.
// The base class implementation simply flows the eoe message to output. Derived classes
// may override if they need to perform special eoe handling.
Status DatasetOp::EoeReceived(int32_t worker_id) {
  std::unique_ptr<DataBuffer> eoe_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE);
  return (out_connector_->Add(static_cast<int>(worker_id), std::move(eoe_buffer)));
}

// Performs handling for when an eof message is received.
// The base class implementation simply flows the eof message to output. Derived classes
// may override if they need to perform special eof handling.
Status DatasetOp::EofReceived(int32_t worker_id) {
  std::unique_ptr<DataBuffer> eof_buffer = std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF);
  return (out_connector_->Add(static_cast<int>(worker_id), std::move(eof_buffer)));
}

// During tree prepare phase, operators may have specific pre-operations to perform depending on
// their role.
Status DatasetOp::PrepareNodePreAction() {
  if (BitTest(tree_->PrepareFlags(), ExecutionTree::kDePrepRepeat)) set_control_flag(kDeOpRepeated);
  return Status::OK();
}
// During tree prepare phase, operators may have specific post-operations to perform depending on
// their role.
Status DatasetOp::PrepareNodePostAction() {
  // If this op does not have any children and it is in a repeat path of the tree...
  if (child_.empty() && BitTest(op_ctrl_flags_, kDeOpRepeated)) {
    // push ourselves onto the tree repeat stack.  Later, the repeat operator
    // above us will consume them.
    tree_->AddToRepeatStack(shared_from_this());
  }

  // Creating Connector object for each op.
  // The consumer of the root node is assumed to be one thread.
  // If multiple threads are consuming from the root node, they will get the ordered data in round robin fashion.
  if (parent_.empty()) {
    this->CreateConnector(num_producers(), 1);
  } else {
    this->CreateConnector(num_producers(), parent_[0]->num_consumers());
  }
  if (out_connector_) {
    RETURN_IF_NOT_OK(out_connector_->Register(tree_->AllTasks()));
  }
  RETURN_IF_NOT_OK(this->RegisterWorkerConnectors());
  return Status::OK();
}

// Getter function.  Base class does not have any special flags setting.
uint32_t DatasetOp::PrepareFlags() const { return ExecutionTree::kDePrepNone; }

// Derived classes may implement the reset function if the operator is stateful and needs
// specific reset handling that is not contained in this common code version of the reset.
Status DatasetOp::Reset() {
  state_ = OpState::kDeOpRunning;
  return Status::OK();
}

// gives a string output for the column map for handy debug printing
std::string DatasetOp::ColumnNameMapAsString() const {
  std::string outStr = "Column name id map: ";
  for (auto &it : column_name_id_map_) {
    outStr += (" " + it.first + ":" + std::to_string(it.second));
  }
  return outStr;
}

// A helper function for providing assignment of the column name map.
// This grabs the map from child 0 and assigns it into this op.
// Can only be used if number of children is 1.
Status DatasetOp::AssignColMapFromChild() {
  if (child_.size() > 1) {
    RETURN_STATUS_UNEXPECTED("Assigning column name map from child only works for single-child operators.");
  }
  // Assign the correct column name map to this op by taking it from the input child.
  // This must be done AFTER the first fetch, but only needs to be done once by the first worker to
  // do the first fetch.
  if (first_fetch_) {
    // If there was a single worker, or this is being called from a master thread in a parallel op,
    // then the mutex is not really needed here, although it's harmless.
    std::unique_lock<std::mutex> lock(column_name_map_mutex_);
    // If the map has not been set up yet, then we are the first one in to set it up. The first_fetch_ (dirty read)
    // bool allows us to avoid acquiring the lock if the map has already been set.
    if (column_name_id_map_.empty()) {
      column_name_id_map_ = child_[0]->column_name_id_map();
      first_fetch_ = false;
      if (column_name_id_map_.empty()) {
        RETURN_STATUS_UNEXPECTED("Child column name map cannot be empty!");
      }
    }
    MS_LOG(INFO) << "Setting column map after first fetch:\n" << DatasetOp::ColumnNameMapAsString();
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
