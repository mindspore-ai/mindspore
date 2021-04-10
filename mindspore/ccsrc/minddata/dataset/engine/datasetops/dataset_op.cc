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
#include "minddata/dataset/engine/datasetops/dataset_op.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <utility>
#include <string>
#include <algorithm>

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/data_buffer.h"
#include "minddata/dataset/engine/db_connector.h"
#ifndef ENABLE_ANDROID
#include "utils/system/crc32c.h"
#include "utils/log_adapter.h"
#else
#include "mindspore/lite/src/common/log_adapter.h"
#endif

namespace mindspore {
namespace dataset {
// Constructor
DatasetOp::DatasetOp(int32_t op_connector_size, std::shared_ptr<SamplerRT> sampler)
    : oc_queue_size_(op_connector_size),
      sampler_(sampler),
      operator_id_(kInvalidOperatorId),
      tree_(nullptr),
      state_(OpState::kDeOpIdle),
      op_total_repeats_(kInfiniteRepeat),
      op_num_repeats_per_epoch_(kInfiniteRepeat),
      op_current_repeats_(0),
      op_current_epochs_(0),
      out_connector_(nullptr),
      dataset_size_(-1),
      num_classes_(-1) {
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

Status DatasetOp::RemoveChild(std::shared_ptr<DatasetOp> child) {
  if (operator_id_ == kInvalidOperatorId) {
    std::string err_msg(
      "Cannot remove child node.  Tree node connections can only"
      "be made if the node belongs to a tree.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // disallow relationships with other trees
  if (tree_ != child->tree_) {
    std::string err_msg(
      "Cannot remove child node.  Tree node connections can only be made if both nodes belong to the same tree.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  child_.erase(std::remove(child_.begin(), child_.end(), child), child_.end());
  child->RemoveParent(this);
  return Status::OK();
}

Status DatasetOp::InsertAsParent(std::shared_ptr<DatasetOp> to_add) {
  for (auto &prev_parent : this->parent_) {
    RETURN_IF_NOT_OK(prev_parent->RemoveChild(shared_from_this()));
    RETURN_IF_NOT_OK(prev_parent->AddChild(to_add));
  }
  RETURN_IF_NOT_OK(to_add->AddChild(shared_from_this()));
  if (tree_->root()->id() == this->id()) {
    RETURN_IF_NOT_OK(tree_->AssignRoot(to_add));
  }
  return Status::OK();
}
// Removes child operator in this operator.
Status DatasetOp::RemoveChildren() {
  for (const auto &child : child_) {
    child->RemoveParent(this);
  }
  child_.clear();

  return Status::OK();
}

// Adds a parent operator to this operator
void DatasetOp::AddParent(DatasetOp *parent) { parent_.push_back(parent); }

// Removes a parent operator from this operator
void DatasetOp::RemoveParent(const DatasetOp *parent) {
  parent_.erase(std::remove(parent_.begin(), parent_.end(), parent), parent_.end());
}

// Removes this node from the tree and connects it's parent/child together
Status DatasetOp::Remove() {
  if (parent_.size() > 1) {
    std::string err_msg("No support for op removal if the operator has more than one parent");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (child_.size() > 1) {
    std::string err_msg("No support for op removal if the operator has more than one child");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Scenario's when removing node B:
  // A -> B -> C
  // A -> B
  // B -> C
  //
  // If we remove B, then first take our child A and update it's parent to be C
  // It's possible the parent is null if we are the root node being removed.
  if (!child_.empty()) {
    // If we have a parent, then assign child's parent to point to our parent.
    if (!parent_.empty()) {
      CHECK_FAIL_RETURN_UNEXPECTED(parent_[0]->Children().size() == 1,
                                   "Removing a node whose parent has more than 1 child is not supported.");
      child_[0]->parent_[0] = parent_[0];
    } else {
      // We don't have a parent, so we are the root node being removed.
      // clear the parent list of our child so that it becomes the new root.
      child_[0]->parent_.clear();
      RETURN_IF_NOT_OK(tree_->AssignRoot(child_[0]));
    }
  }

  // Next, if we had a parent, then set it's child to be our child.
  if (!parent_.empty()) {
    // if we have a child, then set our parent to point to it
    if (!child_.empty()) {
      parent_[0]->child_[0] = child_[0];
    } else {
      // We don't have a child, so clear the child list of the current
      // parent because it will be empty once we are removed.
      parent_[0]->child_.clear();
    }
  }

  // Finally, clear "this" op's parent and child pointers since we have just
  // disconnected it from the tree and invalidate it's fields.
  child_.clear();
  parent_.clear();
  operator_id_ = kInvalidOperatorId;
  tree_ = nullptr;

  return Status::OK();
}

// Getter function to get a shared pointer to our child
std::shared_ptr<DatasetOp> DatasetOp::child(int32_t child_index) const {
  std::shared_ptr<DatasetOp> return_op = nullptr;
  if (child_.empty()) {
    return return_op;
  }
  MS_ASSERT(child_index < static_cast<int>(child_.size()));
  // Return a shared pointer
  return child_[child_index];
}

// Getter function to get the parent pointer
void DatasetOp::Parent(DatasetOp **parent, int32_t parent_index) const {
  if (parent_.empty()) {
    // common case if this is a root node
    *parent = nullptr;
  } else {
    MS_ASSERT(parent_index < static_cast<int>(parent_.size()));
    *parent = parent_[parent_index];
  }
}

// Getter function to get all of our parents.
std::vector<DatasetOp *> DatasetOp::parents() const { return parent_; }

// Creates the connector within this operator
void DatasetOp::CreateConnector(int32_t num_producers, int32_t num_consumers) {
  MS_LOG(DEBUG) << "Creating connector in tree operator: " << operator_id_ << ". Producer: " << num_producers
                << ". Consumer: " << num_consumers << ".";
  if (oc_queue_size_ > 0) {
    out_connector_ = std::make_unique<DbConnector>(num_producers,  // The number of producers
                                                   num_consumers,  // Only one consumer (the training App)
                                                   oc_queue_size_);
  } else {
    // Some op's may choose not to have an output connector
    MS_LOG(DEBUG) << "Bypassed connector creation for tree operator: " << operator_id_ << ".";
    out_connector_ = nullptr;
  }
}

// A print method typically used for debugging.  showAll of true will recursively descend to child prints
void DatasetOp::Print(std::ostream &out, bool show_all) const {
  // When show_all is false, we display a 1 liner piece of text for the op.
  // When show_all is true, we display more detailed output for the op.
  // Derived printers should show their own header info, then call base class printer, followed by
  // derived-specific items.

  // Always show the id and name as first line regardless if this summary or detailed print
  out << "(" << std::setw(2) << operator_id_ << ") <" << Name() << ">:";

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
    out << "\nConnector queue size   : " << oc_queue_size_ << "\nTotal repeats : " << op_total_repeats_
        << "\nNumber repeats per epoch : " << op_num_repeats_per_epoch_;
    if (sampler_) {
      out << "\nSampler:\n";
      sampler_->SamplerPrint(out, show_all);
    }
  }
}

Status DatasetOp::GetNextRow(TensorRow *const row) {
  RETURN_UNEXPECTED_IF_NULL(child_[0]);
  return child_[0]->GetNextRow(row);
}

// Gets the next buffer from the given child
Status DatasetOp::GetNextBuffer(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, bool retry_if_eoe) {
  // pop is a blocked call and will throw an interruption if the whole group shuts down.
  RETURN_IF_NOT_OK(out_connector_->PopWithRetry(static_cast<int>(worker_id), p_buffer, retry_if_eoe));
  return Status::OK();
}

// Gets the next buffer from the given child .  This function also has built-in eoe and eof
// message handling so that child classes don't have to manually code pass-through logic when
// those messages are received.
Status DatasetOp::GetNextInput(std::unique_ptr<DataBuffer> *p_buffer, int32_t worker_id, int32_t child_index) {
  if (child_.size() == 0) {
    return this->GetNextBuffer(p_buffer, worker_id);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(child_index < child_.size(),
                               "Invalid data, child index too big : " + std::to_string(child_index));
  std::shared_ptr<DatasetOp> child = child_[child_index];
  std::unique_ptr<DataBuffer> buf;
  RETURN_IF_NOT_OK(child->GetNextBuffer(&buf, worker_id));
  // Loop until non EOE is received
  while (buf->eoe()) {
    UpdateRepeatAndEpochCounter();
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

// Gets the number of classes
Status DatasetOp::GetNumClasses(int64_t *num_classes) {
  if (child_.size() == 1) {
    return child_[0]->GetNumClasses(num_classes);
  } else if (child_.size() > 1) {
    // It is okay for dataset to have more than 1 child, GetNumClasses shouldn't fail in this case.
    // This is done mostly for cache, which injects cache lookup/merge operators. Cache path will
    // always be in front of the child_ structure, so we get num classes from the last child.
    return child_[child_.size() - 1]->GetNumClasses(num_classes);
  } else {
    // when num classes isn't found, the default behavior is to return -1
    MS_LOG(WARNING) << "Num classes not defined for : " << Name();
    *num_classes = -1;
    return Status::OK();
  }
}

Status DatasetOp::GetClassIndexing(std::vector<std::pair<std::string, std::vector<int32_t>>> *output_class_indexing) {
  if (child_.size() == 1) {
    return child_[0]->GetClassIndexing(output_class_indexing);
  } else if (child_.size() > 1) {
    // It is okay for dataset to have more than 1 child, GetClassIndexing shouldn't fail in this case.
    // This is done mostly for cache, which injects cache lookup/merge operators. Cache path will
    // always be in the front of the child_ structure, so we get data from the last child.
    return child_[child_.size() - 1]->GetClassIndexing(output_class_indexing);
  } else {
    *output_class_indexing = {};
    RETURN_STATUS_UNEXPECTED("Trying to get class index from leaf node, missing override");
  }
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

// During tree prepare phase, operators may have specific post-operations to perform depending on their role.
Status DatasetOp::PrepareOperator() {
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

  // Generate the column name map for the current op.
  RETURN_IF_NOT_OK(this->ComputeColMap());

  return Status::OK();
}

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

// Computing the assignment of the column name map.
// This just inherits the column map from its first child, can only be used if the number of children is 1.
// Operations changing the column map must overwrite this function.
Status DatasetOp::ComputeColMap() {
  if (child_.size() > 1) {
    RETURN_STATUS_UNEXPECTED("Assigning column name map from child only works for single-child operators.");
  }
  if (column_name_id_map_.empty()) {
    column_name_id_map_ = child_[0]->column_name_id_map();
    if (column_name_id_map_.empty()) {
      RETURN_STATUS_UNEXPECTED("Child column name map cannot be empty!");
    }
    MS_LOG(DEBUG) << "Setting column map:\n" << DatasetOp::ColumnNameMapAsString();
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

// Getter for the sampler, and it also removes the sampler from the op
Status DatasetOp::FetchRemoveSampler(std::shared_ptr<SamplerRT> *sampler) {
  *sampler = sampler_;  // It's okay if it sampler_ points to nullptr
  sampler_.reset();     // clear our member-copy of this pointer.  We no longer have this sampler
  return Status::OK();
}

#ifndef ENABLE_ANDROID
uint32_t DatasetOp::GenerateCRC(const std::shared_ptr<DatasetOp> &op) {
  std::stringstream ss;
  op->tree_->Print(ss, op);
  std::string ss_str = ss.str();

  // Filter out the Num workers field when generating the check sum
  ss_str = std::regex_replace(ss_str, std::regex("Num workers.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex("\\[workers.*?\\]"), "");
  ss_str = std::regex_replace(ss_str, std::regex("Connector queue size.*\n"), "");

  // Filter out tcp/ip information
  ss_str = std::regex_replace(ss_str, std::regex("Hostname.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex("Port.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex("Number of rpc workers.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex("Prefetch size.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex("Local client support.*\n"), "");

  // Filter out Number of rows when generating the check sum
  ss_str = std::regex_replace(ss_str, std::regex("Number of rows.*\n"), "");

  // Filter out the Operator control flags field when generating the check sum
  ss_str = std::regex_replace(ss_str, std::regex("Operator control flags.*\n"), "");

  // Filter out the Device id field to allow cache sharing for a distributed run of the same pipeline
  ss_str = std::regex_replace(ss_str, std::regex("Device id.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex("device_id.*\n"), "");

  // Filter out the operator id field
  ss_str = std::regex_replace(ss_str, std::regex(" *Parent.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex(" *Child.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex(R"(\(\s*\d+?\))"), "");

  // Doesn't matter whether there is any parent node above CacheOp or not.
  ss_str = std::regex_replace(ss_str, std::regex("Number of parents.*\n"), "");

  // Filter out shuffle seed from ShuffleOp
  ss_str = std::regex_replace(ss_str, std::regex("Shuffle seed.*\n"), "");

  // Filter out the total repeats and number repeats per epoch field
  ss_str = std::regex_replace(ss_str, std::regex("Total repeats.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex("Number repeats per epoch.*\n"), "");

  // The Cache crc and Server cache id field is different when creating new cache_client and re-using the same
  // cache_client later. So we filter out these two fields to allow cache sharing.
  ss_str = std::regex_replace(ss_str, std::regex("Cache crc.*\n"), "");
  ss_str = std::regex_replace(ss_str, std::regex("Server cache id.*\n"), "");

  MS_LOG(DEBUG) << "Printing the tree for generating crc:\n" << ss_str;

  uint32_t cache_crc = system::Crc32c::GetMaskCrc32cValue(ss_str.c_str(), ss_str.length());
  return cache_crc;
}
#endif

void DatasetOp::UpdateRepeatAndEpochCounter() {
  op_current_repeats_++;
  if (op_current_repeats_ % op_num_repeats_per_epoch_ == 0) op_current_epochs_++;
  MS_LOG(DEBUG) << Name() << " current repeats: " << op_current_repeats_ << ", current epochs: " << op_current_epochs_;
}

int64_t DatasetOp::GetTreeBatchSize() {
  if (child_.size() == 1) {
    return child_[0]->GetTreeBatchSize();
  } else if (child_.size() > 1) {
    // It is okay for dataset to have more than 1 child, GetBatchSize shouldn't fail in this case.
    // This is done mostly for cache, which injects cache lookup/merge operators. Cache path will
    // always be in front of the child_ structure, so we get data from the last child.
    return child_[child_.size() - 1]->GetTreeBatchSize();
  } else {
    return 1;
  }
}

int64_t DatasetOp::GetTreeRepeatCount() {
  if (child_.size() == 1) {
    return child_[0]->GetTreeRepeatCount();
  } else if (child_.size() > 1) {
    // It is okay for dataset to have more than 1 child, GetRepeatCount shouldn't fail in this case.
    // This is done mostly for cache, which injects cache lookup/merge operators. Cache path will
    // always be in front of the child_ structure, so we get data from the last child.
    return child_[child_.size() - 1]->GetTreeRepeatCount();
  } else {
    return 1;
  }
}
}  // namespace dataset
}  // namespace mindspore
