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
#include "dataset/engine/execution_tree.h"
#include <iostream>
#include <string>
#include "dataset/engine/datasetops/dataset_op.h"
#include "dataset/engine/datasetops/shuffle_op.h"
#include "dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
// Constructor
ExecutionTree::ExecutionTree() : id_count_(0) {
  tg_ = std::make_unique<TaskGroup>();
  tree_state_ = kDeTStateInit;
  prepare_flags_ = kDePrepNone;
}

// Destructor
ExecutionTree::~ExecutionTree() { (void)tg_->ServiceStop(); }

// Associates a DatasetOp with this tree. This assigns a valid node id to the operator and
// provides it with a link to the tree. A node cannot form any relationships (parent/child) with
// other nodes unless they are associated with the same tree.
Status ExecutionTree::AssociateNode(const std::shared_ptr<DatasetOp> &op) {
  if (tree_state_ != kDeTStateInit && tree_state_ != kDeTStateBuilding) {
    std::string err_msg =
      "Invalid tree state for adding a node. Current state: " + std::to_string(static_cast<int>(tree_state_)) +
      " Expected states: " + std::to_string(static_cast<int>(kDeTStateInit)) + " or " +
      std::to_string(static_cast<int>(kDeTStateBuilding));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Enter the building state if we were not already there
  tree_state_ = kDeTStateBuilding;

  // Assign an id to the operator
  op->set_id(id_count_);
  id_count_++;

  // Assign our tree into the op so that each op has a link back to the tree
  op->set_tree(this);
  return Status::OK();
}

// Sets the root node of the tree
Status ExecutionTree::AssignRoot(const std::shared_ptr<DatasetOp> &op) {
  // Tree must be in building state before we can assign root to it
  if (tree_state_ != kDeTStateBuilding) {
    std::string err_msg =
      "Invalid tree state for assigning a root node. Current state: " + std::to_string(static_cast<int>(tree_state_)) +
      " Expected state: " + std::to_string(static_cast<int>(kDeTStateBuilding));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // If they didn't already call AssociateNode for this node before calling AssignRoot,
  // then do so now.
  if (op->operator_id_ == DatasetOp::kInvalidOperatorId) {
    RETURN_IF_NOT_OK(this->AssociateNode(op));
  }

  // Then add it as the root.
  root_ = op;

  // The tree has an assigned root now and it's ready to be prepared.
  tree_state_ = kDeTStatePrepare;
  return Status::OK();
}

// A print method typically used for debugging
void ExecutionTree::Print(std::ostream &out, bool show_all) const {
  out << "Total number of nodes in the ExecutionTree (may or may not be connected nodes): " << id_count_
      << "\nTree state: " << static_cast<int>(tree_state_) << "\n";
  if (root_ != nullptr) {
    // Just call the printer on the root node.  Each node descends to it's children to print them if
    // showAll is true.
    root_->Print(out, show_all);
  }
}

// Start the execution of the tree
Status ExecutionTree::Launch() {
  // Tree must be built and prepared before it can be launched!
  if (tree_state_ != kDeTStateReady) {
    std::string err_msg =
      "Invalid tree state for launching tree. Current state: " + std::to_string(static_cast<int>(tree_state_)) +
      " Expected state: " + std::to_string(static_cast<int>(kDeTStateReady));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  for (auto itr = this->begin(); itr != this->end(); ++itr) {
    // An inlined operator is one that has an output connector size of 0, and it does not
    // require a thread to execute.  Instead, the work of this operator is executed inlined
    // from the tree node directly above it (or in the case of a root node, it runs from within
    // the launching tree/user thread.  Do not exec any thread for an inlined op.
    itr->state_ = DatasetOp::OpState::kDeOpRunning;
    if (!itr->inlined()) {
      RETURN_IF_NOT_OK(tg_->CreateAsyncTask("Op launched, OperatorId:" + std::to_string(itr->id()), std::ref(*itr)));
      // Set the state of the Operator as running. This only matters in Leaf ops, CacheOp and TakeOp
    }
  }
  tree_state_ = kDeTStateExecuting;
  return Status::OK();
}

// A function that traverse the tree in postorder then save the results in nodes
void ExecutionTree::Iterator::PostOrderTraverse(const std::shared_ptr<DatasetOp> &node) {
  if (node == nullptr) {
    return;
  }
  for (int32_t i = 0; i < node->child_.size(); ++i) {
    PostOrderTraverse(node->child_[i]);
  }
  nodes_.push_back(node);
}

ExecutionTree::Iterator::Iterator(const std::shared_ptr<DatasetOp> &root) : ind_(0) {
  // post-order traverse the tree, if root is null, it return
  PostOrderTraverse(root);
  nodes_.emplace_back(nullptr);
}

// Given the number of workers, launches the worker entry function for each. Essentially a
// wrapper for the TaskGroup handling that is stored inside the execution tree.
Status ExecutionTree::LaunchWorkers(int32_t num_workers, std::function<Status(uint32_t)> func) {
  // Launch the workers
  for (int32_t i = 0; i < num_workers; ++i) {
    RETURN_IF_NOT_OK(tg_->CreateAsyncTask("Parallel Op Worker", std::bind(func, i)));
  }
  return Status::OK();
}

// The driver of the prepare phase of the execution tree. The prepare phase will recursively
// walk the tree to perform modifications to the tree or specific nodes within the tree to get
// it ready for execution.
Status ExecutionTree::Prepare() {
  // Tree must be in pending prepare state before we can assign root to it
  if (tree_state_ != kDeTStatePrepare) {
    std::string err_msg =
      "Invalid tree state for preparing the tree. Current state: " + std::to_string(static_cast<int>(tree_state_)) +
      " Expected state: " + std::to_string(static_cast<int>(kDeTStatePrepare));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  // Start the recursive prepare
  RETURN_IF_NOT_OK(this->PrepareNode(root_));
  tree_state_ = kDeTStateReady;
  return Status::OK();
}

// Recursive function used during prepare phase to visit a node and drive any pre- and post-
// node actions during a tree walk.
Status ExecutionTree::PrepareNode(const std::shared_ptr<DatasetOp> &dataset_op) {
  int32_t num_children = dataset_op->child_.size();

  // Before going down into children, make any prepare flags updates based on this
  // operator.
  uint32_t op_prep_flags = dataset_op->PrepareFlags();
  // Sanity check.  In future we can support nested repeats.  for now it's not allowed.
  // If somebody above us already set the repeat flag, and now we are another repeat...
  if (BitTest(op_prep_flags, kDePrepRepeat) && BitTest(prepare_flags_, kDePrepRepeat)) {
    std::string err_msg("Nested RepeatOp detected! This is not supported yet.");
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  BitSet(&prepare_flags_, op_prep_flags);

  // Now, descend to children
  for (int32_t i = 0; i < num_children; ++i) {
    RETURN_IF_NOT_OK(this->PrepareNode(dataset_op->child_[i]));
  }

  // No more children, now we execute any prepare actions before going back up the
  // the tree on recursive function exit
  RETURN_IF_NOT_OK(dataset_op->PrepareNodeAction());

  // Then clear the flags from this op now that we have prepared it.
  BitClear(&prepare_flags_, op_prep_flags);
  return Status::OK();
}

// Adds an operator to the repeat stack during prepare phase.
void ExecutionTree::AddToRepeatStack(std::shared_ptr<DatasetOp> dataset_op) { repeat_stack_.push(dataset_op); }

// Pops an operator from the repeat stack during prepare phase.
std::shared_ptr<DatasetOp> ExecutionTree::PopFromRepeatStack() {
  std::shared_ptr<DatasetOp> top_op = nullptr;
  if (!repeat_stack_.empty()) {
    top_op = repeat_stack_.top();
    repeat_stack_.pop();
  }
  return top_op;
}
}  // namespace dataset
}  // namespace mindspore
