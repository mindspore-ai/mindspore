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
#ifndef DATASET_ENGINE_EXECUTION_TREE_H_
#define DATASET_ENGINE_EXECUTION_TREE_H_

#include <functional>
#include <memory>
#include <stack>
#include <string>
#include <vector>
#include "dataset/engine/datasetops/dataset_op.h"
#include "dataset/util/status.h"
#include "mindspore/ccsrc/dataset/engine/perf/profiling.h"

namespace mindspore {
namespace dataset {
// Forward declares
class TaskGroup;
class DatasetOp;
class Monitor;

class ExecutionTree {
 public:
  // Prepare flags used during tree prepare phase
  enum PrepareFlags {
    kDePrepNone = 0,
    kDePrepRepeat = 1  //  Processing a repeat operation
  };

  // State flags for the lifecycle of the tree
  enum TreeState {
    kDeTStateInit = 0,   // The freshly initialized state after construction
    kDeTStateBuilding,   // The tree is being built, nodes are being added
    kDeTStatePrepare,    // The tree has been assigned a root node and is pending prepare
    kDeTStateReady,      // The tree has been prepared and is ready to be launched
    kDeTStateExecuting,  // The tree has been launched and is executing
    kDeTStateFinished    // The tree has been drained, dataset iterator received EOF
  };

  class Iterator {
   public:
    // Constructor
    // @param root The root node to start iterating from
    explicit Iterator(const std::shared_ptr<DatasetOp> &root = nullptr);

    // Destructor
    ~Iterator() {}

    Iterator &operator++() {
      ++ind_;
      return *this;
    }  // prefix ++ overload
    Iterator operator++(int) {
      Iterator it = *this;
      it.ind_ = ind_;
      ind_++;
      return it;
    }  // post-fix ++ overload
    Iterator &operator--() {
      --ind_;
      return *this;
    }  // prefix -- overload
    Iterator operator--(int) {
      Iterator it = *this;
      it.ind_ = ind_;
      ind_--;
      return it;
    }                                                 // post-fix -- overload
    DatasetOp &operator*() { return *nodes_[ind_]; }  // dereference operator
    std::shared_ptr<DatasetOp> operator->() { return nodes_[ind_]; }

    // getter function
    // @return Shared pointer to the current operator
    std::shared_ptr<DatasetOp> get() { return nodes_[ind_]; }

    bool operator!=(const Iterator &rhs) { return nodes_[ind_] != rhs.nodes_[rhs.ind_]; }

    int32_t NumNodes() { return nodes_.size(); }

   private:
    int32_t ind_;                                    // the cur node our Iterator points to
    std::vector<std::shared_ptr<DatasetOp>> nodes_;  // store the nodes in post order
    void PostOrderTraverse(const std::shared_ptr<DatasetOp> &);
  };

  // Constructor
  ExecutionTree();

  // Destructor
  ~ExecutionTree();

  // Associates a DatasetOp with this tree. This assigns a valid node id to the operator and
  // provides it with a link to the tree. A node cannot form any relationships (parent/child) with
  // other nodes unless they are associated with the same tree.
  // @param op - The operator to associate
  // @return Status - The error code return
  Status AssociateNode(const std::shared_ptr<DatasetOp> &op);

  // Sets the root node of the tree
  // @param op - The operator to assign as root
  // @return Status - The error code return
  Status AssignRoot(const std::shared_ptr<DatasetOp> &op);

  // Start the execution of the tree
  // @return Status - The error code return
  Status Launch();

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  void Print(std::ostream &out) const;

  // Returns an iterator positioned at the start
  // @return Iterator - The iterator
  ExecutionTree::Iterator begin(const std::shared_ptr<DatasetOp> &root = nullptr) const {
    return Iterator(root == nullptr ? root_ : root);
  }

  // Returns an iterator positioned at the end
  // @return Iterator - The iterator
  ExecutionTree::Iterator end() const { return Iterator(nullptr); }

  // << Stream output operator overload
  // @notes This allows you to write the debug print info using stream operators
  // @param out - reference to the output stream being overloaded
  // @param exe_tree - reference to the execution tree to display
  // @return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, ExecutionTree &exe_tree) {
    exe_tree.Print(out);
    return out;
  }

  // Given the number of workers, launches the worker entry function for each. Essentially a
  // wrapper for the TaskGroup handling that is stored inside the execution tree.
  // @param num_workers - The number of workers to launch
  // @param func - The function entry point that workers will execute
  // @return Status - The error code return
  Status LaunchWorkers(int32_t num_workers, std::function<Status(uint32_t)> func);

  // Getter method
  // @return shared_ptr to the root operator
  std::shared_ptr<DatasetOp> root() const { return root_; }

  // Getter method
  // @return the prepare flags
  uint32_t PrepareFlags() const { return prepare_flags_; }

  // The driver of the prepare phase of the execution tree.
  // Prepare phase consists of three sub phases
  //
  // 1. PrepareTreePreAction()
  //    Compulsory transformation/action pre optimization.
  //    For example, CacheOp Insertion
  //
  // 2. Optimize()
  //    Optimization transformation/action, optional
  //    For example, MapOp Fusion
  //
  // 3. PrepareTreePostAction()
  //    Compulsory transformation/action post optimization.
  //    For example, repeatOp inlining
  //
  // @return Status - The error code return
  Status Prepare();

  // Compulsory transformation/action pre optimization.
  // @return Status - The error code return
  Status PrepareTreePreAction();

  // Compulsory transformation/action post optimization.
  // @return Status - The error code return
  Status PrepareTreePostAction();

  // Optimization transformation/action, optional.
  // @return Status - The error code return
  Status Optimize();

  // The DEPRECATED driver of the prepare phase of the execution tree. The prepare phase will recursively
  // walk the tree to perform modifications to the tree or specific nodes within the tree to get
  // it ready for execution.
  // @return Status - The error code return
  Status PrepareDeprecated();

  // Recursive function used during prepare phase to visit a node and drive any pre- and post-
  // node actions during a tree walk.
  // @param op - The dataset op to work on
  // @return Status - The error code return
  Status PrepareNode(const std::shared_ptr<DatasetOp> &dataset_op);

  // Adds an operator to the repeat stack during prepare phase.
  // @param op - The dataset op to work add to repeat stack
  // @return Status - The error code return
  void AddToRepeatStack(std::shared_ptr<DatasetOp> dataset_op);

  // Pops an operator from the repeat stack during prepare phase.
  // @return shared_ptr to the popped operator
  std::shared_ptr<DatasetOp> PopFromRepeatStack();

  // Return the pointer to the TaskGroup
  // @return raw pointer to the TaskGroup
  TaskGroup *AllTasks() const { return tg_.get(); }

  // Return if the ExecutionTree is finished (iterator receives EOF).
  // @return Bool - true is ExecutionTree is finished
  bool isFinished() const { return tree_state_ == TreeState::kDeTStateFinished; }

  // Set the ExecutionTree to Finished state.
  void SetFinished() { tree_state_ = TreeState::kDeTStateFinished; }

  // Getter for profiling manager, no ownership
  ProfilingManager *GetProfilingManager() { return profiling_manager_.get(); }

 private:
  // A helper functions for doing the recursive printing
  // @param dataset_op - The dataset op to print
  // @param indent - an indent string for aligning child levels in output
  // @param last - an indicator if it's the last child or not
  // @param detailed - should it display the detailed node output or the summary line
  void PrintNode(std::ostream &out, const std::shared_ptr<DatasetOp> &dataset_op, std::string indent, bool last,
                 bool detailed) const;

  std::unique_ptr<TaskGroup> tg_;                        // Class for worker management
  std::shared_ptr<DatasetOp> root_;                      // The root node of the tree
  int32_t id_count_;                                     // Counter for generating operator id's
  uint32_t prepare_flags_;                               // Flags used during tree prepare
  TreeState tree_state_;                                 // Tracking the current tree state
  std::stack<std::shared_ptr<DatasetOp>> repeat_stack_;  // A stack used during prepare phase
  std::unique_ptr<Monitor> perf_monitor_;                // Performance Monitor
  std::unique_ptr<ProfilingManager> profiling_manager_;  // Profiling manager
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_EXECUTION_TREE_H_
