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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_EXECUTION_TREE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_EXECUTION_TREE_H_

#include <functional>
#include <memory>
#include <stack>
#include <string>
#include <vector>
#ifndef ENABLE_ANDROID
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__)
#include <sys/sysinfo.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#endif
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/util/status.h"
#include "mindspore/ccsrc/minddata/dataset/engine/perf/profiling.h"
namespace mindspore {
namespace dataset {
// Forward declares
class TaskGroup;
class DatasetOp;
class Pass;
using OptPass = std::vector<std::unique_ptr<Pass>>;
class ExecutionTree {
 public:
  // Prepare flags used during tree prepare phase
  enum PrepareFlags {
    kDePrepNone = 0,
    kDePrepRepeat = 1,  //  Processing a repeat operation
    kDePrepCache = 2    //  Processing a cache operation
  };

  // State flags for the lifecycle of the tree
  enum TreeState {
    kDeTStateInit = 0,   // The freshly initialized state after construction
    kDeTStateBuilding,   // The tree is being built, nodes are being added
    kDeTStatePrepare,    // The tree has been assigned a root node and is pending prepare
    kDeTStateReady,      // The tree has been prepared and is ready to be launched
    kDeTStateExecuting,  // The tree has been launched and is executing
    kDeTStateEpochEnd,   // The tree has been received end of epoch signal, just for profiling
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

    bool operator==(const Iterator &rhs) { return nodes_[ind_] == rhs.nodes_[rhs.ind_]; }

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
  // @return Status The status code returned
  Status AssociateNode(const std::shared_ptr<DatasetOp> &op);

  // Sets the root node of the tree
  // @param op - The operator to assign as root
  // @return Status The status code returned
  Status AssignRoot(const std::shared_ptr<DatasetOp> &op);

  // Start the execution of the tree
  // @return Status The status code returned
  Status Launch();

  /// A print method typically used for debugging
  /// \param out - The output stream to write output to
  void Print(std::ostream &out, const std::shared_ptr<DatasetOp> &op = nullptr) const;

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
  // @return Status The status code returned
  Status LaunchWorkers(int32_t num_workers, std::function<Status(uint32_t)> func, std::string name = "");

  // Getter method
  // @return shared_ptr to the root operator
  std::shared_ptr<DatasetOp> root() const { return root_; }

  // Getter method
  // @return the prepare flags
  uint32_t PrepareFlags() const { return prepare_flags_; }

  // The driver of the prepare phase of the execution tree.
  // Prepare phase consists of three sub phases
  //
  // 1. PreAction()
  //    Compulsory transformation/action pre optimization.
  //    For example, CacheOp Insertion
  //
  // 2. Optimize()
  //    Optimization transformation/action, optional
  //    For example, MapOp Fusion
  //
  // 3. PostAction()
  //    Compulsory transformation/action post optimization.
  //    For example, repeatOp inlining
  //
  // @return Status The status code returned
  Status Prepare(int num_epochs = -1, bool partial = false);

  // Compulsory transformation/action pre optimization.
  // @return Status The status code returned
  Status PreAction();

  // Compulsory transformation/action post optimization.
  // @return Status The status code returned
  Status PostAction();

  // The DEPRECATED driver of the prepare phase of the execution tree. The prepare phase will recursively
  // walk the tree to perform modifications to the tree or specific nodes within the tree to get
  // it ready for execution.
  // @param Total number of epochs that will be run on this tree
  // @return Status The status code returned
  Status PrepareDeprecated();

  // Recursive function used during prepare phase to visit a node and drive any pre- and post-
  // node actions during a tree walk.
  // @param op - The dataset op to work on
  // @return Status The status code returned
  Status PrepareNode(const std::shared_ptr<DatasetOp> &dataset_op);

  // Return the pointer to the TaskGroup
  // @return raw pointer to the TaskGroup
  TaskGroup *const AllTasks() const { return tg_.get(); }

  // Return if the ExecutionTree is at end of epoch status
  // @return bool - true is ExecutionTree is end of epoch status
  bool IsEpochEnd() const { return tree_state_ == TreeState::kDeTStateEpochEnd; }

  // Set the ExecutionTree to EOE state
  void SetEpochEnd() { tree_state_ = TreeState::kDeTStateEpochEnd; }

  // Set the ExecutionTree to executing state
  void SetExecuting() { tree_state_ = TreeState::kDeTStateExecuting; }

  // Return if the ExecutionTree is finished (iterator receives EOF).
  // @return Bool - true is ExecutionTree is finished
  bool isFinished() const { return tree_state_ == TreeState::kDeTStateFinished; }

  // Return if the ExecutionTree is ready.
  // @return Bool - true is ExecutionTree is ready
  bool isPrepared() const {
    return tree_state_ == TreeState::kDeTStateReady || tree_state_ == kDeTStateExecuting ||
           tree_state_ == kDeTStateFinished;
  }

  // Set the ExecutionTree to Finished state.
  void SetFinished() { tree_state_ = TreeState::kDeTStateFinished; }

  // Getter for profiling manager, no ownership
  ProfilingManager *GetProfilingManager() { return profiling_manager_.get(); }

  // Getter function to get the total number of epochs to be run on this tree.
  // @return total number of epochs
  int32_t num_epochs() { return num_epochs_; }

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
  int32_t num_epochs_;                                   // Total number of epochs to run for this tree
  std::unique_ptr<ProfilingManager> profiling_manager_;  // Profiling manager
  bool partially_prepare_;                               // Temp: during migration to IR, if true, run remaining passes.
#if defined(ENABLE_GPUQUE) || defined(ENABLE_TDTQUE)
  // This rank_id is for numa and device_queue, one process work with only one rank_id,
  // for standalone scenario, this rank_id may come from env 'CUDA_VISIBLE_DEVICES',
  // but for distribute scenario, this rank_id come from _get_global_rank() in python
  int32_t rank_id_;
  bool numa_enable_;
  void *handle_;
#endif
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_EXECUTION_TREE_H_
