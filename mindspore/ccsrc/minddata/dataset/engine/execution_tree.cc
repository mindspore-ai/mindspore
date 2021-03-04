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
#include "minddata/dataset/engine/execution_tree.h"
#include <iostream>
#include <string>
#include <limits>
#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/datasetops/device_queue_op.h"
#include "minddata/dataset/engine/perf/profiling.h"
#include "minddata/dataset/engine/perf/monitor.h"
#if defined(ENABLE_GPUQUE) || defined(ENABLE_TDTQUE)
#include "minddata/dataset/util/numa_interface.h"
#endif
#include "minddata/dataset/util/task_manager.h"

namespace mindspore {
namespace dataset {
// Constructor
ExecutionTree::ExecutionTree() : id_count_(0), tree_state_(kDeTStateInit) {
  tg_ = std::make_unique<TaskGroup>();
  profiling_manager_ = std::make_unique<ProfilingManager>(this);
#if defined(ENABLE_GPUQUE) || defined(ENABLE_TDTQUE)
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  rank_id_ = cfg->rank_id();
  numa_enable_ = cfg->numa_enable();
  handle_ = nullptr;
#endif
}

// Destructor
ExecutionTree::~ExecutionTree() {
#if defined(ENABLE_GPUQUE) || defined(ENABLE_TDTQUE)
  if (numa_enable_) {
    if (handle_ != nullptr) {
      ReleaseLibrary(handle_);
    }
  }
#if defined(ENABLE_TDTQUE)
  DeviceQueueOp *op = dynamic_cast<DeviceQueueOp *>(root_.get());
  if (op != nullptr) {
    op->StopWaiting();
  }
#endif
#endif
  (void)tg_->ServiceStop();
}

// Associates a DatasetOp with this tree. This assigns a valid node id to the operator and
// provides it with a link to the tree. A node cannot form any relationships (parent/child) with
// other nodes unless they are associated with the same tree.
Status ExecutionTree::AssociateNode(const std::shared_ptr<DatasetOp> &op) {
  // If we are already a part of the tree, no-op
  if (op->tree_ == this) {
    return Status::OK();
  }
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

  return Status::OK();
}

// A print method typically used for debugging
void ExecutionTree::Print(std::ostream &out, const std::shared_ptr<DatasetOp> &op) const {
  out << "Execution tree summary:\n"
      << "-----------------------\n";
  this->PrintNode(out, op == nullptr ? root_ : op, "", true, false);
  out << "\nExecution tree operator details:\n"
      << "--------------------------------\n";
  this->PrintNode(out, op == nullptr ? root_ : op, "", true, true);
}

// A helper functions for doing the recursive printing
void ExecutionTree::PrintNode(std::ostream &out, const std::shared_ptr<DatasetOp> &dataset_op, std::string indent,
                              bool last, bool detailed) const {
  // Decide which printer to use based on detailed arg.
  if (!detailed) {
    out << indent << "+- " << *dataset_op;
    indent += (last ? "    " : "|   ");
  } else {
    dataset_op->Print(out, detailed);
  }

  // Descend to children
  for (int32_t i = 0; i < dataset_op->child_.size(); ++i) {
    this->PrintNode(out, dataset_op->child_[i], indent, (i == (dataset_op->child_.size() - 1)), detailed);
  }
}

// Start the execution of the tree
Status ExecutionTree::Launch() {
  // opencv limit too many threads
#if !defined(_WIN32) && !defined(_WIN64) && !defined(__APPLE__) && !defined(ENABLE_ANDROID)
#if defined(ENABLE_GPUQUE) || defined(ENABLE_TDTQUE)
  // Here we do numa bind for performance optimization, as our test result,
  // if we do numa bind when get_dataset_size launch a tree, we'll get a
  // better performance than only we do numa bind at the time _To_Device
  // launch a tree. Our numa bind work is a process level bind, bind with
  // both cpu and memory and we choose numa_node with a polling logic:
  // numa_bind_id = rank_id_ % (numa_max_node() + 1)
  // Now we only support GPU scenario and the single process scenario of Ascend,
  // now we remove the target_link of numa with _c_dataengine, and user can use
  // a config api to control whether to open numa feature.
  if (numa_enable_ && rank_id_ >= 0) {
    if (handle_ == nullptr) {
      handle_ = GetNumaAdapterHandle();
      if (handle_ == nullptr) {
        RETURN_STATUS_UNEXPECTED("Numa package (libnuma.so) not found.");
      }
    }
    RETURN_IF_NOT_OK(NumaBind(handle_, rank_id_));
    MS_LOG(INFO) << "Numa bind memory and cpu successful.";
  }
#endif
  int32_t thread_num = get_nprocs();
  if (thread_num == 0) {
    std::string err_msg = "Invalid thread number.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (thread_num > 8)
    cv::setNumThreads(8);
  else
    cv::setNumThreads(thread_num);
#endif

  // Tree must be built and prepared before it can be launched!
  if (tree_state_ != kDeTStatePrepared) {
    std::string err_msg =
      "Invalid tree state for launching tree. Current state: " + std::to_string(static_cast<int>(tree_state_)) +
      " Expected state: " + std::to_string(static_cast<int>(kDeTStatePrepared));
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // Profiling infrastructures need to be initialized before Op launching
  if (profiling_manager_->IsProfilingEnable()) {
    // Setup profiling manager
    RETURN_IF_NOT_OK(profiling_manager_->Initialize());
    // Launch Monitor Thread
    RETURN_IF_NOT_OK(profiling_manager_->LaunchMonitor());
  }

  std::ostringstream ss;
  ss << *this;
  MS_LOG(DEBUG) << "Printing the tree before launch tasks:\n" << ss.str();
  for (auto itr = this->begin(); itr != this->end(); ++itr) {
    // An inlined operator is one that has an output connector size of 0, and it does not
    // require a thread to execute.  Instead, the work of this operator is executed inlined
    // from the tree node directly above it (or in the case of a root node, it runs from within
    // the launching tree/user thread.  Do not exec any thread for an inlined op.
    itr->state_ = DatasetOp::OpState::kDeOpRunning;
    if (!itr->inlined()) {
      RETURN_IF_NOT_OK(tg_->CreateAsyncTask(itr->NameWithID(), std::ref(*itr), nullptr, itr->id()));
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
Status ExecutionTree::LaunchWorkers(int32_t num_workers, std::function<Status(uint32_t)> func, std::string name,
                                    int32_t operator_id) {
  int32_t num_cpu_threads = GlobalContext::Instance()->config_manager()->num_cpu_threads();
  // this performs check that num_workers is positive and not unreasonably large which could happen
  // for example, un-initialized variable. uint16 max is 65536 which is large enough to cover everything
  CHECK_FAIL_RETURN_UNEXPECTED(num_workers > 0 && num_workers < std::numeric_limits<uint16_t>::max(),
                               name + "'s num_worker=" + std::to_string(num_workers) + ", is negative or too large.");
  // Launch the workers
  if (num_workers > num_cpu_threads) {
    MS_LOG(WARNING) << name + " is launched with " << std::to_string(num_workers) << " worker threads which exceeds "
                    << std::to_string(num_cpu_threads) << ", the maximum number of threads on this CPU.";
  }
  for (int32_t i = 0; i < num_workers; ++i) {
    RETURN_IF_NOT_OK(tg_->CreateAsyncTask(name, std::bind(func, i), nullptr, operator_id));
  }
  return Status::OK();
}

// Walks the tree to perform modifications to the tree in post-order to get it ready for execution.
Status ExecutionTree::Prepare() {
  if (root_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("Please assign one operator as the root of this tree.");
  }

  std::vector<std::shared_ptr<DatasetOp>> fifo;
  std::shared_ptr<DatasetOp> op = root_;
  size_t index = 0;

  // Build a FIFO queue with the root at the beginning and continue adding its descendants to the queue.
  fifo.push_back(op);
  do {
    op = fifo[index];
    fifo.insert(fifo.end(), op->child_.begin(), op->child_.end());
    ++index;
  } while (index < fifo.size());

  // By iterating from the end of the FIFO queue, we simulate the post-order walk.
  for (auto rit = fifo.crbegin(); rit != fifo.crend(); ++rit) {
    RETURN_IF_NOT_OK((*rit)->PrepareOperator());
  }

  // The tree is prepared.
  tree_state_ = kDeTStatePrepared;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
