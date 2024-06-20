/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/perf/auto_tune.h"
#include "minddata/dataset/engine/perf/dataset_iterator_tracing.h"

namespace mindspore {
namespace dataset {
class DatasetNode;
class TreeModifier;
class ToDevice;
class IteratorConsumer;

class TreeAdapter {
#ifndef ENABLE_SECURITY
  friend ProfilingManager;
  friend TreeConsumer;
  friend ToDevice;
  friend IteratorConsumer;
  friend AutoTune;
#endif
  friend TreeModifier;

 public:
  // this flag is used to indicate the purpose of the creation of this tree adapter (type of the tree_consumer).
  // Currently there are 3 types of consumer, Iterator, Getter and TDT/Vocab/Save ...
  // To avoid premature optimization, the last type (TDT/Vocab/Save) is regarded as Iterator for now.
  enum UsageFlag { kDeIterator = 0, kDeGetter = 1, kDeReset = 2 };

  explicit TreeAdapter(UsageFlag usage = kDeIterator);

  ~TreeAdapter() = default;

  // This function performs syntax checking, semantics checking, optimizes, and then builds
  // the Execution tree.
  Status Compile(const std::shared_ptr<DatasetNode> &input_ir, int32_t num_epochs = -1, int64_t global_step = 0,
                 int64_t dataset_size = -1);

  // Return the root node of the IR after cloned from the parsed IR tree
  std::shared_ptr<DatasetNode> RootIRNode() const { return root_ir_; }

  const ExecutionTree *GetExecutionTree() const { return tree_.get(); }

  // This is the main method TreeConsumer uses to interact with TreeAdapter
  // 1. GetNext will Launch() the ExeTree on its first call by iterator (tree is already prepared)
  // 2. GetNext will return empty row when eoe/eof is obtained
  Status GetNext(TensorRow *);

  // unique_ptr overloads operator bool(), will return false if it doesn't manage an object
  // This is needed by Iterator to get data by 'GetNext'.
  std::weak_ptr<DatasetOp> GetRoot() const { return tree_ ? tree_->root() : nullptr; }

  // This function will return the column_name_map once BuildAndPrepare() is called
  std::unordered_map<std::string, int32_t> GetColumnNameMap() const { return column_name_map_; }

  // This function returns the TaskGroup associated with ExeTree. This is needed by DeviceQueueConsumer
  // to be able to launch a thread. BuildAndPrepare needs to be called before this function
  TaskGroup *const AllTasks() const { return tree_ ? tree_->AllTasks() : nullptr; }

  Status Launch();

  // Set optional optimization pass
  void SetOptimize(bool value) { optimize_ = value; }

  // Optional optimizations status
  bool OptimizationEnabled() const { return optimize_; }

  // Return Offload Json
  nlohmann::json GetOffloadJson();
#ifndef ENABLE_SECURITY
  /// \brief Setter for Profiling Manager
  Status SetProfilingManagerPtr(const std::shared_ptr<ProfilingManager> &profiling_manager,
                                std::shared_ptr<Tracing> tracing_node = nullptr) {
    profiling_manager_ = profiling_manager;
    if (tracing_node != nullptr) {
      tracing_ = std::dynamic_pointer_cast<DatasetIteratorTracing>(tracing_node);
    }
    return Status::OK();
  }

  /// \brief Getter for profiling manager, no ownership
  ProfilingManager *GetProfilingManager() { return profiling_manager_.get(); }
#endif

 protected:
  // Run the mandatory pass checking the syntax and semantics of the IR tree
  Status PrePass(const std::shared_ptr<DatasetNode> &ir);

  // Run the optional optimization pass on the IR tree
  static Status Optimize(const std::shared_ptr<DatasetNode> &ir);

  // Run the mandatory pass augmenting the IR tree
  Status PostPass(const std::shared_ptr<DatasetNode> &ir);

#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  // Insert SendBridgeOp and ReceiveBridgeOp to the tree
  Status InsertSendReceiveOp();

  // Split the tree to send tree and receive tree
  Status SplitBySendReceiveOp();
#endif

  Status CheckTreeIfNull();

  // Build an Execution tree
  Status Build(const std::shared_ptr<DatasetNode> &root_ir, int64_t init_epoch = 0);

  // This RECURSIVE function walks the (optimized) IR tree in DFS to build its corresponding Execution tree.
  Status BuildExecutionTreeRecur(const std::shared_ptr<DatasetNode> &ir, std::shared_ptr<DatasetOp> *op);

  // Adjust the pipeline (eg, move rng_ forward) if in reset mode
  Status AdjustReset(const int64_t epoch_num);

  std::unordered_map<std::string, int32_t> column_name_map_;
  std::shared_ptr<DatasetNode> input_ir_;
  std::shared_ptr<DatasetNode> root_ir_;
#if !defined(__APPLE__) && !defined(BUILD_LITE) && !defined(_WIN32) && !defined(_WIN64) && !defined(__ANDROID__) && \
  !defined(ANDROID)
  // Launch the subprocess
  Status LaunchSubprocess();

  // The subprocess is changed to daemon and do nothing, just waiting for the main process exit
  void SubprocessDaemonLoop();

  // the send tree, like: xxDataset -> map -> ... -> batch -> send
  std::unique_ptr<ExecutionTree> send_tree_;
  // the receive tree, like: receive -> iterator / data_queue
  std::unique_ptr<ExecutionTree> receive_tree_;

  pid_t parent_process_id_;  // parent process id
  pid_t process_id_;         // current process id
  pid_t sub_process_id_;     // sub process id
#endif

  // 1. the tree holder, the send_tree_ will be moved to it and launched in independent dataset process
  // 2. the tree holder, the receive_tree_ will be moved to it and launched in main dataset process
  std::unique_ptr<ExecutionTree> tree_;
  bool optimize_;  // Flag to enable optional optimization pass
#ifndef ENABLE_SECURITY
  std::shared_ptr<ProfilingManager> profiling_manager_;  // Profiling manager
  std::shared_ptr<DatasetIteratorTracing> tracing_;      // trace profiling data
#endif
  int32_t cur_batch_num_;           // current batch number, used for profiling
  int32_t cur_connector_size_;      // current connector size of root op, used for profiling
  int32_t cur_connector_capacity_;  // current connector capacity of root op, used for profiling
  UsageFlag usage_;                 // usage of this tree adapter (type of consumer)
  bool launched_;
  // State flags for the lifecycle of the tree
  enum CompileState {
    kCompileStateInit = 0,      // The freshly initialized state
    kCompileStateIRGraphBuilt,  // User code has been parsed and its IR graph built
    kCompileStateIRTreeCloned,  // IR tree has been cloned from the IR graph
    kCompileStateOptimized,     // IR tree has been optimized
    kCompileStateReady          // Execution tree is generated from the optimized IR
  };
  CompileState tree_state_;
  nlohmann::json offload_json_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_H_
