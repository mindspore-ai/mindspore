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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"
#include "minddata/dataset/engine/perf/dataset_iterator_tracing.h"

namespace mindspore {
namespace dataset {
class DatasetNode;

class TreeAdapter {
 public:
  TreeAdapter() = default;

  ~TreeAdapter() = default;

  // This function performs syntax checking, semantics checking, optimizes, and then builds
  // the Execution tree.
  Status Compile(std::shared_ptr<DatasetNode> root_ir, int32_t num_epochs = -1);

  // This is the main method TreeConsumer uses to interact with TreeAdapter
  // 1. GetNext will Launch() the ExeTree on its first call by iterator (tree is already prepared)
  // 2. GetNext will return empty row when eoe/eof is obtained
  Status GetNext(TensorRow *);

  // This function will return the root of the execution tree.
  std::weak_ptr<DatasetOp> GetRoot() { return tree_ != nullptr ? tree_->root() : nullptr; }

  // This function will return the column_name_map once BuildAndPrepare() is called
  std::unordered_map<std::string, int32_t> GetColumnNameMap() const { return column_name_map_; }

  // This function returns the TaskGroup associated with ExeTree. This is needed by DeviceQueueConsumer
  // to be able to launch a thread. BuildAndPrepare needs to be called before this function
  TaskGroup *AllTasks() const { return tree_ != nullptr ? tree_->AllTasks() : nullptr; }

  Status Launch() const;

  // Set optional optimization pass
  void SetOptimize(bool value) { optimize_ = value; }

  // function to override override the pre-pass
  void SetPrePassOverride(std::function<OptPass(OptPass)> pre_pass_override) { pre_pass_override_ = pre_pass_override; }

  // Optional optimizations status
  bool OptimizationEnabled() const { return optimize_; }

 private:
  // This function runs a mandatory pass checking the syntax and semantics of the IR tree.
  Status PrePass(std::shared_ptr<DatasetNode> ir);

  // This function runs an optional optimization pass on the IR tree.
  Status Optimize(std::shared_ptr<DatasetNode> ir);

  // This function runs a mandatory pass augmenting the IR tree before the execution.
  Status PostPass(std::shared_ptr<DatasetNode> ir);

  // This RECURSIVE function walks the (optimized) IR tree in DFS to build its corresponding Execution tree.
  Status BuildExecutionTree(std::shared_ptr<DatasetNode> ir, std::shared_ptr<DatasetOp> *op);

  std::unique_ptr<DataBuffer> cur_db_;
  std::unordered_map<std::string, int32_t> column_name_map_;
  std::unique_ptr<ExecutionTree> tree_;  // current connector capacity of root op, used for profiling
  int32_t num_epochs_;
  bool optimize_;                                      // Flag to enable optional optimization pass
  std::shared_ptr<DatasetIteratorTracing> tracing_;    // trace profiling data
  int32_t cur_batch_num_;                              // current batch number, used for profiling
  int32_t cur_connector_size_;                         // current connector size of root op, used for profiling
  int32_t cur_connector_capacity_;                     // current connector capacity of root op, used for profiling
  std::function<OptPass(OptPass)> pre_pass_override_;  // function ptr that overrides pre pass, called in PrePrepare()
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_H_
