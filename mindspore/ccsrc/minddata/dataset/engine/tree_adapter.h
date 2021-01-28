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
  // this flag is used to indicate the purpose of the creation of this tree adapter (type of the tree_consumer).
  // Currently there are 3 types of consumer, Iterator, Getter and TDT/Vocab/Save ...
  // To avoid premature optimization, the last type (TDT/Vocab/Save) is regarded as Iterator for now.
  enum UsageFlag { kDeIterator = 0, kDeGetter = 1 };

  explicit TreeAdapter(UsageFlag flag = kDeIterator);

  ~TreeAdapter() = default;

  // This function performs syntax checking, semantics checking, optimizes, and then builds
  // the Execution tree.
  Status Compile(std::shared_ptr<DatasetNode> root_ir, int32_t num_epochs = -1);

  // Return the root node of the IR after cloned from the parsed IR tree
  std::shared_ptr<DatasetNode> RootIRNode() const { return root_ir_; }

  // This is the main method TreeConsumer uses to interact with TreeAdapter
  // 1. GetNext will Launch() the ExeTree on its first call by iterator (tree is already prepared)
  // 2. GetNext will return empty row when eoe/eof is obtained
  Status GetNext(TensorRow *);

  // unique_ptr overloads operator bool(), will return false if it doesn't manage an object
  std::weak_ptr<DatasetOp> GetRoot() { return tree_ ? tree_->root() : nullptr; }

  // This function will return the column_name_map once BuildAndPrepare() is called
  std::unordered_map<std::string, int32_t> GetColumnNameMap() const { return column_name_map_; }

  // This function returns the TaskGroup associated with ExeTree. This is needed by DeviceQueueConsumer
  // to be able to launch a thread. BuildAndPrepare needs to be called before this function
  TaskGroup *const AllTasks() const { return tree_ ? tree_->AllTasks() : nullptr; }

  Status Launch() const;

  // Set optional optimization pass
  void SetOptimize(bool value) { optimize_ = value; }

  // Optional optimizations status
  bool OptimizationEnabled() const { return optimize_; }

 private:
  // Run the mandatory pass checking the syntax and semantics of the IR tree
  Status PrePass(std::shared_ptr<DatasetNode> ir);

  // Run the optional optimization pass on the IR tree
  Status Optimize(std::shared_ptr<DatasetNode> ir);

  // Run the mandatory pass augmenting the IR tree
  Status PostPass(std::shared_ptr<DatasetNode> ir);

  // Build an Execution tree
  Status Build(std::shared_ptr<DatasetNode> root_ir);

  // This RECURSIVE function walks the (optimized) IR tree in DFS to build its corresponding Execution tree.
  Status BuildExecutionTreeRecur(std::shared_ptr<DatasetNode> ir, std::shared_ptr<DatasetOp> *op);

  std::unique_ptr<DataBuffer> cur_db_;
  std::unordered_map<std::string, int32_t> column_name_map_;
  std::shared_ptr<DatasetNode> root_ir_;
  std::unique_ptr<ExecutionTree> tree_;              // current connector capacity of root op, used for profiling
  bool optimize_;                                    // Flag to enable optional optimization pass
  std::shared_ptr<DatasetIteratorTracing> tracing_;  // trace profiling data
  int32_t cur_batch_num_;                            // current batch number, used for profiling
  int32_t cur_connector_size_;                       // current connector size of root op, used for profiling
  int32_t cur_connector_capacity_;                   // current connector capacity of root op, used for profiling
  UsageFlag usage_;                                  // usage of this tree adapter (type of consumer)
  // State flags for the lifecycle of the tree
  enum CompileState {
    kCompileStateInit = 0,      // The freshly initialized state
    kCompileStateIRGraphBuilt,  // User code has been parsed and its IR graph built
    kCompileStateIRTreeCloned,  // IR tree has been cloned from the IR graph
    kCompileStateOptimized,     // IR tree has been optimized
    kCompileStateReady          // Execution tree is generated from the optimized IR
  };
  CompileState tree_state_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_H_
