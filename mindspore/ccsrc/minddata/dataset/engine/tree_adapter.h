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

namespace mindspore {
namespace dataset {
namespace api {
class DatasetNode;
}
class TreeAdapter {
 public:
  TreeAdapter() = default;

  ~TreeAdapter() = default;

  // This will construct an ExeTree from a Dataset root and Prepare() the ExeTree
  // This function is only meant to be called once and needs to be called before GetNext
  // ExeTree will be launched when the first GetNext is called
  Status BuildAndPrepare(std::shared_ptr<api::DatasetNode> root, int32_t num_epoch = -1);

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

 private:
  // This RECURSIVE function converts IR nodes into DatasetOp in ExecutionTree. IR could build a vector of ops. In
  // such case, the first node is returned. Op is added as child when the current function returns.
  Status DFSBuildTree(std::shared_ptr<api::DatasetNode> ir, std::shared_ptr<DatasetOp> *op);

  std::unique_ptr<DataBuffer> cur_db_;
  std::unordered_map<std::string, int32_t> column_name_map_;
  std::unique_ptr<ExecutionTree> tree_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_H_
