/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_LITE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_LITE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

namespace mindspore {
namespace dataset {

class TensorRow;
class DatasetNode;

class TreeAdapterLite {
 public:
  TreeAdapterLite();

  ~TreeAdapterLite() = default;

  Status BuildTree(std::shared_ptr<DatasetNode> root_ir);

  // Get rows equal to num_rows
  Status GetNextRow(TensorRow *const row);

  std::unordered_map<std::string, int32_t> GetColumnNameMap() const { return tree_->root()->column_name_id_map(); }

 private:
  // This RECURSIVE function walks the (optimized) IR tree in DFS to build its corresponding Execution tree.
  Status BuildExecutionTreeRecur(std::shared_ptr<DatasetNode> ir, std::shared_ptr<DatasetOp> *op);

  std::shared_ptr<DatasetOp> root_;  // current connector capacity of root op, used for profiling
  std::unique_ptr<ExecutionTree> tree_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TREE_ADAPTER_LITE_H_
