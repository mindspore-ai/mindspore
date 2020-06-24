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

#include <memory>
#include <vector>
#include "dataset/engine/opt/pre/map_column_reorder.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/engine/datasetops/map_op.h"
#include "dataset/engine/datasetops/project_op.h"

namespace mindspore {
namespace dataset {

Status MapColumnReorder::RunOnTree(ExecutionTree *tree, bool *modified) {
  std::vector<std::shared_ptr<MapOp>> to_process;

  // Pass 1, search for all MapOp with column orders
  for (auto &op : *tree) {
    if (auto mapOp = std::dynamic_pointer_cast<MapOp>(op.shared_from_this())) {
      if (mapOp->ColumnsOrder().size() != 0) {
        to_process.push_back(mapOp);
      }
    }
  }

  // Pass 2, insert nodes for all MapOp
  for (auto node : to_process) {
    std::shared_ptr<ProjectOp::Builder> builder = std::make_shared<ProjectOp::Builder>(node->ColumnsOrder());
    std::shared_ptr<ProjectOp> op;
    RETURN_IF_NOT_OK(builder->Build(&op));
    RETURN_IF_NOT_OK(tree->AssociateNode(op));
    RETURN_IF_NOT_OK(node->InsertAsParent(op));
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
