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

#include "minddata/dataset/engine/tree_adapter_lite.h"

namespace mindspore {
namespace dataset {

TreeAdapterLite::TreeAdapterLite() : root_(nullptr) { tree_ = std::make_unique<ExecutionTree>(); }

Status TreeAdapterLite::BuildExecutionTreeRecur(std::shared_ptr<DatasetNode> ir, std::shared_ptr<DatasetOp> *const op) {
  // Build the DatasetOp ExecutionTree from the optimized IR tree
  std::vector<std::shared_ptr<DatasetOp>> ops;
  RETURN_IF_NOT_OK(ir->Build(&ops));

  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "Unable to build node.");

  (*op) = ops.front();  // return the first op to be added as child by the caller of this function

  RETURN_IF_NOT_OK(tree_->AssociateNode(*op));

  for (size_t i = 1; i < ops.size(); i++) {
    RETURN_IF_NOT_OK(tree_->AssociateNode(ops[i]));
    RETURN_IF_NOT_OK(ops[i - 1]->AddChild(ops[i]));
  }

  // Build the children of IR, once they return, add the return value to *op
  for (std::shared_ptr<DatasetNode> child_ir : ir->Children()) {
    std::shared_ptr<DatasetOp> child_op;
    RETURN_IF_NOT_OK(BuildExecutionTreeRecur(child_ir, &child_op));
    RETURN_IF_NOT_OK(ops.back()->AddChild(child_op));  // append children to the last of ops
  }

  return Status::OK();
}

Status TreeAdapterLite::BuildTree(std::shared_ptr<DatasetNode> root_ir) {
  RETURN_UNEXPECTED_IF_NULL(root_ir);
  RETURN_IF_NOT_OK(BuildExecutionTreeRecur(root_ir, &root_));
  RETURN_IF_NOT_OK(tree_->AssignRoot(root_));
  return Status::OK();
}

Status TreeAdapterLite::GetNextRow(TensorRow *const row) {
  RETURN_UNEXPECTED_IF_NULL(root_);
  RETURN_IF_NOT_OK(root_->GetNextRow(row));
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
