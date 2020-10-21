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

#include "minddata/dataset/engine/tree_adapter.h"

#include "minddata/dataset/core/client.h"
#include "minddata/dataset/include/datasets.h"

namespace mindspore {
namespace dataset {

Status TreeAdapter::BuildAndPrepare(std::shared_ptr<api::Dataset> root_ir, int32_t num_epoch) {
  // Check whether this function has been called before. If so, return failure
  CHECK_FAIL_RETURN_UNEXPECTED(tree_ == nullptr, "ExecutionTree is already built.");
  RETURN_UNEXPECTED_IF_NULL(root_ir);

  // This will evolve in the long run
  tree_ = std::make_unique<ExecutionTree>();

  std::shared_ptr<DatasetOp> root_op;
  RETURN_IF_NOT_OK(DFSBuildTree(root_ir, &root_op));
  RETURN_IF_NOT_OK(tree_->AssignRoot(root_op));

  // Prepare the tree
  RETURN_IF_NOT_OK(tree_->Prepare(num_epoch));

  // After the tree is prepared, the col_name_id_map can safely be obtained
  column_name_map_ = tree_->root()->column_name_id_map();

  return Status::OK();
}

Status TreeAdapter::GetNext(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(tree_);
  RETURN_UNEXPECTED_IF_NULL(row);
  row->clear();  // make sure row is empty
  // When cur_db_ is a nullptr, it means this is the first call to get_next, launch ExecutionTree
  if (cur_db_ == nullptr) {
    RETURN_IF_NOT_OK(tree_->Launch());
    RETURN_IF_NOT_OK(tree_->root()->GetNextBuffer(&cur_db_));  // first buf can't be eof or empty buf with none flag
    RETURN_OK_IF_TRUE(cur_db_->eoe());                         // return empty tensor if 1st buf is a ctrl buf (no rows)
  }

  CHECK_FAIL_RETURN_UNEXPECTED(!cur_db_->eof(), "EOF has already been reached.");

  if (cur_db_->NumRows() == 0) {  // a new row is fetched if cur buf is empty or a ctrl buf
    RETURN_IF_NOT_OK(tree_->root()->GetNextBuffer(&cur_db_));
    RETURN_OK_IF_TRUE(cur_db_->eoe() || cur_db_->eof());  // return empty if this new buffer is a ctrl flag
  }

  RETURN_IF_NOT_OK(cur_db_->PopRow(row));
  return Status::OK();
}

Status TreeAdapter::DFSBuildTree(std::shared_ptr<api::Dataset> ir, std::shared_ptr<DatasetOp> *op) {
  std::vector<std::shared_ptr<DatasetOp>> ops = ir->Build();
  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "Unable to build node.");

  (*op) = ops.front();  // return the first op to be added as child by the caller of this function
  RETURN_IF_NOT_OK(tree_->AssociateNode(*op));

  for (size_t i = 1; i < ops.size(); i++) {
    RETURN_IF_NOT_OK(tree_->AssociateNode(ops[i]));
    RETURN_IF_NOT_OK(ops[i - 1]->AddChild(ops[i]));
  }

  // Build the children of ir, once they return, add the return value to *op
  for (std::shared_ptr<api::Dataset> child_ir : ir->children) {
    std::shared_ptr<DatasetOp> child_op;
    RETURN_IF_NOT_OK(DFSBuildTree(child_ir, &child_op));
    RETURN_IF_NOT_OK(ops.back()->AddChild(child_op));  // append children to the last of ops
  }

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
