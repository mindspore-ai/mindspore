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
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/engine/opt/pre/input_validation_pass.h"

namespace mindspore {
namespace dataset {

Status TreeAdapter::PrePass(std::shared_ptr<DatasetNode> ir) {
  // Vector of actions in validation pass
  std::vector<std::unique_ptr<NodePass>> validations;

  MS_LOG(INFO) << "Running pre pass loops.";
  validations.push_back(std::make_unique<InputValidationPass>());

  // Vector of flags for each action
  // Apply validation actions
  for (auto i = 0; i < validations.size(); i++) {
    auto modified = false;
    // InputValidationPass does not change the IR tree. We don't need to capture the "modified" value.
    RETURN_IF_NOT_OK(validations[i]->Run(ir, &modified));
  }

  // Vector of actions in pre-pass phase
  std::vector<std::unique_ptr<Pass>> actions;

  // We will gradually move CacheErrorPass, EpochInjectionPass, CacheTransformPass
  // from ExecutionTree::PrepareTreePreAction to here.

  // Vector of flags for each action
  std::vector<bool> modified(actions.size(), false);
  // Apply pre-pass actions
  for (auto i = 0; i < actions.size(); i++) {
    auto m = false;
    RETURN_IF_NOT_OK(actions[i]->Run(ir, &m));
    modified[i] = m;
  }
  MS_LOG(INFO) << "Pre pass complete.";
  return Status::OK();
}

Status TreeAdapter::Optimize(std::shared_ptr<DatasetNode> ir) {
  // Vector of optimizations
  std::vector<std::unique_ptr<NodePass>> optimizations;
  MS_LOG(INFO) << "Running optimization pass loops";

  // We will gradually move TensorOpFusionPass from ExecutionTree::Optimize to here.

  // Vector of flags for each optimization
  std::vector<bool> modified(optimizations.size(), false);
  // Apply optimization pass actions
  for (auto i = 0; i < optimizations.size(); i++) {
    auto m = false;
    RETURN_IF_NOT_OK(optimizations[i]->Run(ir, &m));
    modified[i] = m;
  }
  MS_LOG(INFO) << "Optimization pass complete.";
  return Status::OK();
}

Status TreeAdapter::PostPass(std::shared_ptr<DatasetNode> ir) {
  // Vector of actions in post-pass phase
  std::vector<std::unique_ptr<Pass>> actions;
  MS_LOG(INFO) << "Running post pass loops.";

  // We will gradually move RepeatPass from ExecutionTree::PrepareTreePostAction to here.

  // Vector of flags for each action
  std::vector<bool> modified(actions.size(), false);
  for (auto i = 0; i < actions.size(); i++) {
    auto m = false;
    RETURN_IF_NOT_OK(actions[i]->Run(ir, &m));
    modified[i] = m;
  }
  MS_LOG(INFO) << "Post passes complete.";
  return Status::OK();
}

Status TreeAdapter::BuildExecutionTree(std::shared_ptr<DatasetNode> ir, std::shared_ptr<DatasetOp> *op) {
  // Build the DatasetOp ExecutionTree from the optimized IR tree
  std::vector<std::shared_ptr<DatasetOp>> ops = ir->Build();
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
    RETURN_IF_NOT_OK(BuildExecutionTree(child_ir, &child_op));
    RETURN_IF_NOT_OK(ops.back()->AddChild(child_op));  // append children to the last of ops
  }

  return Status::OK();
}

Status TreeAdapter::Compile(std::shared_ptr<DatasetNode> root_ir, int32_t num_epochs) {
  num_epochs_ = num_epochs;
  optimize_ = true;  // Always ON (temporary)

  RETURN_UNEXPECTED_IF_NULL(root_ir);

  // Pre-pass of the IR tree
  RETURN_IF_NOT_OK(PrePass(root_ir));

  // Optional phase of optimization
  if (optimize_) {
    RETURN_IF_NOT_OK(Optimize(root_ir));
  }

  // Post-pass of the IR tree
  RETURN_IF_NOT_OK(PostPass(root_ir));

  // This will evolve in the long run
  tree_ = std::make_unique<ExecutionTree>();

  std::shared_ptr<DatasetOp> root_op;
  RETURN_IF_NOT_OK(BuildExecutionTree(root_ir, &root_op));
  RETURN_IF_NOT_OK(tree_->AssignRoot(root_op));

  // Note: We will gradually move the pre pass, optimizer pass, and post pass
  //       on ExecutionTree to perform on IR tree.
  // Prepare the tree
  RETURN_IF_NOT_OK(tree_->Prepare(num_epochs));

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

Status TreeAdapter::Launch() const {
  CHECK_FAIL_RETURN_UNEXPECTED(tree_ != nullptr, "Tree is a nullptr.");
  return tree_->Launch();
}

}  // namespace dataset
}  // namespace mindspore
