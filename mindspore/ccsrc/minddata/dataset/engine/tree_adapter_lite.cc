/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/ir/datasetops/root_node.h"
#include "minddata/dataset/engine/opt/pass.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/opt/post/repeat_pass.h"
#endif
#include "minddata/dataset/engine/opt/pre/debug_mode_pass.h"
#include "minddata/dataset/engine/opt/pre/deep_copy_pass.h"
#include "minddata/dataset/engine/opt/pre/epoch_ctrl_pass.h"
#include "minddata/dataset/engine/opt/pre/input_validation_pass.h"
#include "minddata/dataset/engine/opt/pre/node_removal_pass.h"

namespace mindspore {
namespace dataset {

TreeAdapterLite::TreeAdapterLite() : root_(nullptr) {
  // Create ExecutionTree.
  tree_ = std::make_unique<ExecutionTree>();
}

Status TreeAdapterLite::BuildExecutionTreeRecur(std::shared_ptr<DatasetNode> ir, std::shared_ptr<DatasetOp> *const op) {
  RETURN_UNEXPECTED_IF_NULL(ir);
  RETURN_UNEXPECTED_IF_NULL(op);
  // Build the DatasetOp ExecutionTree from the optimized IR tree
  std::vector<std::shared_ptr<DatasetOp>> ops;
  RETURN_IF_NOT_OK(ir->Build(&ops));

  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "Unable to build node: " + ir->Name());

  (*op) = ops.front();  // return the first op to be added as child by the caller of this function

  for (size_t i = 0; i < ops.size(); i++) {
    RETURN_IF_NOT_OK(tree_->AssociateNode(ops[i]));
    if (i > 0) {
      RETURN_IF_NOT_OK(ops[i - 1]->AddChild(ops[i]));
    }
  }

  // Build the children of IR, once they return, add the return value to *op
  for (const std::shared_ptr<DatasetNode> &child_ir : ir->Children()) {
    std::shared_ptr<DatasetOp> child_op;
    RETURN_IF_NOT_OK(BuildExecutionTreeRecur(child_ir, &child_op));
    RETURN_IF_NOT_OK(ops.back()->AddChild(child_op));  // append children to the last of ops
  }

  return Status::OK();
}

Status TreeAdapterLite::BuildTree(std::shared_ptr<DatasetNode> root_ir) {
  RETURN_UNEXPECTED_IF_NULL(root_ir);
  // Build the Execution tree from the child of the IR root node, which represent the root of the input IR tree as a Top
  // Node is added to IR tree.
  RETURN_IF_NOT_OK(BuildExecutionTreeRecur(root_ir->Children()[0], &root_));
  RETURN_IF_NOT_OK(tree_->AssignRoot(root_));
  // Prepare the tree
  RETURN_IF_NOT_OK(tree_->Prepare(true));
  return Status::OK();
}

Status TreeAdapterLite::GetNextRow(TensorRow *const row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  RETURN_UNEXPECTED_IF_NULL(root_);
  row->reset();  // Ensure TensorRow is empty and flags are initialized
  RETURN_IF_NOT_OK(root_->GetNextRowPullMode(row));
  if (row->eoe()) {  // return empty tensor if 1st buf is a ctrl buf (no rows)
    MS_LOG(INFO) << "End of data iteration.";
    return Status::OK();
  }
  if (row->eof()) {
    std::string err = "EOF buffer encountered. User tries to fetch data beyond the specified number of epochs.";
    RETURN_STATUS_UNEXPECTED(err);
  }
  return Status::OK();
}

Status TreeAdapterLite::PrePass(std::shared_ptr<DatasetNode> ir) const {
  RETURN_UNEXPECTED_IF_NULL(ir);
  // Vector of actions in pre-pass phase
  std::vector<std::unique_ptr<IRPass>> actions;
  MS_LOG(INFO) << "Prepare PrePass loops.";
  (void)actions.emplace_back(std::make_unique<InputValidationPass>());
  (void)actions.emplace_back(std::make_unique<NodeRemovalPass>());
  (void)actions.emplace_back(std::make_unique<EpochCtrlPass>());
  if (GlobalContext::config_manager()->get_debug_mode()) {
    (void)actions.emplace_back(std::make_unique<DebugModePass>());
  }
  // Apply pre-pass actions
  for (size_t i = 0; i < actions.size(); i++) {
    auto m = false;
    RETURN_IF_NOT_OK(actions[i]->Run(ir, &m));
  }
  MS_LOG(INFO) << "PrePass completed.";
  return Status::OK();
}

Status TreeAdapterLite::PostPass(std::shared_ptr<DatasetNode> ir) const {
  RETURN_UNEXPECTED_IF_NULL(ir);
  // Vector of actions in post-pass phase
  std::vector<std::unique_ptr<IRPass>> actions;
#ifndef ENABLE_ANDROID
  MS_LOG(INFO) << "Running repeat pass.";
  (void)actions.emplace_back(std::make_unique<RepeatPass>());
  bool modified = false;
  RETURN_IF_NOT_OK(actions[0]->Run(ir, &modified));
  MS_LOG(INFO) << "Repeat pass completed.";
#endif
  return Status::OK();
}

Status TreeAdapterLite::Compile(const std::shared_ptr<DatasetNode> &input_ir, int32_t num_epochs) {
  RETURN_UNEXPECTED_IF_NULL(input_ir);
  input_ir_ = input_ir;
  MS_LOG(INFO) << "Input plan:" << '\n' << *input_ir << '\n';

  // Clone the input IR tree and insert under the root node
  // Create a root node to host the new copy of the input IR tree
  // This is done so that the PrePass will process and modify the tree
  // without changing the tree associated with the user code.
  // The tree from the user code is permitted to form a graph where any node
  // is consumed by more than one parent. However, this cloning process here
  // will break the graph into a tree by copying each consumption of a node into a new copy.
  DeepCopyPass cloning_tree;
  bool m = false;
  RETURN_IF_NOT_OK(cloning_tree.Run(input_ir, &m));
  std::shared_ptr<RootNode> root_ir = cloning_tree.Root();
  root_ir->SetNumEpochs(num_epochs);

  MS_LOG(INFO) << "Plan before PrePass:" << '\n' << *root_ir << '\n';
  // Pre-pass of the IR tree
  RETURN_IF_NOT_OK(PrePass(root_ir));
  MS_LOG(INFO) << "Plan after PrePass:" << '\n' << *root_ir << '\n';

  RETURN_IF_NOT_OK(PostPass(root_ir));
  MS_LOG(INFO) << "Plan after PostPass:" << '\n' << *root_ir << '\n';
  root_ir_ = root_ir;

  RETURN_IF_NOT_OK(BuildTree(root_ir));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
