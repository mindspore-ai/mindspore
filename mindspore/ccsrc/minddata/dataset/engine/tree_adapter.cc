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

#include "minddata/dataset/engine/tree_adapter.h"

#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/ir/datasetops/root_node.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/opt/optional/tensor_op_fusion_pass.h"
#include "minddata/dataset/engine/opt/pre/cache_transform_pass.h"
#include "minddata/dataset/engine/opt/post/repeat_pass.h"
#endif
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/engine/opt/post/auto_worker_pass.h"
#ifdef ENABLE_PYTHON
#include "minddata/dataset/engine/opt/post/generator_node_pass.h"
#endif
#include "minddata/dataset/engine/opt/pre/cache_validation_pass.h"
#include "minddata/dataset/engine/opt/pre/deep_copy_pass.h"
#include "minddata/dataset/engine/opt/pre/epoch_ctrl_pass.h"
#include "minddata/dataset/engine/opt/pre/getter_pass.h"
#include "minddata/dataset/engine/opt/pre/input_validation_pass.h"
#include "minddata/dataset/engine/opt/pre/node_removal_pass.h"

namespace mindspore {
namespace dataset {

TreeAdapter::TreeAdapter(UsageFlag usage) : usage_(usage), tree_state_(kCompileStateInit) {
  optimize_ = common::GetEnv("OPTIMIZE") == "true";

  // Initialize profiling parameters
  cur_batch_num_ = 0;
  cur_connector_size_ = 0;
  cur_connector_capacity_ = 0;
}

Status TreeAdapter::PrePass(std::shared_ptr<DatasetNode> ir) {
  // Vector of actions in pre-pass phase
  std::vector<std::unique_ptr<IRPass>> actions;

  MS_LOG(INFO) << "Running pre pass loops.";
  actions.emplace_back(std::make_unique<InputValidationPass>());
  actions.emplace_back(std::make_unique<CacheValidationPass>());
  actions.emplace_back(std::make_unique<NodeRemovalPass>());
  actions.emplace_back(std::make_unique<EpochCtrlPass>());
  if (usage_ == kDeGetter) actions.emplace_back(std::make_unique<GetterPass>());
#ifndef ENABLE_ANDROID
  actions.emplace_back(std::make_unique<CacheTransformPass>());
#endif
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
  std::vector<std::unique_ptr<IRNodePass>> optimizations;
  MS_LOG(INFO) << "Running optimization pass loops";
#ifndef ENABLE_ANDROID
  optimizations.emplace_back(std::make_unique<TensorOpFusionPass>());
#endif
  // Apply optimization pass actions
  for (auto i = 0; i < optimizations.size(); i++) {
    bool modified = false;
    RETURN_IF_NOT_OK(optimizations[i]->Run(ir, &modified));
  }
  MS_LOG(INFO) << "Optimization pass complete.";
  return Status::OK();
}

Status TreeAdapter::PostPass(std::shared_ptr<DatasetNode> ir) {
  // Vector of actions in post-pass phase
  std::vector<std::unique_ptr<IRPass>> actions;
  MS_LOG(INFO) << "Running post pass loops.";

  // AutoWorkerPass should ideally precede CacheTransForm Pass to avoid complications of the setting
  if (GlobalContext::config_manager()->auto_num_workers() && usage_ == kDeIterator) {
    // skip this for getter pass
    actions.emplace_back(std::make_unique<AutoWorkerPass>());
  }
#ifdef ENABLE_PYTHON
  actions.emplace_back(std::make_unique<GeneratorNodePass>());
#endif
#ifndef ENABLE_ANDROID
  actions.emplace_back(std::make_unique<RepeatPass>());
#endif
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

Status TreeAdapter::BuildExecutionTreeRecur(std::shared_ptr<DatasetNode> ir, std::shared_ptr<DatasetOp> *const op) {
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

Status TreeAdapter::Build(std::shared_ptr<DatasetNode> root_ir) {
  // This will evolve in the long run
  tree_ = std::make_unique<ExecutionTree>();
  // disable profiling if this is only a getter pass
  if (usage_ == kDeGetter) tree_->GetProfilingManager()->DisableProfiling();
  // Build the Execution tree from the child of the IR root node, which represent the root of the input IR tree
  std::shared_ptr<DatasetOp> root_op;
  RETURN_IF_NOT_OK(BuildExecutionTreeRecur(root_ir->Children()[0], &root_op));
  RETURN_IF_NOT_OK(tree_->AssignRoot(root_op));

  // Note: We will gradually move the pre pass, optimizer pass, and post pass
  //       on ExecutionTree to perform on IR tree.
  // Prepare the tree
  RETURN_IF_NOT_OK(tree_->Prepare());

  // After the tree is prepared, the col_name_id_map can safely be obtained
  column_name_map_ = tree_->root()->column_name_id_map();

  return Status::OK();
}

Status TreeAdapter::Compile(std::shared_ptr<DatasetNode> input_ir, int32_t num_epochs) {
  RETURN_UNEXPECTED_IF_NULL(input_ir);

  tree_state_ = kCompileStateIRGraphBuilt;
  MS_LOG(INFO) << "Input plan:" << '\n' << *input_ir << '\n';

  // Clone the input IR tree and insert under the root node
  // Create a root node to host the new copy of the input IR tree
  // This is done so that the compilation will process and modify the tree
  // without changing the tree associated with the user code.
  // The tree from the user code is permitted to form a graph where any node
  // is consumed by more than one parent. However, this cloning process here
  // will break the graph into a tree by copying each consumption of a node into a new copy.
  bool m = false;
  DeepCopyPass cloning_tree;
  RETURN_IF_NOT_OK(cloning_tree.Run(input_ir, &m));
  std::shared_ptr<RootNode> root_ir = cloning_tree.Root();
  root_ir->SetNumEpochs(num_epochs);

  tree_state_ = kCompileStateIRTreeCloned;
  MS_LOG(INFO) << "Plan before optimization:" << '\n' << *root_ir << '\n';

  // Pre-pass of the IR tree
  RETURN_IF_NOT_OK(PrePass(root_ir));

  // Optional phase of optimization
  if (optimize_) {
    RETURN_IF_NOT_OK(Optimize(root_ir));
  }

  // Post-pass of the IR tree
  RETURN_IF_NOT_OK(PostPass(root_ir));

  tree_state_ = kCompileStateOptimized;
  MS_LOG(INFO) << "Plan after optimization:" << '\n' << *root_ir << '\n';
  // Remember the root node
  root_ir_ = root_ir;

  RETURN_IF_NOT_OK(Build(root_ir_));
  tree_state_ = kCompileStateReady;

  return Status::OK();
}

Status TreeAdapter::GetNext(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(tree_);
  RETURN_UNEXPECTED_IF_NULL(row);
  row->clear();  // make sure row is empty

  bool isProfilingEnable = tree_->GetProfilingManager()->IsProfilingEnable();

  // When cur_db_ is a nullptr, it means this is the first call to get_next, launch ExecutionTree
  if (cur_db_ == nullptr) {
    RETURN_IF_NOT_OK(tree_->Launch());
    // Profiling
    std::shared_ptr<Tracing> node;
    Status s = tree_->GetProfilingManager()->GetTracingNode(kDatasetIteratorTracingName, &node);
    if (s.IsOk()) {
      tracing_ = std::dynamic_pointer_cast<DatasetIteratorTracing>(node);
      cur_connector_size_ = tree_->root()->ConnectorSize();
      cur_connector_capacity_ = tree_->root()->ConnectorCapacity();
    }
    RETURN_IF_NOT_OK(tree_->root()->GetNextBuffer(&cur_db_));  // first buf can't be eof or empty buf with none flag
    if (cur_db_->eoe()) {                                      // return empty tensor if 1st buf is a ctrl buf (no rows)
      MS_LOG(INFO) << "End of data iteration.";
      if (isProfilingEnable) {
        tree_->SetEpochEnd();
      }
      return Status::OK();
    }
  }

  CHECK_FAIL_RETURN_UNEXPECTED(!cur_db_->eof(), "EOF has already been reached.");

  if (cur_db_->NumRows() == 0) {  // a new row is fetched if cur buf is empty or a ctrl buf
    RETURN_IF_NOT_OK(tree_->root()->GetNextBuffer(&cur_db_));
    if (cur_db_->eoe()) {  // return empty if this new buffer is a ctrl flag
      MS_LOG(INFO) << "End of data iteration.";
      if (isProfilingEnable) {
        tree_->SetEpochEnd();
      }
      return Status::OK();
    }
    if (cur_db_->eof()) {
      tree_->SetFinished();
      std::string err = "EOF buffer encountered. Users try to fetch data beyond the specified number of epochs.";
      RETURN_STATUS_UNEXPECTED(err);
    }
  }
  RETURN_IF_NOT_OK(cur_db_->PopRow(row));
  // Record profiling info
  if (tracing_ != nullptr) {
    uint64_t end_time = ProfilingTime::GetCurMilliSecond();
    cur_batch_num_++;
    RETURN_IF_NOT_OK(
      tracing_->Record(CONNECTOR_DEPTH, cur_connector_capacity_, cur_batch_num_, cur_connector_size_, end_time));
  }
  return Status::OK();
}

Status TreeAdapter::Launch() const {
  CHECK_FAIL_RETURN_UNEXPECTED(tree_ != nullptr, "Tree is a nullptr.");
  return tree_->Launch();
}

}  // namespace dataset
}  // namespace mindspore
