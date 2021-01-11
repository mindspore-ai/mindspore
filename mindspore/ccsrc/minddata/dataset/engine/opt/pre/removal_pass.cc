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

#include <vector>
#include <algorithm>
#include "minddata/dataset/engine/opt/pre/removal_pass.h"
#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {

RemovalPass::RemovalNodes::RemovalNodes() : is_caching_(false) {}

#ifndef ENABLE_ANDROID
// Identifies the subtree below this node as a cached descendant tree.
Status RemovalPass::RemovalNodes::PreRunOnNode(std::shared_ptr<CacheOp> node, bool *const modified) {
  *modified = false;
  MS_LOG(INFO) << "Removal pass: CacheOp found, identified descendant tree.";
  is_caching_ = true;
  return Status::OK();
}

// Resets the tracking of the cache within the tree
Status RemovalPass::RemovalNodes::RunOnNode(std::shared_ptr<CacheOp> node, bool *const modified) {
  *modified = false;
  MS_LOG(INFO) << "Removal pass: cache descendant tree complete.";
  is_caching_ = false;
  return Status::OK();
}
#endif

// Perform ShuffleOp removal check.
Status RemovalPass::RemovalNodes::RunOnNode(std::shared_ptr<ShuffleOp> node, bool *const modified) {
  *modified = false;
  // If we are in a cache descendant tree, then this shuffle op needs to be removed
  if (is_caching_) {
    MS_LOG(INFO) << "ShuffleOp identified for removal (CacheOp is in ascendant tree)";
    nodes_to_remove_.push_back(std::static_pointer_cast<DatasetOp>(node));
  }
  return Status::OK();
}

// constructor
RemovalPass::RemovalPass() {}

// Walk the tree to collect the nodes to remove, then removes them.
Status RemovalPass::RunOnTree(ExecutionTree *tree, bool *const modified) {
  MS_LOG(INFO) << "Pre pass: removal pass started.";
  // Create the removal node pass which can identify which nodes need to be removed.
  std::unique_ptr<RemovalPass::RemovalNodes> removal_nodes = std::make_unique<RemovalPass::RemovalNodes>();
  RETURN_IF_NOT_OK(removal_nodes->Run(tree, modified));

  // Then, execute the removal of any nodes that were set up for removal
  for (auto node : removal_nodes->nodes_to_remove()) {
    RETURN_IF_NOT_OK(node->Remove());
  }
  MS_LOG(INFO) << "Pre pass: removal pass complete.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
