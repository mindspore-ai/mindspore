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
#include "minddata/dataset/engine/opt/pre/node_removal_pass.h"
#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/engine/ir/datasetops/shuffle_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#include "minddata/dataset/engine/ir/datasetops/take_node.h"

namespace mindspore {
namespace dataset {

NodeRemovalPass::RemovalNodes::RemovalNodes() : is_caching_(false) {}

// Identifies the subtree below this node as a cached descendant tree.
Status NodeRemovalPass::RemovalNodes::Visit(std::shared_ptr<DatasetNode> node, bool *modified) {
  *modified = false;
  MS_LOG(INFO) << "Node removal pass: Operation with cache found, identified descendant tree.";
  if (node->IsCached()) {
    is_caching_ = true;
  }
  return Status::OK();
}

// Resets the tracking of the cache within the tree
Status NodeRemovalPass::RemovalNodes::VisitAfter(std::shared_ptr<DatasetNode> node, bool *modified) {
  *modified = false;
  MS_LOG(INFO) << "Removal pass: Descendant walk is complete.";
  if (is_caching_ && node->IsLeaf()) {
    // Mark this leaf node to indicate it is a descendant of an operator with cache.
    // This is currently used in non-mappable dataset (leaf) nodes to not add a ShuffleOp in DatasetNode::Build().
    node->HasCacheAbove();
  }
  is_caching_ = false;
  return Status::OK();
}

// Perform RepeatNode removal check.
Status NodeRemovalPass::RemovalNodes::Visit(std::shared_ptr<RepeatNode> node, bool *modified) {
  *modified = false;
  if (node->Count() == 1) {
    nodes_to_remove_.push_back(std::static_pointer_cast<DatasetNode>(node));
  }
  return Status::OK();
}

// Perform ShuffleNode removal check.
Status NodeRemovalPass::RemovalNodes::Visit(std::shared_ptr<ShuffleNode> node, bool *modified) {
  *modified = false;
  return Status::OK();
}

// Perform SkipNode removal check.
Status NodeRemovalPass::RemovalNodes::Visit(std::shared_ptr<SkipNode> node, bool *modified) {
  *modified = false;
  if (node->Count() == 0) {
    nodes_to_remove_.push_back(std::static_pointer_cast<DatasetNode>(node));
  }
  return Status::OK();
}

// Perform TakeNode removal check.
Status NodeRemovalPass::RemovalNodes::Visit(std::shared_ptr<TakeNode> node, bool *modified) {
  *modified = false;
  if (node->Count() == -1) {
    nodes_to_remove_.push_back(std::static_pointer_cast<DatasetNode>(node));
  }
  return Status::OK();
}

// constructor
NodeRemovalPass::NodeRemovalPass() {}

// Walk the tree to collect the nodes to remove, then removes them.
Status NodeRemovalPass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *modified) {
  MS_LOG(INFO) << "Pre pass: node removal pass started.";
  // Create the removal node pass which can identify which nodes need to be removed.
  std::unique_ptr<NodeRemovalPass::RemovalNodes> removal_nodes = std::make_unique<NodeRemovalPass::RemovalNodes>();
  RETURN_IF_NOT_OK(removal_nodes->Run(root_ir, modified));

  // Then, execute the removal of any nodes that were set up for removal
  for (auto node : removal_nodes->nodes_to_remove()) {
    RETURN_IF_NOT_OK(node->Remove());
  }
  MS_LOG(INFO) << "Pre pass: node removal pass complete.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
