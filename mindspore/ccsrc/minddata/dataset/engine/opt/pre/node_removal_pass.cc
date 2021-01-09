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

#include "minddata/dataset/engine/opt/pre/node_removal_pass.h"
#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#include "minddata/dataset/engine/ir/datasetops/take_node.h"

namespace mindspore {
namespace dataset {

NodeRemovalPass::RemovalNodes::RemovalNodes() {}

// Perform RepeatNode removal check.
Status NodeRemovalPass::RemovalNodes::Visit(std::shared_ptr<RepeatNode> node, bool *const modified) {
  *modified = false;
  if (node->Count() == 1) {
    nodes_to_remove_.push_back(std::static_pointer_cast<DatasetNode>(node));
  }
  return Status::OK();
}

// Perform SkipNode removal check.
Status NodeRemovalPass::RemovalNodes::Visit(std::shared_ptr<SkipNode> node, bool *const modified) {
  *modified = false;
  if (node->Count() == 0) {
    nodes_to_remove_.push_back(std::static_pointer_cast<DatasetNode>(node));
  }
  return Status::OK();
}

// Perform TakeNode removal check.
Status NodeRemovalPass::RemovalNodes::Visit(std::shared_ptr<TakeNode> node, bool *const modified) {
  *modified = false;
  if (node->Count() == -1) {
    nodes_to_remove_.push_back(std::static_pointer_cast<DatasetNode>(node));
  }
  return Status::OK();
}

// constructor
NodeRemovalPass::NodeRemovalPass() {}

// Walk the tree to collect the nodes to remove, then removes them.
Status NodeRemovalPass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  MS_LOG(INFO) << "Pre pass: node removal pass started.";
  // Create the removal node pass which can identify which nodes need to be removed.
  std::unique_ptr<NodeRemovalPass::RemovalNodes> removal_nodes = std::make_unique<NodeRemovalPass::RemovalNodes>();
  RETURN_IF_NOT_OK(removal_nodes->Run(root_ir, modified));

  // Then, execute the removal of any nodes that were set up for removal
  for (auto node : removal_nodes->nodes_to_remove()) {
    RETURN_IF_NOT_OK(node->Drop());
  }
  MS_LOG(INFO) << "Pre pass: node removal pass complete.";
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
