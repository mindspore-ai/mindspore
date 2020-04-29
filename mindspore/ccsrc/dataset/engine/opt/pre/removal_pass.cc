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
#include "dataset/engine/opt/pre/removal_nodes.h"
#include "dataset/engine/opt/pre/removal_pass.h"
#include "dataset/engine/execution_tree.h"

namespace mindspore {
namespace dataset {

// constructor
RemovalPass::RemovalPass() {}

// Runs a removal_nodes pass first to find out which nodes to remove, then removes them.
Status RemovalPass::RunOnTree(ExecutionTree *tree, bool *modified) {
  MS_LOG(INFO) << "Pre pass: removal pass started.";
  // Create the removal node pass which can identify which nodes need to be removed.
  std::unique_ptr<Pass> removal_nodes = std::make_unique<RemovalNodes>(this);
  RETURN_IF_NOT_OK(removal_nodes->Run(tree, modified));

  // Then, execute the removal of any nodes that were set up for removal
  for (auto node : removal_nodes_) {
    node->Remove();
  }
  MS_LOG(INFO) << "Pre pass: removal pass complete.";
  return Status::OK();
}

// Adds an operator to the list of operators to be removed
void RemovalPass::AddToRemovalList(std::shared_ptr<DatasetOp> dataset_op) { removal_nodes_.push_back(dataset_op); }
}  // namespace dataset
}  // namespace mindspore
