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

#include <string>
#include <nlohmann/json.hpp>
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/engine/opt/pre/deep_copy_pass.h"
#include "minddata/dataset/engine/ir/datasetops/root_node.h"

namespace mindspore {
namespace dataset {

DeepCopyPass::DeepCopyPass() {
  root_ = std::make_shared<RootNode>();
  parent_ = root_.get();
}

Status DeepCopyPass::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {
  *modified = true;
  // Do a nested-loop walk to check whether a node has the same child more than once.
  // This is an artificial restriction. We can support it since we will do a clone of the input tree in this pass.
  // Example:  ds2 = ds1 + ds1;
  auto children = node->Children();
  if (children.size() > 0) {
    for (auto it1 = children.begin(); it1 != children.end() - 1; ++it1) {
      for (auto it2 = it1 + 1; it2 != children.end(); ++it2) {
        if (*it1 == *it2) {
          std::string err_msg = "The same node " + (*it1)->Name() + " is a child of its parent more than once.";
          RETURN_STATUS_UNEXPECTED(err_msg);
        }
      }
    }
  }

  // Clone a new copy of this node
  std::shared_ptr<DatasetNode> new_node = node->Copy();
  // Temporary fix to set the num_workers to each cloned node.
  // This can be improved by adding a new method in the base class DatasetNode to transfer the properties to
  // the cloned node. Each derived class's Copy() will need to include this method.
  new_node->SetNumWorkers(node->num_workers());
  // This method below assumes a DFS walk and from the first child to the last child.
  // Future: A more robust implementation that does not depend on the above assumption.
  RETURN_IF_NOT_OK(parent_->AppendChild(new_node));

  // Then set this node to be a new parent to accept a copy of its next child
  parent_ = new_node.get();
  return Status::OK();
}

Status DeepCopyPass::VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) {
  *modified = true;
  // After visit the node, move up to its parent
  parent_ = parent_->parent_;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
