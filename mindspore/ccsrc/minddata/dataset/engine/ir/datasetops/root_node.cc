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

#include "minddata/dataset/engine/ir/datasetops/root_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for RootNode
RootNode::RootNode(std::shared_ptr<DatasetNode> child) : DatasetNode() {
  // The root node's parent must remain nullptr, which is set in the constructor of DatasetNode.
  AddChild(child);
  num_epochs_ = 0;
}

std::shared_ptr<DatasetNode> RootNode::Copy() {
  auto node = std::make_shared<RootNode>(nullptr);
  node->SetNumEpochs(num_epochs_);
  return node;
}

void RootNode::Print(std::ostream &out) const { out << Name(); }

Status RootNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // root node doesn't build a runtime Op. this function should return Status::Error when called.
  std::string err_msg = "Root node doesn't build a runtime Op";
  MS_LOG(ERROR) << err_msg;
  RETURN_STATUS_UNEXPECTED(err_msg);
}

// Function to validate the parameters for RootNode
Status RootNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (num_epochs_ <= 0 && num_epochs_ != -1) {
    std::string err_msg =
      "RootNode: num_epochs should be either -1 or positive integer, num_epochs: " + std::to_string(num_epochs_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (parent_ != nullptr) {
    std::string err_msg = "Internal error: root node should not have a parent";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (children_.size() != 1) {
    std::string err_msg = "Internal error: root node should have one child node";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (children_[0] == nullptr) {
    std::string err_msg = "Internal error: root node's child is a null pointer";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status RootNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<RootNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status RootNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<RootNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
