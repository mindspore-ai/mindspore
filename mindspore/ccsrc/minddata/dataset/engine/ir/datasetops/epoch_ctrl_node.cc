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

#include "minddata/dataset/engine/ir/datasetops/epoch_ctrl_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for EpochCtrlNode
EpochCtrlNode::EpochCtrlNode(std::shared_ptr<DatasetNode> child, int32_t num_epochs) : num_epochs_(num_epochs) {
  // The root node's parent must set to null pointer.
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> EpochCtrlNode::Copy() {
  auto node = std::make_shared<EpochCtrlNode>(num_epochs_);
  return node;
}

void EpochCtrlNode::Print(std::ostream &out) const { out << Name() + "(epoch:" + std::to_string(num_epochs_) + ")"; }

// Function to build the EpochCtrlOp
Status EpochCtrlNode::Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops) {
  node_ops->push_back(std::make_shared<EpochCtrlOp>(num_epochs_));
  return Status::OK();
}

// Function to validate the parameters for EpochCtrlNode
Status EpochCtrlNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (num_epochs_ <= 0 && num_epochs_ != -1) {
    std::string err_msg =
      "EpochCtrlNode: num_epochs should be either -1 or positive integer, num_epochs: " + std::to_string(num_epochs_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (children_.size() != 1 || children_[0] == nullptr) {
    std::string err_msg = "Internal error: epoch control node should have one child node";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
