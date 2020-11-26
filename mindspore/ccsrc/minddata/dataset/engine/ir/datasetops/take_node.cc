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

#include "minddata/dataset/engine/ir/datasetops/take_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/take_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

// Constructor for TakeNode
TakeNode::TakeNode(std::shared_ptr<DatasetNode> child, int32_t count) : take_count_(count) { this->AddChild(child); }

std::shared_ptr<DatasetNode> TakeNode::Copy() {
  auto node = std::make_shared<TakeNode>(nullptr, take_count_);
  return node;
}

void TakeNode::Print(std::ostream &out) const { out << Name() + "(num_rows:" + std::to_string(take_count_) + ")"; }

// Function to build the TakeOp
std::vector<std::shared_ptr<DatasetOp>> TakeNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<TakeOp>(take_count_, connector_que_size_));
  return node_ops;
}

// Function to validate the parameters for TakeNode
Status TakeNode::ValidateParams() {
  if (take_count_ <= 0 && take_count_ != -1) {
    std::string err_msg =
      "TakeNode: take_count should be either -1 or positive integer, take_count: " + std::to_string(take_count_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
