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

#include "minddata/dataset/engine/ir/datasetops/shuffle_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for ShuffleNode
ShuffleNode::ShuffleNode(std::shared_ptr<DatasetNode> child, int32_t shuffle_size, bool reset_every_epoch)
    : shuffle_size_(shuffle_size), shuffle_seed_(GetSeed()), reset_every_epoch_(reset_every_epoch) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> ShuffleNode::Copy() {
  auto node = std::make_shared<ShuffleNode>(nullptr, shuffle_size_, reset_every_epoch_);
  return node;
}

void ShuffleNode::Print(std::ostream &out) const {
  out << (Name() + "(shuffle_size:" + std::to_string(shuffle_size_) +
          ",reset_every_epoch:" + (reset_every_epoch_ ? "true" : "false") + ")");
}

// Function to build the ShuffleOp
Status ShuffleNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto op = std::make_shared<ShuffleOp>(shuffle_size_, shuffle_seed_, connector_que_size_, reset_every_epoch_);
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Function to validate the parameters for ShuffleNode
Status ShuffleNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateScalar("Shuffle", "shuffle_size", shuffle_size_, {1}, true));
  return Status::OK();
}

Status ShuffleNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["buffer_size"] = shuffle_size_;
  args["reset_each_epoch"] = reset_every_epoch_;
  *out_json = args;
  return Status::OK();
}

Status ShuffleNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> ds,
                              std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("buffer_size") != json_obj.end(), "Failed to find buffer_size");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("reset_each_epoch") != json_obj.end(), "Failed to find reset_each_epoch");
  int32_t buffer_size = json_obj["buffer_size"];
  bool reset_every_epoch = json_obj["reset_each_epoch"];
  *result = std::make_shared<ShuffleNode>(ds, buffer_size, reset_every_epoch);
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status ShuffleNode::Accept(IRNodePass *const p, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(p);
  RETURN_UNEXPECTED_IF_NULL(modified);
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<ShuffleNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status ShuffleNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(p);
  RETURN_UNEXPECTED_IF_NULL(modified);
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<ShuffleNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
