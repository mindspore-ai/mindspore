/**
 * Copyright 20202-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "minddata/dataset/engine/datasetops/take_op.h"
#include "minddata/dataset/engine/opt/pass.h"
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
Status TakeNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto op = std::make_shared<TakeOp>(take_count_, connector_que_size_);
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Function to validate the parameters for TakeNode
Status TakeNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (take_count_ <= 0 && take_count_ != -1) {
    std::string err_msg =
      "TakeNode: take_count should be either -1 or positive integer, take_count: " + std::to_string(take_count_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Get Dataset size
Status TakeNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows;
  RETURN_IF_NOT_OK(children_[0]->GetDatasetSize(size_getter, estimate, &num_rows));
  *dataset_size = std::min(static_cast<int64_t>(take_count_), num_rows);
  dataset_size_ = *dataset_size;
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status TakeNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<TakeNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status TakeNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<TakeNode>(), modified);
}

Status TakeNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["count"] = take_count_;
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
