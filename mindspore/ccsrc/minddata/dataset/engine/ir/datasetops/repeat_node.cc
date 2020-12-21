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

#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

RepeatNode::RepeatNode(std::shared_ptr<DatasetNode> child, int32_t count) : repeat_count_(count) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> RepeatNode::Copy() {
  auto node = std::make_shared<RepeatNode>(nullptr, this->repeat_count_);
  return node;
}

void RepeatNode::Print(std::ostream &out) const { out << Name() + "(count:" + std::to_string(repeat_count_) + ")"; }

Status RepeatNode::Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops) {
  node_ops->push_back(std::make_shared<RepeatOp>(repeat_count_));
  return Status::OK();
}

Status RepeatNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (repeat_count_ <= 0 && repeat_count_ != -1) {
    std::string err_msg = "RepeatNode: repeat_count should be either -1 or positive integer, repeat_count_: " +
                          std::to_string(repeat_count_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Get Dataset size
Status RepeatNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                  int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows;
  RETURN_IF_NOT_OK(children_[0]->GetDatasetSize(size_getter, estimate, &num_rows));
  if (num_rows > 0 && repeat_count_ > 0) {
    num_rows = num_rows * repeat_count_;
  }
  *dataset_size = num_rows;
  dataset_size_ = num_rows;
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status RepeatNode::Accept(IRNodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<RepeatNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status RepeatNode::AcceptAfter(IRNodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<RepeatNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
