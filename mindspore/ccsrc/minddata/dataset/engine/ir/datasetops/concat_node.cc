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

#include "minddata/dataset/engine/ir/datasetops/concat_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/concat_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Function to build ConcatOp
ConcatNode::ConcatNode(const std::vector<std::shared_ptr<DatasetNode>> &datasets,
                       const std::shared_ptr<SamplerObj> &sampler,
                       const std::vector<std::pair<int, int>> &children_flag_and_nums,
                       const std::vector<std::pair<int, int>> &children_start_end_index)
    : sampler_(sampler),
      children_flag_and_nums_(children_flag_and_nums),
      children_start_end_index_(children_start_end_index) {
  nary_op_ = true;
  for (auto const &child : datasets) AddChild(child);
}

std::shared_ptr<DatasetNode> ConcatNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->Copy();
  // create an empty vector to copy a concat
  auto node = std::make_shared<ConcatNode>(std::vector<std::shared_ptr<DatasetNode>>(), sampler,
                                           children_flag_and_nums_, children_start_end_index_);
  return node;
}

void ConcatNode::Print(std::ostream &out) const { out << Name(); }

Status ConcatNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (children_.size() < 2) {
    std::string err_msg = "ConcatNode: concatenated datasets are not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (find(children_.begin(), children_.end(), nullptr) != children_.end()) {
    std::string err_msg = "ConcatNode: concatenated datasets should not be null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // Either one of children_flag_and_nums_ or children_start_end_index_ should be non-empty.
  if ((children_flag_and_nums_.empty() && !children_start_end_index_.empty()) ||
      (!children_flag_and_nums_.empty() && children_start_end_index_.empty())) {
    std::string err_msg = "ConcatNode: children_flag_and_nums and children_start_end_index should be used together";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

Status ConcatNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  if (children_flag_and_nums_.empty() || children_start_end_index_.empty()) {
    node_ops->push_back(std::make_shared<ConcatOp>(connector_que_size_));
  } else {
    node_ops->push_back(std::make_shared<ConcatOp>(connector_que_size_, sampler_->Build(), children_flag_and_nums_,
                                                   children_start_end_index_));
  }

  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status ConcatNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<ConcatNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status ConcatNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<ConcatNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
