/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/model/node.h"
#include <algorithm>
#include <sstream>
#include <utility>
#include "abstract/utils.h"

namespace mindspore::graphkernel::inner {
void Node::SetBaseInfo(const NodeBaseList &baseinfo) {
  this->shape = baseinfo[0].shape;
  this->type = baseinfo[0].type;
  this->format = baseinfo[0].format;
  if (baseinfo.size() > 1) {
    outputs_ = baseinfo;
  }
}

std::string Node::ToString() const {
  std::ostringstream oss;
  oss << debug_name() << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    oss << shape[i];
    if (i + 1 < shape.size()) {
      oss << ",";
    }
  }
  oss << "]{" << TypeIdToString(type) << "x" << format << "}";
  return oss.str();
}

abstract::AbstractBasePtr Node::ToAbstract() const {
  if (outputs_.empty()) {
    return std::make_shared<abstract::AbstractTensor>(TypeIdToType(this->type), this->shape);
  }
  AbstractBasePtrList abs_list(outputs_.size());
  (void)std::transform(outputs_.cbegin(), outputs_.cend(), abs_list.begin(), [](const NodeBase &node) {
    return std::make_shared<abstract::AbstractTensor>(TypeIdToType(node.type), node.shape);
  });
  return std::make_shared<abstract::AbstractTuple>(std::move(abs_list));
}

void Node::AddInput(const NodePtr &new_input) {
  MS_EXCEPTION_IF_NULL(new_input);
  new_input->AddUser(this, inputs_.size());
  (void)inputs_.emplace_back(new_input);
}

void Node::SetInput(size_t i, const NodePtr &new_input) {
  MS_EXCEPTION_IF_NULL(new_input);
  if (i >= inputs_.size()) {
    MS_LOG(EXCEPTION) << "The index " << i << " is out of the inputs range [0, " << inputs_.size() << ")";
  }
  auto &old_input = inputs_[i];
  old_input->RemoveUser(this, i);
  new_input->AddUser(this, i);
  inputs_[i] = new_input;
}

void Node::SetInputs(const NodePtrList &inputs) {
  ClearInputs();
  inputs_.reserve(inputs.size());
  for (const auto &inp : inputs) {
    AddInput(inp);
  }
}

void Node::ClearInputs() noexcept {
  if (!inputs_.empty()) {
    // remove the original inputs
    for (size_t i = 0; i < inputs_.size(); i++) {
      inputs_[i]->RemoveUser(this, i);
    }
    inputs_.clear();
  }
}

void Node::ReplaceWith(const NodePtr &other_node) {
  if (this->users_.empty()) {
    return;
  }
  // the users_ will be changed, so we copy the users before traversal
  auto users = this->users_;
  for (auto &user : users) {
    for (const auto &idx : user.second) {
      user.first->SetInput(idx, other_node);
    }
  }
}

void Node::RemoveUser(Node *const user, size_t index) {
  if (auto iter = users_.find(user); iter != users_.end()) {
    (void)iter->second.erase(index);
    if (iter->second.empty()) {
      (void)users_.erase(iter);
    }
  }
}

size_t Node::tensor_size(bool in_bytes) const {
  size_t size = LongToSize(abstract::ShapeSize(this->shape));
  return in_bytes ? abstract::TypeIdSize(this->type) * size : size;
}
}  // namespace mindspore::graphkernel::inner
