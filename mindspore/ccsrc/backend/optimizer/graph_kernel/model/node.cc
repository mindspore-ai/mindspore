/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/model/node.h"

#include "mindspore/core/ir/dtype/type_id.h"
#include "mindspore/core/ir/value.h"
#include "mindspore/core/ir/tensor.h"
#include "mindspore/core/utils/shape_utils.h"
#include "utils/utils.h"
#include "backend/kernel_compiler/common_utils.h"

namespace mindspore {
namespace opt {
namespace graphkernel {
void Node::DumpTensor(std::ostringstream &os) const {
  os << name_ << "[";
  for (size_t i = 0; i < shape.size(); i++) {
    os << shape[i];
    if (i + 1 < shape.size()) os << ",";
  }
  os << "]{" << kernel::TypeId2String(type) << "x" << format << "}";
}

void Node::AddInput(const NodePtr &new_input) {
  MS_EXCEPTION_IF_NULL(new_input);
  new_input->AddUser(this, inputs_.size());
  (void)inputs_.emplace_back(new_input);
}

void Node::SetInput(size_t i, const NodePtr &new_input) {
  MS_EXCEPTION_IF_NULL(new_input);
  if (i >= inputs_.size()) {
    MS_LOG(EXCEPTION) << "The index " << i << " is out of the inputs range " << inputs_.size();
  }
  auto &old_input = inputs_[i];
  old_input->RemoveUser(this, i);
  new_input->AddUser(this, i);
  inputs_[i] = new_input;
}

void Node::SetInputs(const NodePtrList &inputs) {
  if (!inputs_.empty()) {
    // remove the original inputs
    for (size_t i = 0; i < inputs_.size(); i++) {
      inputs_[i]->RemoveUser(this, i);
    }
    inputs_.clear();
  }
  inputs_.reserve(inputs.size());
  for (const auto &inp : inputs) {
    AddInput(inp);
  }
}

void Node::ReplaceWith(const NodePtr &other_node) {
  if (this->users_.empty()) return;
  // copy the users before traversal
  auto users = this->users_;
  for (auto &user : users) {
    for (auto idx : user.second) {
      user.first->SetInput(idx, other_node);
    }
  }
}
}  // namespace graphkernel
}  // namespace opt
}  // namespace mindspore
