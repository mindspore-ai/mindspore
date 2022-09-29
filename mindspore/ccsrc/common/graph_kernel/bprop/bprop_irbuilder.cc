/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "common/graph_kernel/bprop/bprop_irbuilder.h"

#include <algorithm>
#include <queue>
#include <set>
#include "include/common/utils/utils.h"

namespace mindspore {
namespace expander {
namespace bprop {
bool BpropIRBuilder::Run(const NodePtrList &inputs, const DAttr &attrs, std::vector<CNodePtr> *outputs,
                         DoutUser *dout_user) {
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(dout_user);
  if (!BpropIRBuilderFactory::Instance().HasOp(name())) {
    return false;
  }
  inputs_ptr_ = &inputs;
  attrs_ptr_ = &attrs;
  auto func = BpropIRBuilderFactory::Instance().GetBuilder(name());
  auto output_nodes = func(this);
  outputs->reserve(output_nodes.size());
  (void)std::transform(output_nodes.cbegin(), output_nodes.cend(), std::back_inserter(*outputs),
                       [](const NodePtr &node) {
                         auto cnode = node->get<CNodePtr>();
                         MS_EXCEPTION_IF_NULL(cnode);
                         return cnode;
                       });
  FindDoutUsers(*outputs, dout_user);
  return true;
}

void BpropIRBuilder::FindDoutUsers(const std::vector<CNodePtr> &outputs, DoutUser *dout_user) const {
  std::set<AnfNodePtr> visited;
  // do not visit the inputs again.
  std::for_each(inputs_ptr_->cbegin(), inputs_ptr_->cend(),
                [&visited](const NodePtr &node) { visited.insert(node->get()); });

  std::queue<CNodePtr> que;
  std::for_each(outputs.cbegin(), outputs.cend(), [&que](const CNodePtr &cnode) { que.push(cnode); });

  AnfNodePtr dout = inputs_ptr_->back()->get();
  while (!que.empty()) {
    auto node = que.front();
    que.pop();
    for (size_t i = 1; i < node->size(); ++i) {
      const auto &inp = node->input(i);
      if (inp == dout) {
        (void)dout_user->emplace_back(node, i);
      }
      if (inp->isa<CNode>() && visited.count(inp) == 0) {
        (void)visited.insert(inp);
        que.push(inp->cast<CNodePtr>());
      }
    }
  }
}

ValuePtr BpropIRBuilder::GetAttr(const std::string &attr) const {
  auto iter = attrs_ptr_->find(attr);
  if (iter != attrs_ptr_->end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "The attr " << attr << " does not exist in op " << name();
  return nullptr;
}

NodePtr BpropIRBuilder::GetInput(size_t i) const {
  if (i >= inputs_ptr_->size()) {
    MS_LOG(EXCEPTION) << "For " << name_ << ", the index " << i << " is out of range of inputs size "
                      << inputs_ptr_->size();
  }
  return (*inputs_ptr_)[i];
}

ShapeVector BpropIRBuilder::GetShape(const NodePtr &node) const {
  auto abs = node->get()->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto shape = abs->BuildShape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->isa<abstract::Shape>()) {
    return shape->cast<abstract::ShapePtr>()->shape();
  } else if (shape->isa<abstract::SequenceShape>()) {
    MS_LOG(EXCEPTION) << "The output of node " << node->get()->ToString() << " is a tuple.";
  }
  return {};
}

TypePtr BpropIRBuilder::GetDtype(const NodePtr &node) const {
  auto abs = node->get()->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  auto dtype = abs->BuildType();
  MS_EXCEPTION_IF_NULL(dtype);
  if (dtype->isa<TensorType>()) {
    return dtype->cast<TensorTypePtr>()->element();
  } else if (dtype->isa<Tuple>()) {
    MS_LOG(EXCEPTION) << "The output of node " << node->get()->ToString() << " is a tuple.";
  }
  return dtype;
}

ValuePtr BpropIRBuilder::GetAttr(const NodePtr &node, const std::string &attr) const {
  auto p = GetCNodePrimitive(node->get());
  MS_EXCEPTION_IF_NULL(p);
  return p->GetAttr(attr);
}
}  // namespace bprop
}  // namespace expander
}  // namespace mindspore
