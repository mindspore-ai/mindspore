/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "ir/func_graph.h"
#include "ir/visitor.h"

namespace mindspore {
void AnfIrVisitor::Visit(const AnfNodePtr &node) { node->accept(this); }

void AnfIrVisitor::Visit(const CNodePtr &cnode) {
  for (auto &weak_input : cnode->weak_inputs()) {
    auto input = weak_input.lock();
    MS_EXCEPTION_IF_NULL(input);
    Visit(input);
  }
}

void AnfIrVisitor::Visit(const ValueNodePtr &vnode) {
  auto func_graph = GetValuePtr<FuncGraph>(vnode);
  if (func_graph != nullptr) {
    Visit(func_graph->output());
  }
}

void AnfIrVisitor::Visit(const ParameterPtr &) {}

VisitFuncType AnfIrVisitor::Match(const PrimitivePtr &prim, const std::vector<PredicateFuncType> &funcs) {
  return [prim, funcs, this](const AnfNodePtr &node) {
    if (!IsPrimitiveCNode(node, prim)) {
      return;
    }

    auto &inputs = node->cast_ptr<CNode>()->inputs();
    auto funcs_size = funcs.size();
    auto inputs_size = inputs.size();

    // Check the inputs are matched with the predicate functions.
    if (funcs_size > 0) {
      // Use the predicate function list to check the number of inputs.
      if (funcs_size != (inputs_size - 1)) {
        return;
      }

      // Check inputs.
      for (size_t i = 0; i < funcs_size; ++i) {
        if (!funcs[i](inputs[i + 1])) {
          return;
        }
      }
    }

    // Visit argument inputs.
    for (size_t i = 1; i < inputs_size; ++i) {
      this->Visit(inputs[i]);
    }
  };
}
}  // namespace mindspore
