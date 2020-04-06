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

#include "ir/visitor.h"
#include "ir/func_graph.h"

namespace mindspore {
AnfNodePtr AnfVisitor::operator()(const opt::OptimizerPtr &, const AnfNodePtr &) { return nullptr; }
void AnfVisitor::Visit(const AnfNodePtr &node) { node->accept(this); }

void AnfVisitor::Visit(const CNodePtr &cnode) {
  for (auto &input : cnode->inputs()) {
    Visit(input);
  }
}

void AnfVisitor::Visit(const ValueNodePtr &vnode) {
  if (IsValueNode<FuncGraph>(vnode)) {
    auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
    Visit(func_graph->output());
  }
}

void AnfVisitor::Visit(const ParameterPtr &) {}

VisitFuncType AnfVisitor::Match(const PrimitivePtr &prim, const std::vector<opt::PredicateFuncType> &funcs) {
  auto fn = [prim, funcs, this](const AnfNodePtr &node) {
    if (!IsPrimitiveCNode(node, prim)) {
      return;
    }

    auto &inputs = node->cast<CNodePtr>()->inputs();
    auto funcs_size = funcs.size();
    auto inputs_size = inputs.size();

    // check the inputs are matched with the predicate functions
    if (funcs_size > 0) {
      // use the predicate function list to check the number of inputs
      if (funcs_size != (inputs_size - 1)) {
        return;
      }

      // check per input
      for (size_t i = 0; i < funcs_size; i++) {
        if (!funcs[i](inputs[i + 1])) {
          return;
        }
      }
    }

    // visit the inputs
    for (size_t i = 1; i < inputs_size; i++) {
      this->Visit(inputs[i]);
    }
  };

  return fn;
}
}  // namespace mindspore
