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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GRAD_PARTIAL_TRANSFORM_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GRAD_PARTIAL_TRANSFORM_H_

#include <vector>
#include <algorithm>

#include "frontend/optimizer/irpass.h"
#include "mindspore/core/ops/array_ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {S_Prim_grad, {UpackGraph, Partial{fg, args},}} -> {Partial{{S_Prim_grad, ...}, args}}
class GradPartialTransform : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto grad_cnode = dyn_cast<CNode>(node);
    if (grad_cnode == nullptr || grad_cnode->inputs().empty()) {
      MS_LOG(INTERNAL_EXCEPTION) << "GradPartialTransform encounter invalid node: " << node->DebugString();
    }
    const auto &value = GetCNodeValueWithoutDoSignature(grad_cnode);
    if (value == nullptr || !value->isa<prim::GradOperation>()) {
      return nullptr;
    }
    auto unpack_graph_node = grad_cnode->input(1);
    auto prim = GetCNodePrimitive(unpack_graph_node);
    if (prim == nullptr || !prim->isa<prim::UnpackGraphPrimitive>()) {
      return nullptr;
    }
    auto unpack_graph_cnode = dyn_cast<CNode>(unpack_graph_node);
    MS_EXCEPTION_IF_NULL(unpack_graph_cnode);
    auto partial_node = unpack_graph_cnode->input(1);
    if (!IsPrimitiveCNode(partial_node, prim::kPrimPartial)) {
      return nullptr;
    }
    if (transformed_nodes_.count(node) != 0) {
      return nullptr;
    }
    auto partial_cnode = dyn_cast<CNode>(partial_node);
    MS_EXCEPTION_IF_NULL(partial_cnode);
    AnfNodePtrList inputs = {NewValueNode(prim::kPrimPartial), node};
    constexpr auto ignored_partial_input_count = 2;
    (void)std::transform(partial_cnode->inputs().cbegin() + ignored_partial_input_count, partial_cnode->inputs().cend(),
                         std::back_inserter(inputs), [](const AnfNodePtr &inp) { return inp; });

    auto new_node = grad_cnode->func_graph()->NewCNodeInOrder(inputs);
    (void)transformed_nodes_.emplace(node);
    return new_node;
  }

 private:
  mindspore::HashSet<AnfNodePtr> transformed_nodes_;
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_GRAD_PARTIAL_TRANSFORM_H_
