/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_J_NODE_AND_USER_REMATCH_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_J_NODE_AND_USER_REMATCH_H_

#include <memory>

#include "include/frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"

namespace mindspore {
namespace opt {
namespace irpass {
// The J node should have only one user.
// %0 = J(net)
// %1 = %0(x)
// %2 = %0(y)
// =>
// %0 = J(net)
// %1 = %0(x)
// %2 = J(net)
// %3 = %2(y)
class JNodeAndUserRematch : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override {
    auto cnode = dyn_cast<CNode>(node);
    if (cnode == nullptr || cnode->empty()) {
      return nullptr;
    }
    auto input0 = cnode->input(0);
    if (!IsPrimitiveCNode(input0, prim::kPrimJ)) {
      return nullptr;
    }
    auto j_cnode = input0->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(j_cnode);
    auto manager = optimizer->manager();
    MS_EXCEPTION_IF_NULL(manager);
    const auto &users = manager->node_users()[j_cnode];
    if (users.size() <= 1) {
      return nullptr;
    }

    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    auto new_j_cnode = fg->NewCNodeInOrder(j_cnode->inputs());
    new_j_cnode->CloneCNodeInfo(j_cnode);
    auto inputs = cnode->inputs();
    inputs[0] = new_j_cnode;
    return fg->NewCNodeInOrder(inputs);
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_J_NODE_AND_USER_REMATCH_H_
