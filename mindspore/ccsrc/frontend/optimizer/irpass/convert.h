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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONVERT_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONVERT_H_

#include <vector>

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"
#include "frontend/operator/ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
// {prim::kPrimPrint, Xs} -> {prim::kPrimPrint, {prim::kPrinMakeTuple, Xs}}
class PrintTupleWrapper : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsPrimitiveCNode(node, prim::kPrimPrint)) {
      return nullptr;
    }

    // already be {prim::kPrimPrint, {prim::kPrinMakeTuple, Xs}}
    auto cnode = node->cast<CNodePtr>();
    if (cnode->size() == 2 && IsPrimitiveCNode(cnode->input(1), prim::kPrimMakeTuple)) {
      return nullptr;
    }

    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));

    // {prim::kPrimPrint, Xs}
    auto &inputs = cnode->inputs();
    (void)args.insert(args.end(), inputs.begin() + 1, inputs.end());

    // {prim::kPrinMakeTuple, Xs}
    auto fg = node->func_graph();
    auto tuple = NewCNode(args, fg);
    auto print = GetValueNode<PrimitivePtr>(cnode->input(0));
    return NewCNode({NewValueNode(print), tuple}, fg);
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // #ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CONVERT_H_
