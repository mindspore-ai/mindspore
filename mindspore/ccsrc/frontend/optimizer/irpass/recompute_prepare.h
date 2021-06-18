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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RECOMPUTE_PREPARE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RECOMPUTE_PREPARE_H_

#include <unordered_set>
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
namespace irpass {
class SetCellOutputNoRecompute : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    if (!IsValueNode<FuncGraph>(node)) {
      return nullptr;
    }

    auto fg = GetValueNode<FuncGraphPtr>(node);
    if (fg == nullptr || !fg->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
      return nullptr;
    }
    auto output = fg->output();
    if (output == nullptr) {
      return nullptr;
    }
    if (output->isa<CNode>()) {
      std::unordered_set<CNodePtr> real_outputs;
      GetRealOutputNodes(output, &real_outputs);
      for (const auto &real_output : real_outputs) {
        // Set the attr of cnode in case of shared primitives.
        real_output->AddAttr(kAttrRecompute, MakeValue(false));
      }
    }
    fg->erase_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE);
    return nullptr;
  }

  void GetRealOutputNodes(const AnfNodePtr &output, std::unordered_set<CNodePtr> *real_outputs) {
    MS_EXCEPTION_IF_NULL(output);
    MS_EXCEPTION_IF_NULL(real_outputs);
    if (!output->isa<CNode>()) {
      return;
    }
    auto output_cnode = output->cast<CNodePtr>();
    if (IsPrimitiveCNode(output_cnode, prim::kPrimDepend) || IsPrimitiveCNode(output_cnode, prim::kPrimTupleGetItem)) {
      GetRealOutputNodes(output_cnode->input(kRealInputIndexInDepend), real_outputs);
    } else if (IsPrimitiveCNode(output_cnode, prim::kPrimMakeTuple)) {
      auto &inputs = output_cnode->inputs();
      for (size_t i = 1; i < inputs.size(); ++i) {
        GetRealOutputNodes(output_cnode->input(i), real_outputs);
      }
    } else {
      real_outputs->insert(output_cnode);
    }
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RECOMPUTE_PREPARE_H_
