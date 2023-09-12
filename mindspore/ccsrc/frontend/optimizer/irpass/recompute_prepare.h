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

#include "utils/hash_set.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/anf_visitor.h"
#include "include/common/utils/parallel_context.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
namespace irpass {
class SetCellOutputNoRecompute : public AnfVisitor {
 public:
  AnfNodePtr operator()(const OptimizerPtr &, const AnfNodePtr &node) override {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    const auto no_cell_reuse = context->CellReuseLevel() == CellReuseLevel::kNoCellReuse;
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
      mindspore::HashSet<CNodePtr> real_outputs;
      GetRealOutputNodes(output, &real_outputs);
      if (OutputAllNodes(real_outputs)) {
        MS_LOG(WARNING)
          << "All nodes in the graph " << fg->ToString()
          << " are the output nodes, which are set to not be recomputed. If you want to set these nodes to "
             "be recomputed, use the api recompute() of Primitive.";
      }
      for (const auto &real_output : real_outputs) {
        // Set the attr of cnode in case of shared primitives.
        if (no_cell_reuse) {
          real_output->AddAttr(kAttrRecompute, MakeValue(false));
        }

        if (parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kSemiAutoParallel ||
            parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kAutoParallel) {
          auto prim = GetCNodePrimitive(real_output);
          if (prim->HasAttr(kAttrSliceActivation) && GetValue<bool>(prim->GetAttr(kAttrSliceActivation))) {
            real_output->AddAttr(kAttrSliceActivation, MakeValue(true));
          }
        }
      }
    }
    if (no_cell_reuse) {
      fg->erase_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE);
    }
    return nullptr;
  }

  void GetRealOutputNodes(const AnfNodePtr &output, mindspore::HashSet<CNodePtr> *real_outputs) {
    MS_EXCEPTION_IF_NULL(output);
    MS_EXCEPTION_IF_NULL(real_outputs);
    auto output_cnode = output->cast<CNodePtr>();
    if (output_cnode == nullptr) {
      return;
    }
    auto input0 = output_cnode->input(0);
    MS_EXCEPTION_IF_NULL(input0);
    if (IsPrimitive(input0, prim::kPrimDepend) || IsPrimitive(input0, prim::kPrimTupleGetItem)) {
      GetRealOutputNodes(output_cnode->input(kRealInputIndexInDepend), real_outputs);
    } else if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
      auto &inputs = output_cnode->inputs();
      for (size_t i = 1; i < inputs.size(); ++i) {
        GetRealOutputNodes(output_cnode->input(i), real_outputs);
      }
    } else if (IsValueNode<FuncGraph>(input0)) {
      auto fg = GetValueNode<FuncGraphPtr>(input0);
      GetRealOutputNodes(fg->output(), real_outputs);
    } else if (input0->isa<CNode>()) {
      auto abs = input0->abstract();
      if (abs == nullptr || !abs->isa<abstract::AbstractFunction>()) {
        return;
      }
      auto abs_func = abs->cast<abstract::AbstractFunctionPtr>();
      if (abs_func->isa<abstract::AbstractFuncUnion>()) {
        auto visit_fn = [this, &real_outputs](const abstract::AbstractFuncAtomPtr &poss) {
          auto abs_fg = GetAbstractFuncGraph(poss);
          if (abs_fg != nullptr) {
            GetRealOutputNodes(abs_fg->output(), real_outputs);
          }
        };
        abs_func->Visit(visit_fn);
        return;
      }
      auto fg = GetAbstractFuncGraph(abs_func);
      if (fg != nullptr) {
        GetRealOutputNodes(fg->output(), real_outputs);
      }
    } else {
      real_outputs->insert(output_cnode);
    }
  }

  FuncGraphPtr GetAbstractFuncGraph(const abstract::AbstractFunctionPtr &abs) const {
    if (abs->isa<abstract::FuncGraphAbstractClosure>()) {
      auto abstract_func_graph = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
      return abstract_func_graph->func_graph();
    }
    if (abs->isa<abstract::PartialAbstractClosure>()) {
      auto abstract_partial_func = abs->cast<abstract::PartialAbstractClosurePtr>();
      auto abstract_fn = abstract_partial_func->fn();
      if (abstract_fn != nullptr && abstract_fn->isa<abstract::FuncGraphAbstractClosure>()) {
        auto abstract_func_graph = abstract_fn->cast<abstract::FuncGraphAbstractClosurePtr>();
        return abstract_func_graph->func_graph();
      }
    }
    return nullptr;
  }

  bool OutputAllNodes(const mindspore::HashSet<CNodePtr> &real_outputs) const {
    for (const auto &cnode : real_outputs) {
      const auto &inputs = cnode->inputs();
      for (const auto &input : inputs) {
        auto input_cnode = input->cast<CNodePtr>();
        if (input_cnode == nullptr || IsPrimitiveCNode(input_cnode, prim::kPrimLoad)) {
          continue;
        }
        if (real_outputs.find(input_cnode) == real_outputs.end()) {
          return false;
        }
      }
    }
    return true;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_RECOMPUTE_PREPARE_H_
