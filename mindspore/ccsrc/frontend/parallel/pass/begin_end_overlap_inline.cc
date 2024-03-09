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

#include "frontend/parallel/pass/begin_end_overlap_inline.h"
#include <memory>
#include <list>
#include "mindspore/core/ops/framework_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "abstract/abstract_function.h"
#include "ir/func_graph_cloner.h"

namespace mindspore {
namespace parallel {
namespace {
bool IsLazyInlineBackward(const FuncGraphPtr &bg) {
  for (auto &entry : bg->func_graph_cnodes_index()) {
    auto cnode = entry.first->first->cast<CNodePtr>();
    auto index = entry.first->second;
    if (index == 1 && IsPrimitive(cnode->inputs().at(0), prim::kPrimPartial)) {
      // To find real calling.
      auto fg = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      if (fg->has_attr(FUNC_GRAPH_FLAG_NO_INLINE)) {
        return true;
      } else {
        return false;
      }
    }
  }
  return false;
}

FuncGraphPtr GetAbstractFunc(const CNodePtr &node) {
  if (node->input(0)->isa<CNode>() && node->input(0)->abstract() != nullptr) {
    auto abs = node->input(0)->abstract();
    if (abs->isa<abstract::FuncGraphAbstractClosure>()) {
      const auto &abstract_func_graph = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
      return abstract_func_graph->func_graph();
    } else if (abs->isa<abstract::PartialAbstractClosure>()) {
      const auto &abstract_partial_func = abs->cast<abstract::PartialAbstractClosurePtr>();
      const auto &abstract_fn = abstract_partial_func->fn();
      if (abstract_fn->isa<abstract::FuncGraphAbstractClosure>()) {
        const auto &abstract_func_graph = abstract_fn->cast<abstract::FuncGraphAbstractClosurePtr>();
        return abstract_func_graph->func_graph();
      }
    }
  }
  return nullptr;
}

void InlineExpandFuncGraph(const CNodePtr &expanding_node, const FuncGraphPtr &expanded_graph) {
  auto main_graph = expanding_node->func_graph();
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = Manage(main_graph, true);
    main_graph->set_manager(mng);
  }
  MS_EXCEPTION_IF_NULL(expanding_node);
  AnfNodePtrList inp(expanding_node->inputs().begin() + 1, expanding_node->inputs().end());
  // expand bg node from partial
  auto out =
    InlineClone(expanded_graph, main_graph, inp, expanding_node->input(0)->scope(), expanding_node->debug_info());
  (void)mng->Replace(expanding_node, out);
}

// expand bg node from partial
void InlineExpandPartialFuncGraph(const CNodePtr &expanding_node, const FuncGraphPtr &expanded_graph,
                                  const AnfNodePtrList &partial_params) {
  auto main_graph = expanding_node->func_graph();
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = Manage(main_graph, true);
    main_graph->set_manager(mng);
  }
  MS_EXCEPTION_IF_NULL(expanding_node);
  AnfNodePtrList inp(expanding_node->inputs().begin() + 1, expanding_node->inputs().end());
  (void)inp.insert(inp.begin(), partial_params.begin(), partial_params.end());
  auto out =
    InlineClone(expanded_graph, main_graph, inp, expanding_node->input(0)->scope(), expanding_node->debug_info());
  (void)mng->Replace(expanding_node, out);
}

bool SkipBeginEndOverlapInline(const FuncGraphPtr &graph, FuncGraphPtr *fg, FuncGraphPtr *bg, CNodePtrList *fg_call,
                               CNodePtrList *bg_call) {
  std::list<CNodePtr> graph_orders = graph->GetOrderedCnodes();
  for (auto &node : graph_orders) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsValueNode<FuncGraph>(node->input(0))) {
      FuncGraphPtr sub_graph = node->input(0)->cast<ValueNodePtr>()->value()->cast<FuncGraphPtr>();
      MS_EXCEPTION_IF_NULL(sub_graph);
      if (sub_graph->has_attr(FUNC_GRAPH_FLAG_NO_INLINE)) {
        (void)fg_call->emplace_back(node);
        *fg = sub_graph;
      }
    } else {
      auto func = GetAbstractFunc(node);
      if (func != nullptr && IsLazyInlineBackward(func)) {
        *bg = func;
        (void)bg_call->emplace_back(node);
      }
    }
  }
  constexpr size_t mini_micro_size = 2;
  return fg_call->size() < mini_micro_size || bg_call->size() < mini_micro_size;
}

}  // namespace

void BeginEndOverlapInlineOpt(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    return;
  }
  MS_LOG(INFO) << "Begin end overlap inline start.";
  // find micro fg call
  FuncGraphPtr bg;
  FuncGraphPtr fg;
  CNodePtrList fg_call;
  CNodePtrList bg_call;
  if (SkipBeginEndOverlapInline(graph, &fg, &bg, &fg_call, &bg_call)) {
    return;
  }

  // Inline the last micro fg
  InlineExpandFuncGraph(fg_call.back(), fg);
  // Inline the last micro bg
  AnfNodePtrList last_micro_bg_partial_params;
  CNodePtr last_micro_bg_partial_call;
  for (auto &entry : bg->func_graph_cnodes_index()) {
    auto cnode = entry.first->first->cast<CNodePtr>();
    auto index = entry.first->second;
    if (index == 1 && IsPrimitive(cnode->inputs().at(0), prim::kPrimPartial)) {
      // The partial node is in the root graph after last micro forward inline
      if (graph == cnode->func_graph()) {
        last_micro_bg_partial_call = cnode;
        (void)last_micro_bg_partial_params.insert(last_micro_bg_partial_params.begin(),
                                                  cnode->inputs().begin() + kIndex2, cnode->inputs().end());
        break;
      }
    }
  }
  InlineExpandPartialFuncGraph(bg_call.back(), bg, last_micro_bg_partial_params);

  // Inline the first micro fg
  InlineExpandFuncGraph(fg_call[0], fg);
  AnfNodePtrList first_micro_bg_partial_params;
  for (auto &entry : bg->func_graph_cnodes_index()) {
    auto cnode = entry.first->first->cast<CNodePtr>();
    auto index = entry.first->second;
    if (index == 1 && IsPrimitive(cnode->inputs().at(0), prim::kPrimPartial)) {
      // The partial node is in the root graph after first micro forward inline.
      MS_EXCEPTION_IF_NULL(fg);
      if (graph == cnode->func_graph() && cnode != last_micro_bg_partial_call) {
        (void)first_micro_bg_partial_params.insert(first_micro_bg_partial_params.begin(),
                                                   cnode->inputs().begin() + kIndex2, cnode->inputs().end());
        break;
      }
    }
  }
  // Inline the first micro bg
  InlineExpandPartialFuncGraph(bg_call[0], bg, first_micro_bg_partial_params);
}
}  // namespace parallel
}  // namespace mindspore
