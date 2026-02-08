/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/interleave_split_concat_branches.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <queue>
#include <stack>
#include <unordered_map>
#include "primitive/array_op_name.h"
#include "primitive/math_op_name.h"
#include "primitive/other_op_name.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/pass/interleave_branches_utils.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"
#include "frontend/jit/ps/graph_circle_handler.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace parallel {
auto const kEnableInterleave = "enable_interleave";

void InterleaveSplitConcatBranches(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_ENABLE_INTERLEAVE_SPLIT_CONCAT_BRANCH);
  if (!is_enable) {
    return;
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  circle_handler::SetAttrToDepend(graph);
  std::vector<InterLeaveScopePtr> interleave_scopes;

  // Use unique id to find correspond backward ops
  std::unordered_map<std::string, bool> target_unique_id_map;
  HashMap<std::string, CNodePtr> matmul_unique_id_map;
  size_t scope_id = 0;
  HashMap<CNodePtr, CNodePtr> matmul_grad_dual_map;
  for (const auto &child_graph : manager->func_graphs()) {
    MS_EXCEPTION_IF_NULL(child_graph);
    auto graph_orders = child_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    InterLeaveScopePtr current_scope = nullptr;
    std::stack<InterLeaveScopePtr> scope_stack;

    for (const auto &node : origin_nodes_topological) {
      MS_EXCEPTION_IF_NULL(node);
      EraseInterLeaveBranchAttr(node);
      UpdateMatMulGradDualMap(node, &matmul_unique_id_map, &matmul_grad_dual_map);
      // Only split and concat node is the target node
      bool is_split = IsPrimitiveCNode(node, prim::kPrimSplit);
      bool is_concat = IsPrimitiveCNode(node, prim::kPrimConcat);
      if (!is_split && !is_concat) {
        continue;
      }

      bool forward = true;
      auto prim = common::AnfAlgo::GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(prim);
      // Node with enable interleave flag or its backward node
      bool optimize_matmul_dw_order = false;
      if (!prim->HasAttr(kEnableInterleave)) {
        if (!node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
          continue;
        }
        auto iter_forward =
          target_unique_id_map.find(GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrForwardUniqueId)));
        if (iter_forward == target_unique_id_map.end()) {
          continue;
        }
        optimize_matmul_dw_order = iter_forward->second;
        forward = false;
      }

      if (forward && node->HasPrimalAttr(kPrimalAttrUniqueId)) {
        auto enable_flag = GetValue<int64_t>(prim->GetAttr(kEnableInterleave));
        (void)target_unique_id_map.emplace(GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrUniqueId)),
                                           enable_flag == kEnableOptimizeMatMulDwOrderFlag);
      }

      if (is_split) {
        current_scope = std::make_shared<InterLeaveScope>();
        current_scope->graph = child_graph;
        current_scope->fork_node = node;
        current_scope->forward = forward;
        current_scope->scope_id = ++scope_id;
        if (optimize_matmul_dw_order) {
          current_scope->matmul_grad_dual_map = &matmul_grad_dual_map;
        } else {
          current_scope->matmul_grad_dual_map = nullptr;
        }
        scope_stack.push(current_scope);
      }

      if (is_concat) {
        if (scope_stack.empty()) {
          continue;
        }
        current_scope = scope_stack.top();
        current_scope->merge_node = node;
        interleave_scopes.emplace_back(current_scope);
        scope_stack.pop();
      }
    }
  }

  // Handle each split/concat scope
  for (auto &interleave_scope : interleave_scopes) {
    InterleaveParallelBranches(interleave_scope);
  }

  circle_handler::DetectAndRevertGraphCircle(graph, manager, "InterleaveSplitConcatBranches",
                                             "enable_interleave_split_concat_branch");
}
}  // namespace parallel
}  // namespace mindspore
