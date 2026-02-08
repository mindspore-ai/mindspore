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

#include "frontend/parallel/pass/interleave_parallel_branches.h"

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
#include "primitive/nn_op_name.h"
#include "primitive/other_op_name.h"
#include "primitive/sequence_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/pass/interleave_branches_utils.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"
#include "frontend/jit/ps/graph_circle_handler.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"

namespace mindspore {
namespace parallel {
constexpr auto kAttrParallelBranch = "parallel_branch";
constexpr auto kAttrTempBranchId = "temp_branch_id";
constexpr auto kAttrTempScopeId = "temp_scope_id";
constexpr size_t kMaxDepthToGetForkNode = 10;

// get valid branch shared node (ignore getitem and cast)
CNodePtr GetValidSharedNode(const CNodePtr &node, size_t recursive_depth = 0) {
  if (node == nullptr || recursive_depth > kMaxDepthToGetForkNode) {
    return nullptr;
  }

  static const std::set<std::string> kInvalidNodeName = {kCastOpName, kRmsNormGradOpName, kTupleGetItemOpName};
  auto node_name = common::AnfAlgo::GetCNodeName(node);
  if (kInvalidNodeName.find(node_name) != kInvalidNodeName.end()) {
    ++recursive_depth;
    auto input_node = node->input(1);
    if (input_node == nullptr) {
      return nullptr;
    }
    return GetValidSharedNode(input_node->cast<CNodePtr>(), recursive_depth);
  }
  return node;
}

// find branch fork node from merge node for backward branch parallel
CNodePtr GetForkNode(const CNodePtr &merge_node, size_t scope_id) {
  MS_EXCEPTION_IF_NULL(merge_node);
  auto entry_node = merge_node;
  if (IsPrimitiveCNode(merge_node, prim::kPrimConcat)) {
    auto make_tuple = merge_node->input(1);
    MS_EXCEPTION_IF_NULL(make_tuple);
    entry_node = make_tuple->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(entry_node);
  }

  auto branch_outputs = entry_node->inputs();
  size_t interleave_branch_num = branch_outputs.size() - 1;
  if (interleave_branch_num <= 1) {
    return nullptr;
  }

  std::queue<CNodePtr> to_visit;
  auto scope_id_value = MakeValue<size_t>(scope_id);
  for (size_t i = 0; i < interleave_branch_num; ++i) {
    auto branch_id = i + 1;
    auto branch_id_value = MakeValue<size_t>(branch_id);
    auto &branch_output = branch_outputs[branch_id];
    MS_EXCEPTION_IF_NULL(branch_output);
    auto branch_output_cnode = branch_output->cast<CNodePtr>();
    if (branch_output_cnode == nullptr) {
      return nullptr;
    }
    branch_output_cnode->AddAttr(kAttrTempScopeId, scope_id_value);
    branch_output_cnode->AddAttr(kAttrTempBranchId, branch_id_value);
    to_visit.emplace(branch_output_cnode);
  }

  auto shared_branch_id_value = MakeValue<size_t>(0);
  CNodePtr fork_node = nullptr;
  while (!to_visit.empty()) {
    auto node = to_visit.front();
    to_visit.pop();
    MS_EXCEPTION_IF_NULL(node);
    auto branch_id_value = node->GetAttr(kAttrTempBranchId);
    for (auto &input : node->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>()) {
        continue;
      }

      auto input_cnode = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      if (!input_cnode->HasAttr(kAttrTempScopeId)) {
        input_cnode->AddAttr(kAttrTempScopeId, scope_id_value);
      } else if (input_cnode->GetAttr(kAttrTempScopeId) != scope_id_value) {
        continue;
      }

      if (!input_cnode->HasAttr(kAttrTempBranchId)) {
        input_cnode->AddAttr(kAttrTempBranchId, branch_id_value);
        to_visit.emplace(input_cnode);
        continue;
      }

      if (input_cnode->GetAttr(kAttrTempBranchId) != branch_id_value) {
        input_cnode->AddAttr(kAttrTempBranchId, shared_branch_id_value);
        fork_node = GetValidSharedNode(input_cnode);
      }
    }
  }

  return fork_node;
}

void InterleaveParallelBranches(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_ENABLE_INTERLEAVE_PARALLEL_BRANCH);
  if (!is_enable) {
    return;
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  circle_handler::SetAttrToDepend(graph);
  std::vector<InterLeaveScopePtr> interleave_scopes;
  size_t scope_id = 0;
  std::unordered_map<std::string, bool> target_unique_id_map;
  HashMap<std::string, CNodePtr> matmul_unique_id_map;
  HashMap<CNodePtr, CNodePtr> matmul_grad_dual_map;
  for (const auto &child_graph : manager->func_graphs()) {
    MS_EXCEPTION_IF_NULL(child_graph);
    auto graph_orders = child_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    for (size_t i = 0; i < origin_nodes_topological.size(); ++i) {
      const auto &node = origin_nodes_topological[i];
      MS_EXCEPTION_IF_NULL(node);
      // clean scope and branch flags
      EraseInterLeaveBranchAttr(node);
      UpdateMatMulGradDualMap(node, &matmul_unique_id_map, &matmul_grad_dual_map);
      bool forward = true;
      auto prim_node = node->input(0);
      if (prim_node == nullptr || GetValuePtr<Primitive>(prim_node) == nullptr) {
        continue;
      }
      auto prim = common::AnfAlgo::GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(prim);
      // backward merge node
      bool optimize_matmul_dw_order = false;
      if (!prim->HasAttr(kAttrParallelBranch)) {
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
      if (!AnfUtils::IsRealKernel(node)) {
        continue;
      }
      auto current_scope = std::make_shared<InterLeaveScope>();
      current_scope->graph = child_graph;
      current_scope->fork_node = nullptr;
      current_scope->merge_node = GetValidSharedNode(node);
      current_scope->forward = forward;
      current_scope->scope_id = ++scope_id;
      if (optimize_matmul_dw_order) {
        current_scope->matmul_grad_dual_map = &matmul_grad_dual_map;
      } else {
        current_scope->matmul_grad_dual_map = nullptr;
      }

      if (forward && !common::AnfAlgo::IsRecompute(node)) {
        auto fork_node = GetForkNode(node, scope_id);
        current_scope->fork_node = fork_node;
        if (fork_node != nullptr && fork_node->HasPrimalAttr(kPrimalAttrUniqueId)) {
          auto enable_flag = GetValue<int64_t>(prim->GetAttr(kAttrParallelBranch));
          (void)target_unique_id_map.emplace(GetValue<std::string>(fork_node->GetPrimalAttr(kPrimalAttrUniqueId)),
                                             enable_flag == kEnableOptimizeMatMulDwOrderFlag);
        }
      }

      (void)interleave_scopes.emplace_back(current_scope);
    }
  }

  for (auto &interleave_scope : interleave_scopes) {
    InterleaveParallelBranches(interleave_scope, true);
  }
  circle_handler::DetectAndRevertGraphCircle(graph, manager, "InterleaveParallelBranches",
                                             "enable_interleave_parallel_branch");
}
}  // namespace parallel
}  // namespace mindspore
