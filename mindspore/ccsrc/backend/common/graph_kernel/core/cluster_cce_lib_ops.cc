/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/core/cluster_cce_lib_ops.h"
#include <algorithm>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/nn_ops.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/graph_builder.h"

namespace mindspore::graphkernel {

std::vector<PrimitivePtr> ClusterCceLibOps::GetClusterableOpList() {
  std::vector<OpWithLevel> cce_lib_ops_with_level = {
    {kAllTarget, OpLevel_0, prim::kPrimPagedAttention},
    {kAllTarget, OpLevel_0, prim::kPrimPagedAttentionMask},
    {kAllTarget, OpLevel_0, prim::kPrimReshapeAndCache},
    {kAllTarget, OpLevel_0, prim::kPrimMatMul},
  };
  const auto &flags = GraphKernelFlags::GetInstance();
  return GkUtils::GetValidOps(cce_lib_ops_with_level, flags.fusion_ops_level, flags.enable_cce_lib_ops,
                              flags.enable_cce_lib_ops_only, flags.disable_cce_lib_ops);
}

bool ClusterCceLibOps::IsClusterableOp(const AnfNodePtr &node) {
  if (AnfUtils::IsGraphKernel(node)) {
    // only fuse itself.
    return false;
  }
  if (GkUtils::IsKeepBasicNode(node)) {
    return false;
  }
  if (GkUtils::CceOpNotFusion(node)) {
    return false;
  }
  bool node_in_oplist = std::any_of(op_list_.begin(), op_list_.end(),
                                    [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  if (!node_in_oplist) {
    return false;
  }
  return true;
}

std::vector<size_t> GetPerm(const CNodePtr &custom_node, const CNodePtr &old_node) {
  auto custom_node_inputs = custom_node->inputs();
  auto old_node_inputs = old_node->inputs();
  std::vector<size_t> perm;
  perm.emplace_back(0);
  for (size_t i = 1; i < custom_node_inputs.size(); i++) {
    auto node = custom_node_inputs[i];
    size_t idx = 1;
    while (idx < old_node_inputs.size() && node != old_node_inputs[idx]) {
      idx++;
    }
    perm.emplace_back(idx);
  }
  return perm;
}
std::vector<AnfNodePtr> ReorderInputs(const std::vector<size_t> &perm, const std::vector<AnfNodePtr> &old_inputs,
                                      bool inputs_has_prim) {
  auto cmp = [](std::pair<const size_t, const size_t> first, std::pair<const size_t, const size_t> second) {
    return first.first > second.first;
  };
  std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>, decltype(cmp)> queue(cmp);
  for (size_t i = 0; i < old_inputs.size(); i++) {
    auto perm_idx = inputs_has_prim ? i : i + 1;
    std::pair<size_t, size_t> pair(perm[perm_idx], i);
    queue.push(pair);
    auto name = old_inputs[i]->fullname_with_scope();
  }
  std::vector<AnfNodePtr> new_inputs;
  while (!queue.empty()) {
    auto pair = queue.top();
    queue.pop();
    if (inputs_has_prim && pair.second == 0) {
      continue;
    }
    new_inputs.emplace_back(old_inputs[pair.second]);
  }
  return new_inputs;
}
void ClusterCceLibOps::ReorderSubGraphInputs(const AnfNodePtr &custom_node, const AnfNodePtrList &old_nodes) {
  auto old_node = old_nodes[0];
  if (!custom_node->isa<CNode>() || !old_node->isa<CNode>()) {
    return;
  }
  auto old_cnode = old_node->cast<CNodePtr>();
  auto custom_cnode = custom_node->cast<CNodePtr>();
  if (old_nodes.size() != 1 || !AnfUtils::IsGraphKernel(custom_node)) {
    return;
  }
  auto fg = GetCNodeFuncGraph(custom_cnode);
  auto nodes = TopoSort(fg->output());
  for (auto &node : nodes) {
    bool node_in_oplist = std::any_of(op_list_.begin(), op_list_.end(),
                                      [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
    if (!node_in_oplist) {
      continue;
    }

    auto name = node->fullname_with_scope();
    // Generate a permutation array 'perm' for the input from custom_cnode to the original cnode based on the original
    // cnode and custom_cnode
    auto perm = GetPerm(custom_cnode, old_cnode);
    std::vector<size_t> no_need_to_change(perm.size());
    size_t idx = 0;
    std::generate(no_need_to_change.begin(), no_need_to_change.end(), [&idx]() { return idx++; });
    // If 'perm' is strictly increasing, it indicates that the input order is correct, and no adjustment is needed.
    if (no_need_to_change == perm) {
      return;
    }
    // If 'perm' is not strictly increasing, adjust the input of custom_cnode its self and the subgraph of custom_cnode
    // according to 'perm'.
    auto fg_inputs = ReorderInputs(perm, fg->get_inputs(), false);
    fg->set_parameters(fg_inputs);
    auto cnode_inputs = ReorderInputs(perm, custom_cnode->inputs(), true);
    std::vector<AnfNodePtr> fn_inputs{NewValueNode(fg)};
    (void)fn_inputs.insert(fn_inputs.end(), cnode_inputs.cbegin(), cnode_inputs.cend());
    custom_cnode->set_inputs(fn_inputs);
  }
}

void ClusterCceLibOps::CreateFuncGraph(const FuncGraphPtr &func_graph, const std::vector<size_t> &nodes_id) {
  AnfNodePtrList old_nodes;
  (void)std::transform(nodes_id.begin(), nodes_id.end(), std::back_inserter(old_nodes),
                       [this](size_t id) { return this->nodes_[id]; });
  auto new_node = ReplaceNodesWithGraphKernelNode(old_nodes, func_graph, "cce_op");
  ReorderSubGraphInputs(new_node, old_nodes);
  if (new_node->isa<CNode>()) {
    auto cnode = new_node->cast<CNodePtr>();
    cnode->AddAttr("use_akg_cce", MakeValue(true));
  }
  if (GraphKernelFlags::GetInstance().dump_as_text) {
    DumpClusterInfo(old_nodes, new_node);
  }
}
}  // namespace mindspore::graphkernel
