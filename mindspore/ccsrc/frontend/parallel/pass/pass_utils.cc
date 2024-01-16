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

#include "frontend/parallel/pass/pass_utils.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <string>

#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/operator/ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/nn_optimizer_ops.h"

namespace mindspore {
namespace parallel {
bool IsForwardNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  return !(cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId) || cnode->HasAttr(kAttrDuplicated));
}

bool IsDxMatMul(const CNodePtr &matmul_node) {
  std::queue<AnfNodePtr> cnode_queue;
  std::vector<AnfNodePtr> visited;
  for (size_t i = 1; i < matmul_node->size(); ++i) {
    cnode_queue.push(matmul_node->input(i));
    visited.push_back(matmul_node->input(i));
  }

  std::vector<AnfNodePtr> res;
  while (!cnode_queue.empty()) {
    auto queue_front = cnode_queue.front();
    cnode_queue.pop();
    if (!IsSomePrimitiveList(queue_front->cast<CNodePtr>(), {prim::kPrimLoad->name(), prim::kPrimDepend->name()})) {
      res.push_back(queue_front);
      continue;
    }
    auto cnode_queue_end = queue_front->cast<CNodePtr>();
    if (std::find(visited.begin(), visited.end(), cnode_queue_end->input(1)) != visited.end()) {
      continue;
    }
    cnode_queue.push(cnode_queue_end->input(1));
    visited.push_back(cnode_queue_end->input(1));
  }
  for (const auto &node : res) {
    if (node->isa<Parameter>()) {
      return true;
    }
    if (IsPrimitiveCNode(node, prim::kPrimAllGather)) {
      return true;
    }
  }
  return false;
}

bool IsDwMatMul(const CNodePtr &matmul_node) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (!cell_reuse) {
    return !IsDxMatMul(matmul_node);
  }
  MS_EXCEPTION_IF_NULL(matmul_node->func_graph());
  auto next_nodes = GetOutputNodesWithFilter(matmul_node, [&](const AnfNodePtr &anode) {
    return IsPrimitiveCNode(anode, prim::kPrimLoad) || IsPrimitiveCNode(anode, prim::kPrimCast) ||
           IsPrimitiveCNode(anode, prim::kPrimDepend);
  });
  for (const auto &next_node : next_nodes) {
    if (IsPrimitiveCNode(next_node.first, prim::kPrimAssignAdd)) {
      return true;
    }
  }
  return false;
}

void ExtractBackwardMatMul(const std::vector<CNodePtr> &origin_nodes_topological,
                           std::unordered_map<CNodePtr, CNodePtr> *backward_matmul_dx_dw_map) {
  std::unordered_map<std::string, std::vector<CNodePtr>> backward_matmul_map;
  for (const auto &node : origin_nodes_topological) {
    if (IsForwardNode(node) || !IsPrimitiveCNode(node, prim::kPrimMatMul)) {
      continue;
    }
    auto matmul_cnode = node->cast<CNodePtr>();
    if (!matmul_cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      continue;
    }
    auto matmul_unique_id = GetValue<std::string>(matmul_cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId));
    backward_matmul_map[matmul_unique_id].push_back(matmul_cnode);
  }

  for (const auto &matmul_list_pair : backward_matmul_map) {
    if (matmul_list_pair.second.size() != 2) {
      continue;
    }
    auto matmul_list = matmul_list_pair.second;
    if (IsDwMatMul(matmul_list[1])) {
      (*backward_matmul_dx_dw_map)[matmul_list[0]] = matmul_list[1];
    } else if (IsDwMatMul(matmul_list[0])) {
      (*backward_matmul_dx_dw_map)[matmul_list[1]] = matmul_list[0];
    }
  }
  MS_LOG(WARNING) << "backward_matmul_dx_dw_map size:" << backward_matmul_dx_dw_map->size();
}

std::string AnfNodeInfo(const AnfNodePtr &anf_node) {
  std::string unique_id;
  if (anf_node->isa<CNode>() && anf_node->cast<CNodePtr>()->HasPrimalAttr("unique_id")) {
    unique_id = GetValue<std::string>(anf_node->cast<CNodePtr>()->GetPrimalAttr("unique_id"));
  }
  if (anf_node->isa<CNode>() && anf_node->cast<CNodePtr>()->HasPrimalAttr("forward_unique_id")) {
    unique_id = GetValue<std::string>(anf_node->cast<CNodePtr>()->GetPrimalAttr("forward_unique_id"));
  }
  return unique_id;
}
}  // namespace parallel
}  // namespace mindspore
