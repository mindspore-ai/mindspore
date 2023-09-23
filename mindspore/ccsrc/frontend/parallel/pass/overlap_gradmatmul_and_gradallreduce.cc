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

#include "frontend/parallel/pass/overlap_gradmatmul_and_gradallreduce.h"
#include <memory>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <queue>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace parallel {
namespace {
using Pattern = std::vector<std::pair<PrimitivePtr, int64_t>>;
const size_t count_ten = 10;

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

void ExtractForwardMatMul(const std::vector<CNodePtr> &origin_nodes_topological,
                          std::vector<std::string> *forward_matmul_unique_id_list) {
  for (auto &node : origin_nodes_topological) {
    if (!IsForwardNode(node) || !IsPrimitiveCNode(node, prim::kPrimMatMul)) {
      continue;
    }
    auto matmul_cnode = node->cast<CNodePtr>();
    if (!matmul_cnode->HasPrimalAttr(kPrimalAttrUniqueId)) {
      continue;
    }
    auto matmul_unique_id = GetValue<std::string>(matmul_cnode->GetPrimalAttr(kPrimalAttrUniqueId));
    (*forward_matmul_unique_id_list).push_back(matmul_unique_id);
  }
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
    if (IsDxMatMul(matmul_list[0])) {
      (*backward_matmul_dx_dw_map)[matmul_list[0]] = matmul_list[1];
    } else if (IsDxMatMul(matmul_list[1])) {
      (*backward_matmul_dx_dw_map)[matmul_list[1]] = matmul_list[0];
    }
  }
}

std::vector<CNodePtr> GetCommInputMatMulNode(const AnfNodePtr &node,
                                             const std::unordered_map<CNodePtr, CNodePtr> &backward_matmul_dx_dw_map,
                                             size_t count_num, size_t loop_max = 100) {
  std::vector<CNodePtr> result;
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(node);
  size_t loop = 0;
  while (!anf_queue.empty()) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    if (result.size() == count_num || loop >= loop_max) {
      return result;
    }
    if (!queue_end->isa<CNode>()) {
      continue;
    }
    auto cnode_queue_end = queue_end->cast<CNodePtr>();
    if (cnode_queue_end->HasAttr(kAttrDuplicated)) {
      continue;
    }
    if (IsPrimitiveCNode(cnode_queue_end, prim::kPrimMatMul) &&
        backward_matmul_dx_dw_map.count(cnode_queue_end->cast<CNodePtr>()) > 0) {
      result.push_back(queue_end->cast<CNodePtr>());
    }
    auto input_size = cnode_queue_end->size();
    if (IsPrimitiveCNode(cnode_queue_end, prim::kPrimDepend)) {
      input_size = 2;
    }
    for (size_t i = 1; i < input_size; ++i) {
      anf_queue.push(cnode_queue_end->input(i));
    }
    loop++;
  }
  return result;
}

void InsertDepend(const FuncGraphManagerPtr &manager, const CNodePtr &comm_i1, const CNodePtr &matmul_i) {
  auto comm_i1_input = comm_i1->input(1);
  auto matmul_i_input = matmul_i->input(1);
  std::vector<AnfNodePtr> depend1_inputs{NewValueNode(prim::kPrimDepend), matmul_i_input, comm_i1_input};
  auto depend_node1 = matmul_i_input->func_graph()->NewCNode(depend1_inputs);
  depend_node1->set_abstract(matmul_i_input->abstract()->Clone());
  depend_node1->AddAttr("matmul_grad_depend1", MakeValue(true));
  depend_node1->AddAttr(kAttrCommInputDepend, MakeValue(true));
  MS_EXCEPTION_IF_NULL(depend_node1);
  manager->SetEdge(matmul_i, 1, depend_node1);

  auto comm_i1_output = manager->node_users()[comm_i1].front().first;
  std::vector<AnfNodePtr> depend2_inputs{NewValueNode(prim::kPrimDepend), comm_i1, matmul_i};
  auto depend_node2 = comm_i1->func_graph()->NewCNode(depend2_inputs);

  depend_node2->set_abstract(comm_i1->abstract()->Clone());
  depend_node2->AddAttr("matmul_grad_depend2", MakeValue(true));
  MS_EXCEPTION_IF_NULL(depend_node2);
  manager->SetEdge(comm_i1_output, manager->node_users()[comm_i1].front().second, depend_node2);
}

void OverLapGradMatMul(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological,
                       const std::unordered_map<CNodePtr, CNodePtr> &backward_matmul_dx_dw_map,
                       const std::vector<std::string> &forward_matmul_unique_id_list) {
  std::set<CNodePtr> matched_matmul_list;
  for (const auto &node : origin_nodes_topological) {
    if (!IsSomePrimitiveList(node, {ALL_GATHER, REDUCE_SCATTER, ALL_REDUCE})) {
      continue;
    }
    if (IsForwardNode(node) || node->HasAttr(kAttrDuplicated)) {
      continue;
    }
    if (!node->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      continue;
    }
    auto input_matmul_dx_nodes = GetCommInputMatMulNode(node, backward_matmul_dx_dw_map, count_ten);
    if (input_matmul_dx_nodes.empty()) {
      MS_LOG(WARNING) << "comm node:" << node->fullname_with_scope() << " cannot find input matmuls";
    }
    std::sort(
      input_matmul_dx_nodes.begin(), input_matmul_dx_nodes.end(), [&](const CNodePtr &cnode1, const CNodePtr &cnode2) {
        auto id1 = GetValue<std::string>(cnode1->GetPrimalAttr(kPrimalAttrForwardUniqueId));
        auto id2 = GetValue<std::string>(cnode2->GetPrimalAttr(kPrimalAttrForwardUniqueId));
        size_t index1 = std::find(forward_matmul_unique_id_list.begin(), forward_matmul_unique_id_list.end(), id1) -
                        forward_matmul_unique_id_list.begin();
        size_t index2 = std::find(forward_matmul_unique_id_list.begin(), forward_matmul_unique_id_list.end(), id2) -
                        forward_matmul_unique_id_list.begin();
        return index1 < index2;
      });
    for (const auto &matmul : input_matmul_dx_nodes) {
      if (matched_matmul_list.count(matmul) > 0) {
        continue;
      }
      // insert depend
      MS_LOG(WARNING) << "insert depend for comm node:" << node->fullname_with_scope() << " and "
                      << backward_matmul_dx_dw_map.at(matmul)->fullname_with_scope();
      InsertDepend(manager, node, backward_matmul_dx_dw_map.at(matmul));
      matched_matmul_list.insert(matmul);
      break;
    }
  }
}

void ExtractForwardNodes(const std::vector<CNodePtr> &origin_nodes_topological,
                         std::vector<std::string> *forward_comm_node_unique_id_list,
                         std::vector<std::string> *forward_matmul_unique_id_list) {
  MS_EXCEPTION_IF_NULL(forward_comm_node_unique_id_list);
  MS_EXCEPTION_IF_NULL(forward_matmul_unique_id_list);
  for (auto &node : origin_nodes_topological) {
    if (!IsForwardNode((node))) {
      continue;
    }
    if (!node->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      continue;
    }
    auto prim = GetCNodePrimitive(node);
    auto instance_name = prim->instance_name();
    if (instance_name.find("forward_op") == std::string::npos) {
      continue;
    }
    auto matmul_node = RealInputNode(node, 1);
    if (!IsPrimitiveCNode(matmul_node, prim::kPrimMatMul)) {
      continue;
    }
    auto matmul_cnode = matmul_node->cast<CNodePtr>();
    if (!matmul_cnode->HasPrimalAttr(kPrimalAttrUniqueId)) {
      continue;
    }
    auto forward_comm_node_unique_id = GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrForwardCommNodeUniqueId));
    (*forward_comm_node_unique_id_list).push_back(forward_comm_node_unique_id);
    auto matmul_unique_id = GetValue<std::string>(matmul_cnode->GetPrimalAttr(kPrimalAttrUniqueId));
    (*forward_matmul_unique_id_list).push_back(matmul_unique_id);
  }
}

void ExtractBackwardNodes(const std::vector<CNodePtr> &origin_nodes_topological,
                          const std::vector<std::string> &forward_comm_node_unique_id_list,
                          const std::vector<std::string> &forward_matmul_unique_id_list,
                          std::vector<CNodePtr> *backward_comm_node_list, std::vector<CNodePtr> *backward_matmul_list) {
  for (auto &node : origin_nodes_topological) {
    if (!node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      continue;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimMatMul)) {
      continue;
    }
    auto matmul_cnode = node->cast<CNodePtr>();
    auto forward_unique_id = GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrForwardUniqueId));
    if (std::find(forward_matmul_unique_id_list.begin(), forward_matmul_unique_id_list.end(), forward_unique_id) ==
        forward_matmul_unique_id_list.end()) {
      continue;
    }
    auto pre_cnode = RealInputNode(matmul_cnode, 1)->cast<CNodePtr>();
    if (pre_cnode == nullptr || !pre_cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      (*backward_matmul_list).push_back(matmul_cnode);
      continue;
    }

    auto pre_cnode_forward_comm_unique_id =
      GetValue<std::string>(pre_cnode->GetPrimalAttr(kPrimalAttrForwardCommNodeUniqueId));
    if (std::find(forward_comm_node_unique_id_list.begin(), forward_comm_node_unique_id_list.end(),
                  pre_cnode_forward_comm_unique_id) != forward_comm_node_unique_id_list.end()) {
      (*backward_comm_node_list).push_back(pre_cnode);
    }
  }
  std::sort(
    backward_comm_node_list->begin(), backward_comm_node_list->end(),
    [&](const CNodePtr &cnode1, const CNodePtr &cnode2) {
      auto id1 = GetValue<std::string>(cnode1->GetPrimalAttr(kPrimalAttrForwardCommNodeUniqueId));
      auto id2 = GetValue<std::string>(cnode2->GetPrimalAttr(kPrimalAttrForwardCommNodeUniqueId));
      size_t index1 = std::find(forward_comm_node_unique_id_list.begin(), forward_comm_node_unique_id_list.end(), id1) -
                      forward_comm_node_unique_id_list.begin();
      size_t index2 = std::find(forward_comm_node_unique_id_list.begin(), forward_comm_node_unique_id_list.end(), id2) -
                      forward_comm_node_unique_id_list.begin();
      return index1 > index2;
    });
  std::sort(
    backward_matmul_list->begin(), backward_matmul_list->end(), [&](const CNodePtr &cnode1, const CNodePtr &cnode2) {
      auto id1 = GetValue<std::string>(cnode1->GetPrimalAttr(kPrimalAttrForwardUniqueId));
      auto id2 = GetValue<std::string>(cnode2->GetPrimalAttr(kPrimalAttrForwardUniqueId));
      size_t index1 = std::find(forward_matmul_unique_id_list.begin(), forward_matmul_unique_id_list.end(), id1) -
                      forward_matmul_unique_id_list.begin();
      size_t index2 = std::find(forward_matmul_unique_id_list.begin(), forward_matmul_unique_id_list.end(), id2) -
                      forward_matmul_unique_id_list.begin();
      return index1 > index2;
    });
}

void DoOverLapWay2(const FuncGraphManagerPtr &manager, const FuncGraphPtr &forward_graph,
                   const FuncGraphPtr &backward_graph) {
  std::list<CNodePtr> forward_orders = forward_graph->GetOrderedCnodes();
  std::vector<CNodePtr> forward_origin_nodes_topological(forward_orders.cbegin(), forward_orders.cend());
  std::list<CNodePtr> backward_orders = backward_graph->GetOrderedCnodes();
  std::vector<CNodePtr> backward_origin_nodes_topological(backward_orders.cbegin(), backward_orders.cend());
  std::vector<std::string> forward_matmul_unique_id_list;
  ExtractForwardMatMul(forward_origin_nodes_topological, &forward_matmul_unique_id_list);
  std::unordered_map<CNodePtr, CNodePtr> backward_matmul_dx_dw_map;
  ExtractBackwardMatMul(backward_origin_nodes_topological, &backward_matmul_dx_dw_map);
  OverLapGradMatMul(manager, backward_origin_nodes_topological, backward_matmul_dx_dw_map,
                    forward_matmul_unique_id_list);
}

void DoOverLap(const FuncGraphManagerPtr &manager, const FuncGraphPtr &forward_graph,
               const FuncGraphPtr &backward_graph) {
  std::list<CNodePtr> forward_orders = forward_graph->GetOrderedCnodes();
  std::vector<CNodePtr> forward_origin_nodes_topological(forward_orders.cbegin(), forward_orders.cend());
  std::list<CNodePtr> backward_orders = backward_graph->GetOrderedCnodes();
  std::vector<CNodePtr> backward_origin_nodes_topological(backward_orders.cbegin(), backward_orders.cend());
  std::vector<std::string> forward_comm_node_unique_id_list;
  std::vector<std::string> forward_matmul_unique_id_list;
  std::vector<CNodePtr> backward_comm_node_list;
  std::vector<CNodePtr> backward_matmul_list;
  ExtractForwardNodes(forward_origin_nodes_topological, &forward_comm_node_unique_id_list,
                      &forward_matmul_unique_id_list);
  ExtractBackwardNodes(backward_origin_nodes_topological, forward_comm_node_unique_id_list,
                       forward_matmul_unique_id_list, &backward_comm_node_list, &backward_matmul_list);
  if (backward_comm_node_list.size() != backward_matmul_list.size() || backward_comm_node_list.empty()) {
    MS_LOG(INFO) << "backward_comm_node_list.size():" << backward_comm_node_list.size()
                 << ", backward_matmul_list.size():" << backward_matmul_list.size();
    return;
  }
  for (size_t i = 0; i < backward_matmul_list.size() - 1; ++i) {
    auto matmul_i = backward_matmul_list[i];
    auto comm_i1 = backward_comm_node_list[i + 1];
    if (matmul_i->HasPrimalAttr(MICRO) || comm_i1->HasPrimalAttr(MICRO)) {
      if (!(matmul_i->HasPrimalAttr(MICRO) && comm_i1->HasPrimalAttr(MICRO))) {
        continue;
      }
      auto comm_micro = GetValue<int64_t>(comm_i1->GetPrimalAttr(MICRO));
      auto matmul_micro = GetValue<int64_t>(matmul_i->GetPrimalAttr(MICRO));
      if (comm_micro != matmul_micro) {
        continue;
      }
    }
    InsertDepend(manager, comm_i1, matmul_i);
  }
}
}  // namespace

void OverlapGradMatmulAndGradAllreduce(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_GRAD_COMM_OVERLAP);
  if (!is_enable) {
    return;
  }
  auto manager = graph->manager();

  FuncGraphPtr forward_graph = graph;
  FuncGraphPtr backward_graph = graph;
  for (const auto &each_graph : manager->func_graphs()) {
    if (IsCellReuseForwardGraph(each_graph)) {
      forward_graph = each_graph;
      backward_graph = GetCellReuseBackwardGraph(forward_graph);
      if (backward_graph == nullptr) {
        MS_LOG(WARNING)
          << "Failed to find backward cell reuse graph, skip pass 'overlap_gradmatmul_and_gradallreduce'.";
        return;
      }
      break;
    }
  }
  if (!parallel::ParallelContext::GetInstance()->enable_micro_interleaved()) {
    DoOverLap(manager, forward_graph, backward_graph);
    return;
  }
  DoOverLapWay2(manager, forward_graph, backward_graph);
}
}  // namespace parallel
}  // namespace mindspore
