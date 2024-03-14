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

#include "frontend/parallel/pass/overlap_grad_comm.h"
#include <memory>
#include <vector>
#include <list>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <queue>
#include "mindspore/core/ops/math_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace parallel {
namespace {
const size_t loop_count = 1000;

std::vector<CNodePtr> GetDwRelyNodes(const CNodePtr &dw_matmul) {
  // second input is the recompute node
  std::vector<CNodePtr> rely_nodes;
  std::queue<CNodePtr> cnode_queue;
  std::set<AnfNodePtr> visited;
  if (dw_matmul->input(kIndex2)->isa<CNode>()) {
    cnode_queue.push(dw_matmul->input(kIndex2)->cast<CNodePtr>());
  }
  if (dw_matmul->input(kIndex1)->isa<CNode>()) {
    cnode_queue.push(dw_matmul->input(kIndex1)->cast<CNodePtr>());
  }
  while (!cnode_queue.empty()) {
    auto queue_front = cnode_queue.front();
    cnode_queue.pop();
    for (size_t i = 1; i < queue_front->size(); ++i) {
      if (std::find(visited.begin(), visited.end(), queue_front->input(i)) != visited.end()) {
        continue;
      }
      (void)visited.insert(queue_front->input(i));
      if (!IsPrimitiveCNode(queue_front->input(i))) {
        continue;
      }
      auto input_cnode = queue_front->input(i)->cast<CNodePtr>();
      cnode_queue.push(input_cnode);
      if (input_cnode->HasAttr(kAttrDuplicated)) {
        continue;
      }
      rely_nodes.push_back(input_cnode);
      if (rely_nodes.size() > loop_count) {
        break;
      }
    }
  }
  return rely_nodes;
}

void InsertDwMatmulDepend(const FuncGraphPtr &backward_graph, const std::vector<CNodePtr> &dw_matmul_list) {
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_COMM_OVERLAP)) {
    return;
  }
  auto manager = backward_graph->manager();
  for (size_t i = 0; i + 1 < dw_matmul_list.size(); ++i) {
    auto cur_dw_matmul = dw_matmul_list[i];
    auto next_dw_matmul = dw_matmul_list[i + 1];
    std::vector<AnfNodePtr> depend4_inputs{NewValueNode(prim::kPrimDepend), next_dw_matmul->input(kIndex1),
                                           cur_dw_matmul};
    auto depend_node4 = backward_graph->NewCNode(depend4_inputs);
    depend_node4->set_abstract(next_dw_matmul->input(kIndex1)->abstract()->Clone());
    depend_node4->AddAttr("grad_comm_depend4", MakeValue(true));
    manager->SetEdge(next_dw_matmul, kIndex1, depend_node4);
  }
}

void InsertDependForDxAndGradComm(const FuncGraphPtr &backward_graph, const std::vector<CNodePtr> &dx_matmul_list,
                                  const std::unordered_map<CNodePtr, CNodePtr> &backward_matmul_dx_dw_map,
                                  const std::unordered_map<CNodePtr, std::vector<CNodePtr>> &dx_grad_comm_map) {
  auto manager = backward_graph->manager();
  std::vector<CNodePtr> matched_dx_list;
  // there are two comm node when opt sharding not fully
  std::vector<std::vector<CNodePtr>> grad_comm_list;
  std::vector<CNodePtr> dw_matmul_list;
  for (size_t i = 0; i < dx_matmul_list.size(); ++i) {
    auto cur_dx_matmul = dx_matmul_list[i];
    auto dw_matmul = backward_matmul_dx_dw_map.at(cur_dx_matmul);
    auto grad_comm = dx_grad_comm_map.at(cur_dx_matmul);
    // Check dw_matmul inputs contains cur_dx_matmul or not, if contains, dx_matmul ++
    auto dw_rely_nodes = GetDwRelyNodes(dw_matmul);
    CNodePtr dx_matmul = nullptr;
    for (size_t j = i; j < dx_matmul_list.size(); ++j) {
      if (std::find(matched_dx_list.begin(), matched_dx_list.end(), dx_matmul_list[j]) != matched_dx_list.end()) {
        continue;
      }
      if (std::find(dw_rely_nodes.begin(), dw_rely_nodes.end(), dx_matmul_list[j]) != dw_rely_nodes.end()) {
        continue;
      }
      dx_matmul = dx_matmul_list[j];
      matched_dx_list.push_back(dx_matmul_list[j]);
      break;
    }
    if (!dx_matmul) {
      continue;
    }
    if (grad_comm.size() > SIZE_TWO) {
      continue;
    }
    if (grad_comm.size() == SIZE_TWO) {
      if (IsPrimitiveCNode(grad_comm.front(), prim::kPrimReduceScatter)) {
        auto tmp = grad_comm.front();
        grad_comm[kIndex0] = grad_comm[kIndex1];
        grad_comm[kIndex1] = tmp;
      }
    }
    // insert depend
    MS_LOG(INFO) << "insert depend for comm node:" << grad_comm.front()->fullname_with_scope()
                 << ", unique id:" << AnfNodeInfo(grad_comm.front())
                 << ", dx_matmul: " << dx_matmul->fullname_with_scope() << ", unique id:" << AnfNodeInfo(dx_matmul)
                 << ", dw_matmul: " << dw_matmul->fullname_with_scope() << ", unique id:" << AnfNodeInfo(dw_matmul);
    // grad comm -> dx_matmul
    auto grad_comm_input = grad_comm.front()->input(kIndex1);
    auto dx_matmul_input = dx_matmul->input(kIndex1);
    std::vector<AnfNodePtr> depend1_inputs{NewValueNode(prim::kPrimDepend), dx_matmul_input, grad_comm_input};
    auto depend_node1 = backward_graph->NewCNode(depend1_inputs);
    depend_node1->set_abstract(dx_matmul_input->abstract()->Clone());
    depend_node1->AddAttr("grad_comm_depend1", MakeValue(true));
    manager->SetEdge(dx_matmul, kIndex1, depend_node1);
    // dx_matmul -> grad comm output
    auto comm_output_users = manager->node_users()[grad_comm.back()];
    for (const auto &comm_output_pair : comm_output_users) {
      if (!IsPrimitiveCNode(comm_output_pair.first)) {
        continue;
      }
      if (IsPrimitiveCNode(comm_output_pair.first, prim::kPrimDepend) && comm_output_pair.second == kIndex2) {
        continue;
      }
      std::vector<AnfNodePtr> depend2_inputs{NewValueNode(prim::kPrimDepend), grad_comm.back(), dx_matmul};
      auto depend_node2 = backward_graph->NewCNode(depend2_inputs);
      depend_node2->set_abstract(grad_comm.back()->abstract()->Clone());
      depend_node2->AddAttr("grad_comm_depend2", MakeValue(true));
      manager->SetEdge(comm_output_pair.first, comm_output_pair.second, depend_node2);
    }
    grad_comm_list.push_back(grad_comm);
    dw_matmul_list.push_back(dw_matmul);
  }
  for (size_t i = 0; i + 1 < grad_comm_list.size(); ++i) {
    auto grad_comm_node = grad_comm_list[i].back();
    auto next_grad_comm_node = grad_comm_list[i + 1].front();
    auto grad_comm_node_users = manager->node_users()[grad_comm_node];
    if (grad_comm_node_users.empty()) {
      continue;
    }
    auto grad_comm_node_user = grad_comm_node_users.front().first;
    std::vector<AnfNodePtr> depend3_inputs{NewValueNode(prim::kPrimDepend), next_grad_comm_node->input(kIndex1),
                                           grad_comm_node_user};
    auto depend_node3 = backward_graph->NewCNode(depend3_inputs);
    depend_node3->set_abstract(next_grad_comm_node->input(kIndex1)->abstract()->Clone());
    depend_node3->AddAttr("grad_comm_depend3", MakeValue(true));
    manager->SetEdge(next_grad_comm_node, kIndex1, depend_node3);
  }
  InsertDwMatmulDepend(backward_graph, dw_matmul_list);
}

std::unordered_map<CNodePtr, std::vector<CNodePtr>> ExtractDxGradCommMap(
  const std::vector<CNodePtr> &origin_nodes_topological,
  const std::unordered_map<CNodePtr, CNodePtr> &backward_matmul_dx_dw_map) {
  std::unordered_map<CNodePtr, std::vector<CNodePtr>> backward_matmul_dx_grad_comm_map;
  std::vector<CNodePtr> backward_matmul_dx_grad_comm_vector;
  for (const auto &node : origin_nodes_topological) {
    if (!node->HasPrimalAttr(kPrimalAttrMirrorUserId) || !IsSomePrimitiveList(node, {ALL_REDUCE, REDUCE_SCATTER})) {
      continue;
    }
    auto user_id = GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrMirrorUserId));
    auto pos =
      std::find_if(backward_matmul_dx_dw_map.begin(), backward_matmul_dx_dw_map.end(), [&](const auto &key_node) {
        if (!key_node.first->HasPrimalAttr(kPrimalAttrMirrorUserId)) {
          return false;
        }
        return GetValue<std::string>(key_node.first->GetPrimalAttr(kPrimalAttrMirrorUserId)) == user_id;
      });
    if (pos == backward_matmul_dx_dw_map.end()) {
      MS_LOG(INFO) << "cannot match comm node:" << node->fullname_with_scope() << ", id:" << AnfNodeInfo(node);
      continue;
    }
    auto matched_dx_node = pos->first;
    backward_matmul_dx_grad_comm_map[matched_dx_node].push_back(node);
    backward_matmul_dx_grad_comm_vector.push_back(matched_dx_node);
  }
  if (!parallel::ParallelContext::GetInstance()->enable_fine_grained_micro_interleaved()) {
    return backward_matmul_dx_grad_comm_map;
  }
  MS_LOG(INFO) << "Enabled fine grained micro interleaved.";
  // One grad comm match multi dx node
  for (const auto &dx_cnode : backward_matmul_dx_grad_comm_vector) {
    auto dw_cnode = backward_matmul_dx_dw_map.at(dx_cnode);
    for (const auto &dx_dw : backward_matmul_dx_dw_map) {
      if (dx_dw.second != dw_cnode) {
        continue;
      }
      if (dx_dw.first == dx_cnode) {
        continue;
      }
      backward_matmul_dx_grad_comm_map[dx_dw.first] = backward_matmul_dx_grad_comm_map[dx_cnode];
    }
  }
  return backward_matmul_dx_grad_comm_map;
}

void OverlapDxAndGradComm(const FuncGraphPtr &backward_graph) {
  std::list<CNodePtr> backward_orders = backward_graph->GetOrderedCnodes();
  std::vector<CNodePtr> backward_origin_nodes_topological(backward_orders.cbegin(), backward_orders.cend());
  std::unordered_map<CNodePtr, CNodePtr> backward_matmul_dx_dw_map;
  ExtractBackwardMatMul(backward_origin_nodes_topological, &backward_matmul_dx_dw_map);
  ExtendDxDwMap(backward_origin_nodes_topological, &backward_matmul_dx_dw_map);
  auto dx_grad_comm_map = ExtractDxGradCommMap(backward_origin_nodes_topological, backward_matmul_dx_dw_map);
  std::vector<CNodePtr> dx_matmul_list;
  for (const auto &dx_matmul : backward_origin_nodes_topological) {
    if (!IsPrimitiveCNode(dx_matmul, prim::kPrimMatMul) && !IsPrimitiveCNode(dx_matmul, prim::kPrimMatMulV2)) {
      continue;
    }
    if (dx_grad_comm_map.count(dx_matmul) == 0) {
      continue;
    }
    if (dx_matmul->HasAttr(INTERLEAVED_OVERLAP_MATMUL)) {
      continue;
    }
    dx_matmul_list.push_back(dx_matmul);
  }
  InsertDependForDxAndGradComm(backward_graph, dx_matmul_list, backward_matmul_dx_dw_map, dx_grad_comm_map);
}

}  // namespace

void OverlapGradComm(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto soc_version = ms_context->ascend_soc_version();
  if (soc_version != "ascend910" && soc_version != "ascend910b") {
    return;
  }
  auto is_enable = ms_context->get_param<bool>(MS_CTX_GRAD_COMM_OVERLAP);
  if (!is_enable) {
    return;
  }

  auto manager = graph->manager();
  FuncGraphPtr backward_graph = graph;
  for (const auto &each_graph : manager->func_graphs()) {
    if (IsCellReuseForwardGraph(each_graph)) {
      auto forward_graph = each_graph;
      // need to using the inlined backward_graph
      backward_graph = GetCellReuseBackwardGraph(forward_graph);
      if (backward_graph == nullptr) {
        MS_LOG(WARNING)
          << "Failed to find backward cell reuse graph, skip pass 'overlap_gradmatmul_and_gradallreduce'.";
        return;
      }
      break;
    }
  }
  OverlapDxAndGradComm(backward_graph);
}
}  // namespace parallel
}  // namespace mindspore
