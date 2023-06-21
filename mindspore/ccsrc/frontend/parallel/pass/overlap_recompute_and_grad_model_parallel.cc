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

#include "frontend/parallel/pass/overlap_recompute_and_grad_model_parallel.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <queue>
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/other_ops.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
constexpr size_t kSize2 = 2;
namespace {
void ExtractRecomputeSubGraph(const std::vector<CNodePtr> &origin_nodes_topological,
                              mindspore::HashMap<int32_t, std::vector<CNodePtr>> *recomputed_block_node_in_orders,
                              mindspore::HashMap<int32_t, std::vector<CNodePtr>> *recompute_block_node_in_orders,
                              mindspore::HashMap<int32_t, std::vector<CNodePtr>> *recomputed_grad_node) {
  for (const auto &cnode : origin_nodes_topological) {
    if (!cnode->HasAttr(kAttrRecomputeSubGraph)) {
      continue;
    }
    auto recompute_block_id = GetValue<size_t>(cnode->GetAttr(kAttrRecomputeSubGraph));
    if (cnode->HasAttr(kAttrDuplicated)) {
      if ((*recompute_block_node_in_orders).find(recompute_block_id) == (*recompute_block_node_in_orders).end()) {
        (*recompute_block_node_in_orders)[recompute_block_id] = {cnode};
      } else {
        (*recompute_block_node_in_orders)[recompute_block_id].push_back(cnode);
      }
    } else if (!cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      if ((*recomputed_block_node_in_orders).find(recompute_block_id) == (*recomputed_block_node_in_orders).end()) {
        (*recomputed_block_node_in_orders)[recompute_block_id] = {cnode};
      } else {
        (*recomputed_block_node_in_orders)[recompute_block_id].push_back(cnode);
      }
    } else {
      if ((*recomputed_grad_node).find(recompute_block_id) == (*recomputed_grad_node).end()) {
        (*recomputed_grad_node)[recompute_block_id] = {cnode};
      } else {
        (*recomputed_grad_node)[recompute_block_id].push_back(cnode);
      }
    }
  }
}

std::vector<CNodePtr> NodeUsersInRecomputeSubGraph(const CNodePtr &cnode, std::function<bool(const CNodePtr &)> match) {
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<CNodePtr> res;
  std::queue<CNodePtr> cnode_queue;
  cnode_queue.push(cnode);
  while (!cnode_queue.empty()) {
    auto queue_end = cnode_queue.front();
    cnode_queue.pop();
    auto user_set = manager->node_users()[queue_end];
    for (auto &pair : user_set) {
      if (!pair.first->isa<CNode>()) {
        continue;
      }
      auto user_cnode = pair.first->cast<CNodePtr>();
      if (std::find(res.begin(), res.end(), user_cnode) != res.end()) {
        continue;
      }
      if (match(user_cnode)) {
        cnode_queue.push(user_cnode);
        res.push_back(user_cnode);
        continue;
      }
    }
  }
  return res;
}

void ExtractCommNodes(const mindspore::HashMap<int32_t, std::vector<CNodePtr>> &origin_node_map,
                      mindspore::HashMap<int32_t, std::vector<CNodePtr>> *dst_node_map) {
  for (const auto &sub_graph : origin_node_map) {
    auto sub_graph_id = sub_graph.first;
    (*dst_node_map)[sub_graph_id] = {};
    for (const auto &sub_cnode : sub_graph.second) {
      if (!sub_cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
        continue;
      }
      (*dst_node_map)[sub_graph_id].push_back(sub_cnode);
    }
  }
}

std::vector<CNodePtr> SrcNodeNoRelyInputs(const CNodePtr &src_node, const std::vector<CNodePtr> &dst_node_users) {
  std::vector<CNodePtr> res;
  std::queue<CNodePtr> cnode_queue;
  cnode_queue.push(src_node);
  while (!cnode_queue.empty()) {
    auto queue_end = cnode_queue.front();
    cnode_queue.pop();
    if (std::find(dst_node_users.begin(), dst_node_users.end(), queue_end) == dst_node_users.end()) {
      res.push_back(queue_end);
      continue;
    }
    for (size_t i = 1; i < queue_end->size(); ++i) {
      if (!queue_end->input(i)->isa<CNode>()) {
        continue;
      }
      auto input_cnode = queue_end->input(i)->cast<CNodePtr>();
      cnode_queue.push(input_cnode);
    }
  }
  return res;
}

bool IsNotCareCNode(const CNodePtr &cnode) {
  return IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) ||
         IsPrimitiveCNode(cnode, prim::kPrimLoad) || IsPrimitiveCNode(cnode, prim::kPrimUpdateState) ||
         IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem);
}

CNodePtr GetSrcNode(const CNodePtr &src_node_output) {
  CNodePtr src_node = nullptr;
  for (size_t i = 1; i < src_node_output->size(); ++i) {
    if (src_node_output->input(i)->isa<CNode>()) {
      src_node = src_node_output->input(i)->cast<CNodePtr>();
    }
  }
  return src_node;
}
}  // namespace

void OverlapRecomputeAndGradModelParallel(const FuncGraphPtr &graph) {
  if ((parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
       parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel)) {
    return;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_RECOMPUTE_COMM_OVERLAP);
  if (!is_enable) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recomputed_block_node_in_orders;
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recompute_block_node_in_orders;
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recomputed_grad_node;
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recomputed_grad_comm_node_in_orders;
  mindspore::HashMap<int32_t, std::vector<CNodePtr>> recompute_block_comm_node_in_orders;
  ExtractRecomputeSubGraph(origin_nodes_topological, &recomputed_block_node_in_orders, &recompute_block_node_in_orders,
                           &recomputed_grad_node);
  ExtractCommNodes(recomputed_grad_node, &recomputed_grad_comm_node_in_orders);
  std::for_each(recomputed_grad_comm_node_in_orders.begin(), recomputed_grad_comm_node_in_orders.end(),
                [](auto &vector_pair) { std::reverse(vector_pair.second.begin(), vector_pair.second.end()); });
  ExtractCommNodes(recompute_block_node_in_orders, &recompute_block_comm_node_in_orders);
  // In recompute.cc, the grad_block input has a depend to the recompute block already.
  for (const auto &recompute_grad_sub_graph : recomputed_grad_comm_node_in_orders) {
    auto sub_graph_id = recompute_grad_sub_graph.first;
    auto recomputed_grad_comm_nodes = recomputed_grad_comm_node_in_orders[sub_graph_id];
    auto recompute_block_comm_nodes = recompute_block_comm_node_in_orders[sub_graph_id];
    size_t overlap_size = 2 * recompute_block_comm_nodes.size() + 1;
    size_t max_iter_num = recompute_block_comm_nodes.size();
    while (overlap_size > 0) {
      auto recompute_begin_index = max_iter_num - overlap_size / 2;
      auto grad_begin_index = max_iter_num - (overlap_size - 1) / 2;
      if (grad_begin_index >= recomputed_grad_comm_nodes.size()) {
        break;
      }
      CNodePtr src_node_output;
      CNodePtr dst_node_input;
      if (overlap_size % kSize2 == 1) {
        src_node_output = (recompute_begin_index == max_iter_num) ? recompute_block_node_in_orders[sub_graph_id].back()
                                                                  : recompute_block_comm_nodes[recompute_begin_index];
        dst_node_input = recomputed_grad_comm_nodes[grad_begin_index];
      } else {
        dst_node_input = recompute_block_comm_nodes[recompute_begin_index];
        src_node_output = recomputed_grad_comm_nodes[grad_begin_index];
      }
      CNodePtr src_node = GetSrcNode(src_node_output);
      if (src_node == nullptr) {
        continue;
      }
      auto dst_nodes = manager->node_users()[dst_node_input];
      for (const auto &dst_node_pair : dst_nodes) {
        if (!dst_node_pair.first->isa<CNode>()) {
          continue;
        }
        auto dst_node = dst_node_pair.first->cast<CNodePtr>();
        MS_LOG(INFO) << "The dst node is:" << dst_node->DebugString() << ", " << dst_node->fullname_with_scope();
        // Check whether src_node is the user of dst_node, if it is, adjust the src node toward its input.
        auto dst_node_users = NodeUsersInRecomputeSubGraph(dst_node, [&](const CNodePtr &cnode) {
          return std::find(recompute_block_node_in_orders[sub_graph_id].begin(),
                           recompute_block_node_in_orders[sub_graph_id].end(),
                           cnode) != recompute_block_node_in_orders[sub_graph_id].end() ||
                 std::find(recomputed_grad_node[sub_graph_id].begin(), recomputed_grad_node[sub_graph_id].end(),
                           cnode) != recomputed_grad_node[sub_graph_id].end() ||
                 IsNotCareCNode(cnode) || IsPrimitiveCNode(cnode, prim::kPrimAllGather) ||
                 IsPrimitiveCNode(cnode, prim::kPrimAddN);
        });
        dst_node_users.push_back(dst_node);
        // Insert depend src_node->depend->dst_node.
        auto src_node_no_rely_inputs = SrcNodeNoRelyInputs(src_node, dst_node_users);
        // Find the last input ordered by executed order.
        auto new_src_node = *std::max_element(
          src_node_no_rely_inputs.begin(), src_node_no_rely_inputs.end(),
          [&](const CNodePtr &cnode1, const CNodePtr &cnode2) {
            size_t cnode_iter1 =
              (size_t)(std::find(origin_nodes_topological.begin(), origin_nodes_topological.end(), cnode1) -
                       origin_nodes_topological.begin());
            size_t cnode_iter2 =
              (size_t)(std::find(origin_nodes_topological.begin(), origin_nodes_topological.end(), cnode2) -
                       origin_nodes_topological.begin());
            cnode_iter1 = IsNotCareCNode(cnode1) ? 0 : cnode_iter1;
            cnode_iter2 = IsNotCareCNode(cnode2) ? 0 : cnode_iter2;
            return cnode_iter1 < cnode_iter2;
          });
        MS_LOG(INFO) << "The origin_src_node is " << src_node->DebugString()
                     << "new_src_node is: " << new_src_node->DebugString();
        // Insert depend src_node->depend->dst_node.
        std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), dst_node_input, new_src_node};
        auto depend_node = graph->NewCNode(depend_input);
        depend_node->AddAttr("recompute_grad_depend", MakeValue<bool>(true));
        depend_node->set_abstract(dst_node_input->abstract()->Clone());
        manager->SetEdge(dst_node, dst_node_pair.second, depend_node);
      }
      overlap_size--;
    }
  }
}
}  // namespace parallel
}  // namespace mindspore
