/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/process_send_recv_for_ge.h"

#include <memory>
#include <vector>
#include <string>
#include <queue>
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "ops/sequence_ops.h"
#include "ops/other_ops.h"
#include "ops/framework_ops.h"
#include "utils/ms_context.h"

namespace mindspore::opt {
namespace {
constexpr const char kNeedAllGather[] = "parallel_optimizer_allgather";
constexpr const char kNodeCloseFollowing[] = "node_close_following";
constexpr const char kNodeWithoutOutput[] = "node_without_output";

bool IsCommOps(const AnfNodePtr &node) {
  static const PrimitiveSet kCommunicationOpsPrim = {prim::kPrimSend,
                                                     prim::kPrimReceive,
                                                     prim::kPrimAllReduce,
                                                     prim::kPrimAllGather,
                                                     prim::kPrimReduceScatter,
                                                     prim::kPrimAllToAll,
                                                     prim::kPrimAllSwap,
                                                     prim::kPrimAllToAllv,
                                                     prim::kPrimNeighborExchange,
                                                     prim::kPrimNeighborExchangeV2,
                                                     prim::kPrimNeighborExchangeV2Grad};
  return IsOneOfPrimitiveCNode(node, kCommunicationOpsPrim);
}

void ProcessNodeWithoutOutput(const FuncGraphPtr &graph, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  /*
  The cut boundary must have an address
  Node---->Depend---->Tensormove
  */
  auto value_node = NewValueNode(MakeValue(std::make_shared<tensor::Tensor>(1)));
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(value_node->value());
  auto value_abs = value_node->value()->ToAbstract();
  MS_EXCEPTION_IF_NULL(value_abs);
  value_node->set_abstract(value_abs);
  auto depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), value_node, node});
  MS_EXCEPTION_IF_NULL(depend);
  depend->set_abstract(value_abs);
  auto tensor_move = graph->NewCNode({NewValueNode(prim::kPrimTensorMove), depend});
  tensor_move->set_abstract(value_abs);
  node->AddPrimalAttr(kNodeWithoutOutput, MakeValue(true));
  depend->AddPrimalAttr(kNodeCloseFollowing, MakeValue(true));
  tensor_move->AddPrimalAttr(kNodeCloseFollowing, MakeValue(true));
  (void)manager->Replace(node, tensor_move);
}

void AddAllGatherRecvDepend(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtr return_node = graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = TopoSort(return_node);
  std::vector<AnfNodePtr> all_gather_nodes;
  std::vector<CNodePtr> recv_nodes;

  for (auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimAllGather)) {
      auto cnode = node->cast<CNodePtr>();
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      const auto &instance_name = prim->instance_name();
      if (instance_name.find(kNeedAllGather) != std::string::npos) {
        (void)all_gather_nodes.emplace_back(cnode);
      }
    } else if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode->HasPrimalAttr(parallel::PIPELINE_BEGIN)) {
        auto pipeline_begin = GetValue<int64_t>(cnode->GetPrimalAttr(parallel::PIPELINE_BEGIN));
        if (pipeline_begin == 0) {
          (void)recv_nodes.emplace_back(cnode);
        }
      }
    }
  }

  if (all_gather_nodes.empty() || recv_nodes.empty()) {
    return;
  }

  for (auto &recv_node : recv_nodes) {
    auto before_input = recv_node->input(1);
    std::vector<AnfNodePtr> input = {NewValueNode(prim::kPrimDepend), before_input};
    (void)std::transform(all_gather_nodes.begin(), all_gather_nodes.end(), std::back_inserter(input),
                         [](const AnfNodePtr &v) { return v; });
    auto new_depend = graph->NewCNode(input);
    new_depend->set_abstract(before_input->abstract());
    manager->SetEdge(recv_node, 1, new_depend);
  }
}

bool IsCall(const AnfNodePtr &node) {
  if (!utils::isa<CNodePtr>(node)) {
    return false;
  }
  return IsValueNode<FuncGraph>(node->cast<CNodePtr>()->input(0));
}

bool IsClosure(const AnfNodePtr &node) {
  if (!utils::isa<CNodePtr>(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsCall(cnode)) {
    return true;
  }
  if (IsPrimitiveCNode(cnode->input(0), prim::kPrimTupleGetItem)) {
    auto tuple_get_node = cnode->input(0)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_get_node);
    auto get_from_node = tuple_get_node->input(kIndex1);
    return IsCall(get_from_node);
  }
  return false;
}

void FindFollowing(const AnfNodePtr &send_node, std::map<AnfNodePtr, std::set<AnfNodePtr>> *clouse_to_send) {
  std::queue<AnfNodePtr> node_queue;
  auto seen = NewSeenGeneration();
  node_queue.push(send_node);
  while (!node_queue.empty()) {
    auto top_node = node_queue.front();
    node_queue.pop();
    top_node->seen_ = seen;
    if (IsClosure(top_node)) {
      (void)(*clouse_to_send)[top_node].insert(send_node);
    }
    auto top_cnode = dyn_cast_ptr<CNode>(top_node);
    if (top_cnode == nullptr) {
      continue;
    }
    for (auto &next : top_cnode->inputs()) {
      if (next->seen_ == seen) {
        continue;
      }
      node_queue.push(next);
    }
  }
}

void AddSendClosureDepend(const FuncGraphPtr &graph) {
  std::set<AnfNodePtr> has_processed;
  // send is must exec after clouse
  std::map<AnfNodePtr, std::set<AnfNodePtr>> clouse_to_send;
  std::vector<AnfNodePtr> all_send;
  MS_EXCEPTION_IF_NULL(graph);
  AnfNodePtr return_node = graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = TopoSort(return_node);

  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }
    FindFollowing(node, &clouse_to_send);
    all_send.push_back(node);
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : all_nodes) {
    if (IsClosure(node)) {
      auto iter = clouse_to_send.find(node);
      if (iter == clouse_to_send.end()) {
        continue;
      }
      for (auto &send_node : all_send) {
        // must exec after clouse, can not add depend
        auto send_iter = iter->second.find(send_node);
        if (send_iter != iter->second.end()) {
          continue;
        }
        // has processed send node, need not add depend
        auto processed_iter = has_processed.find(send_node);
        if (processed_iter != has_processed.end()) {
          continue;
        }
        auto cnode = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_CHECK_FAIL(cnode->inputs().size() >= 1,
                                   "CNode inputs size is less equal than 1, cnode: " + cnode->DebugString());
        auto before_input = cnode->input(1);
        std::vector<AnfNodePtr> input = {NewValueNode(prim::kPrimDepend), before_input, send_node};
        auto new_depend = graph->NewCNode(input);
        new_depend->set_abstract(before_input->abstract());
        manager->SetEdge(cnode, 1, new_depend);
        (void)has_processed.insert(send_node);
      }
    }
  }
}
}  // namespace

void ProcessSendRecvForGE(const FuncGraphPtr &graph) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  static const bool is_enable_ge = (context->backend_policy() == "ge");
  static const bool is_cell_reuse =
    (common::GetEnv("MS_DEV_CELL_REUSE") == "1" || common::GetEnv("MS_DEV_CELL_REUSE") == "2");
  if (!is_enable_ge || !is_cell_reuse) {
    return;
  }
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto stages = parallel_context->pipeline_stage_split_num();
  if (stages <= 1) {
    return;
  }
  // pipeline optimizer allgather exec before micro 0 recv;
  AddAllGatherRecvDepend(graph);
  // prev micro send exec before closure
  AddSendClosureDepend(graph);

  AnfNodePtr last_need_depend = nullptr;
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtr return_node = graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = TopoSort(return_node);

  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->func_graph() != graph) {
      continue;
    }
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }

    if (IsCommOps(node) || IsClosure(node)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (last_need_depend != nullptr && cnode->inputs().size() > 1) {
        auto before_input = node->cast<CNodePtr>()->input(1);
        auto new_depend = graph->NewCNode({NewValueNode(prim::kPrimDepend), before_input, last_need_depend});
        new_depend->set_abstract(before_input->abstract());
        manager->SetEdge(node, 1, new_depend);
      }
      last_need_depend = node;
    }
  }

  for (auto &node : all_nodes) {
    if (IsOneOfPrimitiveCNode(node, {prim::kPrimSend, prim::kPrimNPUClearFloatStatusV2})) {
      ProcessNodeWithoutOutput(graph, node->cast<CNodePtr>());
    }
  }
}
}  // namespace mindspore::opt
