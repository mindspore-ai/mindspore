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

#include "frontend/parallel/pass/micro_interleaved_order_control.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <utility>
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/utils.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kGradientsFlag = "Gradients";
const size_t interleaved_size = 2;
const size_t node_size_two = 2;
const size_t node_size_three = 3;
constexpr char kAttrFineGrainedInterleavedIndex[] = "fine_grained_interleaved_index";
constexpr char kAttrFineGrainedInterleavedBlockIndex[] = "fine_grained_interleaved_block";
using interleaved_node_pair_vector = std::vector<std::pair<size_t, std::vector<CNodePtr>>>;
bool IsBpropNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return node->fullname_with_scope().find(kGradientsFlag) == 0;
}

bool CheckCommNodeEqual(const CNodePtr comm_node1, const CNodePtr comm_node2) {
  auto prim1 = GetCNodePrimitive(comm_node1);
  auto prim2 = GetCNodePrimitive(comm_node2);
  if (!IsCommunicationOp(prim1) && !IsCommunicationOp(prim2)) {
    return true;
  }
  if (prim1->type_name() != prim2->type_name()) {
    MS_LOG(INFO) << "Type of two comm node is not euqal";
    return false;
  }
  if (!prim1->HasAttr(parallel::GROUP) || !prim2->HasAttr(parallel::GROUP)) {
    return false;
  }
  auto group1 = GetValue<std::string>(prim1->GetAttr(parallel::GROUP));
  auto group2 = GetValue<std::string>(prim2->GetAttr(parallel::GROUP));
  if (group1 != group2) {
    MS_LOG(INFO) << "Group of two comm node is not euqal.";
    return false;
  }
  auto shape1 = dyn_cast<abstract::Shape>(comm_node1->Shape());
  auto shape2 = dyn_cast<abstract::Shape>(comm_node2->Shape());
  if (shape1 == nullptr || shape2 == nullptr) {
    return false;
  }
  if (shape1->shape() != shape2->shape()) {
    return false;
  }
  return true;
}

std::unordered_map<int64_t, std::vector<CNodePtr>> ExtractBlockIdCommNode(
  const std::vector<CNodePtr> &origin_nodes_topological, std::string block_index = "micro") {
  std::unordered_map<int64_t, std::vector<CNodePtr>> result_map;
  for (size_t i = 0; i < origin_nodes_topological.size(); ++i) {
    auto cnode = origin_nodes_topological[i];
    if ((!cnode->HasAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER) &&
         !cnode->HasAttr("fine_grained_interleaved_border")) ||
        !cnode->HasAttr(parallel::MICRO_INTERLEAVED_INDEX) || cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }
    if (!cnode->HasPrimalAttr(block_index)) {
      continue;
    }
    auto block_id = GetValue<int64_t>(cnode->GetPrimalAttr(block_index));
    if (block_index == kAttrFineGrainedInterleavedBlockIndex && cnode->HasPrimalAttr(MICRO)) {
      auto micro_id = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
      block_id = micro_id * 1000 + block_id;
    }
    result_map[block_id].push_back(cnode);
  }
  return result_map;
}

bool ExtractInterLeavedCommNode(const std::vector<CNodePtr> &origin_nodes_topological, bool is_forward,
                                interleaved_node_pair_vector *micro_interleaved_fp_bp_node_list, int64_t block_id = -1,
                                std::string block_index = "micro") {
  std::vector<std::pair<std::pair<size_t, size_t>, CNodePtr>> micro_interleaved_fp_bp_node_list0;
  std::vector<std::pair<std::pair<size_t, size_t>, CNodePtr>> micro_interleaved_fp_bp_node_list1;
  for (size_t i = 0; i < origin_nodes_topological.size(); ++i) {
    auto cnode = origin_nodes_topological[i];
    if (!cnode->HasAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER) ||
        !cnode->HasAttr(parallel::MICRO_INTERLEAVED_INDEX) || cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }

    if (is_forward == IsBpropNode(cnode)) {
      continue;
    }

    if (block_id >= 0 && cnode->HasPrimalAttr(block_index) &&
        GetValue<int64_t>(cnode->GetPrimalAttr(block_index)) != block_id) {
      continue;
    }
    if (block_id >= 0 && !cnode->HasPrimalAttr(block_index)) {
      MS_LOG(INFO) << "communication cnode :" << cnode->DebugString() << " dose not contains " << block_index
                   << " info.";
      continue;
    }
    size_t micro_interleaved_fp_bp_comm_order =
      GetValue<size_t>(cnode->GetAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER));
    size_t micro_interleaved_index = GetValue<size_t>(cnode->GetAttr(parallel::MICRO_INTERLEAVED_INDEX));
    if (micro_interleaved_index == 0) {
      micro_interleaved_fp_bp_node_list0.push_back({{micro_interleaved_fp_bp_comm_order, i}, cnode});
    } else if (micro_interleaved_index == 1) {
      micro_interleaved_fp_bp_node_list1.push_back({{micro_interleaved_fp_bp_comm_order, i}, cnode});
    } else {
      MS_LOG(INFO) << "The micro interleaved num can only be 2.";
      return false;
    }
  }
  if (micro_interleaved_fp_bp_node_list0.size() != micro_interleaved_fp_bp_node_list1.size()) {
    return false;
  }
  std::sort(micro_interleaved_fp_bp_node_list0.begin(), micro_interleaved_fp_bp_node_list0.end(),
            [](auto pair1, auto pair2) { return pair1.first.first < pair2.first.first; });
  std::sort(micro_interleaved_fp_bp_node_list1.begin(), micro_interleaved_fp_bp_node_list1.end(),
            [](auto pair1, auto pair2) { return pair1.first.first < pair2.first.first; });
  for (size_t i = 0; i < micro_interleaved_fp_bp_node_list0.size(); ++i) {
    std::vector<CNodePtr> fp_bp_node_same_id;
    if (micro_interleaved_fp_bp_node_list0[i].first.first != micro_interleaved_fp_bp_node_list1[i].first.first) {
      return false;
    }
    fp_bp_node_same_id.push_back(micro_interleaved_fp_bp_node_list0[i].second);
    fp_bp_node_same_id.push_back(micro_interleaved_fp_bp_node_list1[i].second);
    (*micro_interleaved_fp_bp_node_list)
      .push_back({micro_interleaved_fp_bp_node_list0[i].first.second, fp_bp_node_same_id});
  }
  std::sort((*micro_interleaved_fp_bp_node_list).begin(), (*micro_interleaved_fp_bp_node_list).end(),
            [](auto pair1, auto pair2) { return pair1.first < pair2.first; });
  return true;
}

void InsertDepend(const FuncGraphManagerPtr &manager, const CNodePtr &comm_node_a, const CNodePtr &comm_node_b,
                  const CNodePtr &next_comm_node_a) {
  MS_LOG(INFO) << "comm_node_a:" << comm_node_a->fullname_with_scope()
               << ", comm_node_b:" << comm_node_b->fullname_with_scope();
  if (next_comm_node_a->size() < node_size_two || !IsPrimitiveCNode(next_comm_node_a->input(1)) ||
      comm_node_b->size() < node_size_two || !IsPrimitiveCNode(comm_node_b->input(1))) {
    return;
  }
  auto next_comm_node_a_input_node = next_comm_node_a->input(1)->cast<CNodePtr>();
  auto comm_node_b_input_node = comm_node_b->input(1)->cast<CNodePtr>();
  // comm_node_b_input -> depend -> comm_node_a_output
  std::vector<AnfNodePtr> depend1_inputs{NewValueNode(prim::kPrimDepend), comm_node_a, comm_node_b_input_node};
  auto depend_node1 = comm_node_a->func_graph()->NewCNode(depend1_inputs);
  depend_node1->set_abstract(comm_node_a->abstract()->Clone());
  depend_node1->AddAttr("micro_interleaved_depend1", MakeValue(true));
  MS_EXCEPTION_IF_NULL(depend_node1);
  manager->Replace(comm_node_a, depend_node1);
  // next_comm_node_a_input -> depend -> comm_node_b_output
  std::vector<AnfNodePtr> depend2_inputs{NewValueNode(prim::kPrimDepend), comm_node_b, next_comm_node_a_input_node};
  auto depend_node2 = next_comm_node_a_input_node->func_graph()->NewCNode(depend2_inputs);
  depend_node2->AddAttr("micro_interleaved_depend2", MakeValue(true));
  depend_node2->set_abstract(comm_node_b->abstract()->Clone());
  MS_EXCEPTION_IF_NULL(depend_node2);
  manager->Replace(comm_node_b, depend_node2);
}

void InsertInterleavedNodesDepend(const FuncGraphManagerPtr &manager,
                                  const interleaved_node_pair_vector &micro_interleaved_node_list) {
  for (size_t i = 0; i + 1 < micro_interleaved_node_list.size(); ++i) {
    auto comm_node_list = micro_interleaved_node_list[i].second;
    auto next_comm_node_list = micro_interleaved_node_list[i + 1].second;
    auto comm_node_a = comm_node_list[0];
    auto comm_node_b = comm_node_list[1];
    auto next_comm_node_a = next_comm_node_list[0];
    InsertDepend(manager, comm_node_a, comm_node_b, next_comm_node_a);
  }
}

void InsertDependBetweenInterleavedNodes(const FuncGraphManagerPtr &manager,
                                         const interleaved_node_pair_vector &micro_interleaved_node_list,
                                         bool add_second_depend = false) {
  for (size_t i = 0; i + 1 < micro_interleaved_node_list.size(); ++i) {
    auto comm_node_list = micro_interleaved_node_list[i].second;
    auto next_comm_node_list = micro_interleaved_node_list[i + 1].second;
    auto comm_node_b = comm_node_list[1];
    auto next_comm_node_a = next_comm_node_list[0];
    // comm_node_b -> next_comm_node_a
    auto next_comm_node_a_input_node = next_comm_node_a->input(1)->cast<CNodePtr>();
    std::vector<AnfNodePtr> depend1_inputs{NewValueNode(prim::kPrimDepend), next_comm_node_a_input_node, comm_node_b};
    auto depend_node1 = comm_node_b->func_graph()->NewCNode(depend1_inputs);
    depend_node1->set_abstract(next_comm_node_a_input_node->abstract()->Clone());
    depend_node1->AddAttr("micro_interleaved_comm_depend1", MakeValue(true));
    MS_EXCEPTION_IF_NULL(depend_node1);
    manager->Replace(next_comm_node_a_input_node, depend_node1);
    if (!add_second_depend) {
      continue;
    }
    // comm_node_a -> comm_node_b
    auto comm_node_a = comm_node_list[0];
    auto comm_node_b_input_node = comm_node_b->input(1)->cast<CNodePtr>();
    std::vector<AnfNodePtr> depend2_inputs{NewValueNode(prim::kPrimDepend), comm_node_b_input_node, comm_node_a};
    auto depend_node2 = comm_node_a->func_graph()->NewCNode(depend2_inputs);
    depend_node2->set_abstract(comm_node_b_input_node->abstract()->Clone());
    depend_node2->AddAttr("micro_interleaved_comm_depend2", MakeValue(true));
    manager->Replace(comm_node_b_input_node, depend_node2);
  }
}

CNodePtr GetInputBorderNode(const CNodePtr &node) {
  std::queue<CNodePtr> anf_queue;
  anf_queue.push(node);
  size_t loop_count = 0;
  while (!anf_queue.empty() && loop_count < 100) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    for (size_t i = 1; i < queue_end->size(); ++i) {
      if (IsPrimitiveCNode(queue_end->input(i))) {
        auto queue_end_input_cnode = queue_end->input(i)->cast<CNodePtr>();
        if (queue_end_input_cnode->HasAttr("fine_grained_interleaved_border")) {
          return queue_end;
        }
        anf_queue.push(queue_end->input(i)->cast<CNodePtr>());
      }
    }
    loop_count++;
  }
  return nullptr;
}

CNodePtr GetOutputBorderNode(const FuncGraphManagerPtr &manager, const CNodePtr &node) {
  std::queue<CNodePtr> anf_queue;
  anf_queue.push(node);
  size_t loop_count = 0;
  while (!anf_queue.empty() && loop_count < 100) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    auto node_users = manager->node_users()[queue_end];
    for (const auto &node_pair : node_users) {
      if (IsPrimitiveCNode(node_pair.first)) {
        auto queue_end_output_cnode = node_pair.first->cast<CNodePtr>();
        if (queue_end_output_cnode->HasAttr("fine_grained_interleaved_border")) {
          return queue_end;
        }
        anf_queue.push(node_pair.first->cast<CNodePtr>());
      }
    }
    loop_count++;
  }
  return nullptr;
}

void InsertDependForEnd(const FuncGraphManagerPtr &manager,
                        const interleaved_node_pair_vector &micro_interleaved_node_list) {
  if (micro_interleaved_node_list.empty()) {
    return;
  }
  auto comm_node_list = micro_interleaved_node_list.back().second;
  auto comm_node_a = comm_node_list[0];
  auto comm_node_b = comm_node_list[1];
  auto end = GetOutputBorderNode(manager, comm_node_a);
  if (!end) {
    MS_LOG(INFO) << "Cannot find end node for micro_interleaved.";
    return;
  }
  InsertDepend(manager, comm_node_a, comm_node_b, end);
}

void InsertDependForBegin(const FuncGraphManagerPtr &manager,
                          const interleaved_node_pair_vector &micro_interleaved_node_list) {
  if (micro_interleaved_node_list.empty()) {
    return;
  }
  auto comm_node_list = micro_interleaved_node_list.front().second;
  auto comm_node_a = comm_node_list[0];
  auto comm_node_b = comm_node_list[1];
  if (!IsPrimitiveCNode(comm_node_b->input(1))) {
    return;
  }
  auto begin = GetInputBorderNode(comm_node_b->input(1)->cast<CNodePtr>());
  if (!begin) {
    return;
  }
  auto begin_input = begin->input(1);
  if (IsPrimitiveCNode(begin, prim::kPrimTupleGetItem)) {
    begin_input = begin;
  }
  auto comm_node_a_input_node = comm_node_a->input(1)->cast<CNodePtr>();
  std::vector<AnfNodePtr> depend2_inputs{NewValueNode(prim::kPrimDepend), begin_input, comm_node_a_input_node};
  auto depend_node2 = comm_node_a_input_node->func_graph()->NewCNode(depend2_inputs);
  depend_node2->AddAttr("micro_interleaved_depend_begin", MakeValue(true));
  depend_node2->set_abstract(begin_input->abstract()->Clone());
  MS_EXCEPTION_IF_NULL(depend_node2);
  manager->Replace(begin_input, depend_node2);
}

void MicroInterleavedOrderControlProcess(const FuncGraphManagerPtr &manager,
                                         const interleaved_node_pair_vector &micro_interleaved_forward_node_list,
                                         const interleaved_node_pair_vector &micro_interleaved_backward_node_list,
                                         const std::vector<CNodePtr> &origin_nodes_topological, int block_id = -1) {
  if (micro_interleaved_forward_node_list.empty() && micro_interleaved_backward_node_list.empty()) {
    MS_LOG(INFO) << "Cannot find micro interleaved nodes.";
    return;
  }
  for (auto &pair : micro_interleaved_forward_node_list) {
    auto cnode_list = pair.second;
    if (!CheckCommNodeEqual(cnode_list[0], cnode_list[1])) {
      MS_LOG(INFO) << cnode_list[0]->DebugString() << " and " << cnode_list[1]->DebugString()
                   << " not match for micro interleaved.";

      return;
    }
  }
  for (auto &pair : micro_interleaved_backward_node_list) {
    auto cnode_list = pair.second;
    if (!CheckCommNodeEqual(cnode_list[0], cnode_list[1])) {
      MS_LOG(INFO) << cnode_list[0]->DebugString() << " and " << cnode_list[1]->fullname_with_scope()
                   << " not match for micro interleaved.";
      return;
    }
  }
  // find the fp_begin node
  InsertDependForBegin(manager, micro_interleaved_forward_node_list);
  InsertInterleavedNodesDepend(manager, micro_interleaved_forward_node_list);
  InsertDependBetweenInterleavedNodes(manager, micro_interleaved_forward_node_list, true);
  InsertDependForEnd(manager, micro_interleaved_forward_node_list);
  InsertDependForBegin(manager, micro_interleaved_backward_node_list);
  InsertInterleavedNodesDepend(manager, micro_interleaved_backward_node_list);
  InsertDependBetweenInterleavedNodes(manager, micro_interleaved_backward_node_list);
  InsertDependForEnd(manager, micro_interleaved_backward_node_list);
}

void MicroInterleavedOrderControlInBlock(const FuncGraphPtr &graph, const FuncGraphManagerPtr &manager,
                                         const std::vector<CNodePtr> &origin_nodes_topological,
                                         const std::string &block_index = "micro") {
  // 1 order forward_node, and the node with same MICRO_INTERLEAVED_FORWARD_COMM_ORDER is the micro interleaved pair
  // nodes.
  MS_EXCEPTION_IF_NULL(parallel::g_device_manager);
  auto micro_comm_nodes_map = ExtractBlockIdCommNode(origin_nodes_topological, block_index);
  for (auto block_iter : micro_comm_nodes_map) {
    auto block_id = block_iter.first;
    auto comm_nodes_list = block_iter.second;
    interleaved_node_pair_vector micro_interleaved_forward_node_list;
    if (!ExtractInterLeavedCommNode(comm_nodes_list, true, &micro_interleaved_forward_node_list)) {
      MS_LOG(INFO) << "Cannot match forward micro interleaved conditions for block: " << block_id << " " << block_index;
    }
    interleaved_node_pair_vector micro_interleaved_backward_node_list;
    if (!ExtractInterLeavedCommNode(comm_nodes_list, false, &micro_interleaved_backward_node_list)) {
      MS_LOG(INFO) << "Cannot match backward micro interleaved conditions for block: " << block_id << " "
                   << block_index;
    }
    MicroInterleavedOrderControlProcess(manager, micro_interleaved_forward_node_list,
                                        micro_interleaved_backward_node_list, origin_nodes_topological, block_id);
  }
  return;
}

bool IsFineGrained(const std::vector<CNodePtr> &origin_nodes_topological) {
  for (size_t i = 0; i < origin_nodes_topological.size(); ++i) {
    auto cnode = origin_nodes_topological[i];
    auto prim = GetCNodePrimitive(cnode);
    if (!prim) {
      continue;
    }
    if (prim->HasAttr(kAttrFineGrainedInterleavedIndex)) {
      return true;
    }
  }
  return false;
}

void CellReuseProcess(const FuncGraphManagerPtr &manager, const std::string &find_rained_block_id = "") {
  for (const auto &each_graph : manager->func_graphs()) {
    if (IsCellReuseForwardGraph(each_graph)) {
      auto forward_graph = each_graph;
      auto backward_graph = GetCellReuseBackwardGraph(forward_graph);
      if (backward_graph == nullptr) {
        MS_LOG(WARNING) << "Failed to find backward cell reuse graph, skip pass.";
        return;
      }
      std::list<CNodePtr> forward_orders = forward_graph->GetOrderedCnodes();
      std::vector<CNodePtr> forward_origin_nodes_topological(forward_orders.cbegin(), forward_orders.cend());
      std::list<CNodePtr> backward_orders = backward_graph->GetOrderedCnodes();
      std::vector<CNodePtr> backward_origin_nodes_topological(backward_orders.cbegin(), backward_orders.cend());
      bool is_fine_rained = IsFineGrained(forward_origin_nodes_topological);
      if (is_fine_rained) {
        MicroInterleavedOrderControlInBlock(forward_graph, manager, forward_origin_nodes_topological,
                                            find_rained_block_id);
        MicroInterleavedOrderControlInBlock(backward_graph, manager, backward_origin_nodes_topological,
                                            find_rained_block_id);
        return;
      }
      interleaved_node_pair_vector micro_interleaved_forward_node_list;
      if (!ExtractInterLeavedCommNode(forward_origin_nodes_topological, true, &micro_interleaved_forward_node_list)) {
        MS_LOG(INFO) << "Cannot match micro interleaved conditions.";
        return;
      }
      interleaved_node_pair_vector micro_interleaved_backward_node_list;
      if (!ExtractInterLeavedCommNode(backward_origin_nodes_topological, false,
                                      &micro_interleaved_backward_node_list)) {
        MS_LOG(INFO) << "Cannot match micro interleaved conditions.";
        return;
      }
      MicroInterleavedOrderControlProcess(manager, micro_interleaved_forward_node_list,
                                          micro_interleaved_backward_node_list, forward_origin_nodes_topological);
      MicroInterleavedOrderControlProcess(manager, micro_interleaved_forward_node_list,
                                          micro_interleaved_backward_node_list, backward_origin_nodes_topological);
    }
  }
}
}  // namespace

void MicroInterleavedOrderControl(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  if (!parallel::ParallelContext::GetInstance()->enable_fine_grained_micro_interleaved()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (common::GetEnv("MS_ENABLE_FRONTEND_SCHEDULING_OPTIMIZATION") == "1") {
    return;
  }
  auto context = MsContext::GetInstance();
  const auto graph_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;

  if (graph_reuse) {
    CellReuseProcess(manager, kAttrFineGrainedInterleavedBlockIndex);
    return;
  }
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  bool is_fine_grained = IsFineGrained(origin_nodes_topological);
  if (is_fine_grained) {
    MicroInterleavedOrderControlInBlock(graph, manager, origin_nodes_topological,
                                        kAttrFineGrainedInterleavedBlockIndex);
    return;
  }
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() == 1) {
    // 1 order forward_node, and the node with same MICRO_INTERLEAVED_FORWARD_COMM_ORDER is the micro interleaved pair
    // nodes.
    interleaved_node_pair_vector micro_interleaved_forward_node_list;
    if (!ExtractInterLeavedCommNode(origin_nodes_topological, true, &micro_interleaved_forward_node_list)) {
      MS_LOG(INFO) << "Cannot match micro interleaved conditions.";
      return;
    }
    interleaved_node_pair_vector micro_interleaved_backward_node_list;
    if (!ExtractInterLeavedCommNode(origin_nodes_topological, false, &micro_interleaved_backward_node_list)) {
      MS_LOG(INFO) << "Cannot match micro interleaved conditions.";
      return;
    }
    MicroInterleavedOrderControlProcess(manager, micro_interleaved_forward_node_list,
                                        micro_interleaved_backward_node_list, origin_nodes_topological);
    return;
  }
  MicroInterleavedOrderControlInBlock(graph, manager, origin_nodes_topological);
}
}  // namespace parallel
}  // namespace mindspore
