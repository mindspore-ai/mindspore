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
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kGradientsFlag = "Gradients";
const size_t interleaved_size = 2;
const size_t node_size_two = 2;
const size_t node_size_three = 3;
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

bool ExtractInterLeavedCommNode(const std::vector<CNodePtr> &origin_nodes_topological, bool is_forward,
                                interleaved_node_pair_vector *micro_interleaved_fp_bp_node_list,
                                int64_t pipeline_micro = -1) {
  std::vector<std::pair<std::pair<size_t, size_t>, CNodePtr>> micro_interleaved_fp_bp_node_list0;
  std::vector<std::pair<std::pair<size_t, size_t>, CNodePtr>> micro_interleaved_fp_bp_node_list1;
  for (size_t i = 0; i < origin_nodes_topological.size(); ++i) {
    auto cnode = origin_nodes_topological[i];
    if (!common::AnfAlgo::IsCommunicationOp(cnode) || !cnode->HasAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER) ||
        !cnode->HasAttr(parallel::MICRO_INTERLEAVED_INDEX) || cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }

    if (is_forward == IsBpropNode(cnode)) {
      continue;
    }

    if (pipeline_micro >= 0 && cnode->HasPrimalAttr(parallel::MICRO) &&
        GetValue<int64_t>(cnode->GetPrimalAttr(parallel::MICRO)) != pipeline_micro) {
      continue;
    }
    if (pipeline_micro >= 0 && !cnode->HasPrimalAttr(parallel::MICRO)) {
      MS_LOG(INFO) << "communication cnode :" << cnode->DebugString() << " dose not contains micro info.";
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

void CreateGroupForMicroInterleaved(const CNodePtr &comm_cnode, size_t micro_interleaved_index) {
  auto comm_prim = GetCNodePrimitive(comm_cnode);
  auto group_name = GetValue<std::string>(comm_prim->GetAttr(parallel::GROUP));
  if (group_name.find("micro_interleaved") != std::string::npos) {
    return;
  }
  auto rank_ids = parallel::g_device_manager->FindRankListByHashName(group_name);
  auto dev_list = parallel::g_device_manager->CreateDeviceListByRankList(rank_ids);
  auto new_group_name = group_name + "_micro_interleaved_" + std::to_string(micro_interleaved_index);
  parallel::Group cur_device_list;
  parallel::g_device_manager->CreateGroup(new_group_name, dev_list, &cur_device_list);
  auto new_comm_prim = comm_prim->Clone();
  new_comm_prim->SetAttrs(comm_prim->attrs());
  new_comm_prim->AddAttr(parallel::GROUP, MakeValue<std::string>(new_group_name));
  comm_cnode->set_input(0, NewValueNode(new_comm_prim));
}

void InsertInterleavedNodesDepend(const FuncGraphManagerPtr &manager,
                                  const interleaved_node_pair_vector &micro_interleaved_node_list) {
  for (size_t i = 0; i < micro_interleaved_node_list.size() - 1; ++i) {
    auto comm_node_list = micro_interleaved_node_list[i].second;
    auto next_comm_node_list = micro_interleaved_node_list[i + 1].second;
    auto comm_node_a = comm_node_list[0];
    auto comm_node_b = comm_node_list[1];
    auto next_comm_node_a = next_comm_node_list[0];
    auto next_comm_node_b = next_comm_node_list[1];
    if (next_comm_node_a->size() < node_size_two || !IsPrimitiveCNode(next_comm_node_a->input(1)) ||
        comm_node_b->size() < node_size_two || !IsPrimitiveCNode(comm_node_b->input(1))) {
      continue;
    }
    auto next_comm_node_a_input_node = next_comm_node_a->input(1)->cast<CNodePtr>();
    auto comm_node_b_input_node = comm_node_b->input(1)->cast<CNodePtr>();
    auto comm_node_a_node_users = manager->node_users()[comm_node_a];
    auto comm_node_b_node_users = manager->node_users()[comm_node_b];
    if (comm_node_a_node_users.empty() || comm_node_b_node_users.empty()) {
      continue;
    }
    auto comm_node_a_output_node = comm_node_a_node_users.front().first;
    auto comm_node_b_output_node = comm_node_b_node_users.front().first;
    // comm_node_b_input -> depend -> comm_node_a_output
    std::vector<AnfNodePtr> depend1_inputs{NewValueNode(prim::kPrimDepend), comm_node_a, comm_node_b_input_node};
    auto depend_node1 = comm_node_a_output_node->func_graph()->NewCNode(depend1_inputs);
    depend_node1->set_abstract(comm_node_a->abstract()->Clone());
    depend_node1->AddAttr("micro_interleaved_depend1", MakeValue(true));
    MS_EXCEPTION_IF_NULL(depend_node1);
    manager->SetEdge(comm_node_a_output_node, comm_node_a_node_users.front().second, depend_node1);
    // next_comm_node_a_input -> depend -> comm_node_b_output
    std::vector<AnfNodePtr> depend2_inputs{NewValueNode(prim::kPrimDepend), comm_node_b, next_comm_node_a_input_node};
    auto depend_node2 = next_comm_node_a_input_node->func_graph()->NewCNode(depend2_inputs);
    depend_node2->AddAttr("micro_interleaved_depend2", MakeValue(true));
    depend_node2->set_abstract(comm_node_b->abstract()->Clone());
    MS_EXCEPTION_IF_NULL(depend_node2);
    manager->SetEdge(comm_node_b_output_node, comm_node_b_node_users.front().second, depend_node2);
  }
}

void CreateExtraGroupForModelParallelCommNode(const std::vector<CNodePtr> &origin_nodes_topological,
                                              const interleaved_node_pair_vector &micro_interleaved_node_list) {
  std::unordered_map<std::string, size_t> unique_id_interleaved_map;
  for (const auto &pair : micro_interleaved_node_list) {
    auto cnode_list = pair.second;
    CreateGroupForMicroInterleaved(cnode_list[0], 0);
    CreateGroupForMicroInterleaved(cnode_list[1], 1);
    if (!IsBpropNode(cnode_list[0]) && cnode_list[0]->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      auto forward_comm_node_unique_id =
        GetValue<std::string>(cnode_list[0]->GetPrimalAttr(kPrimalAttrForwardCommNodeUniqueId));
      unique_id_interleaved_map[forward_comm_node_unique_id] = 0;
    }
    if (!IsBpropNode(cnode_list[1]) && cnode_list[1]->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      auto forward_comm_node_unique_id =
        GetValue<std::string>(cnode_list[1]->GetPrimalAttr(kPrimalAttrForwardCommNodeUniqueId));
      unique_id_interleaved_map[forward_comm_node_unique_id] = 1;
    }
  }

  if (unique_id_interleaved_map.empty()) {
    return;
  }

  for (const auto &cnode : origin_nodes_topological) {
    if (!cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }
    if (!cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      continue;
    }
    auto duplicate_comm_node_unique_id =
      GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrForwardCommNodeUniqueId));
    if (unique_id_interleaved_map.find(duplicate_comm_node_unique_id) == unique_id_interleaved_map.end()) {
      continue;
    }
    CreateGroupForMicroInterleaved(cnode, unique_id_interleaved_map[duplicate_comm_node_unique_id]);
  }
}

void MicroInterleavedOrderControl(const FuncGraphManagerPtr &manager,
                                  const std::vector<CNodePtr> &origin_nodes_topological, int pipeline_micro = -1) {
  // 1 order forward_node, and the node with same MICRO_INTERLEAVED_FORWARD_COMM_ORDER is the micro interleaved pair
  // nodes.
  interleaved_node_pair_vector micro_interleaved_forward_node_list;
  if (!ExtractInterLeavedCommNode(origin_nodes_topological, true, &micro_interleaved_forward_node_list,
                                  pipeline_micro)) {
    MS_LOG(INFO) << "Cannot match micro interleaved conditions.";
    return;
  }
  interleaved_node_pair_vector micro_interleaved_backward_node_list;
  if (!ExtractInterLeavedCommNode(origin_nodes_topological, false, &micro_interleaved_backward_node_list,
                                  pipeline_micro)) {
    MS_LOG(INFO) << "Cannot match micro interleaved conditions.";
    return;
  }

  if (micro_interleaved_forward_node_list.empty() || micro_interleaved_backward_node_list.empty()) {
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
  static const auto micro_interleaved_extra_comm_group = (common::GetEnv("interleaved_extra_group") == "1");
  if (micro_interleaved_extra_comm_group) {
    CreateExtraGroupForModelParallelCommNode(origin_nodes_topological, micro_interleaved_forward_node_list);
    CreateExtraGroupForModelParallelCommNode(origin_nodes_topological, micro_interleaved_backward_node_list);
  }
  InsertInterleavedNodesDepend(manager, micro_interleaved_forward_node_list);
  InsertInterleavedNodesDepend(manager, micro_interleaved_backward_node_list);
}

void MicroInterleavedOrderControlPipeline(const FuncGraphManagerPtr &manager,
                                          const std::vector<CNodePtr> &origin_nodes_topological) {
  // 1 order forward_node, and the node with same MICRO_INTERLEAVED_FORWARD_COMM_ORDER is the micro interleaved pair
  // nodes.
  MS_EXCEPTION_IF_NULL(parallel::g_device_manager);
  size_t pipeline_micro_size = parallel::ParallelContext::GetInstance()->pipeline_micro_size();
  MS_LOG(INFO) << "The pipeline micro size in micro interleaved is: " << pipeline_micro_size;
  for (size_t pipeline_micro_id = 0; pipeline_micro_id < pipeline_micro_size; ++pipeline_micro_id) {
    MicroInterleavedOrderControl(manager, origin_nodes_topological, pipeline_micro_id);
  }
  return;
}
}  // namespace

void MicroInterleavedOrderControl(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  if (!parallel::ParallelContext::GetInstance()->enable_micro_interleaved()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() == 1) {
    MicroInterleavedOrderControl(manager, origin_nodes_topological);
    return;
  }
  MicroInterleavedOrderControlPipeline(manager, origin_nodes_topological);
}
}  // namespace parallel
}  // namespace mindspore
