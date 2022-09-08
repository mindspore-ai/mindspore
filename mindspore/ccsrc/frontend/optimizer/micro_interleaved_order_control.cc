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

#include "frontend/optimizer/micro_interleaved_order_control.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <utility>
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kGradientsFlag = "Gradients";
constexpr auto kMicroInterleavedTag = "micro_interleaved_been_tag";
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

void SpreadMicroInterleavedIndexForForwardCommNodes(const CNodePtr &input_node, size_t micro_interleaved_index,
                                                    int64_t pipeline_micro = -1) {
  std::queue<CNodePtr> node_queue;
  node_queue.push(input_node);
  size_t forward_order = 0;
  while (!node_queue.empty()) {
    auto cnode = node_queue.front();
    node_queue.pop();
    auto cnode_inputs = cnode->inputs();
    auto spread_size = cnode_inputs.size();
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      spread_size = node_size_two;
    }
    for (size_t i = 1; i < spread_size; ++i) {
      auto input = cnode_inputs[i];
      if (!IsPrimitiveCNode(input)) {
        continue;
      }
      if (IsBpropNode(input) || IsPrimitiveCNode(input, prim::kPrimUpdateState)) {
        continue;
      }
      auto input_cnode = input->cast<CNodePtr>();
      if (input_cnode->HasAttr(kMicroInterleavedTag)) {
        continue;
      }
      bool is_pipeline = (pipeline_micro >= 0 && input_cnode->HasPrimalAttr(parallel::MICRO));
      if (is_pipeline && GetValue<int64_t>(input_cnode->GetPrimalAttr(parallel::MICRO)) != pipeline_micro) {
        continue;
      }
      input_cnode->AddAttr(kMicroInterleavedTag, MakeValue<bool>(True));
      node_queue.push(input_cnode);
      if (input_cnode->HasPrimalAttr(parallel::FORWARD_NODE_UNIQUE_ID)) {
        if (pipeline_micro >= 0 && !input_cnode->HasPrimalAttr(parallel::MICRO)) {
          MS_LOG(INFO) << "node :" << input_cnode->DebugString() << " dose not contain micro tag.";
          continue;
        }
        input_cnode->AddAttr(parallel::MICRO_INTERLEAVED_INDEX, MakeValue<size_t>(micro_interleaved_index));
        input_cnode->AddAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER, MakeValue<size_t>(forward_order));
        forward_order++;
      }
    }
  }
}

void LabelMicroInterleavedIndexForBackwardCommNodes(const std::vector<CNodePtr> &all_nodes) {
  mindspore::HashMap<std::string, CNodePtr> forward_comm_nodes_map;
  mindspore::HashMap<std::string, CNodePtr> grad_forward_comm_nodes_map;
  for (auto &cnode : all_nodes) {
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    if (!cnode->HasPrimalAttr(parallel::FORWARD_NODE_UNIQUE_ID)) {
      continue;
    }
    auto forward_node_unique_id = GetValue<std::string>(cnode->GetPrimalAttr(parallel::FORWARD_NODE_UNIQUE_ID));
    if (IsBpropNode(cnode)) {
      grad_forward_comm_nodes_map[forward_node_unique_id] = cnode;
      continue;
    }
    if (cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }
    forward_comm_nodes_map[forward_node_unique_id] = cnode;
  }
  for (auto &pair : grad_forward_comm_nodes_map) {
    if (forward_comm_nodes_map.find(pair.first) == forward_comm_nodes_map.end()) {
      continue;
    }
    auto forward_node = forward_comm_nodes_map[pair.first];
    if (!forward_node->HasAttr(parallel::MICRO_INTERLEAVED_INDEX) ||
        !forward_node->HasAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER)) {
      continue;
    }
    pair.second->AddAttr(parallel::MICRO_INTERLEAVED_INDEX, forward_node->GetAttr(parallel::MICRO_INTERLEAVED_INDEX));
    pair.second->AddAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER,
                         forward_node->GetAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER));
  }
}

void LabelMicroInterleavedIndex(const std::vector<CNodePtr> &all_nodes) {
  CNodePtr micro_interleaved_add = nullptr;
  for (auto &cnode : all_nodes) {
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    if (GetCNodePrimitive(cnode)->HasAttr("micro_interleaved_add_flag")) {
      micro_interleaved_add = cnode;
      break;
    }
  }
  if (micro_interleaved_add == nullptr || micro_interleaved_add->size() != node_size_three) {
    return;
  }
  for (size_t i = 1; i < micro_interleaved_add->size(); ++i) {
    auto input_cnode = micro_interleaved_add->input(i)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    SpreadMicroInterleavedIndexForForwardCommNodes(input_cnode, i - 1);
  }
  LabelMicroInterleavedIndexForBackwardCommNodes(all_nodes);
}

size_t LabelMicroInterleavedIndexLastStage(const std::vector<CNodePtr> &all_nodes) {
  std::vector<CNodePtr> micro_interleaved_add_list;
  for (auto &cnode : all_nodes) {
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    if (GetCNodePrimitive(cnode)->HasAttr("micro_interleaved_add_flag")) {
      micro_interleaved_add_list.push_back(cnode);
    }
  }

  for (auto &micro_interleaved_add : micro_interleaved_add_list) {
    auto pipeline_micro = GetValue<int64_t>(micro_interleaved_add->GetPrimalAttr(parallel::MICRO));
    for (size_t i = 1; i < micro_interleaved_add->size(); ++i) {
      auto input_cnode = micro_interleaved_add->input(i)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      SpreadMicroInterleavedIndexForForwardCommNodes(input_cnode, i - 1, pipeline_micro);
    }
  }
  LabelMicroInterleavedIndexForBackwardCommNodes(all_nodes);
  return micro_interleaved_add_list.size();
}

size_t LabelMicroInterleavedIndexPipelineStage(const std::vector<CNodePtr> &all_nodes) {
  mindspore::HashMap<size_t, std::vector<CNodePtr>> pipeline_end_list_map;
  std::vector<size_t> micro_list;
  for (auto &cnode : all_nodes) {
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    if (IsBpropNode(cnode)) {
      continue;
    }
    if (!cnode->HasPrimalAttr(parallel::PIPELINE_END) || !cnode->HasPrimalAttr(parallel::MICRO)) {
      continue;
    }
    size_t pipeline_end = LongToSize(GetValue<int64_t>(cnode->GetPrimalAttr(parallel::PIPELINE_END)));
    size_t micro = LongToSize(GetValue<int64_t>(cnode->GetPrimalAttr(parallel::MICRO)));
    if (pipeline_end != micro) {
      continue;
    }
    if (pipeline_end_list_map.find(pipeline_end) == pipeline_end_list_map.end()) {
      pipeline_end_list_map[pipeline_end] = {cnode};
      micro_list.push_back(pipeline_end);
    } else {
      pipeline_end_list_map[pipeline_end].push_back(cnode);
    }
  }

  for (size_t i = 0; i < micro_list.size(); ++i) {
    auto pipeline_end_list = pipeline_end_list_map[micro_list[i]];
    if (pipeline_end_list.size() != interleaved_size) {
      continue;
    }
    if (GetCNodePrimitive(pipeline_end_list[0])->HasAttr(parallel::SR_TAG) &&
        GetCNodePrimitive(pipeline_end_list[1])->HasAttr(parallel::SR_TAG)) {
      std::sort(pipeline_end_list.begin(), pipeline_end_list.end(), [](auto cnode1, auto cnode2) {
        return GetValue<int64_t>(GetCNodePrimitive(cnode1)->GetAttr(parallel::SR_TAG)) <
               GetValue<int64_t>(GetCNodePrimitive(cnode2)->GetAttr(parallel::SR_TAG));
      });
    }
    SpreadMicroInterleavedIndexForForwardCommNodes(pipeline_end_list[0], 0, micro_list[i]);
    SpreadMicroInterleavedIndexForForwardCommNodes(pipeline_end_list[1], 1, micro_list[i]);
  }
  LabelMicroInterleavedIndexForBackwardCommNodes(all_nodes);
  return micro_list.size();
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
  InsertInterleavedNodesDepend(manager, micro_interleaved_forward_node_list);
  InsertInterleavedNodesDepend(manager, micro_interleaved_backward_node_list);
}

void MicroInterleavedOrderControlPipeline(const FuncGraphManagerPtr &manager,
                                          const std::vector<CNodePtr> &origin_nodes_topological) {
  // 1 order forward_node, and the node with same MICRO_INTERLEAVED_FORWARD_COMM_ORDER is the micro interleaved pair
  // nodes.
  MS_EXCEPTION_IF_NULL(parallel::g_device_manager);
  auto stage_num = parallel::g_device_manager->stage_num();
  auto stage_id = parallel::g_device_manager->stage_id();
  size_t pipeline_micro_size = 1;
  if (stage_id == stage_num - 1) {
    pipeline_micro_size = LabelMicroInterleavedIndexLastStage(origin_nodes_topological);
  } else {
    pipeline_micro_size = LabelMicroInterleavedIndexPipelineStage(origin_nodes_topological);
  }
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
    LabelMicroInterleavedIndex(origin_nodes_topological);
    MicroInterleavedOrderControl(manager, origin_nodes_topological);
    return;
  }
  MicroInterleavedOrderControlPipeline(manager, origin_nodes_topological);
}
}  // namespace opt
}  // namespace mindspore
