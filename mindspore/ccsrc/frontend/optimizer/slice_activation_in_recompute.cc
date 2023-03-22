/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/slice_activation_in_recompute.h"
#include <memory>
#include <queue>
#include <utility>
#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "frontend/parallel/tensor_layout/construct_operator.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kGradientsFlag = "Gradients";
const int64_t max_loop_size = 100;

CNodePtr CreateStridedSliceCNode(const parallel::Shape &begin, const parallel::Shape &end,
                                 const parallel::Shape &strides, const AnfNodePtr &node) {
  auto slice_op = parallel::CreateStridedSliceOp(0, begin, end, strides);
  auto slice_input = parallel::CreateInput(slice_op, node, parallel::STRIDEDSLICE);
  auto func_graph = node->func_graph();
  CNodePtr new_node = func_graph->NewCNode(slice_input);
  return new_node;
}

CNodePtr CreateAllGatherCNode(const AnfNodePtr &node, const std::string &group) {
  auto op = parallel::CreateAllGatherOp(group);
  auto allgather_input = parallel::CreateInput(op, node, "recompute_slice_allgather");
  auto func_graph = node->func_graph();
  CNodePtr new_node = func_graph->NewCNode(allgather_input);
  return new_node;
}

std::vector<parallel::Group> InferRepeatedRankList(const CNodePtr &cnode) {
  OperatorInfoPtr operator_info = cnode->user_data<parallel::OperatorInfo>();
  std::vector<parallel::TensorInfo> output_info = operator_info->outputs_tensor_info();
  if (output_info.size() != 1) {
    MS_LOG(EXCEPTION) << "The output_info size is wrong, node is" << cnode->DebugString();
  }
  auto tensor_layout = output_info[0].tensor_layout();
  auto tensor_map = tensor_layout.origin_tensor_map();
  std::vector<parallel::Group> groups;
  (void)operator_info->CreateGroupByTensorMap(tensor_map.array(), &groups);
  return groups;
}

bool IsDuplicateNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  if (node->cast<CNodePtr>()->HasAttr(kAttrDuplicated)) {
    return true;
  }
  if (IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad)) {
    auto manager = node->func_graph()->manager();
    auto node_users = manager->node_users()[node];
    return std::any_of(node_users.begin(), node_users.end(),
                       [](auto node_user) { return IsDuplicateNode(node_user.first); });
  }
  return false;
}

void GroupingNextNodes(const CNodePtr &node, std::vector<std::pair<std::shared_ptr<AnfNode>, int>> *duplicate_users,
                       std::vector<std::pair<std::shared_ptr<AnfNode>, int>> *forward_users) {
  auto manager = node->func_graph()->manager();
  auto root_node = node;
  auto node_users = manager->node_users()[root_node];
  for (auto node_user : node_users) {
    if (IsDuplicateNode(node_user.first)) {
      duplicate_users->push_back(node_user);
    } else {
      forward_users->push_back(node_user);
    }
  }
}

void CreateGroupForSliceAllGatherInMicroInterleaved(const CNodePtr &allgather_cnode, const std::string &group_name) {
  auto micro_interleaved_index = GetValue<size_t>(allgather_cnode->GetAttr(parallel::MICRO_INTERLEAVED_TAG));
  auto rank_ids = parallel::g_device_manager->FindRankListByHashName(group_name);
  auto dev_list = parallel::g_device_manager->CreateDeviceListByRankList(rank_ids);
  if (group_name.find("slice_act") != std::string::npos) {
    return;
  }
  auto new_group_name = group_name + "_slice_act_" + std::to_string(micro_interleaved_index);
  parallel::Group cur_device_list;
  (void)parallel::g_device_manager->CreateGroup(new_group_name, dev_list, &cur_device_list);
  auto allgather_prim = GetCNodePrimitive(allgather_cnode);
  (void)allgather_prim->AddAttr(parallel::GROUP, MakeValue<std::string>(new_group_name));
}

void InsertSliceAllGatherNode(const std::vector<std::pair<std::shared_ptr<AnfNode>, int>> &node_users,
                              const std::pair<std::shared_ptr<AnfNode>, int> &forward_node_user,
                              const std::shared_ptr<CNode> &node, std::vector<CNodePtr> *slice_allgathers,
                              int64_t recompute_order_id) {
  auto manager = node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto output_shape = node->abstract()->BuildShape();
  std::vector<int64_t> out_shape_element = output_shape->cast<abstract::ShapePtr>()->shape();
  if (out_shape_element.empty()) {
    return;
  }
  int64_t global_rank_id = parallel::g_device_manager->global_rank();
  int64_t stage_num = parallel::g_device_manager->stage_num();
  int64_t device_num = SizeToLong(parallel::g_device_manager->DeviceNum());
  int64_t stage_device_num = device_num / stage_num;
  int64_t local_rank_id = global_rank_id % stage_device_num;
  auto groups = InferRepeatedRankList(node);
  if (groups.empty()) {
    return;
  }
  auto group = groups[0];
  if (group.GetDevNum() == 0) {
    MS_LOG(EXCEPTION) << "The dev num of group should not be 0.";
  }
  if (out_shape_element[0] % SizeToLong(group.GetDevNum()) != 0) {
    MS_LOG(WARNING) << "The output_shape first dim:" << out_shape_element[0]
                    << " cannot be divisible by the repeated size: " << group.GetDevNum()
                    << "The slice would not activate to this node: " << node->DebugString();
    return;
  }
  int64_t group_deivce_num = SizeToLong(group.GetDevNum());
  std::vector<int64_t> slice_begin(out_shape_element.size(), 0);
  slice_begin[0] = (local_rank_id % group_deivce_num) * (out_shape_element[0] / group_deivce_num);
  std::vector<int64_t> slice_end = out_shape_element;
  slice_end[0] = (local_rank_id % group_deivce_num + 1) * (out_shape_element[0] / group_deivce_num);
  std::vector<int64_t> slice_strides(out_shape_element.size(), 1);
  CNodePtr slice_cnode = CreateStridedSliceCNode(slice_begin, slice_end, slice_strides, node);
  slice_cnode->set_abstract(node->abstract()->Clone());
  std::vector<int64_t> slice_shape = out_shape_element;
  slice_shape[0] = out_shape_element[0] / group_deivce_num;
  std::shared_ptr<abstract::BaseShape> slice_base_shape = std::make_shared<abstract::Shape>(slice_shape);
  slice_cnode->abstract()->set_shape(slice_base_shape);
  for (auto &node_user : node_users) {
    manager->SetEdge(node_user.first, node_user.second, slice_cnode);
  }

  CNodePtr allgather_cnode = CreateAllGatherCNode(slice_cnode, group.name());
  allgather_cnode->set_abstract(node->abstract()->Clone());
  allgather_cnode->AddAttr("recompute_order", MakeValue(recompute_order_id));
  if (node->HasPrimalAttr(parallel::MICRO)) {
    allgather_cnode->AddPrimalAttr(parallel::MICRO, node->GetPrimalAttr(parallel::MICRO));
  }

  if (node->HasAttr(parallel::MICRO_INTERLEAVED_TAG)) {
    allgather_cnode->AddAttr(parallel::MICRO_INTERLEAVED_TAG, node->GetAttr(parallel::MICRO_INTERLEAVED_TAG));
    static const auto micro_interleaved_extra_comm_group = (common::GetEnv("interleaved_extra_group") == "1");
    if (micro_interleaved_extra_comm_group) {
      CreateGroupForSliceAllGatherInMicroInterleaved(allgather_cnode, group.name());
    }
  }

  (void)manager->Replace(slice_cnode, allgather_cnode);
  slice_allgathers->push_back(allgather_cnode);

  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), forward_node_user.first, slice_cnode};
  auto depend_node = node->func_graph()->NewCNode(depend_inputs);
  depend_node->set_abstract(forward_node_user.first->abstract()->Clone());
  depend_node->AddAttr("slice_forward_depend", MakeValue(true));
  MS_EXCEPTION_IF_NULL(depend_node);
  (void)manager->Replace(forward_node_user.first, depend_node);
}

void InsertAllGatherDepend(const FuncGraphPtr &graph, const std::vector<CNodePtr> &slice_allgathers) {
  auto manager = graph->manager();
  auto last_allgather = slice_allgathers.back();
  for (size_t i = slice_allgathers.size() - 1; i > 0; --i) {
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), slice_allgathers[i - 1]->input(1),
                                          slice_allgathers[i]};
    auto depend_node = graph->NewCNode(depend_inputs);
    MS_EXCEPTION_IF_NULL(depend_node);
    depend_node->set_abstract(slice_allgathers[i - 1]->input(1)->abstract()->Clone());
    depend_node->AddAttr("slice_allgather_depend", MakeValue(i));
    manager->SetEdge(slice_allgathers[i - 1], 1, depend_node);
  }
  CNodePtr allgather_depend_node = nullptr;
  auto node_users = manager->node_users()[last_allgather];
  for (auto &node_user : node_users) {
    if (IsPrimitiveCNode(node_user.first) && node_user.first->cast<CNodePtr>()->HasAttr("recompute_depend") &&
        GetValue<bool>(node_user.first->cast<CNodePtr>()->GetAttr("recompute_depend"))) {
      allgather_depend_node = node_user.first->cast<CNodePtr>();
    }
  }
  if (allgather_depend_node == nullptr) {
    MS_LOG(WARNING) << "cannot find the last allgather depend node.";
    return;
  }
  MS_LOG(INFO) << "Insert depend for last slice allgather. The depend node is: "
               << allgather_depend_node->DebugString();

  allgather_depend_node->set_input(1, last_allgather->input(1));
  allgather_depend_node->set_abstract(last_allgather->input(1)->abstract()->Clone());
  allgather_depend_node->AddAttr("last_slice_allgather_depend", MakeValue(true));
  (void)manager->Replace(allgather_depend_node, last_allgather);
  manager->SetEdge(last_allgather, 1, allgather_depend_node);
}

void InsertAllGatherDependWithMicroInterleaved(const FuncGraphPtr &graph,
                                               const std::vector<CNodePtr> &slice_allgathers) {
  if (!parallel::ParallelContext::GetInstance()->enable_micro_interleaved()) {
    InsertAllGatherDepend(graph, slice_allgathers);
    return;
  }
  std::unordered_map<size_t, std::vector<CNodePtr>> slice_allgather_micro_interleaved;
  for (const auto &allgather_node : slice_allgathers) {
    if (!allgather_node->HasAttr(parallel::MICRO_INTERLEAVED_TAG)) {
      MS_LOG(WARNING) << "The slice activation nodes cannot split to multi micro interleaved part.";
      parallel::ParallelContext::GetInstance()->set_enable_micro_interleaved(false);
      InsertAllGatherDepend(graph, slice_allgathers);
      break;
    }
    auto micro_interleaved_index = GetValue<size_t>(allgather_node->GetAttr(parallel::MICRO_INTERLEAVED_TAG));
    slice_allgather_micro_interleaved[micro_interleaved_index].push_back(allgather_node);
  }
  for (const auto &micro_interleaved_allgather_pairs : slice_allgather_micro_interleaved) {
    MS_LOG(INFO) << "Insert allgather depend for micro interleaved index " << micro_interleaved_allgather_pairs.first
                 << ", allgather num is " << micro_interleaved_allgather_pairs.second.size();
    InsertAllGatherDepend(graph, micro_interleaved_allgather_pairs.second);
  }
}

void SpreadRecomputeDepend(const FuncGraphManagerPtr &manager, const std::vector<CNodePtr> &origin_nodes_topological) {
  std::vector<CNodePtr> depend_cnodes;
  for (auto &node : origin_nodes_topological) {
    if (!node->HasAttr("recompute_depend") || !GetValue<bool>(node->GetAttr("recompute_depend"))) {
      continue;
    }
    std::queue<CNodePtr> node_queue;
    node_queue.push(node);
    int64_t loop_size = 0;
    while (!node_queue.empty() && loop_size < max_loop_size) {
      auto cnode = node_queue.front();
      node_queue.pop();
      auto cnode_users = manager->node_users()[cnode];
      for (auto &cnode_user : cnode_users) {
        if (IsPrimitiveCNode(cnode_user.first, prim::kPrimDepend)) {
          depend_cnodes.push_back(cnode_user.first->cast<CNodePtr>());
        } else if (IsPrimitiveCNode(cnode_user.first, prim::kPrimUpdateState)) {
          node_queue.push(cnode_user.first->cast<CNodePtr>());
        }
        loop_size++;
      }
    }
  }
  for (auto &depend_cnode : depend_cnodes) {
    depend_cnode->cast<CNodePtr>()->AddAttr("recompute_depend", MakeValue(true));
  }
}
}  // namespace

void SliceRecomputedActivationNodes(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  SpreadRecomputeDepend(manager, origin_nodes_topological);
  std::vector<CNodePtr> slice_allgathers;
  int64_t recompute_order_id = 0;
  for (auto &node : origin_nodes_topological) {
    if (!node->HasAttr(kAttrSliceActivation) || IsPrimitiveCNode(node, prim::kPrimTranspose) ||
        !node->has_user_data<parallel::OperatorInfo>()) {
      continue;
    }
    std::vector<std::pair<std::shared_ptr<AnfNode>, int>> duplicate_users;
    std::vector<std::pair<std::shared_ptr<AnfNode>, int>> forward_users;
    GroupingNextNodes(node, &duplicate_users, &forward_users);
    if (duplicate_users.empty() || forward_users.empty()) {
      continue;
    }
    InsertSliceAllGatherNode(duplicate_users, forward_users[0], node, &slice_allgathers, recompute_order_id);
    recompute_order_id++;
  }
  if (slice_allgathers.size() == 0) {
    return;
  }
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    int64_t current_micro = -1;
    std::vector<CNodePtr> stage_slice_allgathers;
    for (auto &slice_allgather_node : slice_allgathers) {
      if (!slice_allgather_node->HasPrimalAttr(parallel::MICRO)) {
        MS_LOG(EXCEPTION) << "In pipeline parallel mode, cannot find 'micro' attributes in node.";
      }
      int64_t micro = GetValue<int64_t>(slice_allgather_node->GetPrimalAttr(parallel::MICRO));
      if (micro > current_micro) {
        if (current_micro != -1) {
          MS_LOG(INFO) << "Insert allgather depends, micro is: " << current_micro;
          InsertAllGatherDependWithMicroInterleaved(graph, stage_slice_allgathers);
        }
        stage_slice_allgathers.clear();
        stage_slice_allgathers.push_back(slice_allgather_node);
        current_micro = micro;
      } else if (micro == current_micro) {
        stage_slice_allgathers.push_back(slice_allgather_node);
      } else if (current_micro != -1) {
        MS_LOG(EXCEPTION) << "The micro number dose not match the execution orders in pipeline parallel";
      }
    }
    MS_LOG(INFO) << "Insert last stage allgather depends, micro is: " << current_micro;
    InsertAllGatherDependWithMicroInterleaved(graph, stage_slice_allgathers);
  } else {
    InsertAllGatherDependWithMicroInterleaved(graph, slice_allgathers);
  }
}
}  // namespace opt
}  // namespace mindspore
