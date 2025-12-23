/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/kernel_executor/hyper_offload/hyper_offload_strategy.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace hyper_offload {
namespace {
struct AbsoluteDistance {
  KernelWithIndex node;
  size_t absolute_distance;
};
using AbsoluteDistanceList = std::vector<AbsoluteDistance>;

struct RelativeDistance {
  KernelWithIndex start_node;
  KernelWithIndex end_node;
  size_t distance;
};
using RelativeDistanceList = std::vector<RelativeDistance>;

AbsoluteDistanceList GenerateAbsoluteDistanceList(const UserInfo &user_info, const CNodePtrList &execution_order) {
  auto node = user_info.node.first;
  auto node_iter = std::find(execution_order.begin(), execution_order.end(), node);
  MS_EXCEPTION_IF_CHECK_FAIL(node_iter != execution_order.end(), "Failed to find node in execution order.");
  auto node_distance = std::distance(execution_order.begin(), node_iter);
  AbsoluteDistanceList absolute_distance_list{AbsoluteDistance{user_info.node, node_distance}};
  const auto &node_users = user_info.users;
  for (const auto &node_user : node_users) {
    auto user_node = node_user.first;
    auto node_user_iter = std::find(execution_order.begin(), execution_order.end(), user_node);
    if (node_user_iter == execution_order.end()) {
      MS_LOG(EXCEPTION) << "Failed to find in executor for user node: " << user_node->DebugString() << ", node is "
                        << node->DebugString();
    }
    size_t absolute_distance = std::distance(execution_order.begin(), node_user_iter);
    (void)absolute_distance_list.emplace_back(AbsoluteDistance{node_user, absolute_distance});
  }
  std::sort(absolute_distance_list.begin(), absolute_distance_list.end(),
            [](const auto &e1, const auto &e2) { return e1.absolute_distance < e2.absolute_distance; });
  return absolute_distance_list;
}

RelativeDistanceList GenerateRelativeDistanceList(const AbsoluteDistanceList &absolute_distance_list) {
  RelativeDistanceList relative_distance_list;
  for (size_t i = 1; i < absolute_distance_list.size(); ++i) {
    int current_distance =
      absolute_distance_list[i].absolute_distance - absolute_distance_list[i - 1].absolute_distance;
    (void)relative_distance_list.emplace_back(RelativeDistance{absolute_distance_list[i - 1].node,
                                                               absolute_distance_list[i].node,
                                                               IntToSize(current_distance)});
  }
  return relative_distance_list;
}

ReplaceInfoList GenerateReplaceInfoList(const RelativeDistanceList &relative_distance_list) {
  static int select_distance = HyperOffloadConfig::GetInstance().select_distance();
  std::vector<bool> need_offload;
  for (const auto &cur_distance_element : relative_distance_list) {
    (void)need_offload.emplace_back(((int)cur_distance_element.distance > select_distance));
  }
  ReplaceInfoList replace_info_list;
  if (!std::any_of(need_offload.begin(), need_offload.end(), [](bool offload) { return offload; })) {
    return replace_info_list;
  }
  // With input:
  // relative_distance_list: (((node1, node2), distance1), ((node2, node3), distance2), ((node3, node4), distance3))
  // need_offload: (False, True, False)
  // generate replace distance info:
  // ((node2, (node3, node4)), )
  for (size_t i = 0; i < relative_distance_list.size(); ++i) {
    const auto &new_need_change_node = relative_distance_list[i].end_node;
    if (need_offload[i]) {
      const auto &upper_bound = relative_distance_list[i].start_node;
      std::vector<KernelWithIndex> need_change_nodes{new_need_change_node};
      replace_info_list.emplace_back(ReplaceInfo{upper_bound, need_change_nodes});
      continue;
    }
    if (replace_info_list.empty()) {
      continue;
    }
    (void)replace_info_list.back().replace_rest_nodes.emplace_back(new_need_change_node);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(!replace_info_list.empty(), "Replace info list should not be empty");
  return replace_info_list;
}

std::optional<OffloadInfo> GenerateOffloadInfo(const UserInfo &user_info, const CNodePtrList &execution_order) {
  if (!IsValidOffloadNode(user_info)) {
    return std::nullopt;
  }
  const auto &absolute_distance_list = GenerateAbsoluteDistanceList(user_info, execution_order);
  if (absolute_distance_list.size() == 1) {
    return std::nullopt;
  }
  const auto &relative_distance_list = GenerateRelativeDistanceList(absolute_distance_list);
  const auto &replace_info_list = GenerateReplaceInfoList(relative_distance_list);
  if (replace_info_list.empty()) {
    return std::nullopt;
  }
  return OffloadInfo{user_info.node, replace_info_list};
}

OffloadInfoList GenerateOffloadInfoList(const KernelGraphPtr &kernel_graph, const UserInfoList &user_info_list) {
  const auto &execution_order = kernel_graph->execution_order();
  OffloadInfoList offload_info_list;
  for (const auto &cur_user_info : user_info_list) {
    const auto &offload_info_opt = GenerateOffloadInfo(cur_user_info, execution_order);
    if (!offload_info_opt.has_value()) {
      continue;
    }
    const auto &offload_info = offload_info_opt.value();
    (void)offload_info_list.emplace_back(offload_info);
  }
  return offload_info_list;
}
}  // namespace

OffloadInfoList DistanceBaseHyperOffloadStrategy::Run(const KernelGraphPtr &kernel_graph) {
  const auto &user_info_list = CollectAllNodeUsers(kernel_graph->execution_order());
  const auto &offload_info_list = GenerateOffloadInfoList(kernel_graph, user_info_list);
  MS_LOG(INFO) << "offload_info_list size: " << offload_info_list.size();
  return offload_info_list;
}

OffloadInfoList OffloadInfoFilterByNumber::Filter(const OffloadInfoList& offload_info_list) {
  static int offload_total_number = HyperOffloadConfig::GetInstance().select_num();
  OffloadInfoList filtered_offload_list{offload_info_list.begin(), offload_info_list.begin() + offload_total_number};
  return filtered_offload_list;
}
}  // namespace hyper_offload
}  // namespace ascend
}  // namespace device
}  // namespace mindspore