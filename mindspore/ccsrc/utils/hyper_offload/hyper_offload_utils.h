/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_UTILS_H_
#define MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_UTILS_H_

#include <vector>
#include <optional>

#include "ir/anf.h"
#include "include/utils/anfalgo.h"
#include "include/utils/visible.h"
#include "utils/hyper_offload/hyper_offload_operation.h"

namespace mindspore {
namespace utils {
namespace hyper_offload {
using KernelWithIndex = common::KernelWithIndex;
using HandleNewNodeCallback = std::function<void(const AnfNodePtr &new_node, const AnfNodePtr &context_node)>;

struct UserInfo {
  KernelWithIndex node;
  std::vector<KernelWithIndex> users;
};
using UserInfoList = std::vector<UserInfo>;

struct ReplaceInfo {
  KernelWithIndex replace_start_node;
  std::vector<KernelWithIndex> replace_rest_nodes;
};
using ReplaceInfoList = std::vector<ReplaceInfo>;

struct OffloadInfo {
  KernelWithIndex data_node;
  ReplaceInfoList replace_info_list;
};
using OffloadInfoList = std::vector<OffloadInfo>;

class COMMON_EXPORT HyperOffloadConfig {
 public:
  // Get the singleton instance of HyperOffloadConfig.
  static HyperOffloadConfig &GetInstance();

  HyperOffloadConfig(const HyperOffloadConfig &) = delete;
  void operator=(const HyperOffloadConfig &) = delete;

  bool enable_hyper_offload() const { return enable_hyper_offload_; }
  size_t select_distance() const { return select_distance_; }
  size_t select_num() const { return select_num_; }
  size_t prefetch_distance() const { return prefetch_distance_; }
  size_t release_distance() const { return release_distance_; }

 private:
  HyperOffloadConfig();

  bool enable_hyper_offload_{false};
  size_t select_distance_{0};
  size_t select_num_{0};
  size_t prefetch_distance_{0};
  size_t release_distance_{0};
};

// Check if a node is a D2H (Device to Host) node.
COMMON_EXPORT bool IsD2HNode(const AnfNodePtr &node);
// Check if a node is an H2D (Host to Device) node.
COMMON_EXPORT bool IsH2DNode(const AnfNodePtr &node);
// Check if a node is a View or Inplace node.
COMMON_EXPORT bool IsViewOrInplaceNode(const AnfNodePtr &node);
// Check if a user info indicates a valid offload candidate.
COMMON_EXPORT bool IsValidOffloadNode(const UserInfo &user_info);
// Collect all users for a list of nodes.
COMMON_EXPORT UserInfoList CollectAllNodeUsers(const CNodePtrList &nodes);
// Find the last real user of a node in the execution order.
COMMON_EXPORT std::optional<AnfNodePtr> FindLastRealUser(const AnfNodePtr &node, const UserInfoList &user_info_list);
// Build a ToHost (D2H) node.
COMMON_EXPORT CNodePtr BuildToHostNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                                       const HandleNewNodeCallback &callback = nullptr);
// Build an Inplace ToHost node.
COMMON_EXPORT CNodePtr BuildInplaceToHostNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                                              const AnfNodePtr &update_node, const AnfNodePtr &depend_node,
                                              const HandleNewNodeCallback &callback = nullptr);
// Build a ToDevice (H2D) node.
COMMON_EXPORT CNodePtr BuildToDeviceNode(const FuncGraphPtr &graph, const AnfNodePtr &data_node,
                                         const HandleNewNodeCallback &callback = nullptr);
// Find a parameter by its reference name.
COMMON_EXPORT std::optional<ParameterPtr> FindParameterByRefName(const FuncGraphPtr &graph, const std::string &name);
// Get the remote parameter associated with a node.
COMMON_EXPORT std::optional<ParameterPtr> GetRemoteParameter(const FuncGraphPtr &graph, const AnfNodePtr &node);
// Update the input of a node at a specific index.
COMMON_EXPORT void UpdateNodeInput(const CNodePtr &node, const size_t input_index, const AnfNodePtr &new_input_node);
}  // namespace hyper_offload
}  // namespace utils
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_HYPER_OFFLOAD_HYPER_OFFLOAD_UTILS_H_
