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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HYPER_OFFLOAD_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HYPER_OFFLOAD_UTILS_H_

#include <utility>
#include <vector>
#include <memory>
#include "utils/compile_config.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace hyper_offload {
struct UserInfo {
  KernelWithIndex node;
  std::vector<KernelWithIndex> users;
};
using UserInfoList = std::vector<UserInfo>;

class HyperOffloadConfig {
 public:
  static HyperOffloadConfig &GetInstance() {
    static HyperOffloadConfig instance;
    return instance;
  }

  HyperOffloadConfig(const HyperOffloadConfig &) = delete;
  void operator=(const HyperOffloadConfig &) = delete;

  bool enable_hyper_offload() const { return enable_hyper_offload_; }
  int select_distance() const { return select_distance_; }
  int select_num() const { return select_num_; }
  int prefetch_distance() const { return prefetch_distance_; }
  int release_distance() const { return release_distance_; }

 private:
  HyperOffloadConfig() {
    enable_hyper_offload_ = (common::GetEnv("MS_DEV_HYPER_OFFLOAD") == "1");
    if (enable_hyper_offload_) {
      select_distance_ = std::stoi(common::GetCompileConfig("HYPER_OFFLOAD_SELECT_DISTANCE"));
      select_num_ = std::stoi(common::GetCompileConfig("HYPER_OFFLOAD_SELECT_NUM"));
      prefetch_distance_ = std::stoi(common::GetCompileConfig("HYPER_OFFLOAD_PREFETCH_DISTANCE"));
      release_distance_ = std::stoi(common::GetCompileConfig("HYPER_OFFLOAD_RELEASE_DISTANCE"));
    }
  }

  bool enable_hyper_offload_ = false;
  int select_distance_ = 0;
  int select_num_ = 0;
  int prefetch_distance_ = 0;
  int release_distance_ = 0;
};

bool IsD2HNode(const AnfNodePtr &node);
bool IsH2DNode(const AnfNodePtr &node);
bool IsViewOrInplaceNode(const AnfNodePtr &node);
CNodePtr BuildToHostNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &data_node);
CNodePtr BuildInplaceToHostNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &data_node,
                                const AnfNodePtr &update_node, const AnfNodePtr &depend_node);
CNodePtr BuildToDeviceNode(const KernelGraphPtr &kernel_graph, const AnfNodePtr &data_node);
bool IsValidOffloadNode(const UserInfo &user_info);

UserInfoList CollectAllNodeUsers(const CNodePtrList &nodes);
std::optional<ParameterPtr> FindParameterByRefName(const KernelGraphPtr &kernel_graph, const std::string &name);
std::optional<ParameterPtr> GetRemoteParameter(const KernelGraphPtr &kernel_graph, const mindspore::AnfNodePtr &node);
}  // namespace hyper_offload
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HYPER_OFFLOAD_UTILS_H_
