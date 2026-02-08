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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_CCOOL_COMMUNICATION_GROUP_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_CCOOL_COMMUNICATION_GROUP_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include <nlohmann/json.hpp>
#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "include/runtime/hardware_abstract/collective/communication_group.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/collective/ascend_communication_group.h"
#include "plugin/ascend/res_manager/collective/leaper_trans.h"

namespace mindspore {
namespace device {
namespace ascend {

constexpr char kCcoolDefaultAzId[] = "ccool_default_az_id";

class CcoolCommunicationGroup : public CommunicationGroup {
 public:
  explicit CcoolCommunicationGroup(const std::string &name, const std::vector<uint32_t> &group_ranks,
                                   uint32_t global_rank, uint32_t local_group_rank, uint32_t local_group_size);

  ~CcoolCommunicationGroup() override = default;

  bool Initialize(void *root_info) override;
  bool Finalize() override;

  void *GenerateRootInfo(size_t *root_info_size) override;

  bool InitAscendCommGroup(const std::vector<std::string> &rank_az_map, const std::vector<std::string> &rank_ip_map);

  const std::vector<uint32_t> &GetInterClusterRanks() const;

  const std::vector<uint32_t> &GetInnerClusterRanks() const;

  LeaperConnInfo &GetConnInfo(uint32_t dst_rank);

  void SetHostCommLib(CollectiveCommunicationLib *comm_lib);

 private:
  // The correspondence between rank and az
  std::vector<std::string> rank_az_map_;
  std::map<std::string, std::vector<uint32_t>> group_az_rank_map_;
  std::string az_id_;
  std::vector<uint32_t> inner_cluster_ranks_, inter_cluster_ranks_;

  // HCCL communication domain information
  uint32_t hccl_rank_id_{0};
  CommunicationGroupPtr hccl_group_;
  void *hccl_root_info_{nullptr};
  size_t hccl_root_info_size_{0};

  // rank connection information
  std::map<uint32_t, LeaperConnInfo> rank_conn_info_map_;
  std::vector<std::string> rank_ip_map_;

  // host comm lib support
  std::string host_group_name_;
  CollectiveCommunicationLib *host_comm_lib_instance_{nullptr};
};
using CcoolCommunicationGroupPtr = std::shared_ptr<CcoolCommunicationGroup>;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_CCOOL_COMMUNICATION_GROUP_H_
