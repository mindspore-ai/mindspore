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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MULTI_ASCEND_COLLECTIVE_COMM_LIB_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MULTI_ASCEND_COLLECTIVE_COMM_LIB_H_

#include <map>
#include <memory>
#include <vector>
#include <string>
#include <unordered_set>
#ifndef _WIN32
#include <dlfcn.h>
#include <libgen.h>
#endif
#include "include/runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "include/runtime/hardware_abstract/collective/collective_comm_lib_loader.h"
#include "include/runtime/hardware_abstract/collective/communication_group.h"
#include "plugin/ascend/res_manager/collective/multi_ascend_communication_group.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "plugin/ascend/res_manager/collective/dvm_collective_comm_lib.h"

#ifndef EXPORT_WRAPPER
#define EXPORT_WRAPPER __attribute__((visibility("default")))
#endif

namespace mindspore {
namespace device {
namespace ascend {
using GroupOptions = mindspore::device::GroupOptions;
constexpr char kMACCLGlobalGroupName[] = "hccl_world_group";

class EXPORT_WRAPPER MultiAscendCollectiveCommLib : public CollectiveCommunicationLib {
 public:
  static MultiAscendCollectiveCommLib &GetInstance();

  bool isGroupWithinLocalMachine(const std::vector<uint32_t> &group_ranks);

  bool Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) override;

  bool Finalize() override;

  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                uint32_t local_group_rank, uint32_t local_group_size,
                                const GroupOptions &config = {}) override;

  bool CreateDeviceCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) override;

  bool DestroyCommunicationGroup(const std::string &group_name) override;

  bool DestroyDeviceCommunicationGroup(const std::string &group_name) override;

  uint32_t GetRankId(const std::string &group_name) override;

  uint32_t GetGroupSize(const std::string &group_name) override;

  uint32_t GetLocalRankId(const std::string &group_name) override;

  uint32_t GetLocalGroupSize(const std::string &group_name) override;

  uint32_t GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) override;

  uint32_t GetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group_name) override;

  std::unordered_set<std::string> GetLcclEnabledGroups();

  std::unordered_set<std::string> GetDvmCommEnabledGroups();

  std::string CommName(const std::string &group_name) override;

  bool ResumeHcclComm() override;

  bool CommSwitchNic(const std::vector<uint32_t> &global_ranks, const std::vector<bool> &use_backup) override;

  bool AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                 const std::string &group_name, void *stream = nullptr) override;

  bool AllReduce(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                 CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream = nullptr) override;

  bool Broadcast(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type, uint32_t root_rank,
                 const std::string &group_name, void *stream = nullptr) override;

  bool ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, TypeId data_type,
                     CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream = nullptr) override;

  bool Send(const void *send_buff, size_t count, TypeId data_type, uint32_t peer, const std::string &group_name,
            void *stream = nullptr) override;

  bool Recv(void *recv_buff, size_t count, TypeId data_type, uint32_t peer, const std::string &group_name,
            void *stream = nullptr) override;

 private:
  CollectiveCommunicationLib *ascend_collective_comm_lib_;
  CollectiveCommunicationLib *lowlatency_collective_comm_lib_;
  CollectiveCommunicationLib *dvm_collective_comm_lib_{nullptr};
  std::unordered_set<std::string> lccl_enabled_groups;
  std::unordered_set<std::string> dvm_comm_enabled_groups;
  MultiAscendCollectiveCommLib();
  ~MultiAscendCollectiveCommLib() override = default;
};
}  // namespace ascend

}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_MULTI_ASCEND_COLLECTIVE_COMM_LIB_H_
