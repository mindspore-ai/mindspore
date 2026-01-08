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
#include "plugin/ascend/res_manager/collective/multi_ascend_collective_comm_lib.h"
#include <unordered_set>
#include <vector>
#include <memory>
#include <string>

#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_hal_manager.h"
#include "kernel/ascend/dvm/kernel_mod_impl/dvm_comm_info.h"

namespace mindspore {
namespace device {
namespace ascend {
using GroupOptions = mindspore::device::GroupOptions;

std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string cur_so_path = dl_info.dli_fname;
  return dirname(cur_so_path.data());
#else
  return "";
#endif
}

MultiAscendCollectiveCommLib &MultiAscendCollectiveCommLib::GetInstance() {
  static MultiAscendCollectiveCommLib instance;
  return instance;
}

MultiAscendCollectiveCommLib::MultiAscendCollectiveCommLib() { global_group_name_ = kMACCLGlobalGroupName; }

std::unordered_set<std::string> MultiAscendCollectiveCommLib::GetLcclEnabledGroups() { return lccl_enabled_groups; }

std::unordered_set<std::string> MultiAscendCollectiveCommLib::GetDvmCommEnabledGroups() {
  return dvm_comm_enabled_groups;
}

bool MultiAscendCollectiveCommLib::isGroupWithinLocalMachine(const std::vector<uint32_t> &group_ranks) {
  std::vector<size_t> all_host_hashs = distributed::collective::CollectiveManager::instance()->GetAllHostHashs();
  if (all_host_hashs.empty()) {
    MS_LOG(WARNING) << "The all_host_hashs_ is empty. Please check whether the local rank ids are successfully "
                       "assigned when initializing collective communication";
    return false;
  }
  return std::all_of(group_ranks.begin() + 1, group_ranks.end(),
                     [&](uint32_t rank) { return all_host_hashs[rank] == all_host_hashs[group_ranks[0]]; });
}

bool MultiAscendCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  if (initialized_) {
    return true;
  }
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::AscendHalManager::GetInstance().EnableLccl()) {
    std::string lowlatency_comm_lib_name = GetCurrentDir() + "/ascend/liblowlatency_collective.so";
    auto loader = std::make_shared<CollectiveCommLibLoader>(lowlatency_comm_lib_name);
    MS_EXCEPTION_IF_NULL(loader);
    if (!loader->Initialize()) {
      MS_LOG(EXCEPTION) << "Loading LCCL collective library failed.";
      return false;
    }
    void *collective_comm_lib_handle = loader->collective_comm_lib_ptr();
    MS_EXCEPTION_IF_NULL(collective_comm_lib_handle);

    auto instance_func = DlsymFuncObj(lowlatency_communication_lib_instance, collective_comm_lib_handle);
    lowlatency_collective_comm_lib_ = instance_func();
    MS_EXCEPTION_IF_NULL(lowlatency_collective_comm_lib_);
    MS_LOG(WARNING) << "Loading LCCL because env MS_ENABLE_LCCL is set to on. Pay attention that LCCL only supports "
                       "communication group within single node in KernelByKernel for now.";
    RETURN_IF_FALSE_WITH_LOG(lowlatency_collective_comm_lib_->Initialize(global_rank, global_rank_size, local_rank_id),
                             "Failed to initialize LCCL.");
    MS_LOG(INFO) << "Successfully initialize LCCL.";
  }
#endif
  if (graphkernel::EnableDvmComm()) {
    dvm_collective_comm_lib_ = &DvmCollectiveCommLib::GetInstance();
    MS_EXCEPTION_IF_NULL(dvm_collective_comm_lib_);
    dvm_collective_comm_lib_->Initialize(global_rank, global_rank_size, local_rank_id);
  }
  ascend_collective_comm_lib_ = &AscendCollectiveCommLib::GetInstance();
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  RETURN_IF_FALSE_WITH_LOG(ascend_collective_comm_lib_->Initialize(global_rank, global_rank_size, local_rank_id),
                           "Failed to initialize HCCL.");
  MS_LOG(INFO) << "Successfully initialize HCCL.";
  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool MultiAscendCollectiveCommLib::Finalize() {
  if (!initialized_ || finalized_.load()) {
    return true;
  }

  auto finalize_collective_comm_lib = [](std::string_view comm_lib_name,
                                         CollectiveCommunicationLib *comm_lib_instance) {
    if (comm_lib_instance == nullptr) {
      MS_LOG(WARNING) << "The instance of " << comm_lib_name << " is nullptr!";
      return;
    }
    if (!comm_lib_instance->Finalize()) {
      MS_LOG(WARNING) << "Failed to finalize " << comm_lib_name << ".";
    }
  };

#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::AscendHalManager::GetInstance().EnableLccl()) {
    finalize_collective_comm_lib("LCCL", lowlatency_collective_comm_lib_);
  }
#endif

  if (dvm_collective_comm_lib_ != nullptr) {
    finalize_collective_comm_lib("DVMCCL", dvm_collective_comm_lib_);
  }

  finalize_collective_comm_lib("HCCL", ascend_collective_comm_lib_);

  for (const auto &group : groups_) {
    CHECK_IF_NULL(group.second);
    (void)group.second->Finalize();
  }

  groups_.clear();
  initialized_ = false;
  finalized_ = true;
  return true;
}

bool MultiAscendCollectiveCommLib::DestroyDeviceCommunicationGroup(const std::string &group_name) {
  RETURN_IF_FALSE(ascend_collective_comm_lib_->DestroyDeviceCommunicationGroup(group_name));
  return true;
}

bool MultiAscendCollectiveCommLib::DestroyCommunicationGroup(const std::string &group_name) {
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::AscendHalManager::GetInstance().EnableLccl() &&
      lccl_enabled_groups.find(group_name) != lccl_enabled_groups.end()) {
    RETURN_IF_FALSE_WITH_LOG(lowlatency_collective_comm_lib_->DestroyCommunicationGroup(group_name),
                             "Failed to destroy LCCL communication group " + group_name);
    lccl_enabled_groups.erase(group_name);
    MS_LOG(INFO) << "Successfully destroy LCCL communication group " << group_name;
  }
#endif
  if (dvm_collective_comm_lib_ != nullptr &&
      dvm_comm_enabled_groups.find(group_name) != dvm_comm_enabled_groups.end()) {
    RETURN_IF_FALSE_WITH_LOG(dvm_collective_comm_lib_->DestroyCommunicationGroup(group_name),
                             "Failed to destroy DVM communication group " + group_name);
    dvm_comm_enabled_groups.erase(group_name);
  }
  RETURN_IF_FALSE_WITH_LOG(ascend_collective_comm_lib_->DestroyCommunicationGroup(group_name),
                           "Failed to destroy HCCL communication group " + group_name);
  MS_LOG(INFO) << "Successfully destroy HCCL communication group " << group_name;
  (void)groups_.erase(group_name);
  return true;
}

bool MultiAscendCollectiveCommLib::CommSwitchNic(const std::vector<uint32_t> &global_ranks,
                                                 const std::vector<bool> &use_backup) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  RETURN_IF_FALSE_WITH_LOG(ascend_collective_comm_lib_->CommSwitchNic(global_ranks, use_backup),
                           "Failed to switch network interface card");
  return true;
}

bool MultiAscendCollectiveCommLib::CreateDeviceCommunicationGroup(const std::string &group_name,
                                                                  const std::vector<uint32_t> &group_ranks) {
  RETURN_IF_FALSE(ascend_collective_comm_lib_->CreateDeviceCommunicationGroup(group_name, group_ranks));
  return true;
}

bool MultiAscendCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                            const std::vector<uint32_t> &group_ranks,
                                                            uint32_t local_group_rank, uint32_t local_group_size,
                                                            const GroupOptions &config) {
  if (groups_.count(group_name) != 0) {
    MS_LOG(WARNING) << "the group:" << group_name << "already existed";
    return true;
  }
  MultiAscendCommunicationGroupPtr group = std::make_shared<MultiAscendCommunicationGroup>(
    group_name, group_ranks, global_rank_id_, local_group_rank, local_group_size);
  MS_EXCEPTION_IF_NULL(group);
#ifdef ENABLE_INTERNAL_KERNELS
  if (device::ascend::AscendHalManager::GetInstance().EnableLccl() && isGroupWithinLocalMachine(group_ranks)) {
    RETURN_IF_FALSE_WITH_LOG(lowlatency_collective_comm_lib_->CreateCommunicationGroup(
                               group_name, group_ranks, local_group_rank, local_group_size),
                             "Failed to create LCCL communication group" + group_name);
    CommunicationGroupPtr lccl_group = lowlatency_collective_comm_lib_->GetGroup(group_name);
    MS_EXCEPTION_IF_NULL(lccl_group);
    group->SetLcclGroup(lccl_group);
    lccl_enabled_groups.insert(group_name);
    MS_LOG(INFO) << "Successfully create LCCL communication group " << group_name;
  }
#endif
  if (dvm_collective_comm_lib_ != nullptr) {
    RETURN_IF_FALSE_WITH_LOG(
      dvm_collective_comm_lib_->CreateCommunicationGroup(group_name, group_ranks, local_group_rank, local_group_size),
      "Failed to create DVM communication group" + group_name);
    CommunicationGroupPtr dvm_group = dvm_collective_comm_lib_->GetGroup(group_name);
    MS_EXCEPTION_IF_NULL(dvm_group);
    group->SetDvmCommGroup(dvm_group);
    dvm_comm_enabled_groups.insert(group_name);
  }
  RETURN_IF_FALSE_WITH_LOG(ascend_collective_comm_lib_->CreateCommunicationGroup(
                             group_name, group_ranks, local_group_rank, local_group_size, config),
                           "Failed to create HCCL communication group" + group_name);
  CommunicationGroupPtr hccl_group = ascend_collective_comm_lib_->GetGroup(group_name);
  MS_EXCEPTION_IF_NULL(hccl_group);
  group->SetHcclGroup(hccl_group);
  MS_LOG(INFO) << "Successfully create HCCL communication group " << group_name;

  groups_[group_name] = group;

  return true;
}

std::string MultiAscendCollectiveCommLib::CommName(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->CommName(group_name);
}

uint32_t MultiAscendCollectiveCommLib::GetRankId(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetRankId(group_name);
}

uint32_t MultiAscendCollectiveCommLib::GetGroupSize(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetGroupSize(group_name);
}

uint32_t MultiAscendCollectiveCommLib::GetLocalRankId(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetLocalRankId(group_name);
}

uint32_t MultiAscendCollectiveCommLib::GetLocalGroupSize(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetLocalGroupSize(group_name);
}

uint32_t MultiAscendCollectiveCommLib::GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetWorldRankFromGroupRank(group_name, local_rank);
}

uint32_t MultiAscendCollectiveCommLib::GetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(ascend_collective_comm_lib_);
  return ascend_collective_comm_lib_->GetGroupRankFromWorldRank(world_rank, group_name);
}

bool MultiAscendCollectiveCommLib::ResumeHcclComm() { return ascend_collective_comm_lib_->ResumeHcclComm(); }

bool MultiAscendCollectiveCommLib::AllGather(const void *send_buff, void *recv_buff, size_t send_count,
                                             TypeId data_type, const std::string &group_name, void *stream) {
  return ascend_collective_comm_lib_->AllGather(send_buff, recv_buff, send_count, data_type, group_name, stream);
}

bool MultiAscendCollectiveCommLib::AllReduce(const void *send_buff, void *recv_buff, size_t send_count,
                                             TypeId data_type, CollectiveOpReduceType reduce_op,
                                             const std::string &group_name, void *stream) {
  return ascend_collective_comm_lib_->AllReduce(send_buff, recv_buff, send_count, data_type, reduce_op, group_name,
                                                stream);
}

bool MultiAscendCollectiveCommLib::Broadcast(const void *send_buff, void *recv_buff, size_t send_count,
                                             TypeId data_type, uint32_t root_rank, const std::string &group_name,
                                             void *stream) {
  return ascend_collective_comm_lib_->Broadcast(send_buff, recv_buff, send_count, data_type, root_rank, group_name,
                                                stream);
}

bool MultiAscendCollectiveCommLib::ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count,
                                                 TypeId data_type, CollectiveOpReduceType reduce_op,
                                                 const std::string &group_name, void *stream) {
  return ascend_collective_comm_lib_->ReduceScatter(send_buff, recv_buff, recv_count, data_type, reduce_op, group_name,
                                                    stream);
}

bool MultiAscendCollectiveCommLib::Send(const void *send_buff, size_t count, TypeId data_type, uint32_t peer,
                                        const std::string &group_name, void *stream) {
  return ascend_collective_comm_lib_->Send(send_buff, count, data_type, peer, group_name, stream);
}

bool MultiAscendCollectiveCommLib::Recv(void *recv_buff, size_t count, TypeId data_type, uint32_t peer,
                                        const std::string &group_name, void *stream) {
  return ascend_collective_comm_lib_->Recv(recv_buff, count, data_type, peer, group_name, stream);
}
}  // namespace ascend

using MultiAscendCollectiveCommLib = mindspore::device::ascend::MultiAscendCollectiveCommLib;

CollectiveCommunicationLib *multi_ascend_communication_lib_instance() {
  return &MultiAscendCollectiveCommLib::GetInstance();
}

}  // namespace device
}  // namespace mindspore
