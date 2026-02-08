/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/collective/ascend_communication_group.h"
#include "include/utils/callback.h"
#if !defined(_WIN32) && !defined(_WIN64)
#include <arpa/inet.h>
#endif
#include <map>
#include <vector>
#include <variant>
#include <unordered_map>
#include <algorithm>
#include <string>
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "utils/ms_context.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "include/cluster/topology/cluster_context.h"
#include "plugin/ascend/res_manager/hal_manager/ascend_err_manager.h"
#include "plugin/ascend/res_manager/collective/utils.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr uint32_t kHcclCommConfigNslbDp = 6;
AscendCommunicationGroup::AscendCommunicationGroup(
  const std::string &name, const std::vector<uint32_t> &group_ranks, uint32_t global_rank, uint32_t local_group_rank,
  uint32_t local_group_size,
  const std::unordered_map<std::string, std::variant<int64_t, uint32_t, std::string>> &hccl_config)
    : CommunicationGroup(name, group_ranks, global_rank, local_group_rank, local_group_size),
      unique_id_({}),
      comm_(nullptr),
      hccl_config_(hccl_config) {
  auto ret = memset_s(inner_comm_name_, INNER_COMM_NAME_MAX_LENGTH, 0x00, INNER_COMM_NAME_MAX_LENGTH);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memset_s errorno: " << ret;
  }
}

bool isGroupWithinLocalMachine(const std::vector<uint32_t> &group_ranks) {
  std::vector<size_t> all_host_hashs = distributed::collective::CollectiveManager::instance()->GetAllHostHashs();
  if (all_host_hashs.empty()) {
    MS_LOG(WARNING) << "The all_host_hashs_ is empty. Please check whether the local rank ids are successfully "
                       "assigned when initializing collective communication";
    return false;
  }
  return std::all_of(group_ranks.begin() + 1, group_ranks.end(),
                     [&](uint32_t rank) { return all_host_hashs[rank] == all_host_hashs[group_ranks[0]]; });
}

bool AscendCommunicationGroup::Initialize(void *root_info) {
  if (initialized_) {
    return true;
  }

  if (MsContext::GetInstance()->ascend_soc_version() == kAscendVersion310p &&
      !isGroupWithinLocalMachine(group_ranks_)) {
    MS_LOG(WARNING) << "The device is infer card and this group " << name_
                    << " is inter-node, do not initialize hccl group for this rank list: " << group_ranks_;
    return true;
  }

  if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    // If using hccl CM envs to launch distributed job, no need to call HcclCommInitRootInfo. The group will be
    // initialized in rank table way.
    initialized_ = true;
    return true;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  (void)CALL_ASCEND_API(aclrtSetDevice, device_id);
  uint32_t group_rank;
  auto group_size = size_;
  if (!common::GetEnv(kSimulationLevel).empty()) {
    group_size = 1;
    group_rank = 0;
  } else {
    group_rank = GetGroupRank(global_rank_);
  }
  // Generate HcclCommConfig.
  HcclCommConfig config = CreateHcclCommConfig();
  // Call HCCL initialize comm API.
  bool ret = false;
  std::string rank_table_file_path = common::GetEnv("RANK_TABLE_FILE");
  if (hccl_config_.find("hccl_comm") != hccl_config_.end()) {
    ret = InitByHcclComm();
  } else if (!rank_table_file_path.empty() && common::GetEnv(kSimulationLevel).empty() &&
             !distributed::cluster::ClusterContext::instance()->enable_cross_cluster()) {
    ret = InitByRankTable(rank_table_file_path, group_size, group_rank, &config);
  } else {
    ret = InitByRootInfoConfig(root_info, group_size, group_rank, config);
  }
  if (!ret) {
    return false;
  }
  // Get HCCL comm name which is used in graph sink mode for GE.
  if (HcclGetCommName(comm_, inner_comm_name_) != static_cast<int32_t>(HCCL_SUCCESS)) {
    const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
    MS_LOG(ERROR) << "HcclGetCommName failed. " + error_message;
    return false;
  }
  initialized_ = true;

  // Initialize watch dog for every group.
  static auto reg_hccl_watchdog_cb = GET_COMMON_CALLBACK(RegisterHcclWatchDog, void, uint32_t, uint32_t,
                                                         const std::string &, HcclComm, const std::vector<uint32_t> &);
  if (reg_hccl_watchdog_cb != nullptr) {
    reg_hccl_watchdog_cb(global_rank_, device_id, name_, comm_, group_ranks_);
  }
  // If communication is obtained directly via the external hcom (InitByHcclComm), which does not depend on the
  // synchronization of HCCL, the uniqueID will not be cleared.
  if (group_rank == 0 && hccl_config_.find("hccl_comm") == hccl_config_.end()) {
    distributed::collective::CollectiveManager::instance()->ClearUniqueID(name_);
  }
  (void)CALL_ASCEND_API(aclrtResetDevice, device_id);
  return true;
}

bool AscendCommunicationGroup::Finalize() {
  if (!initialized_) {
    return true;
  }
  if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    // If using hccl CM envs to launch distributed job, comm_ is not initialized. So directly return.
    initialized_ = false;
    return true;
  }

  // This function will be called at a lonesome thread that has no rtContext, so HcclCommDestroy will be failed.
  // Delete these codes when these threads can be bind.
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  (void)CALL_ASCEND_API(aclrtSetDevice, device_id);
  static auto destroy_hccl_watchdog_handler_cb = GET_COMMON_CALLBACK(DestroyWatchDogHandler, void, const std::string &);
  if (destroy_hccl_watchdog_handler_cb != nullptr) {
    destroy_hccl_watchdog_handler_cb(name_);
  }
  (void)HcclCommDestroy(comm_);
  (void)CALL_ASCEND_API(aclrtResetDevice, device_id);
  initialized_ = false;
  comm_ = nullptr;
  return true;
}

void AscendCommunicationGroup::InitHcclCommConfig(HcclCommConfig *config) {
  MS_LOG(INFO) << "Start to initialize communication config by HcclCommConfigInit for " << name_;
  MS_EXCEPTION_IF_NULL(config);
  HcclCommConfigInit(config);
  uint32_t buffsize = GetHcclBufferSize(name_, group_ranks_);
  config->hcclBufferSize = buffsize == 0 ? HCCL_COMM_DEFAULT_BUFFSIZE : buffsize;

  // The hcclDeterministic configured for the communicator is preferentially based on the parameter in the context,
  // or if this parameter is not configured, hcclDeterministic is configured based on the environment variable
  // 'HCCL_DETERMINISTIC'.
  std::string env_hccl_deterministic = common::GetEnv("HCCL_DETERMINISTIC");
  config->hcclDeterministic = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DETERMINISTIC) == "ON"
                                ? 1
                                : (env_hccl_deterministic == "true" ? 1 : 0);
  MS_LOG(INFO) << "End to initialize communication config by HcclCommConfigInit for " << name_;
}

void AscendCommunicationGroup::SetNslbCommConfig(HcclCommConfig *config) {
#if !defined(_WIN32) && !defined(_WIN64)
  MS_EXCEPTION_IF_NULL(config);
  if (!IsSupportConfigParameter(kHcclCommConfigNslbDp)) {
    MS_LOG(DEBUG) << "Unsupported HCCL config version, NSLB-DP feature disabled!";
    return;
  }

  if (common::GetEnv("MS_SCHED_PORT").empty()) {
    MS_LOG(DEBUG) << "MS_SCHED_PORT unconfigured, NSLB-DP feature disabled!";
    return;
  }
  uint32_t master_port = static_cast<uint32_t>(std::stoi(common::GetEnv("MS_SCHED_PORT")));

  std::string master_addr = common::GetEnv("MS_SCHED_HOST");
  if (master_addr.empty()) {
    MS_LOG(DEBUG) << "MS_SCHED_HOST unconfigured, NSLB-DP feature disabled!";
    return;
  }
  struct sockaddr_in sa;
  if (inet_pton(AF_INET, master_addr.c_str(), &sa.sin_addr) != 1) {
    MS_LOG(DEBUG) << "Invalid IPv4 address: " << master_addr << ", NSLB-DP feature disabled!";
    return;
  }
  uint32_t master_ip = ntohl(sa.sin_addr.s_addr);

  // The hcclJobID(uint64_t) is composed of port(uint32_t) in its high-order 32 bits and ip(uint32_t) in its low-order
  // 32 bits.
  uint64_t job_id = (static_cast<uint64_t>(master_port) << 32) | master_ip;
  config->hcclJobID = job_id;
  config->hcclWorldRankID = global_rank_;
  MS_LOG(INFO) << "Successfully set NSLB-DP comm config for group: " << name_ << ", hcclJobID: " << job_id
               << ", hcclWorldRankID: " << global_rank_;
#endif
}

HcclCommConfig AscendCommunicationGroup::CreateHcclCommConfig() {
  HcclCommConfig config;
  InitHcclCommConfig(&config);
  // Set up NSLB-DP config.
  SetNslbCommConfig(&config);

  if (hccl_config_.empty()) {
    return config;
  }

  // Parameters passed to HCCL via API's config have higher priority than those set in the environment variableThe.
  if (hccl_config_.find("hccl_buffer_size") != hccl_config_.end()) {
    if (std::holds_alternative<int64_t>(hccl_config_["hccl_buffer_size"])) {
      int64_t size = std::get<int64_t>(hccl_config_["hccl_buffer_size"]);
      if (size < 0 || size > UINT32_MAX) {
        MS_LOG(EXCEPTION) << "hccl_buffer_size out of uint32_t range: " << size;
      }
      config.hcclBufferSize = static_cast<uint32_t>(size);
    } else if (std::holds_alternative<uint32_t>(hccl_config_["hccl_buffer_size"])) {
      config.hcclBufferSize = std::get<uint32_t>(hccl_config_["hccl_buffer_size"]);
    } else {
      MS_LOG(EXCEPTION)
        << "Failed to set hcclBufferSize. Type of hccl_buffer_size in GroupOptions should be unsigned integer.";
    }
  }

  return config;
}

bool AscendCommunicationGroup::InitByRootInfoConfig(void *root_info, uint32_t group_size, uint32_t group_rank,
                                                    const HcclCommConfig &config) {
  MS_LOG(WARNING) << "Start to initialize communicator by HcclCommInitRootInfoConfig for " << name_
                  << ", hcclBufferSize is " << config.hcclBufferSize << " MB, hcclDeterministic is "
                  << config.hcclDeterministic;
  if (root_info == nullptr) {
    return false;
  }
  unique_id_ = *(static_cast<HcclRootInfo *>(root_info));
  // MindSpore 2.5 uses HcclCommInitRootInfo to avoid using hcclconfig feature.
  if (hccl::HcclAdapter::GetInstance().HcclCommInitRootInfoConfig(static_cast<uint32_t>(group_size), &unique_id_,
                                                                  static_cast<uint32_t>(group_rank), &config,
                                                                  &comm_) != static_cast<int32_t>(HCCL_SUCCESS)) {
    const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
    MS_LOG(ERROR) << "HcclCommInitRootInfoConfig failed. " + error_message;
    return false;
  }
  MS_LOG(WARNING) << "End to initialize communicator by HcclCommInitRootInfoConfig for " << name_;
  return true;
}

bool AscendCommunicationGroup::InitByRankTable(std::string rank_table, uint32_t group_size, uint32_t group_rank,
                                               HcclCommConfig *config) {
  if (name_ == kHCCLGlobalGroupName) {
    // Initialize global communicator by 'HcclCommInitClusterInfoConfig'.
    MS_LOG(WARNING) << "Start to initialize communicator by HcclCommInitClusterInfoConfig for " << name_
                    << ", hcclBufferSize is " << config->hcclBufferSize << " MB, hcclDeterministic is "
                    << config->hcclDeterministic;
    if (hccl::HcclAdapter::GetInstance().HcclCommInitClusterInfoConfig(static_cast<const char *>(rank_table.c_str()),
                                                                       static_cast<uint32_t>(global_rank_), config,
                                                                       &comm_) != static_cast<int32_t>(HCCL_SUCCESS)) {
      const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
      MS_LOG(ERROR) << "HcclCommInitClusterInfoConfig failed. " + error_message;
      return false;
    }
    MS_LOG(WARNING) << "End to initialize communicator by HcclCommInitClusterInfoConfig for " << name_;
  } else {
    // split sub communicator from global communicator by 'HcclCreateSubCommConfig'.
    MS_LOG(WARNING) << "Start to initialize communicator by HcclCreateSubCommConfig for " << name_
                    << ", hcclBufferSize is " << config->hcclBufferSize << " MB, hcclDeterministic is "
                    << config->hcclDeterministic;
    // The HCCL global communicator. This is used as a parameter to segment sub communicator if initializing with
    // 'HcclCreateSubCommConfig'.
    auto global_comm = AscendCollectiveCommLib::GetInstance().HcclCommunicator(kHCCLGlobalGroupName);
    MS_EXCEPTION_IF_NULL(global_comm);
    std::hash<std::string> to_hash;
    size_t sub_comm_id = to_hash(name_);
    if (hccl::HcclAdapter::GetInstance().HcclCreateSubCommConfig(
          &global_comm, static_cast<uint32_t>(group_size), group_ranks_.data(), static_cast<uint64_t>(sub_comm_id),
          static_cast<uint32_t>(group_rank), config, &comm_) != static_cast<int32_t>(HCCL_SUCCESS)) {
      const string &error_message = ErrorManagerAdapter::GetErrorMessage(true);
      MS_LOG(ERROR) << "HcclCreateSubCommConfig failed. " + error_message;
      return false;
    }
    MS_LOG(WARNING) << "End to initialize communicator by HcclCreateSubCommConfig for " << name_;
  }
  return true;
}

bool AscendCommunicationGroup::InitByHcclComm() {
  MS_LOG(INFO) << "Start to initialize communicator by Hcom from hccl config for " << name_;
  if (!std::holds_alternative<int64_t>(hccl_config_["hccl_comm"])) {
    MS_LOG(EXCEPTION) << "Failed to get hcom. Type of hccl_comm in GroupOptions should be int64.";
  }
  comm_ = reinterpret_cast<HcclComm>(static_cast<intptr_t>(std::get<int64_t>(hccl_config_["hccl_comm"])));
  MS_LOG(INFO) << "End to initialize communicator by Hcom from hccl config for " << name_;
  return true;
}

void *AscendCommunicationGroup::GenerateRootInfo(size_t *root_info_size) {
  *root_info_size = sizeof(unique_id_);
  if (!common::GetEnv(kSimulationLevel).empty() && !hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    if (HcclGetRootInfo(&unique_id_) != static_cast<int32_t>(HCCL_SUCCESS)) {
      return nullptr;
    }
    return &unique_id_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  (void)CALL_ASCEND_API(aclrtSetDevice, device_id);
  uint32_t group_rank = GetGroupRank(global_rank_);
  if (group_rank == 0) {
    if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
      // If using hccl CM envs to launch distributed job, no need to call HcclGetRootInfo.
      return &unique_id_;
    }
    if (HcclGetRootInfo(&unique_id_) != static_cast<int32_t>(HCCL_SUCCESS)) {
      MS_LOG(ERROR) << "Failed to get HCCL unique id: " << CALL_ASCEND_API(aclGetRecentErrMsg);
      return nullptr;
    }
  }
  return &unique_id_;
}

bool AscendCommunicationGroup::IsSupportConfigParameter(uint32_t config_parameter) {
  return config_parameter < hccl::HcclAdapter::GetInstance().HcclGetCommConfigCapability();
}

const HcclComm &AscendCommunicationGroup::hccl_communicator() const { return comm_; }

std::string AscendCommunicationGroup::inner_comm_name() const { return inner_comm_name_; }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
