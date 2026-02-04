/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"
#include "include/utils/utils.h"
#include "plugin/ascend/res_manager/hccl_adapter/hccl_adapter.h"
#include "kernel/ascend/hccl/hcom_util.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"

constexpr size_t kPathMax = 4096;
namespace mindspore {
namespace device {
namespace ascend {
using GroupOptions = mindspore::device::GroupOptions;
namespace {
/* Correspondence between data_type and hcom data type in Ascend */
static std::map<int64_t, HcclDataType> kConstOpHcomDataTypeMap = {
  {TypeId::kNumberTypeInt8, HCCL_DATA_TYPE_INT8},     {TypeId::kNumberTypeInt16, HCCL_DATA_TYPE_INT16},
  {TypeId::kNumberTypeInt32, HCCL_DATA_TYPE_INT32},   {TypeId::kNumberTypeFloat16, HCCL_DATA_TYPE_FP16},
  {TypeId::kNumberTypeFloat32, HCCL_DATA_TYPE_FP32},  {TypeId::kNumberTypeInt64, HCCL_DATA_TYPE_INT64},
  {TypeId::kNumberTypeUInt64, HCCL_DATA_TYPE_UINT64}, {TypeId::kNumberTypeUInt8, HCCL_DATA_TYPE_UINT8},
  {TypeId::kNumberTypeUInt16, HCCL_DATA_TYPE_UINT16}, {TypeId::kNumberTypeUInt32, HCCL_DATA_TYPE_UINT32},
  {TypeId::kNumberTypeFloat64, HCCL_DATA_TYPE_FP64},  {TypeId::kNumberTypeBFloat16, HCCL_DATA_TYPE_BFP16},
  {TypeId::kNumberTypeBool, HCCL_DATA_TYPE_INT8},
};

::HcclDataType ConvertHcclType(TypeId type_id) {
  auto iter = kConstOpHcomDataTypeMap.find(type_id);
  if (iter == kConstOpHcomDataTypeMap.end()) {
    if (type_id == TypeId::kNumberTypeComplex64) {
      MS_LOG(INFO) << "HcomDataType Can't support Current Ascend Data Type : Complex64, Convert it to Float32";
      return HCCL_DATA_TYPE_FP32;
    }
    MS_LOG(EXCEPTION) << "HcomDataType can't support Current Ascend Data Type : " << TypeIdLabel(type_id);
  }
  return iter->second;
}
}  // namespace
#define HCCL_RUN_CHECK(op_name, group, op)                         \
  do {                                                             \
    auto hccl_result = static_cast<int64_t>(op);                   \
    if (hccl_result != 0) {                                        \
      MS_LOG(ERROR) << (op_name) << " failed: #" << (group) << "#" \
                    << ", error code: " << hccl_result;            \
      return false;                                                \
    }                                                              \
  } while (0)

#define EXCEPTION_IF_HCCL_RUN_CHECK_FAIL(op_name, group, op)           \
  do {                                                                 \
    auto hccl_result = static_cast<int64_t>(op);                       \
    if (hccl_result != 0) {                                            \
      MS_LOG(EXCEPTION) << (op_name) << " failed: #" << (group) << "#" \
                        << ", error code: " << hccl_result;            \
    }                                                                  \
  } while (0)

#define HCCL_GROUP_CHECK_EMPTY(group)                              \
  do {                                                             \
    if ((group).length() == 0) {                                   \
      MS_LOG(ERROR) << "The length of group name should not be 0"; \
      return false;                                                \
    }                                                              \
  } while (0)

#define EXCEPTION_IF_HCCL_GROUP_CHECK_EMPTY(group)                     \
  do {                                                                 \
    if ((group).length() == 0) {                                       \
      MS_LOG(EXCEPTION) << "The length of group name should not be 0"; \
    }                                                                  \
  } while (0)

#define HCCL_GROUP_CHECK_IS_WORLD(group)                                   \
  do {                                                                     \
    if ((group) == kHcclWorldGroup) {                                      \
      MS_LOG(ERROR) << "The group name should not be " << kHcclWorldGroup; \
      return false;                                                        \
    }                                                                      \
  } while (0)
AscendCollectiveCommLib::AscendCollectiveCommLib() { global_group_name_ = kHCCLGlobalGroupName; }

AscendCollectiveCommLib &AscendCollectiveCommLib::GetInstance() {
  static AscendCollectiveCommLib instance;
  return instance;
}

bool AscendCollectiveCommLib::InitializeHccl() {
  if (initialized_) {
    return true;
  }
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
  MS_LOG(INFO) << "Create hccl_world_group with rank table.";
  auto config_path_str = common::EnvHelper::GetInstance()->GetEnv("MINDSPORE_HCCL_CONFIG_PATH");
  if (config_path_str == nullptr) {
    config_path_str = common::EnvHelper::GetInstance()->GetEnv("RANK_TABLE_FILE");
    if (config_path_str == nullptr) {
      MS_LOG(ERROR) << "The environment variable 'MINDSPORE_HCCL_CONFIG_PATH' or 'RANK_TABLE_FILE' is not set, so get"
                    << " hccl json config failed, please set env 'MINDSPORE_HCCL_CONFIG_PATH' or 'RANK_TABLE_FILE'";
      return false;
    }
  }
  if (strlen(config_path_str) >= kPathMax) {
    MS_LOG(ERROR) << "Invalid environment variable 'MINDSPORE_HCCL_CONFIG_PATH' or 'RANK_TABLE_FILE', the path length"
                  << " should be smaller than " << kPathMax << ", but got " << config_path_str;
    return false;
  }
  auto full_path = realpath(config_path_str, nullptr);
  if (full_path == nullptr) {
    MS_LOG(ERROR) << "Invalid environment variable 'MINDSPORE_HCCL_CONFIG_PATH' or 'RANK_TABLE_FILE', the path is: "
                  << config_path_str << ". Please check (1) whether the path exists, "
                  << "(2) whether the path has the access permission, (3) whether the path is too long. ";
    return false;
  }
  auto rank_id_str = common::GetEnv("RANK_ID");
  if (rank_id_str.empty()) {
    MS_LOG(EXCEPTION) << "Invalid environment variable 'RANK_ID', it should not be empty.";
  }
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  MS_LOG(INFO) << "MINDSPORE_HCCL_CONFIG_PATH : " << full_path << ", RANK_ID: " << rank_id_str;

  hccl::HcclMode hccl_mode = hccl::HcclMode::kGraph;
  if (ms_context->IsKByKExecutorMode()) {
    hccl_mode = hccl::HcclMode::kKernelByKernel;
  }

  bool ret = hccl::HcclAdapter::GetInstance().InitHccl(device_id, rank_id_str, full_path, hccl_mode);
  free(full_path);
  if (!ret) {
    MS_LOG(ERROR) << "Hcom init failed.";
    return false;
  }
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool AscendCollectiveCommLib::Initialize(uint32_t global_rank, uint32_t global_rank_size, uint32_t local_rank_id) {
  if (initialized_) {
    return true;
  }
  try {
    if (!common::GetEnv(kSimulationLevel).empty()) {
      std::string rank_id_str = std::to_string(0);
      (void)hccl::HcclAdapter::GetInstance().InitHccl(local_rank_id, rank_id_str);
    } else if (!common::UseHostCollective()) {
      // Use rank table to launch distribtued job.
      MS_LOG(WARNING)
        << "Launch Ascend distributed job in RankTable manner. This manner will be deprecated in later version of "
           "MindSpore. \n Please switch to 'msrun' or 'mpirun'. You can refer to this link about how to use these "
           "commands: https://www.mindspore.cn/docs/zh-CN/r2.5.0/model_train/parallel/startup_method.html.";
      return InitializeHccl();
    } else {
      if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
        // Use dynamic cluster and hccl's CM envs to launch distributed job. This method is similar to rank table. It
        // only supports to run in graph sink mode.
        MS_LOG(INFO) << "Launch Ascend distributed job using hccl CM envs.";
      }
      std::string rank_id_str = std::to_string(global_rank);
      (void)hccl::HcclAdapter::GetInstance().InitHccl(local_rank_id, rank_id_str);
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Ascend collective communication initialization failed.#dmsg#Framework Error Message:#dmsg#"
                      << e.what();
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool AscendCollectiveCommLib::Finalize() {
  if (!initialized_ || finalized_.load()) {
    return true;
  }

  for (const auto &group : groups_) {
    CHECK_IF_NULL(group.second);
    if (!group.second->Finalize()) {
      return false;
    }
  }

  groups_.clear();
  group_hccl_comm_map_.clear();
  initialized_ = false;
  finalized_ = true;
  return true;
}

bool AscendCollectiveCommLib::DestroyHcclComm() {
  for (auto &group : groups_) {
    CHECK_IF_NULL(group.second);
    if (!group.second->Finalize()) {
      return false;
    }
  }
  group_hccl_comm_map_.clear();
  groups_.clear();
  bool res = hccl::HcclAdapter::GetInstance().FinalizeHccl();
  if (!res) {
    MS_LOG(WARNING) << "Hccl finalize failed";
    return false;
  }
  return true;
}

bool AscendCollectiveCommLib::DestroyDeviceCommunicationGroup(const std::string &group_name) {
  HCCL_GROUP_CHECK_EMPTY(group_name);
  return true;
}

bool AscendCollectiveCommLib::DestroyCommunicationGroup(const std::string &group_name) {
  // If using hccl CM, we reuse rank table launching interfaces.
  if (hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    return DestroyDeviceCommunicationGroup(group_name);
  }

  HCCL_GROUP_CHECK_EMPTY(group_name);
  CHECK_RET((groups_.count(group_name) != 0), true, "The HCCL group " + group_name + " does not exist.");

  if (!groups_[group_name]->Finalize()) {
    MS_LOG(WARNING) << "group finalize failed";
    return false;
  }
  (void)groups_.erase(group_name);
  if (group_hccl_comm_map_.count(group_name)) {
    (void)group_hccl_comm_map_.erase(group_name);
  }
  return true;
}

bool AscendCollectiveCommLib::CreateDeviceCommunicationGroup(const std::string &group_name,
                                                             const std::vector<uint32_t> &group_ranks) {
  HCCL_GROUP_CHECK_EMPTY(group_name);
  return true;
}

bool AscendCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                       const std::vector<uint32_t> &group_ranks,
                                                       uint32_t local_group_rank, uint32_t local_group_size,
                                                       const GroupOptions &config) {
  HCCL_GROUP_CHECK_EMPTY(group_name);
  if (groups_.count(group_name) != 0) {
    // If this group name has already existed, return true instead of throwing exception.
    MS_LOG(INFO) << "The HCCL group " << group_name << " has already existed.";
    return true;
  }

  AscendCommunicationGroupPtr group = std::make_shared<AscendCommunicationGroup>(
    group_name, group_ranks, global_rank_id_, local_group_rank, local_group_size, config.hccl_config);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;

  // If using hccl CM, we reuse rank table launching interfaces.
  // It does not support to create hccl_world_group.
  if (hccl::HcclAdapter::GetInstance().UseHcclCM() && group_name != kHCCLGlobalGroupName) {
    return CreateDeviceCommunicationGroup(group_name, group_ranks);
  }
  return true;
}

HcclComm AscendCollectiveCommLib::HcclCommunicator(const std::string &group_name) {
  if (!common::GetEnv(kSimulationLevel).empty()) {
    return nullptr;
  }
  if (!common::UseHostCollective() || hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    return hccl::HcclAdapter::GetInstance().get_hccl_comm();
  }
  CHECK_RET((groups_.count(group_name) != 0), true, "The HCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<AscendCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->hccl_communicator();
}

HcclComm AscendCollectiveCommLib::GetHcomByGroup(const std::string &group_name) {
  auto iter = group_hccl_comm_map_.find(group_name);
  if (iter == group_hccl_comm_map_.end()) {
    distributed::collective::CollectiveManager::instance()->WaitCommInitDone(group_name);
    auto comm = HcclCommunicator(group_name);
    group_hccl_comm_map_[group_name] = comm;
    return comm;
  }

  return iter->second;
}

std::string AscendCollectiveCommLib::CommName(const std::string &group_name) {
  if (!common::UseHostCollective() || hccl::HcclAdapter::GetInstance().UseHcclCM()) {
    return "";
  }
  CHECK_RET((groups_.count(group_name) != 0), true, "The HCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<AscendCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->inner_comm_name();
}

uint32_t AscendCollectiveCommLib::GetRankId(const std::string &group_name) {
  uint32_t rank_id = 0;
  EXCEPTION_IF_HCCL_RUN_CHECK_FAIL(std::string("get rank_id"), group_name,
                                   hccl::HcclAdapter::GetInstance().HcclGetRankId(group_name, &rank_id));
  return rank_id;
}

uint32_t AscendCollectiveCommLib::GetGroupSize(const std::string &group_name) {
  EXCEPTION_IF_HCCL_GROUP_CHECK_EMPTY(group_name);
  uint32_t rank_size = 0;
  EXCEPTION_IF_HCCL_RUN_CHECK_FAIL(std::string("get rank size"), group_name,
                                   hccl::HcclAdapter::GetInstance().HcclGetRankSize(group_name, &rank_size));
  return rank_size;
}

uint32_t AscendCollectiveCommLib::GetLocalRankId(const std::string &group_name) {
  MS_LOG(EXCEPTION) << "MindSpore does not support ranktable startup, please use msrun to start it.";
}

uint32_t AscendCollectiveCommLib::GetLocalGroupSize(const std::string &group_name) {
  MS_LOG(EXCEPTION) << "MindSpore does not support ranktable startup, please use msrun to start it.";
}

uint32_t AscendCollectiveCommLib::GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) {
  MS_LOG(EXCEPTION) << "MindSpore does not support ranktable startup, please use msrun to start it.";
}

uint32_t AscendCollectiveCommLib::GetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group_name) {
  MS_LOG(EXCEPTION) << "MindSpore does not support ranktable startup, please use msrun to start it.";
}

bool AscendCollectiveCommLib::CommSwitchNic(const std::vector<uint32_t> &global_ranks,
                                            const std::vector<bool> &use_backup) {
  MS_LOG(INFO) << "Switch global ranks: " << global_ranks << ", use backup: " << use_backup;
  // Firstly, traverse the communication domain to check whether there is an intersection
  // between the ranks in the communication domain and the ranks that need to switch NIC.
  for (const auto &kv : groups_) {
    std::vector<uint32_t> sec_global_ranks;
    std::vector<uint32_t> sec_group_ranks;
    std::vector<bool> sec_use_backup_v;
    auto group_ranks = kv.second->group_ranks();
    for (size_t i = 0; i < global_ranks.size(); i++) {
      if (std::find(group_ranks.begin(), group_ranks.end(), global_ranks[i]) != group_ranks.end()) {
        sec_global_ranks.push_back(global_ranks[i]);
        sec_use_backup_v.push_back(use_backup[i]);
        sec_group_ranks.push_back(kv.second->GetGroupRank(global_ranks[i]));
      }
    }

    // If there is no intersection, continue to traverse the next communication domain.
    if (sec_global_ranks.empty()) {
      MS_LOG(INFO) << "group: " << kv.first << ", group ranks(global): " << group_ranks
                   << ", sec global ranks is empty.";
      continue;
    }

    // If there is intersection, obtain the group ranks corresponding to the intersection global ranks,
    // and the hccl comm  corresponding to the group name, then call the HcclCommWorkingDevNicSet to switch NIC.
    bool *sec_use_backup = new bool[sec_use_backup_v.size()];
    for (size_t i = 0; i < sec_use_backup_v.size(); i++) {
      sec_use_backup[i] = sec_use_backup_v[i];
    }

    MS_LOG(INFO) << "group: " << kv.first << ", group ranks(global): " << group_ranks
                 << ", sec global ranks: " << sec_global_ranks << ", sec group ranks: " << sec_group_ranks
                 << ", sec use backup: " << sec_use_backup_v;
    auto ret = hccl::HcclAdapter::GetInstance().HcclCommWorkingDevNicSet(
      GetHcomByGroup(kv.first), sec_group_ranks.data(), sec_use_backup, sec_group_ranks.size());
    delete[] sec_use_backup;
    if (ret != HCCL_SUCCESS) {
      MS_LOG(ERROR) << "group: " << kv.first << " switch nic failed, error code" << ret;
      return false;
    }
    MS_LOG(INFO) << "group: " << kv.first << " switch nic success!";
  }
  MS_LOG(INFO) << "The comm switch nic execute success!";
  return true;
}

bool AscendCollectiveCommLib::ResumeHcclComm() {
  for (auto &group : groups_) {
    auto hccl_comm = HcclCommunicator(group.first);
    HCCL_RUN_CHECK(std::string("resume communicate group"), group.first,
                   hccl::HcclAdapter::GetInstance().HcclCommResume(hccl_comm));
  }
  return true;
}

bool AscendCollectiveCommLib::AllGather(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                        const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(recv_buff);
  MS_EXCEPTION_IF_NULL(stream);
  const auto hccl_data_type = ConvertHcclType(data_type);
  const auto comm = GetHcomByGroup(group_name);
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllGather(const_cast<void *>(send_buff), recv_buff,
                                                                    send_count, hccl_data_type, stream, comm);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAllGather failed, ret:" << hccl_result;
    return false;
  }
  return true;
}

bool AscendCollectiveCommLib::AllReduce(const void *send_buff, void *recv_buff, size_t send_count, TypeId data_type,
                                        CollectiveOpReduceType reduce_op, const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(recv_buff);
  MS_EXCEPTION_IF_NULL(stream);
  const auto hccl_data_type = ConvertHcclType(data_type);
  const auto comm = GetHcomByGroup(group_name);
  const auto &hccl_reduce_type_iter = kHcomOpReduceTypeMap.find(reduce_op);
  if (hccl_reduce_type_iter == kHcomOpReduceTypeMap.end()) {
    MS_LOG(ERROR) << "Can not find hcom reduce type for " << reduce_op;
    return false;
  }
  const auto hccl_reduce_type = hccl_reduce_type_iter->second;

  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllReduce(
    const_cast<void *>(send_buff), recv_buff, send_count, hccl_data_type, hccl_reduce_type, stream, comm);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAllReduce failed, ret:" << hccl_result;
    return false;
  }
  return true;
}

bool AscendCollectiveCommLib::Broadcast(const void *send_buff, void *, size_t send_count, TypeId data_type,
                                        uint32_t root_rank, const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(stream);
  const auto hccl_data_type = ConvertHcclType(data_type);
  const auto comm = GetHcomByGroup(group_name);
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclBroadcast(const_cast<void *>(send_buff), send_count,
                                                                    hccl_data_type, root_rank, stream, comm);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclBroadcast failed, ret: " << hccl_result;
    return false;
  }
  return true;
}

bool AscendCollectiveCommLib::ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, TypeId data_type,
                                            CollectiveOpReduceType reduce_op, const std::string &group_name,
                                            void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(recv_buff);
  MS_EXCEPTION_IF_NULL(stream);
  const auto hccl_data_type = ConvertHcclType(data_type);
  const auto comm = GetHcomByGroup(group_name);
  const auto &hccl_reduce_type_iter = kHcomOpReduceTypeMap.find(reduce_op);
  if (hccl_reduce_type_iter == kHcomOpReduceTypeMap.end()) {
    MS_LOG(ERROR) << "Can not find hcom reduce type for " << reduce_op;
    return false;
  }
  const auto hccl_reduce_type = hccl_reduce_type_iter->second;

  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclReduceScatter(
    const_cast<void *>(send_buff), recv_buff, recv_count, hccl_data_type, hccl_reduce_type, stream, comm);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclReduceScatter failed, ret:" << hccl_result;
    return false;
  }
  return true;
}

bool AscendCollectiveCommLib::Send(const void *send_buff, size_t count, TypeId data_type, uint32_t peer,
                                   const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(send_buff);
  MS_EXCEPTION_IF_NULL(stream);
  const auto hccl_data_type = ConvertHcclType(data_type);
  const auto comm = GetHcomByGroup(group_name);
  auto hccl_result =
    hccl::HcclAdapter::GetInstance().HcclSend(const_cast<void *>(send_buff), count, hccl_data_type, peer, stream, comm);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcomSend failed, ret:" << hccl_result;
    return false;
  }
  return true;
}

bool AscendCollectiveCommLib::Recv(void *recv_buff, size_t count, TypeId data_type, uint32_t peer,
                                   const std::string &group_name, void *stream) {
  MS_EXCEPTION_IF_NULL(recv_buff);
  MS_EXCEPTION_IF_NULL(stream);
  const auto hccl_data_type = ConvertHcclType(data_type);
  const auto comm = GetHcomByGroup(group_name);
  MS_EXCEPTION_IF_NULL(comm);
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclRecv(recv_buff, count, hccl_data_type, peer, stream, comm);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcomReceive failed, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace ascend

using AscendCollectiveCommLib = mindspore::device::ascend::AscendCollectiveCommLib;

CollectiveCommunicationLib *ascend_communication_lib_instance() { return &AscendCollectiveCommLib::GetInstance(); }
}  // namespace device
}  // namespace mindspore
