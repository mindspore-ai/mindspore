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

#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"

constexpr size_t kPathMax = 4096;
namespace mindspore {
namespace device {
namespace ascend {
#define HCCL_RUN_CHECK(op_name, group, op)                          \
  do {                                                              \
    auto hccl_result = static_cast<int64_t>(op);                    \
    if (hccl_result != 0) {                                         \
      MS_LOG(ERROR) << (op_name) << " failed: #" << (group) << "#"; \
      return false;                                                 \
    }                                                               \
  } while (0)

#define HCCL_GROUP_CHECK_EMPTY(group)                              \
  do {                                                             \
    if ((group).length() == 0) {                                   \
      MS_LOG(ERROR) << "The length of group name should not be 0"; \
      return false;                                                \
    }                                                              \
  } while (0)

#define HCCL_GROUP_CHECK_IS_WORLD(group)                                   \
  do {                                                                     \
    if ((group) == kHcclWorldGroup) {                                      \
      MS_LOG(ERROR) << "The group name should not be " << kHcclWorldGroup; \
      return false;                                                        \
    }                                                                      \
  } while (0)
AscendCollectiveCommLib::AscendCollectiveCommLib() { global_group_name_ = kHCCLGlobalGroupName; }

bool AscendCollectiveCommLib::InitializeHccl() {
  if (initialized_) {
    return false;
  }
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
  MS_LOG(INFO) << "Create hccl_world_group with rank table.";
  auto config_path_str = std::getenv("MINDSPORE_HCCL_CONFIG_PATH");
  if (config_path_str == nullptr) {
    config_path_str = std::getenv("RANK_TABLE_FILE");
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

  auto mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  bool ret = hccl::HcclAdapter::GetInstance().InitHccl(
    device_id, rank_id_str, full_path, mode == kGraphMode ? hccl::HcclMode::kGraph : hccl::HcclMode::kPynative);
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
    return false;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
  (void)device_context->GetDeprecatedInterface()->OpenTsd(ms_context);
  try {
    if (!common::UseHostCollective()) {
      return InitializeHccl();
    }
    std::string rank_id_str = std::to_string(global_rank);
    (void)hccl::HcclAdapter::GetInstance().InitHccl(local_rank_id, rank_id_str);
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Ascend collective communication initialization failed.#dmsg#Framework Error Message:#dmsg#"
                      << e.what();
  }
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
  global_rank_id_ = global_rank;
  global_rank_size_ = global_rank_size;
  local_rank_id_ = local_rank_id;
  initialized_ = true;
  finalized_ = false;
  return true;
}

bool AscendCollectiveCommLib::DestroyHcclComm() {
  for (auto &group : groups_) {
    CHECK_IF_NULL(group.second);
    if (!group.second->Finalize()) {
      return false;
    }
  }
  groups_.clear();
  bool res = hccl::HcclAdapter::GetInstance().FinalizeHccl();
  if (!res) {
    MS_LOG(ERROR) << "Hccl finalize failed";
    return false;
  }
  return true;
}

bool AscendCollectiveCommLib::DestroyDeviceCommunicationGroup(const std::string &group_name) {
  HCCL_GROUP_CHECK_EMPTY(group_name);
  HCCL_RUN_CHECK(std::string("destroy communicate group"), group_name,
                 hccl::HcclAdapter::GetInstance().HcclDestroyGroup(group_name));
  return true;
}

bool AscendCollectiveCommLib::CreateDeviceCommunicationGroup(const std::string &group_name,
                                                             const std::vector<uint32_t> &group_ranks) {
  HCCL_GROUP_CHECK_EMPTY(group_name);
  auto rank_size = group_ranks.size();
  HCCL_RUN_CHECK(std::string("create communicate group"), group_name,
                 hccl::HcclAdapter::GetInstance().HcclCreateGroup(group_name, UlongToUint(rank_size),
                                                                  std::vector<unsigned int>(group_ranks).data()));
  return true;
}

bool AscendCollectiveCommLib::CreateCommunicationGroup(const std::string &group_name,
                                                       const std::vector<uint32_t> &group_ranks,
                                                       uint32_t local_group_rank, uint32_t local_group_size) {
  HCCL_GROUP_CHECK_EMPTY(group_name);
  CHECK_RET((groups_.count(group_name) == 0), true, "The HCCL group " + group_name + " has already existed.");

  AscendCommunicationGroupPtr group = std::make_shared<AscendCommunicationGroup>(
    group_name, group_ranks, global_rank_id_, local_group_rank, local_group_size);
  CHECK_IF_NULL(group);
  groups_[group_name] = group;
  return true;
}

HcclComm AscendCollectiveCommLib::HcclCommunicator(const std::string &group_name) {
  if (!common::UseHostCollective()) {
    return hccl::HcclAdapter::GetInstance().get_hccl_comm();
  }
  CHECK_RET((groups_.count(group_name) != 0), true, "The HCCL group " + group_name + " does not existed.");
  auto group = std::dynamic_pointer_cast<AscendCommunicationGroup>(groups_[group_name]);
  CHECK_IF_NULL(group);
  return group->hccl_communicator();
}

uint32_t AscendCollectiveCommLib::GetRankId(const std::string &group_name) {
  uint32_t rank_id = 0;
  HCCL_RUN_CHECK(std::string("get rank_id"), group_name,
                 hccl::HcclAdapter::GetInstance().HcclGetRankId(group_name, &rank_id));
  return rank_id;
}

uint32_t AscendCollectiveCommLib::GetGroupSize(const std::string &group_name) {
  HCCL_GROUP_CHECK_EMPTY(group_name);
  uint32_t rank_size = 0;
  HCCL_RUN_CHECK(std::string("get rank size"), group_name,
                 hccl::HcclAdapter::GetInstance().HcclGetRankSize(group_name, &rank_size));
  return rank_size;
}

uint32_t AscendCollectiveCommLib::GetLocalRankId(const std::string &group_name) {
  uint32_t rank_id = 0;
  HCCL_RUN_CHECK(std::string("get rank_id"), group_name,
                 hccl::HcclAdapter::GetInstance().HcclGetLocalRankId(group_name, &rank_id));
  return rank_id;
}

uint32_t AscendCollectiveCommLib::GetLocalGroupSize(const std::string &group_name) {
  HCCL_GROUP_CHECK_EMPTY(group_name);
  uint32_t rank_size = 0;
  HCCL_RUN_CHECK(std::string("get rank size"), group_name,
                 hccl::HcclAdapter::GetInstance().HcclGetLocalRankSize(group_name, &rank_size));
  return rank_size;
}

uint32_t AscendCollectiveCommLib::GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) {
  uint32_t world_rank_id = 0;
  HCCL_RUN_CHECK(
    std::string("get world rank id"), group_name,
    hccl::HcclAdapter::GetInstance().HcclGetWorldRankFromGroupRank(group_name, local_rank, &world_rank_id));
  return world_rank_id;
}

uint32_t AscendCollectiveCommLib::GetGroupRankFromWorldRank(uint32_t world_rank, const std::string &group_name) {
  uint32_t local_rank_id = 0;
  HCCL_RUN_CHECK(
    std::string("get local rank id"), group_name,
    hccl::HcclAdapter::GetInstance().HcclGetGroupRankFromWorldRank(world_rank, group_name, &local_rank_id));
  return local_rank_id;
}
}  // namespace ascend

using AscendCollectiveCommLib = mindspore::device::ascend::AscendCollectiveCommLib;

CollectiveCommunicationLib *communication_lib_instance() { return &AscendCollectiveCommLib::GetInstance(); }
}  // namespace device
}  // namespace mindspore
