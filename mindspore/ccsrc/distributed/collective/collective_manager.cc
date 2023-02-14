/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "distributed/collective/collective_manager.h"
#include <algorithm>
#include <string>
#include <numeric>
#include <vector>
#include <functional>
#include <csignal>
#include <memory>
#include "utils/ms_context.h"
#include "distributed/recovery/recovery_context.h"

namespace mindspore {
namespace distributed {
namespace collective {
using recovery::RecoveryContext;

CollectiveManager::CollectiveManager()
    : inited_(false),
      finalized_(true),
      need_init_(false),
      need_reinit_(false),
      host_ctx_(nullptr),
      device_ctx_(nullptr),
      host_comm_lib_instance_(nullptr),
      device_comm_lib_instance_(nullptr),
      comm_lib_instance_(nullptr),
      global_rank_id_(0),
      local_rank_id_(0),
      global_rank_size_(0),
      global_group_ranks_({}),
      device_lib_supported_(true),
      need_host_collective_(false) {}

CollectiveManager::~CollectiveManager() {
  if (!finalized_) {
    try {
      (void)Finalize();
    } catch (std::exception &) {
      MS_LOG(ERROR) << "Failed to finalize collective manager.";
    }
  }
  finalized_ = true;
  host_ctx_ = nullptr;
  device_ctx_ = nullptr;
  host_comm_lib_instance_ = nullptr;
  device_comm_lib_instance_ = nullptr;
  comm_lib_instance_ = nullptr;
}

std::shared_ptr<CollectiveManager> CollectiveManager::instance() {
  static std::shared_ptr<CollectiveManager> instance = nullptr;
  if (instance == nullptr) {
    instance.reset(new (std::nothrow) CollectiveManager());
    MS_EXCEPTION_IF_NULL(instance);
  }
  return instance;
}

namespace {
// The wrapper to provide a timeout mechanism for executing functions.
bool ExecuteFuncInThread(const std::function<bool()> &func, const int64_t timeout) {
  bool execute_success = false;
  bool execute_fail = false;
  std::mutex exec_ret_mutex;
  std::condition_variable thread_blocker;

  std::unique_ptr<std::thread> executive_thread = std::make_unique<std::thread>([&] {
    if (!func()) {
      MS_LOG(ERROR) << "Failed to execute function asynchronously";
      std::unique_lock<std::mutex> lock(exec_ret_mutex);
      execute_fail = true;
      thread_blocker.notify_one();
      return;
    }

    {
      std::unique_lock<std::mutex> lock(exec_ret_mutex);
      execute_success = true;
      thread_blocker.notify_one();
    }
  });
  MS_EXCEPTION_IF_NULL(executive_thread);
  executive_thread->detach();

  std::unique_lock<std::mutex> locker(exec_ret_mutex);
  (void)thread_blocker.wait_for(locker, std::chrono::seconds(timeout), [&] { return execute_success || execute_fail; });

  if (!execute_success && !execute_fail) {
    std::string node_id = common::GetEnv("MS_NODE_ID");
#if !defined(_WIN32) && !defined(_WIN64)
    MS_LOG(ERROR) << "Execute function asynchronously timeout, node id: " << node_id << " exit process";
    (void)kill(getpid(), SIGTERM);
#endif
  }
  return execute_success;
}

// In a disaster recovery scenario, the comparison between the current unique id and the last generated unique id
// ensures that the acquired unique id is newly generated, and the latest unique id will be persisted.
bool CheckUniqueIDLatest(const std::string &group_name, size_t root_info_size, const void *root_info) {
  MS_EXCEPTION_IF_NULL(root_info);
  auto persistent_json = RecoveryContext::GetInstance()->persistent_json();
  MS_EXCEPTION_IF_NULL(persistent_json);

  std::string new_unique_id(static_cast<const char *>(root_info), root_info_size);
  std::vector<int> new_unique_id_integer_seq;
  (void)std::transform(new_unique_id.begin(), new_unique_id.end(), std::back_inserter(new_unique_id_integer_seq),
                       [](char c) { return static_cast<int>(c); });

  const char unique_id_str[] = "_unique_id";
  std::string unique_id_key = group_name + unique_id_str;
  if (!persistent_json->Exists(unique_id_key)) {
    persistent_json->Insert(unique_id_key, new_unique_id_integer_seq);
    return true;
  }

  std::vector<int> old_unique_id_integer_seq = persistent_json->Get<std::vector<int>>(unique_id_key);
  if (new_unique_id_integer_seq == old_unique_id_integer_seq) {
    return false;
  }

  persistent_json->Insert(unique_id_key, new_unique_id_integer_seq);
  return true;
}
}  // namespace

bool CollectiveManager::Initialize() {
  need_init_ = true;
  if (inited_ && !need_reinit_) {
    return true;
  }
  need_host_collective_ = common::UseHostCollective();
  device_type_ = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  // need_host_collective_ means using rank_table to initialize collective communication, which is only supported by
  // Ascend. On other types of devices, exception should be thrown.
  if (device_type_ != kAscendDevice && !need_host_collective_) {
    MS_LOG(EXCEPTION) << kDetailedFailureReason;
  }

  MS_LOG(INFO) << "Start initializing collective communication for backend: " << device_type_ << "...";

  if (!need_host_collective_) {
    RETURN_IF_FALSE_WITH_LOG(InitDeviceCommLib(), "Failed to initialize device communication library.");
    comm_lib_instance_ = device_comm_lib_instance_;
  } else {
    // Step 1: Initialize host side collective communication.
    RETURN_IF_FALSE_WITH_LOG(InitHostCommlib(), "Failed to initialize host communication library.");
    comm_lib_instance_ = host_comm_lib_instance_;

    // Step 2, 3 and 4 are for device communication library. So if the training job is only launched on CPU, they will
    // not be necessary.
    // Step 2: Assign local rank id(device id) for this process.
    RETURN_IF_FALSE_WITH_LOG(AssignLocalRank(), "Failed to assign local rank id.");

    // Step 3: Initialize device side collective communication.
    RETURN_IF_FALSE_WITH_LOG(InitDeviceCommLib(), "Failed to initialize device communication library.");

    // Step 4: Create global communication group.
    MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
    auto group_name = device_comm_lib_instance_->global_group_name();
    RETURN_IF_FALSE_WITH_LOG(CreateCommunicationGroup(group_name, global_group_ranks_),
                             "Failed to create group " + group_name);
  }

  MS_LOG(INFO) << "End initializing collective communication for backend: " << device_type_;
  inited_ = true;
  finalized_ = false;
  need_reinit_ = false;
  return true;
}

bool CollectiveManager::GetLocalGroupRankAndSize(const std::vector<uint32_t> &group_ranks, uint32_t *local_group_rank,
                                                 uint32_t *local_group_size) {
  MS_EXCEPTION_IF_NULL(local_group_rank);
  MS_EXCEPTION_IF_NULL(local_group_size);
  auto it =
    std::find_if(group_ranks.begin(), group_ranks.end(), [&](uint32_t rank) { return rank > global_rank_size_; });
  if (it != group_ranks.end()) {
    MS_LOG(ERROR) << "The rank " << *it << "is out of global rank size.";
    return false;
  }
  if (all_host_hashs_.size() != static_cast<size_t>(global_rank_size_)) {
    MS_LOG(ERROR) << "The host hash size should be equal to global rank size " << global_rank_size_ << ", but got "
                  << all_host_hashs_.size();
    return false;
  }
  *local_group_size = static_cast<uint32_t>(std::count_if(group_ranks.begin(), group_ranks.end(), [&](uint32_t rank) {
    return all_host_hashs_[rank] == all_host_hashs_[global_rank_id_];
  }));
  auto pos = find(group_ranks.begin(), group_ranks.end(), global_rank_id_);
  if (pos == group_ranks.end()) {
    *local_group_rank = UINT32_MAX;
    return true;
  }
  *local_group_rank = static_cast<uint32_t>(std::count_if(group_ranks.begin(), pos, [&](uint32_t rank) {
    return all_host_hashs_[rank] == all_host_hashs_[global_rank_id_];
  }));
  return true;
}

bool CollectiveManager::CreateCommunicationGroup(const std::string &group_name,
                                                 const std::vector<uint32_t> &group_ranks) {
  MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
  if (!need_host_collective_) {
    RETURN_IF_FALSE_WITH_LOG(device_comm_lib_instance_->CreateDeviceCommunicationGroup(group_name, group_ranks),
                             "Failed to create device communication group " + group_name);
    return true;
  }
  uint32_t local_group_rank = 0;
  uint32_t local_group_size = 0;
  RETURN_IF_FALSE_WITH_LOG(GetLocalGroupRankAndSize(group_ranks, &local_group_rank, &local_group_size),
                           "GetLocalGroupRankAndSize failed for group " + group_name);
  MS_EXCEPTION_IF_NULL(host_comm_lib_instance_);
  // Step 1: Create communication group on host side.
  RETURN_IF_FALSE_WITH_LOG(
    host_comm_lib_instance_->CreateCommunicationGroup(group_name, group_ranks, local_group_rank, local_group_size),
    "Failed to create host communication group" + group_name);

  // Step 2: Create communication group on device side.
  RETURN_IF_FALSE_WITH_LOG(
    device_comm_lib_instance_->CreateCommunicationGroup(group_name, group_ranks, local_group_rank, local_group_size),
    "Failed to create device communication group" + group_name);

  // Step 3: Generate device information of the root node.
  CommunicationGroupPtr group = device_comm_lib_instance_->GetGroup(group_name);
  MS_EXCEPTION_IF_NULL(group);
  size_t root_info_size = 0;
  void *root_info = group->GenerateRootInfo(&root_info_size);
  MS_EXCEPTION_IF_NULL(root_info);

  bool ret = false;
  // Step 4: Broadcast the device root information to all nodes on host side.
  while (!ret) {
    RETURN_IF_FALSE_WITH_LOG(host_comm_lib_instance_->BroadcastUniqueID(group_name, root_info_size, root_info),
                             "Broadcast for device root info failed on the host side.");
    ret = true;
    // In disaster recovery scenarios, it is necessary to ensure that the unique id obtained from the Scheduler is a
    // newly generated one.
    if (RecoveryContext::GetInstance()->enable_recovery()) {
      ret = CheckUniqueIDLatest(group_name, root_info_size, root_info);
      if (!ret) {
        // The time interval for querying latest unique id from scheduler: 3 second.
        constexpr uint32_t kWaitDuration = 3;
        std::this_thread::sleep_for(std::chrono::seconds(kWaitDuration));
      }
    }
  }

  // Step 5: Initialize communication group on the device side.
  std::function<bool()> init_device_comm_group_func = [&, this]() {
    MS_EXCEPTION_IF_NULL(device_ctx_);
    device_ctx_->Initialize();
    return group->Initialize(root_info);
  };
  MS_LOG(INFO) << "Begin initialize communication group on the device side.";

  // Timeout limit 600 seconds to wait finish initializing device communication group.
  const int64_t kTimeToWait = 600;
  // Initialize communication group on the device side in thread with timeout limit.
  ret = ExecuteFuncInThread(init_device_comm_group_func, kTimeToWait);
  if (!ret) {
    MS_LOG(ERROR) << "Failed to create comm group on device side for " << group_name;
  }
  MS_LOG(INFO) << "End initialize communication group on the device side.";
  return ret;
}

bool CollectiveManager::DestroyCommunicationGroup(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
  if (!need_host_collective_) {
    RETURN_IF_FALSE_WITH_LOG(device_comm_lib_instance_->DestroyDeviceCommunicationGroup(group_name),
                             "Failed to destroy device communication group " + group_name);
    return true;
  }
  MS_EXCEPTION_IF_NULL(host_comm_lib_instance_);
  RETURN_IF_FALSE_WITH_LOG(host_comm_lib_instance_->DestroyCommunicationGroup(group_name),
                           "Failed to destroy host communication group " + group_name);
  RETURN_IF_FALSE_WITH_LOG(device_comm_lib_instance_->DestroyCommunicationGroup(group_name),
                           "Failed to destroy device communication group " + group_name);
  return true;
}

uint32_t CollectiveManager::GetRankId(const std::string &group_name) {
  BY_PASS_SCHED_RANK_ID;
  MS_EXCEPTION_IF_NULL(comm_lib_instance_);
  return comm_lib_instance_->GetRankId(group_name);
}

uint32_t CollectiveManager::GetGroupSize(const std::string &group_name) {
  BY_PASS_SCHED_RANK_SIZE;
  MS_EXCEPTION_IF_NULL(comm_lib_instance_);
  return comm_lib_instance_->GetGroupSize(group_name);
}

uint32_t CollectiveManager::GetLocalRankId(const std::string &group_name) {
  BY_PASS_SCHED_RANK_ID;
  MS_EXCEPTION_IF_NULL(comm_lib_instance_);
  return comm_lib_instance_->GetLocalRankId(group_name);
}

uint32_t CollectiveManager::GetLocalGroupSize(const std::string &group_name) {
  BY_PASS_SCHED_RANK_SIZE;
  MS_EXCEPTION_IF_NULL(comm_lib_instance_);
  return comm_lib_instance_->GetLocalGroupSize(group_name);
}

uint32_t CollectiveManager::GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank) {
  BY_PASS_SCHED_RANK_ID;
  MS_EXCEPTION_IF_NULL(comm_lib_instance_);
  return comm_lib_instance_->GetWorldRankFromGroupRank(group_name, local_rank);
}

uint32_t CollectiveManager::GetGroupRankFromWorldRank(uint32_t global_rank, const std::string &group_name) {
  BY_PASS_SCHED_RANK_ID;
  MS_EXCEPTION_IF_NULL(comm_lib_instance_);
  return comm_lib_instance_->GetGroupRankFromWorldRank(global_rank, group_name);
}

bool CollectiveManager::Finalize() {
  if (!inited_.load() || finalized_.load()) {
    return true;
  }

  std::function<bool()> finalize_func = [&, this]() {
    if (need_host_collective_) {
      MS_EXCEPTION_IF_NULL(host_comm_lib_instance_);
      if (!host_comm_lib_instance_->Finalize()) {
        MS_LOG(WARNING) << "Failed to finalize device communication library.";
      }
    }

    MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
    if (!device_comm_lib_instance_->Finalize()) {
      MS_LOG(WARNING) << "Failed to finalize device communication library.";
    }

    inited_ = false;
    finalized_ = true;
    need_init_ = false;
    return true;
  };

  MS_LOG(INFO) << "Begin finalize collective manager.";

  // Timeout limit 5 seconds to wait to finish finalizing device communication group.
  const int64_t kTimeToWait = 5;
  // Finalize collective manager in thread with timeout limit.
  bool ret = ExecuteFuncInThread(finalize_func, kTimeToWait);

  MS_LOG(INFO) << "End finalize collective manager.";
  return ret;
}

void CollectiveManager::set_global_rank_id(uint32_t global_rank_id) { global_rank_id_ = global_rank_id; }

void CollectiveManager::set_global_rank_size(uint32_t global_rank_size) { global_rank_size_ = global_rank_size; }

uint32_t CollectiveManager::local_rank_id() const { return local_rank_id_; }

bool CollectiveManager::InitHostCommlib() {
  device::DeviceContextKey host_key = {"CPU", 0};
  host_ctx_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_ctx_);
  MS_EXCEPTION_IF_NULL(host_ctx_->device_res_manager_);
  RETURN_IF_FALSE_WITH_LOG(host_ctx_->device_res_manager_->LoadCollectiveCommLib(),
                           "Failed to load communication library on the host side.");

  host_comm_lib_instance_ = host_ctx_->device_res_manager_->collective_comm_lib();
  MS_EXCEPTION_IF_NULL(host_comm_lib_instance_);

  // For some communication libraries, global_rank_id_', 'global_rank_size_' should be set by caller, e.g., when using
  // MindSpore communication. For other communication libraries, global rank id and size is generated by itself, e.g.,
  // OpenMPI, and parameters 'global_rank_id_', 'global_rank_size_' will not be used.
  MS_LOG(INFO) << "Start initializing communication library on host side...";
  RETURN_IF_FALSE_WITH_LOG(host_comm_lib_instance_->Initialize(global_rank_id_, global_rank_size_),
                           "Failed to initialize communication library on host side.");

  if (!global_group_ranks_.empty()) {
    global_group_ranks_.clear();
  }

  // Reassign 'global_rank_id_' and 'global_rank_size_'. Generate global communication group ranks.
  global_rank_id_ = host_comm_lib_instance_->global_rank_id();
  global_rank_size_ = host_comm_lib_instance_->global_rank_size();
  for (uint32_t i = 0; i < global_rank_size_; i++) {
    global_group_ranks_.push_back(i);
  }

  // Create world group on host side for AllGather operation of host name while assigning local rank.
  host_global_group_name_ = host_comm_lib_instance_->global_group_name();
  RETURN_IF_FALSE_WITH_LOG(
    host_comm_lib_instance_->CreateCommunicationGroup(host_global_group_name_, global_group_ranks_, 0, 0),
    "Failed to create host communication group " + host_global_group_name_);
  MS_LOG(INFO) << "Communication library on host side is successfully initialized. Global rank id: " << global_rank_id_
               << ", global rank size: " << global_rank_size_;
  return true;
}

bool CollectiveManager::InitDeviceCommLib() {
  // If library on device side is not supported, replace it with host library.
  if (!device_lib_supported_) {
    device_type_ = kCPUDevice;
  }
  device::DeviceContextKey device_key = {device_type_, local_rank_id_};
  device_ctx_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(device_key);
  MS_EXCEPTION_IF_NULL(device_ctx_);
  // We can initialize device context now because device id(local_rank_id_) is already assigned.
  device_ctx_->Initialize();

  MS_EXCEPTION_IF_NULL(device_ctx_->device_res_manager_);
  RETURN_IF_FALSE_WITH_LOG(device_ctx_->device_res_manager_->LoadCollectiveCommLib(),
                           "Failed to load communication library on the device side.");
  device_comm_lib_instance_ = device_ctx_->device_res_manager_->collective_comm_lib();
  MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);

  MS_LOG(INFO) << "Start initializing communication library on device side...";
  RETURN_IF_FALSE_WITH_LOG(device_comm_lib_instance_->Initialize(global_rank_id_, global_rank_size_, local_rank_id_),
                           "Failed to initialize communication library on device side.");
  MS_LOG(INFO) << "Communication library on device side is successfully initialized.";
  return true;
}

bool CollectiveManager::AssignLocalRank() {
  char host_name[MAX_HOSTNAME_LEN] = {0};
#ifndef _WIN32
  if (gethostname(host_name, MAX_HOSTNAME_LEN) != 0) {
    MS_LOG(ERROR) << "Failed to get host name.";
    return false;
  }
#endif
  MS_LOG(INFO) << "Host name for rank " << global_rank_id_ << " is " << host_name;

  // Generate host name hash for every process. The host names of different physical machine should not be the same so
  // that local rank id won't repeat.
  size_t host_hash = std::hash<std::string>()(host_name);
  const uint32_t kGlobalRankSize = global_rank_size_;
  all_host_hashs_.resize(kGlobalRankSize);
  if (global_rank_id_ >= global_rank_size_) {
    MS_LOG(ERROR) << "The global rank id " << global_rank_id_ << " should be less than global rank size "
                  << global_rank_size_;
    return false;
  }
  all_host_hashs_[global_rank_id_] = host_hash;
  // some case, call init("hccl"), though is one card case and DEVICE_ID is set by user.
  if (global_rank_size_ <= 1) {
    local_rank_id_ = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    return true;
  }
  MS_EXCEPTION_IF_NULL(host_comm_lib_instance_);
  RETURN_IF_FALSE_WITH_LOG(host_comm_lib_instance_->AllGatherHostHashName(host_hash, &all_host_hashs_),
                           "AllGather for host names failed.");

  // Accumulate rank id.
  // In disaster recovery scenario, this function will enter multiple times when the network is reconfigured, so old
  // local rank id need to be cleaned.
  std::vector<uint32_t> world_ranks(global_rank_size_);
  std::iota(world_ranks.begin(), world_ranks.end(), 0);
  uint32_t local_group_size = 0;
  RETURN_IF_FALSE_WITH_LOG(GetLocalGroupRankAndSize(world_ranks, &local_rank_id_, &local_group_size),
                           "GetLocalGroupRankAndSize for world group failed.");
  host_comm_lib_instance_->SetLocalGroupRank(host_comm_lib_instance_->global_group_name(), local_rank_id_);
  host_comm_lib_instance_->SetLocalGroupSize(host_comm_lib_instance_->global_group_name(), local_group_size);
  // No need to reset device_id if library on device side is not supported, e.g., ascend.
  if (device_lib_supported_) {
    MsContext::GetInstance()->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, local_rank_id_);
    MS_LOG(INFO) << "The local rank id assigned for this process is " << local_rank_id_
                 << ". device_id of ms_context is set.";
    common::SetEnv("RANK_ID", std::to_string(global_rank_id_).c_str());
    common::SetEnv("DEVICE_ID", std::to_string(local_rank_id_).c_str());
    common::SetEnv("RANK_SIZE", std::to_string(global_rank_size_).c_str());
  }

  return true;
}
}  // namespace collective
}  // namespace distributed
}  // namespace mindspore
