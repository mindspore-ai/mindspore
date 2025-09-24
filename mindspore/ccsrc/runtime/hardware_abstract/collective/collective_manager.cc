/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "include/backend/distributed/collective/collective_manager.h"
#include <algorithm>
#include <string>
#include <iostream>
#include <map>
#include <sstream>
#include <numeric>
#include <vector>
#include <functional>
#include <csignal>
#include <future>
#include <memory>
#include <cstdint>
#include "utils/anf_utils.h"
#include "utils/ms_context.h"
#include "utils/device_manager_conf.h"
#include "utils/distributed_meta.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"
#include "include/common/callback.h"
#include "runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "runtime/hardware_abstract/collective/dummy_collective_communication_lib.h"
#include "utils/ms_exception.h"
#include "tools/profiler/mstx/mstx_impl.h"
#include "pybind11/pybind11.h"

namespace mindspore {
namespace distributed {
namespace collective {
namespace {
// pipeline parallel group name prefix
const size_t kSizeFour = 4;
const char kPlatFormMccl[] = "mccl";
const char kPipelineGroupNamePrefix[] = "pp-";
}  // namespace

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
      global_rank_size_(1),
      local_rank_size_(1),
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
  inited_ = false;
  finalized_ = true;
  host_ctx_ = nullptr;
  device_ctx_ = nullptr;
  host_comm_lib_instance_ = nullptr;
  device_comm_lib_instance_ = nullptr;
  comm_lib_instance_ = nullptr;
  group_infos_.clear();
  inited_groups_.clear();
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
// We also need to log the functionality of the function.
bool ExecuteFuncInThread(const std::function<bool()> &func, const int64_t timeout, const std::string &func_name,
                         const std::string &functionality) {
  bool execute_success = false;
  bool execute_fail = false;
  std::mutex exec_ret_mutex;
  std::condition_variable thread_blocker;

  std::unique_ptr<std::thread> executive_thread = std::make_unique<std::thread>([&] {
    if (!func()) {
      MS_LOG(ERROR) << "Failed to execute function: " << func_name << " " << functionality
                    << ". Please check error log above.";
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
    MS_LOG(ERROR) << "Execute function: " << func_name << " " << functionality << " timeout, this node id: " << node_id
                  << " exit process";
    (void)kill(getpid(), SIGTERM);
#endif
  }
  return execute_success;
}
}  // namespace

bool CollectiveManager::Initialize() {
  need_init_ = true;
  stop_init_comm_ = false;
  if (inited_ && !need_reinit_) {
    return true;
  }

  need_host_collective_ = common::UseHostCollective();
  std::string device_type = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  // need_host_collective_ means using rank_table to initialize collective communication, which is only supported by
  // Ascend. On other types of devices, exception should be thrown.
  if (device_type != kAscendDevice && !need_host_collective_) {
    MS_LOG(EXCEPTION) << kDetailedFailureReason;
  }

  MS_LOG(INFO) << "Start initializing collective communication for backend: " << device_type << "...";

  // Use dummy collective libs in simulation mode.
  if (!common::GetEnv(kSimulationLevel).empty()) {
    MS_LOG(WARNING) << "This is simulation mode with level " << common::GetEnv(kSimulationLevel)
                    << ". Process's RANK_ID: " << common::GetEnv("RANK_ID")
                    << ", RANK_SIZE: " << common::GetEnv("RANK_SIZE");

    return InitializeDummyCommLib();
  }

  // Initialize real collective libs.
  if (!need_host_collective_) {
    RETURN_IF_FALSE_WITH_LOG(InitDeviceCommLib(), "Failed to initialize device communication library.");
    comm_lib_instance_ = device_comm_lib_instance_;
  } else {
    // Step 1: Initialize host side collective communication.
    PROF_START(InitHostCommlib);
    RETURN_IF_FALSE_WITH_LOG(InitHostCommlib(), "Failed to initialize host communication library.");
    PROF_END(InitHostCommlib);
    comm_lib_instance_ = host_comm_lib_instance_;

    // Step 2, 3 and 4 are for device communication library. So if the training job is only launched on CPU, they will
    // not be necessary.
    // Step 2: Assign local rank id(device id) for this process.
    PROF_START(AssignLocalRank);
    RETURN_IF_FALSE_WITH_LOG(AssignLocalRank(), "Failed to assign local rank id.");
    PROF_END(AssignLocalRank);

    // Step 3: Initialize device side collective communication.
    PROF_START(InitDeviceBackend);
    RETURN_IF_FALSE_WITH_LOG(InitDeviceCommLib(), "Failed to initialize device communication library.");
    PROF_END(InitDeviceBackend);

    // Step 4: Create global communication group asynchronizely
    MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
    bool async = IsAsyncInitGlobalComm();
    GroupOptions config = {};
    config.async = async;
    auto group_name = device_comm_lib_instance_->global_group_name();
    PROF_START(CreateGlobalCommunicationGroup);
    RETURN_IF_FALSE_WITH_LOG(CreateCommunicationGroup(group_name, global_group_ranks_, config),
                             "Failed to create group " + group_name);
    if (async) {
      SubmitCreateDeviceCommTask(group_name);
    }
    PROF_END(CreateGlobalCommunicationGroup);
  }

  MS_LOG(INFO) << "End initializing collective communication for backend: " << device_type;
  inited_ = true;
  finalized_ = false;
  need_reinit_ = false;
  SetDistributedMeta();
  return true;
}

bool CollectiveManager::InitializeDummyCommLib() {
  dummy_comm_lib_instance_ = std::make_shared<device::DummyCollectiveCommunicationLib>();
  comm_lib_instance_ = dummy_comm_lib_instance_.get();
  MS_EXCEPTION_IF_NULL(comm_lib_instance_);
  // Get global rank id, global rank size and local rank(device id).
  if (!common::GetEnv(kEnvRankSize).empty()) {
    global_rank_size_ = LongToUint(std::strtol(common::GetEnv(kEnvRankSize).c_str(), nullptr, kDecimalBase));
  } else {
    global_rank_size_ = kDefaultRankSize;
  }
  if (!common::GetEnv(kEnvRankId).empty()) {
    global_rank_id_ = LongToUint(std::strtol(common::GetEnv(kEnvRankId).c_str(), nullptr, kDecimalBase));
  } else {
    global_rank_id_ = kDefaultRankId;
  }
  local_rank_id_ = global_rank_id_ % kDefaultLocalRankSize;

  RETURN_IF_FALSE_WITH_LOG(comm_lib_instance_->Initialize(global_rank_id_, global_rank_size_, local_rank_id_),
                           "Failed to initialize dummy communication library.");
  device_comm_lib_instance_ = comm_lib_instance_;
  for (uint32_t i = 0; i < global_rank_size_; i++) {
    global_group_ranks_.push_back(i);
  }
  MS_LOG(WARNING) << "Initializing dummy collective communication with rank size: " << global_rank_size_
                  << ", rank id: " << global_rank_id_ << ", local rank id: " << local_rank_id_
                  << ". Real rank size: 1.";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_type = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto kbk = ms_context->IsKByKExecutorMode();
  // If this is Ascend backend and uses host collective(OpenMPI or Dynamic Cluster/msrun), initialize dummy ascend
  // collective lib.
  if (!kbk && device_type == kAscendDevice) {
    MS_LOG(WARNING) << "Initialize dummy Ascend collective communication lib.";
    RETURN_IF_FALSE_WITH_LOG(InitDeviceCommLib(), "Failed to initialize dummy device communication library on Ascend.");

    MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
    // Create dummy device global communication group.
    auto group_name = device_comm_lib_instance_->global_group_name();
    RETURN_IF_FALSE_WITH_LOG(CreateCommunicationGroup(group_name, global_group_ranks_),
                             "Failed to create group " + group_name);
  }

  inited_ = true;
  finalized_ = false;
  need_reinit_ = false;
  SetDistributedMeta();
  return true;
}

bool CollectiveManager::FinalizeDummyCommLib() {
  std::string device_type = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (need_host_collective_ && device_type == kAscendDevice) {
    MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
    if (!device_comm_lib_instance_->Finalize()) {
      MS_LOG(WARNING) << "Failed to finalize dummy device communication library.";
    }
  }
  MS_EXCEPTION_IF_NULL(comm_lib_instance_);
  (void)comm_lib_instance_->Finalize();

  inited_ = false;
  finalized_ = true;
  need_init_ = false;
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
                                                 const std::vector<uint32_t> &group_ranks, const GroupOptions &config) {
  PROF_START(distributed_create_group);
  MS_LOG(INFO) << "Start to create communication group: " << group_name << " " << group_ranks
               << ", async: " << config.async;
  if (std::find(group_ranks.begin(), group_ranks.end(), global_rank_id_) == group_ranks.end()) {
    MS_LOG(WARNING) << "This rank: " << global_rank_id_ << " is not in the group ranks: " << group_ranks
                    << ". This may cause some exception when initializing the group.";
  }
  if (group_map_.count(group_name) == 0) {
    (void)group_infos_.emplace_back(std::make_pair(group_name, group_ranks));
  }
  group_map_[group_name] = group_ranks;

  // Create simulation communication group.
  if (!common::GetEnv(kSimulationLevel).empty()) {
    return CreateSimulationGroup(group_name, group_ranks);
  }

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
  PROF_START(CreateCommunicationGroupOnHostSide);
  RETURN_IF_FALSE_WITH_LOG(host_comm_lib_instance_->CreateCommunicationGroup(group_name, group_ranks, local_group_rank,
                                                                             local_group_size, config),
                           "Failed to create host communication group" + group_name);
  PROF_END(CreateCommunicationGroupOnHostSide);

  // Step 2: Create communication group on device side.
  PROF_START(CreateCommunicationGroupOnDeviceSide);
  RETURN_IF_FALSE_WITH_LOG(device_comm_lib_instance_->CreateCommunicationGroup(
                             group_name, group_ranks, local_group_rank, local_group_size, config),
                           "Failed to create device communication group" + group_name);
  PROF_END(CreateCommunicationGroupOnDeviceSide);

  // save pipeline parallel local rank for silent check
  if (group_name.find(kPipelineGroupNamePrefix) == 0) {
    constexpr char kSilentCheckSetPipelineStage[] = "SilentCheckSetPipelineStage";
    static const auto silent_check_set_pipeline_stage =
      callback::CommonCallback::GetInstance()
        .GetCallback<void, uint32_t, const string &, const std::vector<uint32_t> &, uint32_t>(
          kSilentCheckSetPipelineStage);
    if (silent_check_set_pipeline_stage) {
      silent_check_set_pipeline_stage(local_group_rank, group_name, group_ranks, local_group_size);
    }
  }

  if (config.async) {
    // If this is in async manner, it's user's duty to call SubmitCreateDeviceCommTask and join the result.
    MS_LOG(WARNING) << "This group's communicator is async created " << group_name;
    return true;
  } else {
    // To ensure the initialization order of async and sync created communicators, we invoke submit and wait methods
    // for sync ones.
    SubmitCreateDeviceCommTask(group_name);
    if (!WaitCommInitDone(group_name)) {
      MS_LOG(EXCEPTION) << "Failed to wait for communicator of " << group_name
                        << " init done. Please check ERROR log above.";
    }
  }

  PROF_END(distributed_create_group);
  return true;
}

bool CollectiveManager::DestroyDeviceSideCommunicationGroup(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
  if (!need_host_collective_ || !common::GetEnv(kSimulationLevel).empty()) {
    RETURN_IF_FALSE_WITH_LOG(device_comm_lib_instance_->DestroyDeviceCommunicationGroup(group_name),
                             "Failed to destroy device communication group " + group_name);
    return true;
  }
  RETURN_IF_FALSE_WITH_LOG(device_comm_lib_instance_->DestroyCommunicationGroup(group_name),
                           "Failed to destroy device communication group " + group_name);
  return true;
}

bool CollectiveManager::DestroyCommunicationGroup(const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
  if (!need_host_collective_ || !common::GetEnv(kSimulationLevel).empty()) {
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

void CollectiveManager::RemoveGroupInfoForARF(const std::string &group_name) {
  (void)group_infos_.erase(std::remove_if(group_infos_.begin(), group_infos_.end(),
                                          [group_name](const auto &p) { return p.first == group_name; }),
                           group_infos_.end());
}

std::string CollectiveManager::GetCommName(const std::string &group_name) {
  if (!WaitCommInitDone(group_name)) {
    MS_LOG(EXCEPTION) << "The communucation group " << group_name << " is not successfully initialized.";
  }
  if (!common::GetEnv(kSimulationLevel).empty()) {
    MS_EXCEPTION_IF_NULL(dummy_comm_lib_instance_);
    return dummy_comm_lib_instance_->CommName(group_name);
  }
  MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
  return device_comm_lib_instance_->CommName(group_name);
}

uint32_t CollectiveManager::GetRankId(const std::string &group_name) {
  BY_PASS_SCHED_RANK_ID;
  if (comm_lib_instance_ == nullptr) {
    MS_LOG(DEBUG) << "Distributed module is not initialized, return 0 as rank id.";
    return kIndex0;
  }
  return comm_lib_instance_->GetRankId(group_name);
}

uint32_t CollectiveManager::GetGroupSize(const std::string &group_name) {
  BY_PASS_SCHED_RANK_SIZE;
  if (comm_lib_instance_ == nullptr) {
    MS_LOG(DEBUG) << "Distributed module is not initialized, return 1 as group size.";
    return kSizeOne;
  }
  return comm_lib_instance_->GetGroupSize(group_name);
}

uint32_t CollectiveManager::GetLocalRankId(const std::string &group_name) {
  BY_PASS_SCHED_RANK_ID;
  if (comm_lib_instance_ == nullptr) {
    MS_LOG(DEBUG) << "Distributed module is not initialized, return 0 as local rank id.";
    return kIndex0;
  }
  return comm_lib_instance_->GetLocalRankId(group_name);
}

uint32_t CollectiveManager::GetLocalGroupSize(const std::string &group_name) {
  BY_PASS_SCHED_RANK_SIZE;
  if (comm_lib_instance_ == nullptr) {
    MS_LOG(DEBUG) << "Distributed module is not initialized, return 1 as group size.";
    return kSizeOne;
  }
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

std::vector<uint32_t> CollectiveManager::GetGroupRanks(const std::string &group_name) {
  const auto &group = comm_lib_instance_->GetGroup(group_name);
  if (group == nullptr) {
    MS_LOG(EXCEPTION) << "Group " << group_name << " doesn't include this rank " << global_rank_id_ << " process.";
  }
  return group->group_ranks();
}

bool CollectiveManager::Finalize() {
  if (!inited_.load() || finalized_.load()) {
    return true;
  }

  if (!common::GetEnv(kSimulationLevel).empty() || dummy_comm_lib_instance_ != nullptr) {
    return FinalizeDummyCommLib();
  }

  std::function<bool()> finalize_comm_lib_func = [&, this]() {
    // Ensure all comms are initialized to avoid exiting error.
    (void)WaitAllCommInitDone();
    stop_init_comm_ = true;
    task_queue_blocker_.notify_one();
    group_name_to_result_.clear();
    task_list_.clear();
    if (run_init_comm_task_thread_.joinable()) {
      run_init_comm_task_thread_.join();
    }

    if (need_host_collective_) {
      MS_EXCEPTION_IF_NULL(host_comm_lib_instance_);
      MS_LOG(INFO) << "Start finalizing host communication lib.";
      if (!host_comm_lib_instance_->Finalize()) {
        MS_LOG(WARNING) << "Failed to finalize device communication library.";
      }
      MS_LOG(INFO) << "End finalizing host communication lib.";
    }

    MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);

    MS_LOG(INFO) << "Start finalizing device communication lib.";
    if (!device_comm_lib_instance_->Finalize()) {
      MS_LOG(WARNING) << "Failed to finalize device communication library.";
    }
    MS_LOG(INFO) << "End finalizing device communication lib.";

    inited_ = false;
    finalized_ = true;
    need_init_ = false;
    host_ctx_ = nullptr;
    device_ctx_ = nullptr;
    host_comm_lib_instance_ = nullptr;
    device_comm_lib_instance_ = nullptr;
    comm_lib_instance_ = nullptr;
    local_rank_id_ = 0;
    local_rank_size_ = 1;
    group_infos_.clear();
    inited_groups_.clear();
    group_map_.clear();
    while (!init_comm_task_queue_.empty()) {
      init_comm_task_queue_.pop();
    }
    return true;
  };

  MS_LOG(INFO) << "Begin finalizing collective manager.";

  // Timeout limit 30 seconds to wait to finish finalizing device communication group.
  const int64_t kTimeToWait = 30;
  // Finalize collective manager in thread with timeout limit.
  bool ret = ExecuteFuncInThread(finalize_comm_lib_func, kTimeToWait, "finalize_comm_lib_func",
                                 "to destroy communication groups and finalize communication lib");

  MS_LOG(INFO) << "End finalizing collective manager.";
  return ret;
}

void CollectiveManager::set_global_rank_id(uint32_t global_rank_id) { global_rank_id_ = global_rank_id; }

void CollectiveManager::set_global_rank_size(uint32_t global_rank_size) { global_rank_size_ = global_rank_size; }

uint32_t CollectiveManager::global_rank_id() const { return global_rank_id_; }

uint32_t CollectiveManager::global_rank_size() const { return global_rank_size_; }

uint32_t CollectiveManager::local_rank_id() const { return local_rank_id_; }

uint32_t CollectiveManager::local_rank_size() const { return local_rank_size_; }

bool CollectiveManager::InitHostCommlib() {
  device::DeviceContextKey host_key = {device::DeviceType::kCPU, 0};
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
  std::string device_type = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  uint32_t device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  // If library on device side is not supported, replace it with host library.
  if (!device_lib_supported_) {
    device_type = kCPUDevice;
  }
  device::DeviceContextKey device_key = {device::GetDeviceTypeByName(device_type), device_id};
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
  RETURN_IF_FALSE_WITH_LOG(device_comm_lib_instance_->Initialize(global_rank_id_, global_rank_size_, device_id),
                           "Failed to initialize communication library on device side.");
  if (DistributedMeta::GetInstance()->enable_cross_cluster()) {
    MS_LOG(WARNING) << "Set helper for CCOOL collective communication.";
    device_comm_lib_instance_->SetHelperCommLib(host_comm_lib_instance_);
  }
  MS_LOG(INFO) << "Communication library on device side is successfully initialized.";
  return true;
}

std::map<int, std::vector<int>> ParseRangeString(const std::string &input) {
  std::map<int, std::vector<int>> result;
  std::string content = input.substr(1, input.size() - 2);
  std::vector<std::string> parts;
  std::stringstream ss(content);
  std::string part;
  while (std::getline(ss, part, '|')) {
    part.erase(0, part.find_first_not_of(" \t"));
    part.erase(part.find_last_not_of(" \t") + 1);
    parts.push_back(part);
  }

  for (const auto &p : parts) {
    size_t colon_pos = p.find(':');
    int key = std::stoi(p.substr(0, colon_pos));

    std::string range = p.substr(colon_pos + 1);
    size_t tilde_pos = range.find('~');
    int start = std::stoi(range.substr(0, tilde_pos));
    int end = std::stoi(range.substr(tilde_pos + 1));

    std::vector<int> numbers;
    for (int i = start; i <= end; ++i) {
      numbers.push_back(i);
    }
    result[key] = numbers;
  }
  return result;
}

std::map<std::string, std::vector<int>> ParseModuleString(const std::string &input) {
  std::map<std::string, std::vector<int>> result;
  std::string content = input.substr(1, input.size() - 2);
  size_t pipe_pos = content.find('|');
  std::vector<std::string> parts;
  if (pipe_pos == std::string::npos) {
    parts.push_back(content);
  } else {
    std::stringstream ss(content);
    std::string part;
    while (std::getline(ss, part, '|')) {
      part.erase(0, part.find_first_not_of(" \t"));
      part.erase(part.find_last_not_of(" \t") + 1);
      parts.push_back(part);
    }
  }

  for (const auto &p : parts) {
    size_t colon_pos = p.find(':');
    std::string key = p.substr(0, colon_pos);
    std::string values_str = p.substr(colon_pos + 1);

    std::vector<int> numbers;
    std::stringstream values_ss(values_str);
    std::string num_str;
    while (std::getline(values_ss, num_str, '-')) {
      numbers.push_back(std::stoi(num_str));
    }
    result[key] = numbers;
  }
  return result;
}

std::vector<int> GetSubList(const std::vector<int> &listA, const std::vector<int> &indexListB) {
  std::vector<int> result;
  for (int index : indexListB) {
    if (index >= 0 && index < static_cast<int>(listA.size())) {
      result.push_back(listA[index]);
    }
  }
  return result;
}

std::string VectorToString(const std::vector<int> &vec) {
  std::ostringstream oss;
  for (size_t i = 0; i < vec.size(); ++i) {
    oss << vec[i];
    if (i != vec.size() - 1) {
      oss << ", ";
    }
  }
  return oss.str();
}

void ParseEnvCpuAffinity(const uint32_t &global_rank_id, const uint32_t &local_rank_id) {
  const auto &cpu_affinity_list = common::GetConfigValue(runtime::kRuntimeConf, runtime::kRuntimeCpuAffinityList);
  const auto &cpu_affinity_module = common::GetConfigValue(runtime::kRuntimeConf, runtime::kRuntimeCpuAffinityMoudule);
  const auto &actor_thread_fix_bind =
    common::GetConfigValue(runtime::kRuntimeConf, runtime::kRuntimeActorThreadFixBind);
  std::vector<int> rank_core_reserved;
  if (cpu_affinity_list.empty()) {
    return;
  }

  MS_LOG(INFO) << "Got cpu_affinity_list: " << cpu_affinity_list;
  std::string env_visible_device = common::GetEnv("ASCEND_RT_VISIBLE_DEVICES");
  uint32_t physical_device_id;
  if (env_visible_device.empty()) {
    physical_device_id = local_rank_id;
  } else {
    std::vector<uint32_t> list_visible_device;
    std::stringstream ss(env_visible_device);
    std::string item;
    while (std::getline(ss, item, ',')) {
      list_visible_device.push_back(std::stoi(item));
    }
    std::sort(list_visible_device.begin(), list_visible_device.end());
    physical_device_id = list_visible_device[local_rank_id];
  }

  auto rangeMap = ParseRangeString(cpu_affinity_list);
  rank_core_reserved = rangeMap[physical_device_id];

  std::map<std::string, std::vector<int>> moduleMap;
  if (!cpu_affinity_module.empty()) {
    moduleMap = ParseModuleString(cpu_affinity_module);
  } else {
    moduleMap = {{"runtime", {0, 1, 2, 3, 4}}, {"minddata", {5, 6, 7, 8, 9, 10, 11, 12}}};
  }

  auto it = moduleMap.find("runtime");
  if (it != moduleMap.end()) {
    std::vector<int> runtime_idx_list = it->second;
    std::vector<int> runtime_core_list = GetSubList(rank_core_reserved, runtime_idx_list);
    common::SetEnv("CONFIG_BIND_RUNTIME_LIST", VectorToString(runtime_core_list).c_str());
    common::SetEnv("ACTOR_THREAD_FIX_BIND", actor_thread_fix_bind.c_str());
  }

  it = moduleMap.find("minddata");
  if (it != moduleMap.end()) {
    std::vector<int> data_idx_list = it->second;
    std::vector<int> data_core_list = GetSubList(rank_core_reserved, data_idx_list);
    common::SetEnv("CONFIG_BIND_MINDDATA_LIST", VectorToString(data_core_list).c_str());
  }

  MS_LOG(WARNING) << "Core reserved for global rank[" << global_rank_id << "], local rank[" << local_rank_id << "] is "
                  << rank_core_reserved << ", set env 'CONFIG_BIND_RUNTIME_LIST' to "
                  << common::GetEnv("CONFIG_BIND_RUNTIME_LIST") << ", set env 'CONFIG_BIND_MINDDATA_LIST' to "
                  << common::GetEnv("CONFIG_BIND_MINDDATA_LIST");
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
  MS_LOG(INFO) << "Successfully get all nodes' hostname.";

  // Accumulate rank id.
  std::vector<uint32_t> world_ranks(global_rank_size_);
  std::iota(world_ranks.begin(), world_ranks.end(), 0);
  uint32_t local_group_size = 0;
  RETURN_IF_FALSE_WITH_LOG(GetLocalGroupRankAndSize(world_ranks, &local_rank_id_, &local_group_size),
                           "GetLocalGroupRankAndSize for world group failed.");

  // Need to reset local_rank_id_(device id) if it's set by user.
  SetDeviceIDEnvByRuntimeConf();
  host_comm_lib_instance_->SetLocalGroupRank(host_comm_lib_instance_->global_group_name(), local_rank_id_);
  host_comm_lib_instance_->SetLocalGroupSize(host_comm_lib_instance_->global_group_name(), local_group_size);
  local_rank_size_ = local_group_size;

  MS_LOG(INFO) << "The local rank id assigned for this process is " << local_rank_id_;
  MS_LOG(INFO) << "The env 'DEVICE_ID' assigned for this process is: " << common::GetEnv("DEVICE_ID");
  common::SetEnv("RANK_ID", std::to_string(global_rank_id_).c_str());
  common::SetEnv("RANK_SIZE", std::to_string(global_rank_size_).c_str());
  DeviceManagerConf::GetInstance()->distributed_refresh_device_id(local_rank_id_);
  // When starting with msrun and adding argument '--rank_table_file', device_id of ms_context will be set from env
  // "DEVICE_ID"; here env "DEVICE_ID" is not equal to "local_rank_id_".
  if (!common::GetEnv("RANK_TABLE_FILE").empty()) {
    if (common::GetEnv("DEVICE_ID").empty() || !common::IsStrNumeric(common::GetEnv("DEVICE_ID"))) {
      MS_LOG(EXCEPTION)
        << "Launching distributed job using dynamic cluster or OpenMPI, but the deive id is not imported when set env "
           "'RANK_TABLE_FILE'. Please use msrun startup method to rearrange rank ids based on rank table file or do "
           "not set env 'RANK_TABLE_FILE' to arrange rank ids by default.";
    }
    MsContext::GetInstance()->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, std::stoi(common::GetEnv("DEVICE_ID")));
    MS_LOG(INFO) << "The device_id of ms_context is set to env DEVICE_ID [" << std::stoi(common::GetEnv("DEVICE_ID"))
                 << "].";
  } else {
    MsContext::GetInstance()->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, local_rank_id_);
    common::SetEnv("DEVICE_ID", std::to_string(local_rank_id_).c_str());
    MS_LOG(INFO) << "The device_id of ms_context is set to local rank id [" << local_rank_id_ << "].";
  }

  ParseEnvCpuAffinity(global_rank_id_, local_rank_id_);
  return true;
}

bool CollectiveManager::CreateSimulationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks) {
  // Set local group size to 8 in simulation mode.
  uint32_t local_rank_size = kDefaultLocalRankSize;
  uint32_t local_rank = global_rank_id_ % local_rank_size;
  MS_LOG(WARNING) << "Create dummy communication group with group name: " << group_name
                  << ", group ranks: " << group_ranks << ". Real group size: 1.";
  RETURN_IF_FALSE_WITH_LOG(
    dummy_comm_lib_instance_->CreateCommunicationGroup(group_name, group_ranks, local_rank, local_rank_size),
    "Failed to create dummy communication group " + group_name);

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string device_type = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto kbk = ms_context->IsKByKExecutorMode();
  // If this is Ascend backend and uses host collective(OpenMPI or Dynamic Cluster/msrun), initialize real HCCL
  // communicator through dummy Ascend collective communication lib.
  if (!kbk && device_type == kAscendDevice) {
    MS_LOG(WARNING) << "Create Ascend communication group with group name: " << group_name
                    << ", group ranks: " << group_ranks
                    << ". Real HCCL communicator will be initialized with group size 1.";
    RETURN_IF_FALSE_WITH_LOG(
      device_comm_lib_instance_->CreateCommunicationGroup(group_name, group_ranks, local_rank, local_rank_size),
      "Failed to create dummy device communication group " + group_name);

    CommunicationGroupPtr group = device_comm_lib_instance_->GetGroup(group_name);
    MS_EXCEPTION_IF_NULL(group);
    size_t root_info_size = 0;
    void *root_info = group->GenerateRootInfo(&root_info_size);
    MS_EXCEPTION_IF_NULL(device_ctx_);
    device_ctx_->Initialize();
    auto ret = group->Initialize(root_info);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to create comm group on device side for " << group_name;
    }
  }
  return true;
}

int64_t CollectiveManager::GetCommunicatorInitTimeout() {
  // The default timeout is 600 seconds.
  int64_t default_comm_init_timeout = 600;
  std::string device_type = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device_type == kAscendDevice) {
    std::string str_comm_init_timeout = common::GetEnv("HCCL_CONNECT_TIMEOUT");
    MS_LOG(INFO) << "HCCL_CONNECT_TIMEOUT is " << str_comm_init_timeout << " seconds.";

    // For hccl, we wait until hccl api timeout. So we return ten times of HCCL_CONNECT_TIMEOUT on the host side.
    uint32_t multiple_time = 10;
    return str_comm_init_timeout.empty() ? default_comm_init_timeout : std::stoi(str_comm_init_timeout) * multiple_time;
  }
  return default_comm_init_timeout;
}

bool CollectiveManager::ResumeHcclComm() {
  MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
  if (!device_comm_lib_instance_->ResumeHcclComm()) {
    MS_LOG(EXCEPTION) << "Failed to resume comm group.";
  }
  MS_LOG(INFO) << "Resume hccl comm, and clear force stop state.";
  UCEException::GetInstance().set_force_stop_flag(false);
  return true;
}

bool CollectiveManager::CreateDeviceCommunicator(const std::string &group_name, const int32_t buffsize) {
  MS_LOG(INFO) << "Create device communicator for " << group_name;
  MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
  CommunicationGroupPtr group = device_comm_lib_instance_->GetGroup(group_name);
  bool is_mccl = group_name.compare(kIndex0, kSizeFour, kPlatFormMccl) == 0;
  if (is_mccl) {
    group = host_comm_lib_instance_->GetGroup(group_name);
    MS_LOG(INFO) << "Begin to Create mccl group " << group_name;
  }
  MS_EXCEPTION_IF_NULL(group);

  // Step 1: Generate and broadcast device information of the root node (required for NPU backend without rank table).
  static bool use_ranktable = !common::GetEnv("RANK_TABLE_FILE").empty();
  bool use_cross_cluster = DistributedMeta::GetInstance()->enable_cross_cluster();
  bool ret = false;
  void *root_info;
  if (!is_mccl && (!use_ranktable || use_cross_cluster)) {
    size_t root_info_size = 0;
    PROF_START(GenerateRootInfo);
    root_info = group->GenerateRootInfo(&root_info_size);
    PROF_END(GenerateRootInfo);
    if (root_info == nullptr) {
      MS_LOG(EXCEPTION)
        << "Root info pointer is nullptr. If you are using HCCL, there maybe port conflict in "
           "communication collective library. Please execute 'export HCCL_IF_BASE_PORT=<new port>' to use "
           "different port, or 'npu-smi info' to check if there's residual processes.";
    }

    // Broadcast the device root information to all nodes on host side. CCOOL broadcasts inner comm
    // group's rootinfo within func 'GenerateRootInfo', do not need to broadcast outer comm group's rootinfo.
    if (!use_cross_cluster) {
      PROF_START(BroadcastUniqueID);
      while (!ret) {
        RETURN_IF_FALSE_WITH_LOG(host_comm_lib_instance_->BroadcastUniqueID(group_name, root_info_size, root_info),
                                 "Broadcast for device root info failed on the host side.");
        ret = true;
        MS_LOG(INFO) << "Successfully send/fetch unqiueid for communication group " << group_name;
      }
      PROF_END(BroadcastUniqueID);
    }
  }

  // Step 2: Initialize communication group on the device side.
  std::function<bool()> init_device_comm_group_func = [&, this]() {
    MS_EXCEPTION_IF_NULL(device_ctx_);
    device_ctx_->Initialize();
    return group->Initialize(root_info);
  };
  MS_LOG(WARNING) << "Begin initialize communication group on the device side: " << group_name;
  // Timeout limit in seconds to wait finish initializing device communication group.
  int64_t comm_init_timout = GetCommunicatorInitTimeout();
  PROF_START(InitDeviceCommunicator);
  // Initialize communication group on the device side in thread with timeout limit.
  ret = ExecuteFuncInThread(init_device_comm_group_func, comm_init_timout, "init_device_comm_group_func",
                            "to initialize communicator for group " + group_name);
  PROF_END(InitDeviceCommunicator);
  if (!ret) {
    MS_LOG(ERROR) << "Failed to create comm group on device side for " << group_name;
  }
  MS_LOG(WARNING) << "End initialize communication group on the device side: " << group_name;
  return ret;
}

void CollectiveManager::ClearUniqueID(const std::string &group_name) {
  if (!need_host_collective_) {
    return;
  }
  MS_EXCEPTION_IF_NULL(host_comm_lib_instance_);
  MS_LOG(INFO) << "Start clearing unique id for group: " << group_name;
  host_comm_lib_instance_->ClearUniqueID(group_name);
  MS_LOG(INFO) << "End clearing unique id for group: " << group_name;
}

bool CollectiveManager::CommSwitchNic(const std::vector<uint32_t> &global_ranks, const std::vector<bool> &use_backup) {
  MS_EXCEPTION_IF_NULL(device_comm_lib_instance_);
  MS_LOG(INFO) << "Switch communication network interface card, the global ranks: " << global_ranks
               << ", the use backup: " << use_backup;
  return device_comm_lib_instance_->CommSwitchNic(global_ranks, use_backup);
}

bool CollectiveManager::IsAsyncInitGlobalComm() {
  // Use async manner when three conditions below are satisfied.
  // 1.Runtime dev config is not set to false.
  // 2.This is graph mode.
  // 3.This is NOT using rank table. Ranktable time cost is little and has temporal sequence problems, so use sync
  // manner.
  // 4.This is NOT simulation.
  // 5.This NOT using mpirun. OpenMPI has hanging issues when invoking its interfaces in multiple threads.
  // 6.This is Ascend platform. For early version, we only support to create global comm group for Ascend by default.
  // Otherwise user should control whether using async manner.
  const auto &is_async_str = common::GetConfigValue(runtime::kRuntimeConf, runtime::kRuntimeAsyncInitComm);
  bool async_conf = (is_async_str != "false" && is_async_str != "False");
  bool is_graph = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode;
  bool use_rank_table = !common::GetEnv("RANK_TABLE_FILE").empty();
  bool simulation = !common::GetEnv(kSimulationLevel).empty();
  bool use_mpi = common::UseMPI();
  bool is_ascend = (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  bool result = (async_conf && is_graph && !use_rank_table && !simulation && !use_mpi && is_ascend);
  MS_LOG(INFO) << "Async initialize global comm: " << result << ". async_conf: " << async_conf
               << ", is_graph: " << is_graph << ", use_rank_table: " << use_rank_table << ", simulation: " << simulation
               << ", use_mpi: " << use_mpi << ", is_ascend: " << is_ascend;
  return result;
}

bool CollectiveManager::WaitAllCommInitDone() {
  if (!common::GetEnv(kSimulationLevel).empty()) {
    MS_LOG(INFO) << "This is dry run, no need to wait for communciators init done";
    return true;
  }
  // This is a shared lock so the cost is little, but we need to guarantee there's no thread-safe issue.
  // Because WaitCommInitDone also acquires this lock, we just copy this task_list_ and release the lock immediately to
  // avoid dead lock.
  std::unique_lock<std::mutex> lock(task_queue_mutex_);
  auto group_name_list = task_list_;
  lock.unlock();
  for (const std::string &group_name : group_name_list) {
    if (!WaitCommInitDone(group_name)) {
      MsException::Instance().CheckException();
      return false;
    }
  }
  MS_LOG(INFO) << "All device communictor is initialized. You can launch communication operators after this step.";
  return true;
}

bool CollectiveManager::WaitCommInitDone(const std::string &group_name) {
  if (!common::GetEnv(kSimulationLevel).empty()) {
    MS_LOG(INFO) << "This is dry run, no need to wait for communciator init done for " << group_name;
    return true;
  }

  // When the memory tool msleaks is enabled to capture memory call stacks, it needs to acquire the GIL.
  // If the current thread already holds the GIL and is waiting for other threads, and another thread triggers call
  // stack capture, the GIL must be temporarily released at this point to avoid a deadlock.
  std::optional<py::gil_scoped_release> release;
#ifdef SYSTEM_ENV_POSIX
  if (profiler::MstxImpl::GetInstance().IsMsleaksEnable()) {
    if (Py_IsInitialized() && PyGILState_Check()) {
      release.emplace();
    }
  }
#endif
  std::unique_lock<std::mutex> lock(task_queue_mutex_);
  // If the task is not submitted, throw exception.
  if (!std::any_of(task_list_.begin(), task_list_.end(),
                   [&group_name](const auto &name) { return name == group_name; })) {
    MS_LOG(EXCEPTION)
      << "The HCCL group " << group_name << " does not existed. The group " << group_name
      << " init task may not be submitted yet. Please check if SubmitCreateDeviceCommTask is invoked for this group";
  }
  lock.unlock();

  MS_LOG(DEBUG) << "Start waiting for communciator of " << group_name << " to be done...";
  std::unique_lock<std::mutex> result_lock(init_result_mutex_);
  // This will always unblock because there's timeout window for every device communicator.
  result_blocker_.wait(result_lock, [&]() { return group_name_to_result_.count(group_name) != 0; });
  if (!group_name_to_result_[group_name].first) {
    MS_LOG(EXCEPTION) << "Communicator of group " << group_name
                      << " inited: failed. Result: " << group_name_to_result_[group_name].second;
  }
  MS_LOG(DEBUG) << "Communicator of group " << group_name << " inited: success.";

  return true;
}

void CollectiveManager::SubmitCreateDeviceCommTask(const std::string &group_name, const int32_t buffsize) {
  if (!run_init_comm_task_thread_.joinable()) {
    run_init_comm_task_thread_ = std::thread(&CollectiveManager::RunInitCommTasks, this);
    MS_LOG(INFO) << "Launch init comm thread.";
  }
  std::unique_lock<std::mutex> lock(task_queue_mutex_);
  init_comm_task_queue_.push(std::make_pair(group_name, buffsize));
  task_list_.push_back(group_name);
  task_queue_blocker_.notify_one();
  MS_LOG(INFO) << "Submit init communicator task for " << group_name
               << ". Call 'WaitCommInitDone' later to wait initialization to be done.";
}

void CollectiveManager::RunInitCommTasks() {
  while (!stop_init_comm_) {
    std::unique_lock<std::mutex> lock(task_queue_mutex_);
    // Block until there's task or thread is stopped.
    task_queue_blocker_.wait(lock, [&]() { return !init_comm_task_queue_.empty() || stop_init_comm_; });

    if (stop_init_comm_) {
      MS_LOG(INFO) << "Initialize communciator thread is stopped.";
      break;
    }

    // When execute to this code, the queue should not be empty.
    auto task_element = init_comm_task_queue_.front();
    init_comm_task_queue_.pop();
    // Release the lock to avoid blocking when submitting tasks in asynchronize process.
    lock.unlock();

    std::string group_name = task_element.first;
    int32_t buffsize = task_element.second;
    try {
      MS_LOG(INFO) << "Create device communicator in thread for group: " << group_name;
      if (!CreateDeviceCommunicator(group_name, buffsize)) {
        MS_LOG(EXCEPTION) << "Failed to init communicator asynchronizely for group " << group_name
                          << ". Please check ERROR log.";
      }
      std::unique_lock<std::mutex> result_lock(init_result_mutex_);
      group_name_to_result_[group_name] = std::make_pair(true, "");
      result_blocker_.notify_one();
    } catch (std::exception &e) {
      std::unique_lock<std::mutex> result_lock(init_result_mutex_);
      std::string err_info = "Init communicator for group " + group_name + " exception info: " + e.what();
      group_name_to_result_[group_name] = std::make_pair(false, err_info);
      result_blocker_.notify_one();
      MsException::Instance().SetException();
      // If fail to initialize any communicator, exit immediately and stop the thread.
      stop_init_comm_ = true;
      continue;
    }
  }
}

void CollectiveManager::SetDeviceIDEnvByRuntimeConf() {
  if (!DeviceManagerConf::GetInstance()->is_default_device_id()) {
    uint32_t device_id = DeviceManagerConf::GetInstance()->device_id();
    MS_LOG(INFO) << "Runtime config device id is set by mindspore.set_device to " << device_id;
    local_rank_id_ = device_id;
    common::SetEnv("DEVICE_ID", std::to_string(device_id).c_str());
  }
}

void CollectiveManager::SetDistributedMeta() {
  MS_LOG(DEBUG) << "Set distributed meta data to core module: global rank id: " << global_rank_id_
                << ", global rank size: " << global_rank_size_ << ", local rank id: " << local_rank_id_
                << ", local rank size: " << local_rank_size_;
  DistributedMeta::GetInstance()->set_initialized();
  DistributedMeta::GetInstance()->set_global_rank_id(global_rank_id_);
  DistributedMeta::GetInstance()->set_global_rank_size(global_rank_size_);
  DistributedMeta::GetInstance()->set_local_rank_id(local_rank_id_);
  DistributedMeta::GetInstance()->set_local_rank_size(local_rank_size_);
}

void CollectiveManager::CacheInitedGroups(const std::string &name) {
  MS_LOG(INFO) << "Cache inited group: " << name;
  std::unique_lock<std::mutex> result_lock(cache_mutes_);
  (void)inited_groups_.emplace_back(name);
  MS_LOG(INFO) << "Cache inited group: " << name << " end.";
}

void CollectiveManager::ClearCacheInitedGroups() { inited_groups_.clear(); }

size_t CollectiveManager::InitedGroupSize() {
  std::unique_lock<std::mutex> result_lock(cache_mutes_);
  return inited_groups_.size();
}
}  // namespace collective
}  // namespace distributed
}  // namespace mindspore
