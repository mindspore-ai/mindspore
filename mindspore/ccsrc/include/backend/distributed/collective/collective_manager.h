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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_COLLECTIVE_MANAGER_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_COLLECTIVE_MANAGER_H_

#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <atomic>
#include <utility>
#include <thread>
#include <functional>
#include <mutex>
#include <unordered_map>
#include "utils/ms_utils.h"
#include "include/backend/distributed/constants.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/cluster/cluster_context.h"
#else
#include "include/backend/distributed/cluster/dummy_cluster_context.h"
#endif
#include "runtime/hardware_abstract/collective/collective_communication_lib.h"
#include "runtime/hardware_abstract/collective/communication_group.h"
#include "runtime/hardware_abstract/device_context/device_context_manager.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace distributed {
namespace collective {
using DeviceContext = device::DeviceContext;
using DeviceContextKey = device::DeviceContextKey;
using DeviceContextManager = device::DeviceContextManager;
using CollectiveCommunicationLib = device::CollectiveCommunicationLib;
using CommunicationGroupPtr = device::CommunicationGroupPtr;
using GroupOptions = device::GroupOptions;

// Data types for creating communication groups function.

// Creating communicator function object, refers to CollectiveManager::CreateDeviceCommunicator.
using CreateDeviceCommFunc = std::function<bool(const std::string &)>;
// FIFO queue of communicator initialization tasks. The element is a pair and group name and its communicator's buffer
// size.
using InitCommTaskQueue = std::queue<std::pair<std::string, int32_t>>;
// Group list to be initialized in order. The element will not be deleted.
using InitCommTaskList = std::vector<std::string>;
// Initialization task results for each group.
// Key is group name. Value is a pair of result(true/false) and error info(if false).
using GroupToResultMap = std::unordered_map<std::string, std::pair<bool, std::string>>;

// Interval of initializing each communicator in queue is 300 milliseconds.
const uint32_t kInitCommInterval = 300;

// The collective communication API.
// MindSpore uses OpenMPI on CPU, NCCL on GPU, HCCL on Ascend, to achieve distributed training.
// Besides, MindSpore also has its own communication library which is implemented on the CPU side.
class RUNTIME_HARDWARE_EXPORT CollectiveManager {
 public:
  ~CollectiveManager();
  DISABLE_COPY_AND_ASSIGN(CollectiveManager);
  static std::shared_ptr<CollectiveManager> instance();

  // Initialize the collective communication for distributed training. The backend type is read from MindSpore context.
  bool Initialize(std::string group_name);

  // Finalize the collective communication.
  bool Finalize();

  // Create communication group.
  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks,
                                const GroupOptions &config = {});

  // Destroy the communication group.
  bool DestroyCommunicationGroup(const std::string &group_name);

  void RemoveGroupInfoForARF(const std::string &group_name);

  // Destroy device communication group for resume training.
  bool DestroyDeviceSideCommunicationGroup(const std::string &group_name);

  // Call HcclSetGlobalCommInfo to pass necessary info in world group.
  void SetGlobalCommInfo(CommunicationGroupPtr group, const std::string &group_name);

  // Get the inner comm name of the specified group.
  std::string GetCommName(const std::string &group_name);

  // Get the rank id of this process in the specified group.
  uint32_t GetRankId(const std::string &group_name);

  // Get the size of the specified group.
  uint32_t GetGroupSize(const std::string &group_name);

  uint32_t GetLocalRankId(const std::string &group_name);

  uint32_t GetLocalGroupSize(const std::string &group_name);

  uint32_t GetWorldRankFromGroupRank(const std::string &group_name, uint32_t local_rank);

  uint32_t GetGroupRankFromWorldRank(uint32_t global_rank, const std::string &group_name);

  std::vector<uint32_t> GetGroupRanks(const std::string &group_name);

  std::vector<size_t> GetAllHostHashs() { return all_host_hashs_; }

  // In some cases global rank id and rank size should be set by caller, e.g., when using MindSpore communication
  // framework, they're generated by cluster::ClusterContext.
  void set_global_rank_id(uint32_t global_rank_id);
  void set_global_rank_size(uint32_t global_rank_size);

  uint32_t global_rank_id() const;
  uint32_t local_rank_id() const;
  uint32_t global_rank_size() const;
  uint32_t local_rank_size() const;

  bool need_init() const { return need_init_.load(); }

  // Set whether need reinitialize collective communication.
  void set_need_reinit(bool need_reinit) { need_reinit_ = need_reinit; }
  // Get whether need reinitialize collective communication.
  bool need_reinit() const { return need_reinit_.load(); }

  // Return collective manager is initialized.
  bool initialized() const { return inited_.load(); }
  std::unordered_map<std::string, std::vector<uint32_t>> get_group_map() { return group_map_; }
  std::vector<std::pair<std::string, std::vector<uint32_t>>> get_group_info() { return group_infos_; }

  CollectiveCommunicationLib *device_comm_lib() { return device_comm_lib_instance_; }

  CollectiveCommunicationLib *host_comm_lib() { return host_comm_lib_instance_; }

  void CacheInitedGroups(const std::string &name);
  void ClearCacheInitedGroups();
  size_t InitedGroupSize();

  // Initialize and finalize Dummy communication lib.
  bool InitializeDummyCommLib();
  bool FinalizeDummyCommLib();

  bool ResumeHcclComm();

  // clean unique if after group build success when using ARF
  void ClearUniqueID(const std::string &group_name);

  // switch network interface card between the primary and the secondary NIC.
  bool CommSwitchNic(const std::vector<uint32_t> &global_ranks, const std::vector<bool> &use_backup);

  // Return whether initializing global comm asynchronizely.
  bool IsAsyncInitGlobalComm();

  // This interface will create communicator handle, including preprocessing like root info generating.
  bool CreateDeviceCommunicator(const std::string &group_name, const int32_t buffsize = -1);

  // Submit init task to init_comm_task_queue_.
  void SubmitCreateDeviceCommTask(const std::string &group_name, const int32_t hccl_buffsize = -1);

  // Wait all communicator to be initialized.
  bool WaitAllCommInitDone();

  // Wait specified group's communicator to be initialized.
  bool WaitCommInitDone(const std::string &group_name);

  // Set device id by DeviceManagerConf::set_device.
  void SetDeviceIDEnvByRuntimeConf();

 private:
  CollectiveManager();

  // Initialize communication library on host side.
  bool InitHostCommlib(std::string group_name);

  // Initialize communication library on device side.
  bool InitDeviceCommLib();

  // Assign the local rank id for this process.
  bool AssignLocalRank();

  // Assign local rank and size for each group in current server.
  bool GetLocalGroupRankAndSize(const std::vector<uint32_t> &group_ranks, uint32_t *local_group_rank,
                                uint32_t *local_group_size);

  // Create communication group in simulation mode.
  bool CreateSimulationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks);

  // Get timeout window for communicator initialization.
  int64_t GetCommunicatorInitTimeout();

  // This method will consume the tasks in init_comm_task_queue_.
  void RunInitCommTasks();

  // Set distributed meta data to libmindspore_core.so so that low level modules can access distributed meta data while
  // bypassing CollectiveManager module.
  void SetDistributedMeta();

  void UpdateToDefaultInfo();

  std::atomic_bool inited_;
  std::atomic_bool finalized_;

  // Whether collective communication library should be initialized. This is represents this process is launched as
  // distributed job.
  std::atomic_bool need_init_;

  // Whether need reinitialize collective communication, this value should be set to true once a training process
  // exits unexpectedly is detected.
  std::atomic_bool need_reinit_;

  // The device context on both host and device side. They are used to access the communication library on different
  // devices.
  DeviceContext *host_ctx_;
  DeviceContext *device_ctx_;

  // Host communication library refers to the communication libaray for CPU, e.g., OpenMPI and MindSpore communication
  // framework.
  CollectiveCommunicationLib *host_comm_lib_instance_;

  // Device communication library refers to the communication libaray for NPU or GPU, e.g., NCCL and HCCL.
  // When only CPU backend is used, device communication library should not be initialized.
  CollectiveCommunicationLib *device_comm_lib_instance_;

  // alias of host_comm_lib_instance_ and device_comm_lib_instance_ to avoid condition branch.
  CollectiveCommunicationLib *comm_lib_instance_;

  // Dummy collective communication for single device compile.
  std::shared_ptr<CollectiveCommunicationLib> dummy_comm_lib_instance_;

  // The global rank id of this process. Normally this range is 0 to `total process number - 1`.
  uint32_t global_rank_id_;

  // The local rank id of this process within the same node. This is usually used as device id if rank id is not
  // reassigned from rank table file
  uint32_t local_rank_id_;

  // The global rank size. Normally this is equal to `total process number`.
  uint32_t global_rank_size_;

  // The num of processes on this local node.
  uint32_t local_rank_size_;

  uint32_t default_global_rank_id_;
  uint32_t default_global_rank_size_;
  uint32_t default_local_rank_id_;
  uint32_t default_local_rank_size_;
  std::vector<uint32_t> default_global_group_ranks_;
  std::vector<size_t> default_all_host_hashs_;

  std::unordered_set<std::string> initialized_groups_;

  // Global group ranks.
  std::vector<uint32_t> global_group_ranks_;

  // This member represents whether the collective communication library is supported on the device side. If not, the
  // device side library will be replace by library on the host side.
  bool device_lib_supported_;

  // This member represents whether host collective communication is needed. Currently only effects on Ascend, If is
  // false, it means Ascend use ranktable file.
  bool need_host_collective_;

  // This member uses to assign local rank and size for each group.
  std::vector<size_t> all_host_hashs_;
  std::unordered_map<std::string, std::vector<uint32_t>> group_map_;

  // This thread launches tasks in init_comm_task_queue_ in FIFO order.
  std::thread run_init_comm_task_thread_;
  std::atomic<bool> stop_init_comm_{false};

  // Initializing task for every asynchronizely initialized communication group.
  // For each asynchronizely initialized comm, we need to store and wait for them in order to avoid dead lock.
  mutable std::mutex task_queue_mutex_;
  std::condition_variable task_queue_blocker_;
  InitCommTaskQueue init_comm_task_queue_;
  InitCommTaskList task_list_;

  // Group name to each init comm result map.
  mutable std::mutex init_result_mutex_;
  std::condition_variable result_blocker_;
  GroupToResultMap group_name_to_result_;

  // In the recovery scenario, the creation order of communication groups must be recorded to ensure they are recreated
  // sequentially when rebuilding.
  std::vector<std::pair<std::string, std::vector<uint32_t>>> group_infos_;
  mutable std::mutex cache_mutes_;
  std::vector<std::string> inited_groups_;
};

// For scheduler node, CollectiveManager is not initialized. Return 0 as rank id.
#define BY_PASS_SCHED_RANK_ID                              \
  do {                                                     \
    if (common::GetEnv(kEnvRole) == kEnvRoleOfScheduler) { \
      return static_cast<uint32_t>(0);                     \
    }                                                      \
  } while (0)

// For scheduler node, CollectiveManager is not initialized. Return 1 as rank size.
#define BY_PASS_SCHED_RANK_SIZE                            \
  do {                                                     \
    if (common::GetEnv(kEnvRole) == kEnvRoleOfScheduler) { \
      return static_cast<uint32_t>(1);                     \
    }                                                      \
  } while (0)
}  // namespace collective
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_COLLECTIVE_MANAGER_H_
