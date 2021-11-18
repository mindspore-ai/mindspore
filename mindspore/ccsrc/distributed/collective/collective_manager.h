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
#include <atomic>
#include "utils/ms_utils.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace distributed {
namespace collective {
using DeviceContext = device::DeviceContext;
using DeviceContextKey = device::DeviceContextKey;
using DeviceContextManager = device::DeviceContextManager;

// The collective communication API.
// MindSpore uses OpenMPI on CPU, NCCL on GPU, HCCL on Ascend, to achieve distributed training.
// Besides, MindSpore also has its own communication library which is implemented on the CPU side.
class CollectiveManager {
 public:
  ~CollectiveManager();
  DISABLE_COPY_AND_ASSIGN(CollectiveManager);
  static std::shared_ptr<CollectiveManager> instance();

  // Initialize the collective communication for distributed training with the backend name, e.g., NCCL or HCCL.
  bool Initialize(const std::string &backend, const std::string &global_group_name);

  // Create communication group.
  bool CreateCommunicationGroup(const std::string &group_name, const std::vector<uint32_t> &group_ranks);

  // Destroy the communication group.
  bool DestroyCommunicationGroup(const std::string &group_name);

  // Get the rank id of this process in the specified group.
  uint32_t GetRankId(const std::string &group_name);

  // Get the size of the specified group.
  uint32_t GetGroupSize(const std::string &group_name);

  // Finalize the collective communication.
  bool Finalize();

 private:
  CollectiveManager();

  // Initialize communication library on host side.
  bool InitHostCommlib();

  // Create world communication group on the host side.
  bool CreateHostGlobalCommGroup(const std::string &global_group_name);

  // Initialize communication library on device side.
  bool InitDeviceCommLib(const std::string &backend, uint32_t device_id);

  // Create world communication group on the device side.
  bool CreateDeviceGlobalCommGroup(const std::string &global_group_name);

  std::atomic_bool inited_;
  std::atomic_bool finalized_;

  // The device context on both host and device side. They are used to access the communication library on different
  // devices.
  DeviceContext *host_ctx_;
  DeviceContext *device_ctx_;

  // The dynamically loaded handle for collective communication library by 'dlopen'.
  void *host_comm_lib_;
  void *device_comm_lib_;

  // The global rank id of this process. Normally this range is 0 to `total process number - 1`.
  uint32_t global_rank_id_;

  // The local rank id of this process within the same node. This is usually used as device id.
  uint32_t local_rank_id_;

  // The global rank size. Normally this is equal to `total process number`.
  uint32_t global_rank_size_;

  // Global group ranks.
  std::vector<uint32_t> global_group_ranks_;
};
}  // namespace collective
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_COLLECTIVE_COLLECTIVE_MANAGER_H_
