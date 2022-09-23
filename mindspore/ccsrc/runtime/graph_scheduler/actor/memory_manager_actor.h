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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_MANAGER_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_MANAGER_ACTOR_H_

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <mutex>
#include "utils/hash_map.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::SomasInfo;

// MemoryManagerActor need response to memory alloc and free quickly, so must bind single thread.
class MemoryManagerActor : public ActorBase {
 public:
  MemoryManagerActor() : ActorBase("MemoryManagerActor") {}
  ~MemoryManagerActor() override = default;

  // The process entry of memory alloc.
  void AllocateMemory(const std::vector<DeviceTensor *> *alloc_list, const DeviceContext *device_context,
                      OpContext<DeviceTensor> *const op_context, const AID &from_aid);
  // The process entry of continuous memory alloc, the size of alloc_list_list, size_list_list, total_size_list and
  // device_contexts must be equal.
  void AllocateContinuousMemory(const std::vector<std::vector<DeviceTensorPtr>> *alloc_list_list,
                                const std::vector<std::vector<size_t>> *size_list_list,
                                const std::vector<size_t> *total_size_list,
                                const std::vector<const DeviceContext *> *device_contexts,
                                OpContext<DeviceTensor> *const op_context, const AID &from_aid);
  // device_contexts is from different device, the size of device_contexts must be equal to the alloc_list.
  void AllocateBatchMemory(const std::vector<DeviceTensor *> *alloc_list,
                           const std::vector<const DeviceContext *> *device_contexts,
                           OpContext<DeviceTensor> *const op_context, const AID &from_aid);
  // The process entry of somas memory alloc.
  void AllocateSomasMemory(SomasInfo *const somas_info, const DeviceContext *device_context,
                           OpContext<DeviceTensor> *const op_context, const AID &from_aid);

  // The process entry of memory free.
  void FreeMemory(const std::vector<DeviceTensor *> *free_list, const DeviceContext *device_context,
                  OpContext<DeviceTensor> *const op_context, const AID &from_aid);
  // device_contexts is from different device, the size of device_contexts must be equal to the free_list.
  void FreeBatchMemory(const std::vector<DeviceTensor *> *free_list,
                       const std::vector<const DeviceContext *> *device_contexts,
                       OpContext<DeviceTensor> *const op_context, const AID &from_aid);
  // The process entry of somas memory free.
  void FreeSomasMemory(SomasInfo *const somas_info, const DeviceContext *device_context,
                       OpContext<DeviceTensor> *const op_context, const AID &from_aid);

  // Wait the MemoryManagerActor to finish running all current messages.
  void Wait(OpContext<DeviceTensor> *const op_context, const AID &from_aid);

 private:
  void FreeMemoryByRefCount(DeviceTensor *const device_tensor, const DeviceContext *device_context,
                            const std::string &op_name);

  // When allocate device memory fail, print error log and set op context failed status.
  void SetOpContextMemoryAllocFail(const std::string &kernel_name, const DeviceContext *device_context,
                                   size_t alloc_size, OpContext<DeviceTensor> *const op_context);

  // MemoryManagerActor object is used like a single instance, if one actor allocates memory failed in one batch, which
  // will set fail message info OpContext, major thread will destroy the OpContext object, subsequent actor can not set
  // fail message again, so we record allocating memory fail event by the uuid of the batch, which is key of the set.
  std::set<int> mem_alloc_failed_step_ids_;
  std::mutex mem_alloc_failed_mutex_;

  // The memory free by the ref count maybe triggered concurrently, and the ref count decreased need the lock.
  std::mutex mem_free_mutex_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_MEMORY_MANAGER_ACTOR_H_
