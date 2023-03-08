/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/hal/device/gpu_memory_copy_manager.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace device {
namespace gpu {
void GPUMemCopyManager::Init() {
  CHECK_OP_RET_WITH_EXCEPT(GPUDeviceManager::GetInstance().CreateStream(&swap_out_stream_),
                           "Failed to create CUDA stream of memory swap out.");
  CHECK_OP_RET_WITH_EXCEPT(GPUDeviceManager::GetInstance().CreateStream(&swap_in_stream_),
                           "Failed to create CUDA stream of memory swap in.");
}

void GPUMemCopyManager::AddMemSwapOutTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr) {
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(host_addr.addr);
  CudaDeviceStream event = nullptr;
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::ConstructEvent(&event, cudaEventDisableTiming), "Failed to create CUDA event.");
  DeviceMemPtr device_ptr = const_cast<DeviceMemPtr>(device_address->GetPtr());
  MS_EXCEPTION_IF_NULL(device_ptr);
  device_address->set_status(DeviceAddressStatus::kInDeviceToHost);

  CHECK_OP_RET_WITH_EXCEPT(
    CudaDriver::CopyDeviceMemToHostAsync(host_addr.addr, device_ptr, host_addr.size, swap_out_stream_),
    "Failed to copy device memory to host.");

  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::RecordEvent(event, swap_out_stream_),
                           "Failed to record CUDA event to swap out stream.");
  swap_out_queue_.emplace(device_address, event);
}

void GPUMemCopyManager::AddMemSwapInTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr,
                                         bool profiling, float *cost_time) {
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(host_addr.addr);
  CudaDeviceStream start = nullptr;
  CudaDeviceStream end = nullptr;
  if (profiling) {
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::ConstructEvent(&start), "Failed to create CUDA event.");
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::ConstructEvent(&end), "Failed to create CUDA event.");
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::RecordEvent(start, swap_in_stream_),
                             "Failed to record CUDA event to swap in stream.");
  } else {
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::ConstructEvent(&end, cudaEventDisableTiming), "Failed to create CUDA event.");
  }
  DeviceMemPtr device_ptr = const_cast<DeviceMemPtr>(device_address->GetPtr());
  MS_EXCEPTION_IF_NULL(device_ptr);
  device_address->set_status(DeviceAddressStatus::kInHostToDevice);

  CHECK_OP_RET_WITH_EXCEPT(
    CudaDriver::CopyHostMemToDeviceAsync(device_ptr, host_addr.addr, host_addr.size, swap_in_stream_),
    "Failed to copy host memory to device.");
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::RecordEvent(end, swap_in_stream_),
                           "Failed to record CUDA event to swap in stream.");
  if (profiling) {
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::SyncEvent(start), "Failed to sync event.");
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::SyncEvent(end), "Failed to sync event.");
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::ElapsedTime(cost_time, start, end), "Failed to record elapsed time.");
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::DestroyEvent(start), "Failed to destroy event.");
  }
  swap_in_queue_.emplace(device_address, end);
}

void GPUMemCopyManager::AddMemSwapOutTaskMock(const DeviceAddressPtr &device_address) {
  MS_EXCEPTION_IF_NULL(device_address);
  device_address->set_status(DeviceAddressStatus::kInDeviceToHost);
  swap_out_queue_mock_.emplace(device_address);
}

void GPUMemCopyManager::AddMemSwapInTaskMock(const DeviceAddressPtr &device_address) {
  MS_EXCEPTION_IF_NULL(device_address);
  device_address->set_status(DeviceAddressStatus::kInHostToDevice);
  swap_in_queue_mock_.emplace(device_address);
}

bool GPUMemCopyManager::SyncMemCopyStream(SwapKind swap_kind) {
  if (swap_kind == SwapKind::kDeviceToHost) {
    return GPUDeviceManager::GetInstance().SyncStream(swap_out_stream_);
  } else {
    return GPUDeviceManager::GetInstance().SyncStream(swap_in_stream_);
  }
}

DeviceAddressPtr GPUMemCopyManager::UpdateSwapOutQueue() {
  if (swap_out_queue_.empty()) {
    return nullptr;
  }
  auto &task = swap_out_queue_.front();
  auto device_address = task.first;
  auto &event = task.second;
  bool finish_swap = CudaDriver::QueryEvent(event);
  if (!finish_swap) {
    return nullptr;
  }
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::DestroyEvent(event), "Failed to destroy CUDA event of swap out.");
  swap_out_queue_.pop();
  return device_address;
}

DeviceAddressPtr GPUMemCopyManager::UpdateSwapInQueue() {
  if (swap_in_queue_.empty()) {
    return nullptr;
  }
  auto &task = swap_in_queue_.front();
  auto device_address = task.first;
  auto &event = task.second;
  bool finish_swap = CudaDriver::QueryEvent(event);
  if (!finish_swap) {
    return nullptr;
  }
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::DestroyEvent(event), "Failed to destroy CUDA event of swap in.");
  swap_in_queue_.pop();
  return device_address;
}

DeviceAddressPtr GPUMemCopyManager::UpdateSwapOutQueueMock() {
  if (swap_out_queue_mock_.empty()) {
    return nullptr;
  }
  auto device_address = swap_out_queue_mock_.front();
  swap_out_queue_mock_.pop();
  return device_address;
}

DeviceAddressPtr GPUMemCopyManager::UpdateSwapInQueueMock() {
  if (swap_in_queue_mock_.empty()) {
    return nullptr;
  }
  auto device_address = swap_in_queue_mock_.front();
  swap_in_queue_mock_.pop();
  return device_address;
}

bool GPUMemCopyManager::AllocHostPinnedMem(size_t size, void **addr) const {
  auto alloc_size = CudaDriver::AllocHostPinnedMem(size, addr);
  return alloc_size == size;
}

void GPUMemCopyManager::FreeHostPinnedMem(void *addr) const { CudaDriver::FreeHostPinnedMem(addr); }

void GPUMemCopyManager::ClearSwapQueue() {
  CHECK_OP_RET_WITH_EXCEPT(SyncMemCopyStream(SwapKind::kDeviceToHost), "Failed to sync swap out stream");
  CHECK_OP_RET_WITH_EXCEPT(SyncMemCopyStream(SwapKind::kHostToDevice), "Failed to sync swap in stream");

  while (!swap_out_queue_.empty()) {
    auto &event = swap_out_queue_.front().second;
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::DestroyEvent(event), "Failed to destroy CUDA event of swap out.");
    swap_out_queue_.pop();
  }
  while (!swap_in_queue_.empty()) {
    auto &event = swap_in_queue_.front().second;
    CHECK_OP_RET_WITH_EXCEPT(CudaDriver::DestroyEvent(event), "Failed to destroy CUDA event of swap in.");
    swap_in_queue_.pop();
  }
}

void GPUMemCopyManager::ClearSwapQueueMock() {
  while (!swap_out_queue_mock_.empty()) {
    swap_out_queue_mock_.pop();
  }
  while (!swap_in_queue_mock_.empty()) {
    swap_in_queue_mock_.pop();
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
