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

#include "device/gpu/gpu_memory_copy_manager.h"
#include "device/gpu/gpu_common.h"
#include "device/gpu/gpu_device_manager.h"
#include "session/anf_runtime_algorithm.h"

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
  DeviceEvent event = nullptr;
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::CreateEvent(&event, cudaEventDisableTiming), "Failed to create CUDA event.");
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

void GPUMemCopyManager::AddMemSwapInTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr) {
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(host_addr.addr);
  DeviceEvent event = nullptr;
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::CreateEvent(&event, cudaEventDisableTiming), "Failed to create CUDA event.");
  DeviceMemPtr device_ptr = const_cast<DeviceMemPtr>(device_address->GetPtr());
  MS_EXCEPTION_IF_NULL(device_ptr);
  device_address->set_status(DeviceAddressStatus::kInHostToDevice);

  CHECK_OP_RET_WITH_EXCEPT(
    CudaDriver::CopyHostMemToDeviceAsync(device_ptr, host_addr.addr, host_addr.size, swap_in_stream_),
    "Failed to copy host memory to device.");
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::RecordEvent(event, swap_in_stream_),
                           "Failed to record CUDA event to swap in stream.");
  swap_in_queue_.emplace(device_address, event);
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

bool GPUMemCopyManager::AllocHostPinnedMem(size_t size, void **addr) {
  auto alloc_size = CudaDriver::AllocHostPinnedMem(size, addr);
  return alloc_size == size;
}

void GPUMemCopyManager::FreeHostPinnedMem(void *addr) { CudaDriver::FreeHostPinnedMem(addr); }

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
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
