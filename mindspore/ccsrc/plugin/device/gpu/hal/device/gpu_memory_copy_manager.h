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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_MEMORY_COPY_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_MEMORY_COPY_MANAGER_H_

#include <memory>
#include <queue>
#include <utility>
#include "backend/common/mem_reuse/mem_copy_manager.h"
#include "include/backend/device_address.h"
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "kernel/kernel.h"

// todo: delete with kernel-runtime
namespace mindspore {
namespace device {
namespace gpu {
using mindspore::device::memswap::MemCopyManager;
using mindspore::device::memswap::SwapKind;
class GPUMemCopyManager : public MemCopyManager {
 public:
  GPUMemCopyManager() = default;

  ~GPUMemCopyManager() override = default;

  void Init() override;

  void AddMemSwapOutTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr) override;

  void AddMemSwapInTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr, bool profiling,
                        float *cost_time) override;

  void AddMemSwapOutTaskMock(const DeviceAddressPtr &device_address) override;

  void AddMemSwapInTaskMock(const DeviceAddressPtr &device_address) override;

  bool SyncMemCopyStream(SwapKind swap_kind) override;

  DeviceAddressPtr UpdateSwapOutQueue() override;

  DeviceAddressPtr UpdateSwapInQueue() override;

  DeviceAddressPtr UpdateSwapOutQueueMock() override;

  DeviceAddressPtr UpdateSwapInQueueMock() override;

  bool AllocHostPinnedMem(size_t size, void **addr) const override;

  void FreeHostPinnedMem(void *addr) const override;

  void ClearSwapQueue() override;

  void ClearSwapQueueMock() override;

 private:
  CudaDeviceStream swap_out_stream_{nullptr};
  CudaDeviceStream swap_in_stream_{nullptr};
  std::queue<std::pair<DeviceAddressPtr, CudaDeviceStream>> swap_out_queue_;
  std::queue<std::pair<DeviceAddressPtr, CudaDeviceStream>> swap_in_queue_;
  std::queue<DeviceAddressPtr> swap_out_queue_mock_;
  std::queue<DeviceAddressPtr> swap_in_queue_mock_;
};
using GPUMemCopyManagerPtr = std::shared_ptr<GPUMemCopyManager>;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_MEMORY_COPY_MANAGER_H_
