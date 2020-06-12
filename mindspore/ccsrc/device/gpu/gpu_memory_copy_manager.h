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

#ifndef MINDSPORE_CCSRC_DEVICE_GPU_GPU_MEMORY_COPY_MANAGER_H_
#define MINDSPORE_CCSRC_DEVICE_GPU_GPU_MEMORY_COPY_MANAGER_H_

#include <memory>
#include <queue>
#include <utility>
#include "pre_activate/mem_reuse/mem_copy_manager.h"
#include "device/device_address.h"
#include "device/gpu/cuda_driver.h"
#include "kernel/kernel.h"

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

  void AddMemSwapInTask(const DeviceAddressPtr &device_address, const HostAddress &host_addr) override;

  bool SyncMemCopyStream(SwapKind swap_kind) override;

  DeviceAddressPtr UpdateSwapOutQueue() override;

  DeviceAddressPtr UpdateSwapInQueue() override;

  bool AllocHostPinnedMem(size_t size, void **addr) const override;

  void FreeHostPinnedMem(void *addr) const override;

  void ClearSwapQueue() override;

 private:
  DeviceStream swap_out_stream_{nullptr};
  DeviceStream swap_in_stream_{nullptr};
  std::queue<std::pair<DeviceAddressPtr, DeviceEvent>> swap_out_queue_;
  std::queue<std::pair<DeviceAddressPtr, DeviceEvent>> swap_in_queue_;
};
using GPUMemCopyManagerPtr = std::shared_ptr<GPUMemCopyManager>;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_GPU_GPU_MEMORY_COPY_MANAGER_H_
