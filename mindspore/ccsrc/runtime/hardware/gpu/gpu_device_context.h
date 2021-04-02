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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_GPU_DEVICE_CONTEXT_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_GPU_DEVICE_CONTEXT_H_

#include <vector>
#include <memory>
#include <string>
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/memory_manager.h"

namespace mindspore {
namespace device {
namespace gpu {
class GPUDeviceContext : public DeviceContext {
 public:
  explicit GPUDeviceContext(const DeviceContextKey &device_context_key)
      : DeviceContext(device_context_key), mem_manager_(nullptr), initialized_(false) {}
  ~GPUDeviceContext() override = default;

  // Set device id and initialize device resource, such as stream, cudnn and cublas handle.
  bool Initialize() override;

  // Release device memory, stream, cudnn and cublas handle, etc.
  void Destroy() override;

  bool AllocateMemory(DeviceAddress *const &address, size_t size) const override;
  void FreeMemory(DeviceAddress *const &address) const override;
  bool AllocateContinuousMemory(const std::vector<DeviceAddress *> &addr_list, size_t total_size,
                                const std::vector<size_t> &size_list) const override;

  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) const override;
  DeviceAddressType GetDeviceAddressType() const override { return DeviceAddressType::kGPU; }

  // General graph optimezer ignore device data type and format.
  void OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) const override;
  // Optimize the kernel graph according to device type, such format transform.
  void OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) const override;

  // Optimize the single operator graph for PyNative mode.
  void OptimizeSingleOpGraph(const KernelGraphPtr &graph) const override;

  void SetOperatorInfo(const std::vector<CNodePtr> &nodes) const override;
  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;
  bool LaunchKernel(KernelMod *kernel_mod, const std::vector<AddressPtr> &inputs,
                    const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs) const override;

  bool SyncStream(size_t stream_id = 0) const override;

 private:
  DISABLE_COPY_AND_ASSIGN(GPUDeviceContext);
  bool InitDevice();

  // Operator fusion optimization.
  void FuseOperators(const KernelGraphPtr &graph) const;

  // Update Graph Dynamic Shape Attr.
  void UpdateGraphDynamicShapeAttr(const NotNull<KernelGraphPtr> &graph) const;

  std::shared_ptr<MemoryManager> mem_manager_;
  std::vector<void *> streams_;
  bool initialized_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_GPU_DEVICE_CONTEXT_H_
