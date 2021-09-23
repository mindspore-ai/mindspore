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
  void Initialize() override;

  // Release device memory, stream, cudnn and cublas handle, etc.
  void Destroy() override;

  bool AllocateMemory(DeviceAddress *const &address, size_t size) const override;
  void FreeMemory(DeviceAddress *const &address) const override;
  bool AllocateContinuousMemory(const std::vector<DeviceAddressPtr> &addr_list, size_t total_size,
                                const std::vector<size_t> &size_list) const override;

  DeviceAddressPtr CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) const override;
  DeviceAddressType GetDeviceAddressType() const override { return DeviceAddressType::kGPU; }

  // Optimize the kernel graph for graph mode.
  void OptimizeGraph(const KernelGraphPtr &graph) const override;
  // Optimize the single operator graph for PyNative mode.
  void OptimizeSingleOpGraph(const KernelGraphPtr &graph) const override;

  void SetOperatorInfo(const std::vector<CNodePtr> &nodes) const override;
  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;

  // Infer kernel shape and update abstract info for dynamic shape kernel.
  void UpdateDynamicShape(const CNodePtr &kernel) const override;

  bool LaunchKernel(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                    const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs,
                    bool is_dynamic_shape = false) const override;

  bool SyncStream(size_t stream_id = 0) const override;

  uint32_t GetRankID() const override;

  // Create bucket for every allreduce operator. Bucket is used in PyNative distributed training mode, one bucket
  // handles all resource to launch and sync allreduce operator.
  std::shared_ptr<Bucket> CreateBucket(uint32_t bucket_id, uint32_t bucket_size) const override;

 private:
  DISABLE_COPY_AND_ASSIGN(GPUDeviceContext);
  bool InitDevice();

  // General graph optimezer ignore device data type and format.
  void OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) const;
  // Optimize the kernel graph according to device type, such format transform.
  void OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) const;

  // Operator fusion optimization.
  void FuseOperators(const KernelGraphPtr &graph) const;

  // Update Graph Dynamic Shape Attr.
  void UpdateGraphDynamicShapeAttr(const NotNull<KernelGraphPtr> &graph) const;

  bool BindDeviceToCurrentThread() const;
#ifndef ENABLE_SECURITY
  // Launch a kernel and record the elapsed time end to end.
  bool LaunchKernelWithProfiling(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                                 const std::vector<AddressPtr> &workspace,
                                 const std::vector<AddressPtr> &outputs) const;
#endif
  // Launch a kernel by 'KernelMod' of the kernel.
  bool DoLaunchKernel(KernelMod *kernel_mod, const std::vector<AddressPtr> &inputs,
                      const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs) const;

  // The cublas handle is not thread safety specifically, it is not recommended that multiple threads access the same
  // cublas handle at the same time, so need the launch mutex when multiple threads launch the cublas kernels.
  mutable std::mutex launch_mutex_;

  std::shared_ptr<MemoryManager> mem_manager_;
  std::vector<void *> streams_;
  bool initialized_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_GPU_DEVICE_CONTEXT_H_
