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
#include "runtime/device/auto_mem_offload.h"
#include "plugin/device/gpu/hal/hardware/gpu_deprecated_interface.h"

namespace mindspore {
namespace device {
namespace gpu {
class GPUKernelExecutor;
class GPUDeviceResManager : public DeviceResManager {
 public:
  GPUDeviceResManager() : mem_manager_(nullptr) {}
  ~GPUDeviceResManager() override = default;

  // Set device id and initialize device resource, such as stream, cudnn and cublas handle.
  void Initialize() override;

  // Release device memory, stream, cudnn and cublas handle, etc.
  void Destroy() override;

  bool BindDeviceToCurrentThread(bool force_bind) const override;

  std::shared_ptr<void> AllocateHostMemory(size_t size) const override;

  std::vector<void *> AllocateContinuousMemory(const std::vector<size_t> &size_list) const override;

  size_t GetAvailableMemSize() const override { return mem_manager_->GetAvailableMemSize(); }

  DeviceAddressPtr CreateDeviceAddress(void *const device_ptr, size_t device_size, const string &format, TypeId type_id,
                                       const ShapeVector &shape = ShapeVector(),
                                       const UserDataPtr &user_data = nullptr) const override;

  bool CreateStream(size_t *stream_id) const override;
  bool DestroyStream(size_t stream_id) const override;
  bool SyncStream(size_t stream_id) const override;
  bool SyncAllStreams() const override;

  bool LoadCollectiveCommLib() override;

 protected:
  // Relevant function to allocate and free device memory of raw ptr.
  void *AllocateMemory(size_t size) const override;
  void FreeMemory(void *ptr) const override;

  bool AllocateMemory(DeviceAddress *const &address) const override;

 private:
  friend class GPUKernelExecutor;
  bool InitDevice();
  std::shared_ptr<MemoryManager> mem_manager_;
  std::shared_ptr<MindRTAutoOffloadAdapter> auto_mem_offload_{nullptr};
};

class GPUKernelExecutor : public DeprecatedKernelExecutor {
 public:
  GPUKernelExecutor() = default;
  ~GPUKernelExecutor() override = default;

  void Initialize();
  void Destroy();

  // Optimize the kernel graph for graph mode.
  void OptimizeGraph(const FuncGraphPtr &graph) const override;

  void CreateKernel(const std::vector<CNodePtr> &nodes) const override;

  void PreprocessBeforeRun(const FuncGraphPtr &graph) const override;

  bool LaunchKernel(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                    const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs,
                    size_t stream_id) const override;

  uint32_t GetRankID() const override;

 private:
  // Select the matching backend kernels according to the data type and format of input and output for all
  // execution operators, and set final device data type and format information for backend kernels, device
  // data type and format which replace original data type and format will use for executing kernels.
  void SetOperatorInfo(const KernelGraphPtr &graph) const;

  // General graph optimezer ignore device data type and format.
  void OptimizeGraphWithoutDeviceInfo(const KernelGraphPtr &graph) const;
  // Optimize the kernel graph according to device type, such format transform.
  void OptimizeGraphWithDeviceInfo(const KernelGraphPtr &graph) const;

  // Operator fusion optimization.
  void FuseOperators(const KernelGraphPtr &graph) const;

  // Update kernel ref info before create kernel
  void UpdateKernelRefInfo(const KernelGraphPtr &graph) const;

#ifndef ENABLE_SECURITY
  // Launch a kernel and record the elapsed time end to end.
  bool LaunchKernelWithProfiling(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                                 const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs,
                                 void *stream) const;
#endif
  // Launch a kernel by 'KernelMod' of the kernel.
  bool DoLaunchKernel(const CNodePtr &kernel, const std::vector<AddressPtr> &inputs,
                      const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs,
                      void *stream) const;

  // The cublas handle is not thread safety specifically, it is not recommended that multiple threads access the same
  // cublas handle at the same time, so need the launch mutex when multiple threads launch the cublas kernels.
  mutable std::mutex launch_mutex_;
  GPUDeviceResManager *res_manager_{nullptr};
};

class GPUDeviceContext : public DeviceInterface<GPUKernelExecutor, GPUDeviceResManager> {
 public:
  explicit GPUDeviceContext(const DeviceContextKey &device_context_key)
      : DeviceInterface(device_context_key), initialized_(false) {}
  ~GPUDeviceContext() override = default;

  // Set device id and initialize device resource, such as stream, cudnn and cublas handle.
  void Initialize() override;

  // Release device memory, stream, cudnn and cublas handle, etc.
  void Destroy() override;

  RunMode GetRunMode(const FuncGraphPtr &func_graph) const override { return RunMode::kKernelMode; }

  DeprecatedInterface *GetDeprecatedInterface() override;

 private:
  DISABLE_COPY_AND_ASSIGN(GPUDeviceContext);
  bool initialized_;
  std::unique_ptr<GPUDeprecatedInterface> deprecated_interface_;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_GPU_GPU_DEVICE_CONTEXT_H_
