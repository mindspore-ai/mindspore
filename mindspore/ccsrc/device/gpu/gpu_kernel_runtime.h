/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DEVICE_GPU_GPU_KERNEL_RUNTIME_H_
#define MINDSPORE_CCSRC_DEVICE_GPU_GPU_KERNEL_RUNTIME_H_

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <unordered_map>
#include "device/kernel_runtime.h"
#include "device/kernel_runtime_manager.h"

namespace mindspore {
namespace device {
namespace gpu {
class GPUKernelRuntime : public KernelRuntime {
 public:
  GPUKernelRuntime() = default;
  ~GPUKernelRuntime() override = default;
  bool Init() override;
  void ReleaseDeviceRes() override;
  void AssignMemory(session::KernelGraph *graph) override;
  bool Run(session::KernelGraph *graph) override;

 protected:
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) override;
  bool SyncStream() override;

 private:
  GPUKernelRuntime(const GPUKernelRuntime &);
  GPUKernelRuntime &operator=(const GPUKernelRuntime &);
  bool InitDevice();
  bool device_init_{false};

  // The related functions and members for using dynamic memory pool.
  void InitKernelRefCount(const session::KernelGraph *graph);
  void InitKernelOutputAddress(const session::KernelGraph *graph);
  bool LaunchKernelDynamic(const session::KernelGraph *graph);
  void AllocKernelDynamicRes(const mindspore::kernel::KernelMod &kernel_mod, const mindspore::AnfNodePtr &kernel,
                             AddressPtrList *kernel_inputs, AddressPtrList *kernel_workspaces,
                             AddressPtrList *kernel_outputs);
  void AllocCommunicationOpDynamicRes(const session::KernelGraph *graph);
  void AllocCommunicationOpInputDynamicRes(const mindspore::AnfNodePtr &kernel);
  void AllocCommunicationOpOutputDynamicRes(const mindspore::AnfNodePtr &kernel);
  void FreeKernelDynamicRes(const mindspore::AnfNodePtr &kernel, const AddressPtrList &kernel_workspaces,
                            uint32_t graph_id);
  std::unordered_map<uint32_t, MemReuseUtilPtr> mem_reuse_util_map_;
};
MS_REG_KERNEL_RUNTIME(kGPUDevice, GPUKernelRuntime);
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEVICE_GPU_GPU_KERNEL_RUNTIME_H_
