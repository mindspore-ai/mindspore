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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_KERNEL_RUNTIME_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_KERNEL_RUNTIME_H_

#include <string>
#include <memory>
#include <vector>
#include <set>
#include <utility>
#include <unordered_map>
#include "runtime/device/kernel_runtime.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "backend/optimizer/mem_reuse/mem_swap_manager.h"

namespace mindspore {
namespace device {
namespace gpu {
using mindspore::device::memswap::MemSwapManagerPtr;
class GPUKernelRuntime : public KernelRuntime {
 public:
  GPUKernelRuntime() = default;
  ~GPUKernelRuntime() override = default;
  bool Init() override;
  void ReleaseDeviceRes() override;
  void AssignMemory(session::KernelGraph *graph) override;
  bool Run(session::KernelGraph *graph, Debugger *debugger = nullptr) override;
#ifdef ENABLE_DUMP_E2E
  bool DumpData(session::KernelGraph *graph, Debugger *debugger = nullptr) override;
#endif

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
  void InitKernelWorkspaceAddress(const session::KernelGraph *graph);
  void InitMemorySwapInfo(const session::KernelGraph *graph);
  void SaveGraphOutputNode(const session::KernelGraph *graph);
  bool IsGraphOutput(const session::KernelGraph *graph, const mindspore::AnfNodePtr &kernel) const;
  void ClearKernelOutputAddress(const session::KernelGraph *graph);
  void ClearKernelWorkspaceAddress(const session::KernelGraph *graph);
  void ClearKernelOldOutputAndWorkspace(const session::KernelGraph *graph);
  bool RunOneStep(const session::KernelGraph *graph, Debugger *debugger = nullptr);
  bool SearchMemSwapScheme(const session::KernelGraph *graph, Debugger *debugger = nullptr);
  bool RefineMemSwapScheme(const session::KernelGraph *graph, Debugger *debugger = nullptr);
  bool LaunchKernelDynamic(const session::KernelGraph *graph, Debugger *debugger = nullptr, bool mock = false,
                           bool profiling = false);
  void LaunchKernelWithTimeProfiling(const AnfNodePtr &kernel, const AddressPtrList &inputs,
                                     const AddressPtrList &workspace, const AddressPtrList &outputs);
  bool AttemptMallocMem(const DeviceAddressPtr &device_address, size_t size, bool mock);
  bool AllocKernelDynamicRes(const mindspore::kernel::KernelMod &kernel_mod, const mindspore::AnfNodePtr &kernel,
                             AddressPtrList *kernel_inputs, AddressPtrList *kernel_workspaces,
                             AddressPtrList *kernel_outputs, bool mock);
  bool AllocKernelInputDynamicRes(const mindspore::AnfNodePtr &kernel, AddressPtrList *kernel_inputs, bool mock);
  bool AllocKernelOutputDynamicRes(const mindspore::kernel::KernelMod &kernel_mod, const mindspore::AnfNodePtr &kernel,
                                   AddressPtrList *kernel_outputs, bool mock);
  bool AllocKernelWorkspaceDynamicRes(const mindspore::kernel::KernelMod &kernel_mod,
                                      const mindspore::AnfNodePtr &kernel, AddressPtrList *kernel_workspaces,
                                      bool mock);
  void AllocCommunicationOpDynamicRes(const session::KernelGraph *graph);
  void AllocCommunicationOpInputDynamicRes(const mindspore::AnfNodePtr &kernel);
  void AllocCommunicationOpOutputDynamicRes(const mindspore::AnfNodePtr &kernel);
  void AllocCommunicationOpMemory(bool is_need_alloc_memory, bool is_need_free_memory,
                                  const DeviceAddressPtrList addr_list, size_t total_size,
                                  std::vector<size_t> size_list);
  void FreeKernelDynamicRes(const mindspore::AnfNodePtr &kernel);
  bool UpdateMemorySwapTask(const AnfNodePtr &kernel, bool mock, bool profiling);
  bool AddMemorySwapTask(const AnfNodePtr &kernel, bool mock, bool profiling);
  void UpdateHostSwapInQueue(const DeviceAddressPtr device_address, bool mock);
  void UpdateHostSwapOutQueue(bool mock);
  void ClearSwapInfo(bool mock);
  std::unordered_map<uint32_t, MemReuseUtilPtr> mem_reuse_util_map_;
  std::unordered_map<uint32_t, MemSwapManagerPtr> mem_swap_map_;
  std::unordered_map<uint32_t, bool> is_first_step_map_;
  std::unordered_map<uint32_t, std::set<AnfNodePtr>> graph_output_map_;

  MemReuseUtilPtr mem_reuse_util_{nullptr};
  MemSwapManagerPtr mem_swap_manager_{nullptr};
};
MS_REG_KERNEL_RUNTIME(kGPUDevice, GPUKernelRuntime);
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_KERNEL_RUNTIME_H_
