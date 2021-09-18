/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_RUNTIME_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_RUNTIME_H_
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <utility>
#include <unordered_set>
#include "runtime/device/device_address.h"
#include "ir/tensor.h"
#include "utils/convert_utils.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#include "backend/session/kernel_graph.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/kernel_compiler/kernel.h"
#include "utils/ms_context.h"
#include "runtime/device/memory_manager.h"
#include "runtime/device/memory_scheduler.h"
#include "runtime/device/executor/dynamic_kernel.h"
#include "ir/device_event.h"

using mindspore::tensor::Tensor;
using std::vector;
using TensorPtr = std::shared_ptr<Tensor>;
using mindspore::kernel::AddressPtr;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;

namespace mindspore {
#ifndef ENABLE_DEBUGGER
class Debugger;
#endif
namespace device {
class KernelRuntime {
 public:
  KernelRuntime() = default;
  virtual ~KernelRuntime();
  virtual bool Init() = 0;
  virtual void AssignMemory(const session::KernelGraph &graph);
  void RunOpAssignMemory(const std::vector<tensor::TensorPtr> &input_tensors, const session::KernelGraph &graph,
                         const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node = {});
  void RunOpAssignCommunicationOutput(const AnfNodePtr &node) const;
  void RunOpAssignCommunicationInput(const AnfNodePtr &node) const;
  void RunOpClearMemory(const session::KernelGraph &graph) const;
  void RunOpMallocPre(const session::KernelGraph &graph, const std::vector<tensor::TensorPtr> &input_tensors);
#ifdef ENABLE_DEBUGGER
  static bool DumpDataEnabled();
  static bool DumpDataEnabledIteration();
#endif
  virtual bool LoadData(const session::KernelGraph &graph);
  virtual bool Load(const session::KernelGraph &graph, bool is_task_sink);
  virtual bool Run(const session::KernelGraph &graph, bool is_task_sink) = 0;
  virtual bool GenDynamicKernel(const session::KernelGraph &graph) = 0;
  virtual bool RunDynamicKernelAsync(const session::KernelGraph &graph) = 0;
  bool LaunchKernels(const session::KernelGraph &graph);
  virtual void AssignStaticMemoryInput(const session::KernelGraph &graph);
  virtual void AssignStaticMemoryValueNode(const session::KernelGraph &graph);

  virtual void ClearGraphRuntimeResource(uint32_t graph_id);
  virtual bool SyncStream() = 0;
  virtual bool MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) = 0;
  virtual void ClearGlobalIdleMem() {}
  virtual void CreateContext() {}
  virtual void SetContext() {}
  virtual const void *context() const { return nullptr; }
  uint8_t *MallocMem(MemType type, size_t size, const DeviceAddressPtr &address) {
    return mem_manager_->MallocMem(type, size, address);
  }
  uint8_t *MallocCommunicationMemFromMemPool(size_t size) {
    return mem_manager_->MallocCommunicationMemFromMemPool(size);
  }
  static void GenLaunchArgs(const mindspore::kernel::KernelMod &kernel_mod, const AnfNodePtr &kernel,
                            AddressPtrList *kernel_inputs, AddressPtrList *kernel_workspaces,
                            AddressPtrList *kernel_outputs);

  // for GPU and D to impl
  virtual void ReleaseDeviceRes() {}
  void set_device_id(uint32_t device_id) { device_id_ = device_id; }
  uint32_t device_id() { return device_id_; }

#ifdef ENABLE_DEBUGGER
  // set debugger
  void SetDebugger() {
#if !defined(_WIN32) && !defined(_WIN64)
    debugger_ = Debugger::GetInstance();
#endif
  }
#endif

#ifndef ENABLE_SECURITY
  virtual void PreInit() {}
#endif
  virtual uint64_t GetAvailableMemMaxSize() const { return 0; }
  virtual void GenKernelEvents(const session::KernelGraph &graph);
  virtual std::shared_ptr<DeviceEvent> CreateDeviceEvent() { return nullptr; }
  virtual std::shared_ptr<DeviceEvent> CreateDeviceTimeEvent() { return nullptr; }
  virtual DeviceAddressType GetTargetDeviceAddressType() const = 0;
  virtual void *compute_stream() const { return nullptr; }
  virtual void *communication_stream() const { return nullptr; }
  void UpdateRefNodeOutputMem(const session::KernelGraph &graph);
  virtual DeviceAddressPtr AssignExtraStaticMem(const TensorPtr &tensor, const AnfNodePtr &node, size_t index);
  virtual void *GetModelStream(uint32_t graph_id) const { return nullptr; }

 protected:
  virtual DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                               TypeId type_id) const = 0;
  virtual DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                               TypeId type_id, const KernelWithIndex &node_index) const = 0;
  virtual bool NodeOutputDeviceAddressExist(const AnfNodePtr &node, size_t index);
  virtual bool KernelMemNotReuse(const AnfNodePtr &node);

  void AssignStaticMemory(const session::KernelGraph &graph);
  void AssignDynamicMemory(const session::KernelGraph &graph);
  void AssignNodeOutputMem(MemType type, const AnfNodePtr &node, int index);
  void AssignWorkSpaceMem(MemType type, const AnfNodePtr &node);

  void AssignCommunicationNodeOutputMem(MemType type, const AnfNodePtr &node);
  void AssignCommunicationNodeInputMem(MemType type, const AnfNodePtr &node);
  void AssignCommunicationNodeMem(MemType type, const AnfNodePtr &node);
  bool LaunchKernelWithPynativeProfiling(kernel::KernelMod *kernel_mod, const std::string &op_name,
                                         const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs, void *stream);

  virtual void KernelLaunchProfiling(const std::string &kernel_name) {}

 private:
  void UseMemSchedulerIfNeeded(const session::KernelGraph &graph);
  bool LaunchKernel(const session::KernelGraph &graph, const AnfNodePtr &kernel,
                    const std::shared_ptr<MemScheduler> &mem_scheduler, bool mock = false);
  void ResetNodeAddress(const session::KernelGraph &graph);
  void AssignKernelAddress(const std::shared_ptr<MemScheduler> &mem_scheduler, const AnfNodePtr &kernel,
                           AddressPtrList *kernel_inputs, AddressPtrList *kernel_workspaces,
                           AddressPtrList *kernel_outputs);
  static void GetOrMallocAddress(const std::shared_ptr<MemScheduler> &mem_scheduler,
                                 const DeviceAddress *device_address, const kernel::AddressPtr &kernel_addr);
  void InitGraphInputTensors(const std::shared_ptr<MemScheduler> &mem_scheduler, const session::KernelGraph &graph);
  void SyncNodeOutputTensors(const std::shared_ptr<MemScheduler> &mem_scheduler, const session::KernelGraph &graph,
                             const AnfNodePtr &kernel, bool mock);
  void AssignStaticMemoryOutput(const session::KernelGraph &graph);
  bool LaunchKernelMod(const session::KernelGraph &graph, bool mock = false);
  void LaunchKernelEvent(const std::vector<std::vector<std::function<void()>>> &run_events, size_t index) const;
  void DebugStreamSync(const CNodePtr &kernel);
  static void GenAddrCleanLaunchArgs(const CNodePtr &cnode, AddressPtrList *kernel_inputs,
                                     const std::shared_ptr<MemScheduler> &mem_schedule = nullptr);
  void RunOpAssignInputMemory(const std::vector<tensor::TensorPtr> &input_tensors, const session::KernelGraph &graph);
  void RunOpAssignOutputMemory(const AnfNodePtr &kernel,
                               const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node = {});
  void RunOpAssignWorkSpaceMemory(const AnfNodePtr &kernel);
  void RunOpAssignOutputNodeMemory(const ValuePtr &pre_output_value, const session::KernelGraph &graph);
  void AssignValueNodeTensor(const ValueNodePtr &value_node, const ValuePtr &node_value, size_t output_idx);
  DeviceAddressPtr PreAssignCNodeMemory(const AnfNodePtr &anf_node, size_t index) const;
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  void GetFirstPSEmbeddingCache(const session::KernelGraph &graph, AnfNodePtr *const first_cache_input_index,
                                size_t *const first_cache_size);
  void CheckIfSupportPSEmbeddingCache(const session::KernelGraph &graph);
  void CheckSparsePSEmbeddingCache(const CNodePtr &node);
#endif
  void RunOpGetCommunicationInputInfo(const AnfNodePtr &node, size_t *total_size,
                                      std::vector<DeviceAddressPtr> *address_list,
                                      std::vector<size_t> *align_size_list) const;
  void RunOpGetCommunicationOutputInfo(const AnfNodePtr &node, size_t *total_size, std::vector<size_t> *align_size_list,
                                       std::vector<DeviceAddressPtr> *device_address_list) const;

 protected:
  uint32_t device_id_{0};
  bool pynative_mode_profiling_flag_{false};
#if defined(ENABLE_DEBUGGER) && !defined(_WIN32) && !defined(_WIN64)
  std::shared_ptr<Debugger> debugger_;
#endif
  void *stream_{nullptr};
  void *independent_stream_{nullptr};
  void *communication_stream_{nullptr};
  std::shared_ptr<MemoryManager> mem_manager_{nullptr};
  std::map<uint32_t, std::vector<DynamicKernelPtr>> graph_dynamic_kernel_map_;
  std::map<uint32_t,
           std::pair<std::vector<std::vector<std::function<void()>>>, std::vector<std::vector<std::function<void()>>>>>
    graph_kernel_events_map_;
  MemSchedulerManager mem_scheduler_manager_;
};
using KernelRuntimePtr = std::shared_ptr<KernelRuntime>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_RUNTIME_H_
