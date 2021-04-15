/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
  virtual void AssignMemory(session::KernelGraph *graph);
  void RunOpAssignMemory(const std::vector<tensor::TensorPtr> &input_tensors, session::KernelGraph *graph);
  void RunOpClearMemory(const session::KernelGraph *graph);
  static bool DumpDataEnabled();
  static bool DumpDataEnabledIteration();
  virtual bool LoadData(session::KernelGraph *graph);
  virtual bool Load(session::KernelGraph *graph, bool is_task_sink);
  virtual bool Run(session::KernelGraph *graph, bool is_task_sink) = 0;
  virtual bool GenDynamicKernel(const session::KernelGraph *graph) = 0;
  virtual bool RunDynamicKernelAsync(const session::KernelGraph *graph) = 0;
  bool LaunchKernel(const session::KernelGraph *graph);
  bool LaunchTaskBasedOnSingleKernel(kernel::KernelModPtr kernel_mod_ptr, const AddressPtrList &kernel_inputs,
                                     const AddressPtrList &kernel_outputs,
                                     const AddressPtrList &kernel_workspaces) const;
  virtual void AssignStaticMemoryInput(const session::KernelGraph *graph);
  virtual void AssignStaticMemoryValueNode(session::KernelGraph *graph);
  virtual void ClearGraphRuntimeResource(uint32_t graph_id, const std::vector<AnfNodePtr> &inputs,
                                         const std::unordered_set<ValueNodePtr> &value_nodes,
                                         const std::vector<CNodePtr> &execution_order);
  virtual void ClearOutputAddress(const std::vector<AnfNodePtr> &inputs,
                                  const std::unordered_set<ValueNodePtr> &value_nodes,
                                  const std::vector<CNodePtr> &execution_order);
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
  DeviceAddressPtr AssignSingleOpLaunchMemory(size_t size, const std::string &format, TypeId type);

  // set debugger
  void SetDebugger() {
#if !defined(_WIN32) && !defined(_WIN64)
    debugger_ = Debugger::GetInstance();
#endif
  }

  virtual void PreInit() {}
  virtual uint64_t GetAvailableMemMaxSize() const { return 0; }
  void AddBufferPtr(std::shared_ptr<char[]> ptr) { buffer_ptrs_.push_back(ptr); }
  void FreeAndClearBufferPtrs() { buffer_ptrs_.clear(); }
  virtual DeviceAddressType GetTargetDeviceAddressType() const = 0;
  virtual void *compute_stream() const { return nullptr; }
  virtual void *communication_stream() const { return nullptr; }

 protected:
  virtual DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                               TypeId type_id) = 0;
  virtual bool NodeOutputDeviceAddressExist(const AnfNodePtr &node, size_t index);
  virtual bool KernelMemNotReuse(const AnfNodePtr &node);

  void AssignStaticMemory(session::KernelGraph *graph);
  void AssignDynamicMemory(session::KernelGraph *graph);
  void ReuseAssignDynamicMemory(session::KernelGraph *graph);
  void AssignNodeOutputMem(MemType type, const AnfNodePtr &node, int index);
  void AssignWorkSpaceMem(MemType type, const AnfNodePtr &node);
  void AssignReuseWorkSpaceMem(const AnfNodePtr &node);

  void UpdateRefNodeOutputMem(const session::KernelGraph *graph);

  void AssignCommunicationNodeOutputMem(MemType type, const AnfNodePtr &node);
  void AssignCommunicationNodeInputMem(MemType type, const AnfNodePtr &node);
  void AssignCommunicationNodeMem(MemType type, const AnfNodePtr &node);

  virtual void KernelLaunchProfiling(const std::string &kernel_name) {}

 private:
  void AssignStaticMemoryOutput(const session::KernelGraph *graph);
  bool LaunchKernelMod(const session::KernelGraph &graph);
  static void GenAddrCleanLaunchArgs(const CNodePtr &cnode, AddressPtrList *kernel_inputs);
  size_t CountNodeDeviceMemorySize(const AnfNodePtr &node, size_t output_index);
  void RunOpAssignInputMemory(const std::vector<tensor::TensorPtr> &input_tensors, const session::KernelGraph *graph);
  void RunOpAssignOutputMemory(const AnfNodePtr &kernel);
  void RunOpAssignWorkSpaceMemory(const AnfNodePtr &kernel);
  void RunOpAssignOutputNodeMemory(const ValuePtr &pre_output_value, session::KernelGraph *graph);
  void AssignValueNodeTensor(const ValueNodePtr &value_node, const ValuePtr &node_value, size_t output_idx);
  DeviceAddressPtr PreAssignCNodeMemory(const AnfNodePtr &anf_node, size_t index);
#if (ENABLE_CPU && (ENABLE_D || ENABLE_GPU))
  void GetFirstPSEmbeddingCache(const session::KernelGraph *graph, AnfNodePtr *const first_cache_input_index,
                                size_t *const first_cache_size);
  void CheckIfSupportPSEmbeddingCache(const session::KernelGraph *graph);
  void CheckSparsePSEmbeddingCache(const CNodePtr &node);
#endif

 protected:
  uint32_t device_id_{0};
#if !defined(_WIN32) && !defined(_WIN64)
  std::shared_ptr<Debugger> debugger_;
#endif
  void *stream_{nullptr};
  void *communication_stream_{nullptr};
  std::shared_ptr<MemoryManager> mem_manager_{nullptr};
  std::map<uint32_t, std::vector<DynamicKernelPtr>> graph_dynamic_kernel_map_;
  std::vector<std::shared_ptr<char[]>> buffer_ptrs_ = {};
};
using KernelRuntimePtr = std::shared_ptr<KernelRuntime>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_KERNEL_RUNTIME_H_
