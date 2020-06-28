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

#ifndef MINDSPORE_CCSRC_DEVICE_KERNEL_RUNTIME_H_
#define MINDSPORE_CCSRC_DEVICE_KERNEL_RUNTIME_H_
#include <vector>
#include <memory>
#include <string>
#include <map>

#include "device/device_address.h"
#include "ir/tensor.h"
#include "predict/generator/utils/ir_model_util.h"
#ifdef ENABLE_DUMP_E2E
#include "debug/e2e_dump.h"
#endif
#ifdef ENABLE_DEBUGGER
#include "debug/debugger/debugger.h"
#endif
#include "session/kernel_graph.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/kernel.h"
#include "utils/context/ms_context.h"
#include "device/memory_manager.h"

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
  virtual bool Run(session::KernelGraph *graph);
  virtual bool DumpData(session::KernelGraph *graph);
  virtual bool LoadData(session::KernelGraph *graph, Debugger *debugger);
  virtual bool RunTask(const session::KernelGraph *graph);
  virtual bool GenTask(const session::KernelGraph *graph);
  bool LaunchKernel(const session::KernelGraph *graph);
  virtual void AssignStaticMemoryInput(const session::KernelGraph *graph);
  virtual void AssignStaticMemoryValueNode(session::KernelGraph *graph);
  virtual void ClearGraphRuntimeResource(uint32_t graph_id);
  virtual bool SyncStream() = 0;

#ifdef ENABLE_DUMP_E2E
  DumpConfPtr GetDumpConf();
#endif
  virtual bool LoadTask(const session::KernelGraph *graph);
  // for GPU and D to impl
  virtual void ReleaseDeviceRes() {}
  void set_device_id(uint32_t device_id) { device_id_ = device_id; }

 protected:
  virtual DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                               TypeId type_id) = 0;
  virtual bool NodeOutputDeviceAddressExist(const AnfNodePtr &node, size_t index);
  void AssignStaticMemory(session::KernelGraph *graph);
  void AssignDynamicMemory(session::KernelGraph *graph);
  void ReuseAssignDynamicMemory(session::KernelGraph *graph);
  void AssignNodeOutputMem(int flag, const AnfNodePtr &node, int index);
  void AssignWorkSpaceMem(int flag, const AnfNodePtr &node);
  void AssignReuseWorkSpaceMem(const AnfNodePtr &node);

  void UpdateRefNodeOutputMem(const session::KernelGraph *graph);

  void AssignCommunicationNodeOutputMem(int flag, const AnfNodePtr &node);
  void AssignCommunicationNodeInputMem(const AnfNodePtr &node);
  void AssignCommunicationNodeMem(int flag, const AnfNodePtr &node);
#ifdef ENABLE_DUMP_E2E
  bool SetDumpConf();
#endif

 private:
  void AssignStaticMemoryOutput(const session::KernelGraph *graph);
  void GenLaunchArgs(const session::KernelGraph &graph, const AnfNodePtr &kernel, AddressPtrList *kernel_inputs,
                     AddressPtrList *kernel_workspaces, AddressPtrList *kernel_outputs);
  bool LaunchKernelMod(const session::KernelGraph &graph);
  void GenAddrCleanLaunchArgs(const CNodePtr &cnode, AddressPtrList *kernel_inputs);
  size_t CountNodeDeviceMemorySize(const AnfNodePtr &node, size_t output_index);
  void RunOpAssignInputMemory(const std::vector<tensor::TensorPtr> &input_tensors, const session::KernelGraph *graph);
  void RunOpAssignOutputMemory(const AnfNodePtr &kernel);
  void RunOpAssignWorkSpaceMemory(const AnfNodePtr &kernel);
  void AssignValueNodeTensor(const ValueNodePtr &value_node, const ValuePtr &node_value, size_t output_idx);
  DeviceAddressPtr PreAssignCNodeMemory(const AnfNodePtr &anf_node, size_t index);

 protected:
  uint32_t device_id_{0};
#ifdef ENABLE_DUMP_E2E
  DumpConfPtr dump_conf_ptr_;
#endif
  void *stream_ = nullptr;
  std::shared_ptr<MemoryManager> mem_manager_{nullptr};
};
using KernelRuntimePtr = std::shared_ptr<KernelRuntime>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_KERNEL_RUNTIME_H_
