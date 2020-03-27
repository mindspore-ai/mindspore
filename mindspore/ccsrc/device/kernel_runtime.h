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

#ifndef MINDSPORE_CCSRC_DEVICE_KERNEL_RUNTIME_H_
#define MINDSPORE_CCSRC_DEVICE_KERNEL_RUNTIME_H_
#include <vector>
#include <memory>
#include <string>
#include <map>
#include "pre_activate/mem_reuse/mem_reuse.h"
#include "pre_activate/mem_reuse/mem_reuse_allocator.h"
#include "device/device_address.h"
#include "ir/meta_tensor.h"
#include "predict/generator/utils/ir_model_util.h"
#ifdef ENABLE_DUMP_E2E
#include "debug/e2e_dump.h"
#endif
#include "session/kernel_graph.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/kernel.h"
#include "utils/context/ms_context.h"

// using mindspore::session::KernelGraph;
using mindspore::tensor::Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
using MemReuseUtilPtr = mindspore::memreuse::MemReuseUtilPtr;
using mindspore::kernel::AddressPtr;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;

namespace mindspore {
namespace device {
const int kStaticMem = 0;
const int kDynamicMem = 1;
const int kReuseDynamicMem = 2;
const int kGetAllOuts = -1;

class KernelRuntime {
 public:
  KernelRuntime() = default;
  virtual ~KernelRuntime();
  virtual bool Init() = 0;
  virtual void AssignMemory(session::KernelGraph *graph);
  void RunOpAssignMemory(const std::vector<tensor::TensorPtr> &input_tensors, const session::KernelGraph *graph);
  virtual bool Run(session::KernelGraph *graph);
  virtual bool DumpData(session::KernelGraph *graph);
  virtual bool RunTask(const session::KernelGraph *graph);
  virtual bool GenTask(const session::KernelGraph *graph);
  bool LaunchKernel(const session::KernelGraph *graph);
  virtual void AssignStaticMemoryInput(const session::KernelGraph *graph);

#ifdef ENABLE_DUMP_E2E
  DumpConfPtr GetDumpConf();
#endif
  virtual bool LoadTask(const session::KernelGraph *graph);
  virtual void FreeHostMemory();
  // for GPU and D to impl
  virtual void ReleaseDeviceRes() {}
  void set_device_id(uint32_t device_id) { device_id_ = device_id; }

 protected:
  virtual DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                               TypeId type_id) = 0;
  virtual bool SyncStream() = 0;
  void AssignStaticMemory(session::KernelGraph *graph);
  void AssignDynamicMemory(const session::KernelGraph *graph);
  void ReuseAssignDynamicMemory(session::KernelGraph *graph);
  void AssignNodeOutputMem(int flag, const AnfNodePtr &node, int index);
  void AssignWorkSpaceMem(const AnfNodePtr &node);
  void AssignReuseWorkSpaceMem(const AnfNodePtr &node);
  void AssignCommunicationNodeOutputMem(int flag, const AnfNodePtr &node);
  void UpdateRefNodeOutputMem(const session::KernelGraph *graph);
  void UpdateCommunicationOpInputMem(const AnfNodePtr &node);
  bool IsCommunicationOp(const AnfNodePtr &node);
  size_t GetCommonAlignSize(size_t input_size) const;
  size_t GetCommunicationAlignSize(size_t input_size) const;

  uint8_t *CalDeviceMem(const AnfNodePtr &node, size_t size, int flag, size_t index);
  virtual uint8_t *MallocStaticMem(size_t size, bool communication_mem);
  uint8_t *MallocDynamicMem(size_t size, bool communication_mem);
#ifdef ENABLE_DUMP_E2E
  bool SetDumpConf();
#endif
  // Alloc memory use the dynamic memory pool.
  virtual void *AllocTensorMemDynamic(size_t size);
  // Free memory use the dynamic memory pool.
  virtual void FreeTensorMemDynamic(void *device_ptr);
  virtual void MallocOpMemory(const DeviceAddressPtr address, size_t size, int flag);

 private:
  void AssignStaticMemoryOutput(const session::KernelGraph *graph);
  void AssignStaticMemoryValueNode(session::KernelGraph *graph);
  void GenLaunchArgs(const mindspore::kernel::KernelMod &kernel_mod, const AnfNodePtr &kernel,
                     AddressPtrList *kernel_inputs, AddressPtrList *kernel_workspaces, AddressPtrList *kernel_outputs);
  bool LaunchKernelMod(const session::KernelGraph &graph);
  void GenAddrCleanLaunchArgs(const CNodePtr &cnode, AddressPtrList *kernel_inputs);
  size_t CountNodeDeviceMemorySize(const AnfNodePtr &node, size_t output_index);
  void RunOpAssignInputMemory(const std::vector<tensor::TensorPtr> &input_tensors, const session::KernelGraph *graph);
  void RunOpAssignOutputMemory(const AnfNodePtr &kernel);
  void RunOpAssignWorkSpaceMemory(const AnfNodePtr &kernel);
  void AssignValueNodeTensor(const ValueNodePtr &value_node, const ValuePtr &node_value, size_t output_idx);

 protected:
  uint32_t device_id_{0};
  uint8_t *device_mem_base_{nullptr};
  uint8_t *device_mem_pool_base_{nullptr};
  uint64_t device_mem_size_{0};
  uint64_t device_mem_pool_size_{0};
  uint64_t dynamic_mem_offset_{0};
  uint64_t static_mem_offset_{0};
  const uint64_t mem_align_size_ = 512;
#ifdef ENABLE_DUMP_E2E
  DumpConfPtr dump_conf_ptr_;
#endif
  void *stream_ = nullptr;
  size_t total_static_size_ = 0;
  size_t total_dynamic_size_ = 0;
  MemReuseUtilPtr mem_reuse_util_ptr_{nullptr};

 private:
  uint8_t *reuse_mem_base_{nullptr};
};
using KernelRuntimePtr = std::shared_ptr<KernelRuntime>;
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEVICE_KERNEL_RUNTIME_H_
