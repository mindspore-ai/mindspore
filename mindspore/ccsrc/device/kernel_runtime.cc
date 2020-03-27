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

#include "device/kernel_runtime.h"
#include <utility>
#include <numeric>
#include <functional>
#include "common/utils.h"
#include "common/trans.h"
#include "utils/utils.h"
#include "utils/context/ms_context.h"
#include "operator/ops.h"
#include "pipeline/parse/python_adapter.h"
#include "session/kernel_graph.h"
#include "session/anf_runtime_algorithm.h"
#include "kernel/common_utils.h"
#include "kernel/oplib/oplib.h"
#include "ir/value.h"
using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;
using mindspore::memreuse::BestFitMemReuse;
using mindspore::memreuse::MemReuseUtilPtr;

namespace mindspore {
namespace device {
KernelRuntime::~KernelRuntime() {
  device_mem_base_ = nullptr;
  device_mem_pool_base_ = nullptr;
#ifdef ENABLE_DUMP_E2E
  dump_conf_ptr_ = nullptr;
#endif
  reuse_mem_base_ = nullptr;
  mem_reuse_util_ptr_ = nullptr;
}

bool KernelRuntime::Run(session::KernelGraph *graph) {
  bool ret = false;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_task_sink = context_ptr->enable_task_sink();
  if (is_task_sink) {
    ret = RunTask(graph);
  } else {
    ret = LaunchKernel(graph);
  }
  return ret;
}

// for D to impl
bool KernelRuntime::DumpData(mindspore::session::KernelGraph *graph) {
  if (graph != nullptr) {
    return true;
  }
  return false;
}

// for D to impl
bool KernelRuntime::GenTask(const session::KernelGraph *graph) {
  if (graph != nullptr) {
    return true;
  }
  return false;
}

bool KernelRuntime::LoadTask(const session::KernelGraph *graph) {
  if (graph != nullptr) {
    return true;
  }
  return false;
}

void KernelRuntime::FreeHostMemory() {
  dynamic_mem_offset_ = 0;
  static_mem_offset_ = 0;
}

// for D to impl
bool KernelRuntime::RunTask(const session::KernelGraph *graph) {
  if (graph != nullptr) {
    return true;
  }
  return false;
}

size_t KernelRuntime::CountNodeDeviceMemorySize(const mindspore::AnfNodePtr &node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_index >= AnfAlgo::GetOutputTensorNum(node)) {
    MS_EXCEPTION(ArgumentError) << "output index [" << output_index << "] large than the output size ["
                                << AnfAlgo::GetOutputTensorNum(node) << "] of node!";
  }
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(node, output_index);
  if (output_type_id == kTypeUnknown) {
    output_type_id = AnfAlgo::GetOutputInferDataType(node, output_index);
  }
  size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
  std::vector<size_t> shape = AnfAlgo::GetOutputDeviceShape(node, output_index);
  auto format = AnfAlgo::GetOutputFormat(node, output_index);
  if (shape.empty() && format != kOpFormat_DEFAULT) {
    shape = trans::TransShapeTo4d(shape);
    shape = trans::TransShapeToDevice(shape, format);
  }
  // scalar's output shape is a empty vector
  size_t tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
  return tensor_size;
}

void KernelRuntime::AssignMemory(session::KernelGraph *graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  AssignStaticMemory(graph);
  bool is_enable_mem_reuse = context_ptr->enable_mem_reuse();
  if (is_enable_mem_reuse) {
    ReuseAssignDynamicMemory(graph);
  } else {
    AssignDynamicMemory(graph);
  }
  UpdateRefNodeOutputMem(graph);
}

void KernelRuntime::RunOpAssignMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                      const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // assign memory for input nodes
  RunOpAssignInputMemory(input_tensors, graph);
  for (const auto &cnode : graph->execution_order()) {
    // assign memory for output nodes
    RunOpAssignOutputMemory(cnode);
    // assign memory for workspace
    RunOpAssignWorkSpaceMemory(cnode);
  }
  UpdateRefNodeOutputMem(graph);
}

void KernelRuntime::AssignStaticMemory(session::KernelGraph *graph) {
  AssignStaticMemoryInput(graph);
  AssignStaticMemoryValueNode(graph);
  AssignStaticMemoryOutput(graph);
}

void KernelRuntime::RunOpAssignInputMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                           const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (size_t input_index = 0; input_index < graph->inputs().size(); ++input_index) {
    auto item = graph->inputs()[input_index];
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>()) {
      continue;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      MS_EXCEPTION_IF_NULL(input_tensors[input_index]);
      if (input_tensors[input_index]->device_address().get() != nullptr) {
        AnfAlgo::SetOutputAddr(input_tensors[input_index]->device_address(), index, item.get());
        continue;
      }
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      if (output_type_id == kTypeUnknown) {
        output_type_id = AnfAlgo::GetOutputInferDataType(item, index);
      }
      auto tensor_size = CountNodeDeviceMemorySize(item, index);
      auto device_address =
        CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id);
      MS_EXCEPTION_IF_NULL(device_address);
      MallocOpMemory(device_address, tensor_size, kStaticMem);
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
}

void KernelRuntime::RunOpAssignOutputMemory(const AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    return;
  }
  if (AnfAlgo::GetCNodeName(kernel) == "ApplyMomentum") {
    auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, 0);
    AnfAlgo::SetOutputAddr(device_address, 0, kernel.get());
    return;
  }

  for (size_t i = 0; i < output_sizes.size(); ++i) {
    if (AnfAlgo::OutputAddrExist(kernel, i)) {
      continue;
    }
    std::string output_format = AnfAlgo::GetOutputFormat(kernel, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
    auto device_address = CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type);
    MS_EXCEPTION_IF_NULL(device_address);
    MallocOpMemory(device_address, output_sizes[i], kDynamicMem);
    AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
  }
}

void KernelRuntime::RunOpAssignWorkSpaceMemory(const AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  if (kernel->isa<CNode>()) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_lists.size(); ++i) {
      auto device_address = CreateDeviceAddress(nullptr, workspace_lists[i], "", kTypeUnknown);
      MS_EXCEPTION_IF_NULL(device_address);
      MallocOpMemory(device_address, workspace_lists[i], kDynamicMem);
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());
    }
  }
}

void KernelRuntime::AssignStaticMemoryInput(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &item : graph->inputs()) {
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>()) {
      continue;
    }
    if (AnfAlgo::OutputAddrExist(item, 0)) {
      continue;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      // if graph output is a weight and doesn't link to any cnode,it's data type will be unkonwn
      if (output_type_id == kTypeUnknown) {
        MS_LOG(WARNING) << "It is not suggested to use a lonely weight parameter as the output of graph";
        output_type_id = AnfAlgo::GetOutputInferDataType(item, index);
      }
      auto tensor_size = CountNodeDeviceMemorySize(item, index);
      auto ptr = MallocStaticMem(tensor_size, false);
      auto address = CreateDeviceAddress(ptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id);
      AnfAlgo::SetOutputAddr(address, index, item.get());
    }
  }
}

void KernelRuntime::AssignStaticMemoryOutput(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto nodes = AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  for (const auto &node : nodes) {
    auto item_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0);
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    if (!item_with_index.first->isa<CNode>() || !AnfAlgo::IsRealKernel(item_with_index.first)) {
      continue;
    }
    AssignNodeOutputMem(kStaticMem, item_with_index.first, SizeToInt(item_with_index.second));
  }
}

void KernelRuntime::UpdateRefNodeOutputMem(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &kernels = graph->execution_order();
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);

    auto output_sizes = kernel_mod->GetOutputSizeList();
    if (output_sizes.empty()) {
      MS_LOG(INFO) << "This kernel has no output size.";
      continue;
    }
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      session::AnfWithOutIndex out_pair(kernel, i);
      if (graph->IsInRefOutputMap(out_pair)) {
        auto origin_pair = graph->GetRefCorrespondOutput(out_pair);
        MS_EXCEPTION_IF_NULL(origin_pair.first);
        auto origin_node_output_addr = AnfAlgo::GetMutableOutputAddr(origin_pair.first, origin_pair.second);
        MS_EXCEPTION_IF_NULL(origin_node_output_addr);
        auto cur_node_output_addr = AnfAlgo::GetMutableOutputAddr(kernel, i);
        if (origin_node_output_addr.get() != cur_node_output_addr.get()) {
          MS_LOG(INFO) << "REF address is not same, ref node output need address update";
          MS_LOG(INFO) << "REF origin op is " << origin_pair.first->DebugString() << ", output index is "
                       << origin_pair.second << ", cur op is " << kernel->DebugString() << ", out index is " << i;
          AnfAlgo::SetOutputAddr(origin_node_output_addr, i, kernel.get());
        }
      }
    }
  }
}

void KernelRuntime::AssignCommunicationNodeOutputMem(int flag, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    MS_LOG(INFO) << "This kernel[" << node->DebugString() << "] has no output size.";
    return;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  size_t total_size = 0;
  std::vector<size_t> align_size_list;
  for (uint64_t mem_size : output_sizes) {
    if (context_ptr->enable_hccl()) {
      mem_size = GetCommonAlignSize(mem_size);
    }
    total_size += mem_size;
    align_size_list.emplace_back(mem_size);
  }
  uint8_t *output_ptr = CalDeviceMem(node, total_size, flag, 0);
  for (size_t j = 0; j < align_size_list.size(); ++j) {
    std::string output_format = AnfAlgo::GetOutputFormat(node, j);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(node, j);
    auto address = CreateDeviceAddress(output_ptr, output_sizes[j], output_format, output_type);
    AnfAlgo::SetOutputAddr(address, j, node.get());
    output_ptr += align_size_list[j];
  }
}

void KernelRuntime::UpdateCommunicationOpInputMem(const AnfNodePtr &node) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(node);
  size_t total_size = 0;
  std::vector<std::pair<mindspore::device::DeviceAddress *, size_t>> addr_size;
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(node); ++i) {
    auto address = AnfAlgo::GetPrevNodeMutableOutputAddr(node, i);
    MS_EXCEPTION_IF_NULL(address);
    auto mem_size = address->size();
    if (context_ptr->enable_hccl()) {
      mem_size = GetCommonAlignSize(mem_size);
    }
    total_size += mem_size;
    addr_size.emplace_back(address.get(), mem_size);
  }
  uint8_t *input_ptr = CalDeviceMem(node, total_size, kDynamicMem, 0);
  for (const auto &iter : addr_size) {
    MS_EXCEPTION_IF_NULL(iter.first);
    iter.first->set_ptr(input_ptr);
    input_ptr += iter.second;
  }
}

void KernelRuntime::AssignNodeOutputMem(int flag, const AnfNodePtr &node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsCommunicationOp(node)) {
    UpdateCommunicationOpInputMem(node);
    AssignCommunicationNodeOutputMem(flag, node);
    return;
  }
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    MS_LOG(INFO) << "This kernel[" << node->DebugString() << "] has no output size.";
    return;
  }
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    if ((kGetAllOuts != index) && (SizeToInt(i) != index)) {
      continue;
    }
    if (AnfAlgo::OutputAddrExist(node, i)) {
      MS_LOG(INFO) << "already malloc index:" << i;
      continue;
    }
    auto ptr = CalDeviceMem(node, output_sizes[i], flag, i);
    if (ptr == nullptr) {
      // reused ptr, no need alloc, continue;
      continue;
    }
    std::string output_format = AnfAlgo::GetOutputFormat(node, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(node, i);
    AnfAlgo::SetOutputAddr(CreateDeviceAddress(ptr, output_sizes[i], output_format, output_type), i, node.get());
  }
}

void KernelRuntime::AssignValueNodeTensor(const ValueNodePtr &value_node, const ValuePtr &node_value,
                                          size_t output_idx) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(node_value);
  auto tensor = node_value->cast<TensorPtr>();
  if (tensor == nullptr) {
    MS_LOG(WARNING) << "tensor is null";
    return;
  }
  size_t tensor_size = tensor->data().nbytes();
  auto node_size = CountNodeDeviceMemorySize(value_node, output_idx);
  auto ptr = MallocStaticMem(node_size, false);
  TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
  if (output_type_id == kTypeUnknown) {
    output_type_id = AnfAlgo::GetOutputInferDataType(value_node, output_idx);
  }
  auto address = CreateDeviceAddress(ptr, node_size, AnfAlgo::GetOutputFormat(value_node, output_idx), output_type_id);
  MS_EXCEPTION_IF_NULL(address);
  AnfAlgo::SetOutputAddr(address, output_idx, value_node.get());
  if (!address->SyncHostToDevice(tensor->shape(), tensor_size, tensor->data_type(), tensor->data_c(false))) {
    MS_EXCEPTION(NotExistsError) << "kValueNode SyncHostToDevice fail!" << value_node->DebugString() << "node format is"
                                 << AnfAlgo::GetOutputFormat(value_node, output_idx) << "node dtype is "
                                 << AnfAlgo::GetOutputInferDataType(value_node, output_idx);
  }
}

void KernelRuntime::AssignStaticMemoryValueNode(session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (AnfAlgo::OutputAddrExist(value_node, 0)) {
      MS_LOG(INFO) << "value_node[" << value_node->DebugString() << "] address already exist";
      continue;
    }
    auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<Tensor>()) {
      AssignValueNodeTensor(value_node, node_value, 0);
    } else if (node_value->isa<ValueTuple>()) {
      auto value_tuple = node_value->cast<ValueTuplePtr>();
      if (value_tuple == nullptr) {
        MS_LOG(WARNING) << "value_tuple is null";
        continue;
      }
      size_t i = 0;
      auto value_list = value_tuple->value();
      for (auto value_ptr : value_list) {
        if (value_ptr->isa<Tensor>()) {
          AssignValueNodeTensor(value_node, value_ptr, i++);
        }
      }
    } else if (node_value->isa<StringImm>()) {
      auto value = GetValue<std::string>(node_value);
      size_t tensor_size = value.size();
      auto ptr = MallocStaticMem(tensor_size, false);
      auto address = CreateDeviceAddress(ptr, tensor_size, kOpFormat_DEFAULT, kNumberTypeUInt8);
      MS_EXCEPTION_IF_NULL(address);
      AnfAlgo::SetOutputAddr(address, 0, value_node.get());
      std::vector<int> shape = {1, SizeToInt(tensor_size)};
      if (!address->SyncHostToDevice(shape, tensor_size, kNumberTypeUInt8, value.data())) {
        MS_LOG(EXCEPTION) << "kValueNode SyncHostToDevice fail!";
      }
    }
  }
}

void KernelRuntime::AssignDynamicMemory(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // reset dynamic mem offset
  dynamic_mem_offset_ = 0;
  auto &kernels = graph->execution_order();
  for (auto &kernel : kernels) {
    AssignNodeOutputMem(kDynamicMem, kernel, kGetAllOuts);
    AssignWorkSpaceMem(kernel);
  }
}

void KernelRuntime::ReuseAssignDynamicMemory(session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  dynamic_mem_offset_ = 0;
  MemReuseUtilPtr mem_reuse_util_ptr = std::make_shared<memreuse::MemReuseUtil>();
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  // set all infos
  mem_reuse_util_ptr->SetAllInfo(graph);
  auto bestfit_mem_reuse = std::make_shared<BestFitMemReuse>();
  MS_EXCEPTION_IF_NULL(bestfit_mem_reuse);
  bestfit_mem_reuse->Reuse(mem_reuse_util_ptr.get());
  size_t total_allocated_size = bestfit_mem_reuse->GetAllocatedSize();
  MS_LOG(INFO) << "TotalReuseDynamicSize [" << total_allocated_size << "]";
  auto base_ptr = MallocDynamicMem(total_allocated_size, false);
  reuse_mem_base_ = base_ptr;
  mem_reuse_util_ptr_ = mem_reuse_util_ptr;
  auto &kernels = graph->execution_order();
  for (auto &kernel : kernels) {
    AssignNodeOutputMem(kReuseDynamicMem, kernel, kGetAllOuts);
    AssignReuseWorkSpaceMem(kernel);
  }
}

void KernelRuntime::AssignReuseWorkSpaceMem(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto key = node.get();
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  size_t index = 0;
  auto iter = mem_reuse_util_ptr_->kernel_workspace_refs_.find(key);
  for (auto &size : kernel_mod->GetWorkspaceSizeList()) {
    if (iter != mem_reuse_util_ptr_->kernel_workspace_refs_.end()) {
      if (index >= iter->second.size()) {
        MS_LOG(EXCEPTION) << "index:[" << index << "] is larger than it's workspace size:[" << iter->second.size()
                          << "]";
      }
      auto wk_ref = iter->second[index];
      auto wk_ptr = reuse_mem_base_ + wk_ref->offset_;
      AnfAlgo::SetWorkspaceAddr(CreateDeviceAddress(wk_ptr, size, "", kTypeUnknown), index, node.get());
      index++;
    }
  }
}

void KernelRuntime::AssignWorkSpaceMem(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    size_t index = 0;
    for (auto &size : kernel_mod->GetWorkspaceSizeList()) {
      auto ptr = MallocDynamicMem(size, false);
      AnfAlgo::SetWorkspaceAddr(CreateDeviceAddress(ptr, size, "", kTypeUnknown), index, node.get());
      index++;
    }
  }
}

bool KernelRuntime::IsCommunicationOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_name = AnfAlgo::GetCNodeName(node);
  auto kernel_type = AnfAlgo::GetKernelType(node);
  if (kernel_name == kAllReduceOpName || kernel_type == HCCL_KERNEL) {
    return true;
  }
  return false;
}

uint8_t *KernelRuntime::CalDeviceMem(const AnfNodePtr &node, size_t size, int flag, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint8_t *ptr = nullptr;
  if (IsCommunicationOp(node)) {
    bool communication_mem = false;
    if (context_ptr->enable_hccl()) {
      communication_mem = true;
    }
    if (flag == kStaticMem) {
      ptr = MallocStaticMem(size, communication_mem);
    } else {
      ptr = MallocDynamicMem(size, communication_mem);
    }
    return ptr;
  }

  if (flag == kStaticMem) {
    ptr = MallocStaticMem(size, false);
  } else if (flag == kDynamicMem) {
    ptr = MallocDynamicMem(size, false);
  } else if (flag == kReuseDynamicMem) {
    auto key = node.get();
    auto iter = mem_reuse_util_ptr_->kernel_output_refs_.find(key);
    if (iter != mem_reuse_util_ptr_->kernel_output_refs_.end()) {
      // private member form KernelRuntime
      memreuse::KernelRefCountPtr kernel_ref_count_ptr = mem_reuse_util_ptr_->kernel_output_refs_[key][index];
      if (kernel_ref_count_ptr == nullptr) {
        return ptr;
      }
      ptr = reuse_mem_base_ + kernel_ref_count_ptr->offset_;
    } else {
      MS_LOG(EXCEPTION) << "node [" << AnfAlgo::GetCNodeName(node) << "] don't exist in kernel_output_refs";
    }
  }
  return ptr;
}

void KernelRuntime::GenLaunchArgs(const mindspore::kernel::KernelMod &kernel_mod, const mindspore::AnfNodePtr &kernel,
                                  AddressPtrList *kernel_inputs, AddressPtrList *const kernel_workspaces,
                                  AddressPtrList *kernel_outputs) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  MS_EXCEPTION_IF_NULL(kernel_workspaces);
  MS_EXCEPTION_IF_NULL(kernel_outputs);
  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetCNodeName(cnode) == kAtomicAddrCleanOpName) {
    return GenAddrCleanLaunchArgs(cnode, kernel_inputs);
  }
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    auto real_input = AnfAlgo::GetRealInputIndex(kernel, i);
    auto device_address = AnfAlgo::GetPrevNodeOutputAddr(kernel, real_input);
    kernel::AddressPtr input = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(input);
    input->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(input->addr);
    input->size = device_address->size_;
    kernel_inputs->emplace_back(input);
  }

  for (size_t i = 0; i < kernel_mod.GetOutputSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetOutputAddr(kernel, i);
    kernel::AddressPtr output = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(output);
    output->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(output->addr);
    output->size = device_address->size_;
    kernel_outputs->emplace_back(output);
  }

  for (size_t i = 0; i < kernel_mod.GetWorkspaceSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetWorkspaceAddr(kernel, i);
    kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(workspace);
    workspace->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(workspace->addr);
    workspace->size = device_address->size_;
    kernel_workspaces->emplace_back(workspace);
  }
}

void KernelRuntime::GenAddrCleanLaunchArgs(const CNodePtr &cnode, AddressPtrList *kernel_inputs) {
  if (cnode->inputs().size() != 2) {
    MS_LOG(EXCEPTION) << "atomic Addr clean Node Input nodes not equal 2.";
  }
  auto pre_node = cnode->inputs()[1];
  // set clean output address
  if (AnfAlgo::HasNodeAttr(kAttrAutomicOutputIndexs, pre_node)) {
    auto clean_output_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAutomicOutputIndexs);
    for (auto index : clean_output_indexs) {
      auto device_address = AnfAlgo::GetOutputAddr(pre_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      input->addr = device_address->ptr_;
      MS_EXCEPTION_IF_NULL(input->addr);
      input->size = device_address->size_;
      kernel_inputs->emplace_back(input);
    }
    MS_LOG(INFO) << "AtomicAddClean clean output size:" << clean_output_indexs.size();
  }
  // set clean workspace address
  if (AnfAlgo::HasNodeAttr(kAttrAutomicWorkspaceSize, pre_node)) {
    auto clean_workspaces = AnfAlgo::GetNodeAttr<int>(pre_node, kAttrAutomicWorkspaceSize);
    if (clean_workspaces != 0) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(pre_node, 0);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      workspace->addr = device_address->ptr_;
      MS_EXCEPTION_IF_NULL(workspace->addr);
      workspace->size = device_address->size_;
      kernel_inputs->emplace_back(workspace);
    }
    MS_LOG(INFO) << "AtomicAddClean clean workspace size" << clean_workspaces;
  }
}

bool KernelRuntime::LaunchKernelMod(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);

    AddressPtrList kernel_inputs;
    AddressPtrList kernel_workspaces;
    AddressPtrList kernel_outputs;
    GenLaunchArgs(*kernel_mod, kernel, &kernel_inputs, &kernel_workspaces, &kernel_outputs);
    struct timeval start_time, end_time;
    (void)gettimeofday(&start_time, nullptr);
    auto ret =
      kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, reinterpret_cast<uintptr_t>(stream_));
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed.";
      return false;
    } else {
      if (AnfAlgo::GetKernelType(kernel) == TBE_KERNEL && !SyncStream()) {
        MS_LOG(EXCEPTION) << "SyncStream failed.";
      }
      (void)gettimeofday(&end_time, nullptr);
      const uint64_t kUSecondInSecond = 1000000;
      uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
      cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
      MS_LOG(DEBUG) << "d " << kernel->fullname_with_scope() << " in  " << cost << " us";
    }
  }
  return true;
}

size_t KernelRuntime::GetCommonAlignSize(size_t input_size) const {
  return (input_size + mem_align_size_ + 31) / mem_align_size_ * mem_align_size_;
}

size_t KernelRuntime::GetCommunicationAlignSize(size_t input_size) const {
  return (input_size + mem_align_size_ - 1) / mem_align_size_ * mem_align_size_ + 2 * mem_align_size_;
}

uint8_t *KernelRuntime::MallocStaticMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }
  if (static_mem_offset_ < align_size) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  total_static_size_ += align_size;
  auto offset = static_mem_offset_ - align_size;
  if (dynamic_mem_offset_ > offset) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  static_mem_offset_ = offset;
  if (communication_mem) {
    return device_mem_base_ + offset + mem_align_size_;
  } else {
    return device_mem_base_ + offset;
  }
}

uint8_t *KernelRuntime::MallocDynamicMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }
  uint64_t offset = dynamic_mem_offset_;
  auto new_offset = dynamic_mem_offset_ + align_size;
  if (new_offset > static_mem_offset_) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  total_dynamic_size_ += align_size;
  dynamic_mem_offset_ = new_offset;

  if (communication_mem) {
    return device_mem_base_ + offset + mem_align_size_;
  } else {
    return device_mem_base_ + offset;
  }
}

bool KernelRuntime::LaunchKernel(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!LaunchKernelMod(*graph)) {
    MS_LOG(ERROR) << "LaunchKernelMod failed.";
    return false;
  }
  if (!SyncStream()) {
    MS_LOG(ERROR) << "SyncStream failed.";
    return false;
  }
  return true;
}

void KernelRuntime::MallocOpMemory(const DeviceAddressPtr address, size_t size, int flag) {
  if (flag == kStaticMem) {
    address->ptr_ = MallocStaticMem(size, false);
  } else if (flag == kDynamicMem) {
    address->ptr_ = MallocDynamicMem(size, false);
  } else {
    MS_LOG(EXCEPTION) << "Unknown memory type!";
  }
}

void *KernelRuntime::AllocTensorMemDynamic(size_t size) {
  if (size == 0) {
    MS_LOG(ERROR) << "AllocTensorMemDynamic size is 0.";
  }
  return nullptr;
}

void KernelRuntime::FreeTensorMemDynamic(void *device_ptr) {
  if (device_ptr == nullptr) {
    MS_LOG(ERROR) << "FreeTensorMemDynamic device_ptr is null.";
  }
}

#ifdef ENABLE_DUMP_E2E
bool KernelRuntime::SetDumpConf() {
  dump_conf_ptr_ = std::make_shared<Dump>();
  MS_EXCEPTION_IF_NULL(dump_conf_ptr_);
  bool ret = dump_conf_ptr_->SetDumpConfFromJsonFile();
  return ret;
}

DumpConfPtr KernelRuntime::GetDumpConf() { return dump_conf_ptr_; }
#endif
}  // namespace device
}  // namespace mindspore
