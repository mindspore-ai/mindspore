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
#include "runtime/device/cpu/cpu_kernel_runtime.h"
#include <unistd.h>
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <utility>
#include <algorithm>
#include <functional>
#include <exception>
#include "backend/kernel_compiler/kernel.h"
#include "runtime/device/cpu/cpu_device_address.h"
#include "runtime/device/cpu/cpu_memory_manager.h"
#include "utils/ms_context.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/session_basic.h"
#include "frontend/operator/ops.h"
#include "profiler/device/cpu/cpu_profiling.h"
#include "utils/shape_utils.h"
#include "utils/profile.h"
#include "utils/trace_base.h"
#include "debug/data_dump/cpu_e2e_dump.h"
#include "debug/env_config_parser.h"
#ifdef MEM_REUSE_DEBUG
#include "backend/optimizer/mem_reuse/mem_reuse_checker.h"
#endif

namespace mindspore {
namespace device {
namespace cpu {

bool CPUKernelRuntime::Init() {
  if (initialized_) {
    return true;
  }
  mem_manager_ = std::make_shared<CPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  initialized_ = true;
  return true;
}

const size_t INIT_NODE_REF = 1;
void CPUKernelRuntime::AssignKernelAddress(session::KernelGraph *kernel_graph) {
  AssignValueNodeAddress(kernel_graph);
  AssignInputNodeAddress(kernel_graph);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_enable_mem_reuse = EnvConfigParser::GetInstance().GetSysMemreuse();
  if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
    // disable mem reuse for kPynativeMode
    is_enable_mem_reuse = false;
  }
  if (is_enable_mem_reuse) {
    MS_EXCEPTION_IF_NULL(mem_manager_);
    mem_manager_->ResetDynamicMemory();
    AssignDynamicMemory(kernel_graph);
#ifdef MEM_REUSE_DEBUG
    // Get normal graph ir for memreuse
    mindspore::memreuse::MemReuseChecker::GetInstance().CheckNormalIR(kernel_graph);
#endif
  } else {
    AssignKernelOutputAddress(kernel_graph);
    static_cast<CPUMemoryManager *>(mem_manager_.get())->AssignMemory(kernel_graph);
  }
}

void CPUKernelRuntime::AssignValueNodeAddress(session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &item_node : kernel_graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(item_node);
    if (item_node->isa<ValueNode>()) {
      auto value_node = item_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto node_value = value_node->value();
      MS_EXCEPTION_IF_NULL(node_value);
      if (!node_value->isa<tensor::Tensor>()) {
        continue;
      }
      auto tensor = node_value->cast<TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor->device_address() != nullptr) {
        AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), 0,
                               item_node.get());
        continue;
      }
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item_node, 0);
      if (output_type_id == kTypeUnknown) {
        output_type_id = AnfAlgo::GetOutputInferDataType(item_node, 0);
      }
      size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
      ShapeVector data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(), type_size, std::multiplies<size_t>());
      DeviceAddressPtr address = nullptr;
      address = CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, output_type_id);
      MS_EXCEPTION_IF_NULL(address);
      if (tensor->data_type() == output_type_id) {
        address->ptr_ = tensor->data_c();
      } else {
        address->ptr_ = static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(tensor_size);
        if (!address->SyncHostToDevice(data_shape, LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                       tensor->data_c())) {
          MS_LOG(EXCEPTION) << "Value node sync host to device failed!";
        }
      }
      address->ref_count_ = INIT_NODE_REF;
      AnfAlgo::SetOutputAddr(address, 0, item_node.get());
    }
  }
}

void CPUKernelRuntime::AssignInputNodeAddress(const session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &item : kernel_graph->inputs()) {
    MS_EXCEPTION_IF_NULL(item);
    if (item->isa<Parameter>()) {
      auto output_num = AnfAlgo::GetOutputTensorNum(item);
      for (size_t index = 0; index < output_num; index++) {
        TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
        if (output_type_id == kTypeUnknown) {
          output_type_id = AnfAlgo::GetOutputInferDataType(item, index);
        }
        std::vector<size_t> fmt_shape = AnfAlgo::GetOutputDeviceShape(item, index);
        size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
        size_t tensor_size =
          fmt_shape.empty() ? type_size
                            : std::accumulate(fmt_shape.begin(), fmt_shape.end(), type_size, std::multiplies<size_t>());
        auto format = AnfAlgo::GetOutputFormat(item, index);
        auto address = CreateDeviceAddress(nullptr, tensor_size, format, output_type_id);
        AnfAlgo::SetOutputAddr(address, index, item.get());
      }
    }
  }
}

void CPUKernelRuntime::AssignKernelOutputAddress(const session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto kernels = kernel_graph->execution_order();
  for (auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      AnfAlgo::SetOutputAddr(CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type), i,
                             kernel.get());
    }
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      AnfAlgo::SetWorkspaceAddr(CreateDeviceAddress(nullptr, workspace_sizes[i], kOpFormat_DEFAULT, kNumberTypeFloat32),
                                i, kernel.get());
    }
  }
}

DeviceAddressPtr CPUKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                       TypeId type_id) {
  return std::make_shared<CPUDeviceAddress>(device_ptr, device_size, format, type_id);
}

tensor::TensorPtr CPUKernelRuntime::CreatTensorForOutput(
  session::KernelGraph *kernel_graph, const CNodePtr &node, size_t index,
  std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  size_t output_size = AnfAlgo::GetOutputTensorNum(node);
  if (index >= output_size) {
    MS_LOG(EXCEPTION) << "Invalid input index " << index;
  }
  auto address = AnfAlgo::GetMutableOutputAddr(node, index);
  MS_EXCEPTION_IF_NULL(address);
  TypeId infer_type_id = AnfAlgo::GetOutputInferDataType(node, index);
  TypeId device_type_id = AnfAlgo::GetOutputDeviceDataType(node, index);
  tensor::TensorPtr tensor = kernel_graph->GetInternalOutputTensor(node, index);
  if (tensor == nullptr) {
    auto shape = AnfAlgo::GetOutputInferShape(node, index);
    ShapeVector temp_shape;
    (void)temp_shape.insert(temp_shape.end(), shape.begin(), shape.end());
    size_t type_size = GetTypeByte(TypeIdToType(device_type_id));
    size_t tensor_size = std::accumulate(temp_shape.begin(), temp_shape.end(), type_size, std::multiplies<size_t>());
    if (tensor_size < address->size_) {
      temp_shape.clear();
      temp_shape.emplace_back(address->size_);
    }
    tensor = std::make_shared<tensor::Tensor>(infer_type_id, temp_shape);
    bool is_internal_output = kernel_graph->IsInternalOutput(node, index);
    if (is_internal_output) {
      kernel_graph->AddInternalOutputTensor(node, index, tensor);
    }
  }
  tensor->set_device_address(address);
  if (bound_addresses_.find(address) != bound_addresses_.end()) {
    tensor->set_sync_status(kNeedSyncDeviceToHostImmediately);
  } else {
    if (infer_type_id != device_type_id) {
      size_t type_size = GetTypeByte(TypeIdToType(device_type_id));
      ShapeVector data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(), type_size, std::multiplies<size_t>());
      address->ptr_ = static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(tensor_size);
      tensor->set_sync_status(kNeedSyncDeviceToHostImmediately);
    } else {
      tensor->set_sync_status(kNoNeedSync);
    }
    (void)bound_addresses_.insert(address);
  }
  session::KernelWithIndex node_index(node, index);
  tensor->SetNeedWait(true);
  tensor->SetIsGraphOutput();
  (*tensor_to_node)[tensor] = node_index;
  return tensor;
}

BaseRef CPUKernelRuntime::CreatTensorForOutput(session::KernelGraph *kernel_graph,
                                               const session::KernelWithIndex &kernel_with_index,
                                               std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  auto &input_node = kernel_with_index.first;
  auto index = kernel_with_index.second;
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<CNode>()) {
    auto node = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    if (AnfAlgo::GetCNodeName(input_node) == prim::kPrimMakeTuple->name()) {
      VectorRef ret;
      for (size_t i = 1; i < node->inputs().size(); i++) {
        auto item_with_index = AnfAlgo::VisitKernelWithReturnType(node->input(i), 0);
        auto out = CreatTensorForOutput(kernel_graph, item_with_index, tensor_to_node);
        ret.push_back(out);
      }
      return ret;
    }
    return CreatTensorForOutput(kernel_graph, node, index, tensor_to_node);
  } else if (input_node->isa<Parameter>()) {
    auto iter = input_param_tensor_map_.find(input_node);
    if (iter != input_param_tensor_map_.end()) {
      return iter->second;
    }
  } else if (input_node->isa<ValueNode>()) {
    auto value_node = input_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->value();
  }
  return BaseRef();
}

void CPUKernelRuntime::CreateOutputTensors(session::KernelGraph *kernel_graph,
                                           const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs,
                                           std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  auto &input_nodes = kernel_graph->inputs();
  if (input_nodes.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Input size not equal to input node size!";
  }

  size_t input_idx = 0;
  for (auto &item : input_nodes) {
    MS_EXCEPTION_IF_NULL(item);
    input_param_tensor_map_[item] = inputs[input_idx];
    input_idx++;
  }

  bound_addresses_.clear();
  auto output_nodes = kernel_graph->outputs();
  for (const auto &item : output_nodes) {
    auto item_with_index = AnfAlgo::VisitKernelWithReturnType(item, 0, false);
    auto out = CreatTensorForOutput(kernel_graph, item_with_index, tensor_to_node);
    outputs->push_back(std::move(out));
  }
  input_param_tensor_map_.clear();
}

void CPUKernelRuntime::BindInputTensorAddressPtr(const session::KernelGraph &kernel_graph,
                                                 const std::vector<tensor::TensorPtr> &inputs) {
  auto &input_nodes = kernel_graph.inputs();
  if (input_nodes.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Input size not equal to input node size!";
  }
  size_t input_idx = 0;
  for (auto &item : input_nodes) {
    MS_EXCEPTION_IF_NULL(item);
    if (item->isa<Parameter>() && !HasAbstractMonad(item)) {
      auto address = AnfAlgo::GetMutableOutputAddr(item, 0);
      auto tensor = inputs[input_idx];
      auto tensor_address = tensor->device_address();
      MS_EXCEPTION_IF_NULL(address);
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor_address != nullptr && tensor_address != address &&
          (std::dynamic_pointer_cast<device::DeviceAddress>(tensor_address)->DeviceType() != DeviceAddressType::kCPU ||
           AnfAlgo::IsParameterWeight(item->cast<ParameterPtr>()))) {
        tensor->data_sync(false);
      }
      if (GetTypeByte(TypeIdToType(tensor->data_type())) == GetTypeByte(TypeIdToType(address->type_id_))) {
        address->ptr_ = tensor->data_c();
      } else {
        ShapeVector data_shape = tensor->shape();
        size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(),
                                             GetTypeByte(TypeIdToType(address->type_id_)), std::multiplies<size_t>());
        address->ptr_ = static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(tensor_size);
        if (!address->SyncHostToDevice(data_shape, LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                       tensor->data_c())) {
          MS_LOG(EXCEPTION) << "Parameter node sync host to device failed!";
        }
      }
      if (item->cast<ParameterPtr>()->is_used_by_dynamic_kernel()) {
        auto tensor_shape = tensor->shape();
        std::vector<size_t> shape_tmp;
        (void)std::transform(tensor_shape.begin(), tensor_shape.end(), std::back_inserter(shape_tmp), IntToSize);
        AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(item, 0)}, {shape_tmp}, item.get());
      }
      address->ref_count_ = INIT_NODE_REF;
      tensor->set_device_address(address);
    }
    input_idx++;
  }
}

void CPUKernelRuntime::BindOutputTensorAddressPtr(const VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  for (auto item : *outputs) {
    if (utils::isa<VectorRefPtr>(item)) {
      auto vector_ref = utils::cast<VectorRef>(item);
      BindOutputTensorAddressPtr(&vector_ref);
    } else if (utils::isa<tensor::TensorPtr>(item)) {
      auto tensor = utils::cast<tensor::TensorPtr>(item);
      MS_EXCEPTION_IF_NULL(tensor);
      auto address = tensor->device_address();
      if (address == nullptr) {
        continue;
      }
      auto address_ptr = std::dynamic_pointer_cast<device::DeviceAddress>(address);
      if (tensor->sync_status() == kNoNeedSync) {
        address_ptr->ptr_ = tensor->data_c();
      }
      address_ptr->ref_count_ = INIT_NODE_REF;
    }
  }
}

void CPUKernelRuntime::BindInputOutput(session::KernelGraph *kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                                       VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  BindInputTensorAddressPtr(*kernel_graph, inputs);
  BindOutputTensorAddressPtr(outputs);
}

void CPUKernelRuntime::AddRuntimeAddress(DeviceAddress *address, std::vector<kernel::AddressPtr> *input_list) {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(input_list);
  kernel::AddressPtr input = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(input);
  if (address->ptr_ == nullptr) {
    address->ptr_ = static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(address->size_);
  }
  MS_EXCEPTION_IF_NULL(address->ptr_);
  input->addr = address->ptr_;
  input->size = address->size_;
  input_list->push_back(input);
}

void CPUKernelRuntime::IncreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs) {
  static_cast<CPUMemoryManager *>(mem_manager_.get())->IncreaseSummaryRefCount(summary_outputs);
}

void CPUKernelRuntime::DecreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs) {
  static_cast<CPUMemoryManager *>(mem_manager_.get())->DecreaseSummaryRefCount(summary_outputs);
}

bool CPUKernelRuntime::Run(session::KernelGraph *const kernel_graph, bool is_task_sink) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  static_cast<CPUMemoryManager *>(mem_manager_.get())->IncreaseAddressRefCount(kernel_graph);

  auto kernels = kernel_graph->execution_order();
  auto profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  auto &dump_json_parser = DumpJsonParser::GetInstance();
  dump_json_parser.UpdateDumpIter();
  bool iter_dump_flag = dump_json_parser.GetIterDumpFlag();

  for (const auto &kernel : kernels) {
#ifdef ENABLE_PROFILE
    double start_time = GetTime();
#endif
    if (AnfAlgo::IsDynamicShape(kernel)) {
      AnfAlgo::InferShape(kernel);
    }
    std::vector<kernel::AddressPtr> kernel_inputs;
    std::vector<kernel::AddressPtr> kernel_workspaces;
    std::vector<kernel::AddressPtr> kernel_outputs;
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel);
    for (size_t i = 0; i < input_num; ++i) {
      auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i).get();
      MS_EXCEPTION_IF_NULL(device_address);
      AddRuntimeAddress(device_address, &kernel_inputs);
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel);
    for (size_t i = 0; i < output_num; ++i) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i).get();
      MS_EXCEPTION_IF_NULL(device_address);
      AddRuntimeAddress(device_address, &kernel_outputs);
    }
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(device_address);
      AddRuntimeAddress(device_address, &kernel_workspaces);
    }
    bool ret = true;
    if (profiler_inst->GetEnableFlag()) {
      uint32_t pid = getpid();
      profiler_inst->OpDataProducerBegin(kernel->fullname_with_scope(), pid);
    }
    try {
      ret = kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, 0);
    } catch (std::exception &e) {
      MS_LOG(EXCEPTION) << e.what() << "\nTrace:" << trace::DumpSourceLines(kernel);
    }
    if (iter_dump_flag) {
      CPUE2eDump::DumpCNodeData(kernel);
    }
    if (profiler_inst->GetEnableFlag()) {
      profiler_inst->OpDataProducerEnd();
    }
    if (!ret) {
      MS_LOG(EXCEPTION) << "Launch kernel failed. Trace:" << trace::DumpSourceLines(kernel);
    }
    static_cast<CPUMemoryManager *>(mem_manager_.get())->DecreaseAddressRefCount(kernel);
#ifdef ENABLE_PROFILE
    double cost_time = GetTime() - start_time;
    MS_LOG(INFO) << "cpu kernel: " << kernel->fullname_with_scope() << "  costs " << cost_time * 1e6 << " us";
#endif
  }
  if (iter_dump_flag) {
    CPUE2eDump::DumpParametersAndConst(kernel_graph);
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
