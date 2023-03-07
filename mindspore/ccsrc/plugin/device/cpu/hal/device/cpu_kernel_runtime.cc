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
#include "plugin/device/cpu/hal/device/cpu_kernel_runtime.h"
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <utility>
#include <algorithm>
#include <functional>
#include <exception>
#include "kernel/kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "utils/ms_context.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/session_basic.h"
#include "frontend/operator/ops.h"
#include "plugin/device/cpu/hal/profiler/cpu_profiling.h"
#include "utils/shape_utils.h"
#include "utils/profile.h"
#include "utils/trace_base.h"
#include "debug/data_dump/cpu_e2e_dump.h"
#include "include/common/debug/env_config_parser.h"
#ifdef MEM_REUSE_DEBUG
#include "backend/common/mem_reuse/mem_reuse_checker.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#include "debug/rdr/mem_address_recorder.h"
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
void CPUKernelRuntime::AssignKernelGraphAddress(const session::KernelGraph *kernel_graph) {
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
    MS_EXCEPTION_IF_NULL(kernel_graph);
    AssignDynamicMemory(*kernel_graph);
#ifdef MEM_REUSE_DEBUG
    // Get normal graph ir for memreuse
    mindspore::memreuse::MemReuseChecker::GetInstance().CheckNormalIR(kernel_graph);
#endif
  } else {
    AssignKernelOutputAddress(kernel_graph);
    static_cast<CPUMemoryManager *>(mem_manager_.get())->AssignMemory(kernel_graph);
  }
}

void CPUKernelRuntime::AssignValueNodeAddress(const session::KernelGraph *kernel_graph) {
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
        output_type_id = common::AnfAlgo::GetOutputInferDataType(item_node, 0);
      }
      size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
      ShapeVector data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(), type_size, std::multiplies<size_t>());
      DeviceAddressPtr address = nullptr;
      address = CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, output_type_id);
      address->set_from_persistent_mem(tensor->is_parameter());
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

void CPUKernelRuntime::AssignInputNodeAddress(const session::KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &item : kernel_graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(item);
    if (item->isa<Parameter>()) {
      auto output_num = AnfAlgo::GetOutputTensorNum(item);
      for (size_t index = 0; index < output_num; index++) {
        TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
        if (output_type_id == kTypeUnknown) {
          output_type_id = common::AnfAlgo::GetOutputInferDataType(item, index);
        }
        auto fmt_shape = AnfAlgo::GetOutputDeviceShape(item, index);
        size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
        size_t tensor_size = type_size * SizeOf(fmt_shape);
        auto format = AnfAlgo::GetOutputFormat(item, index);
        auto address = CreateDeviceAddress(nullptr, tensor_size, format, output_type_id);
        address->set_from_persistent_mem(true);
        AnfAlgo::SetOutputAddr(address, index, item.get());
      }
    }
  }
}

void CPUKernelRuntime::AssignKernelOutputAddress(const session::KernelGraph *kernel_graph) const {
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
                                                       TypeId type_id) const {
  return std::make_shared<CPUDeviceAddress>(device_ptr, device_size, format, type_id);
}

DeviceAddressPtr CPUKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                       TypeId type_id, const KernelWithIndex &node_index) const {
  return std::make_shared<CPUDeviceAddress>(device_ptr, device_size, format, type_id, node_index);
}

tensor::TensorPtr CPUKernelRuntime::CreateTensorForOutput(session::KernelGraph *kernel_graph, const CNodePtr &node,
                                                          size_t index, std::set<DeviceAddressPtr> *bound_addresses) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(bound_addresses);
  size_t output_size = AnfAlgo::GetOutputTensorNum(node);
  if (index >= output_size) {
    MS_LOG(EXCEPTION) << "For node " << node->DebugString() << ", index " << index << " exceed output size "
                      << output_size;
  }
  auto address = AnfAlgo::GetMutableOutputAddr(node, index);
  MS_EXCEPTION_IF_NULL(address);
  TypeId infer_type_id = common::AnfAlgo::GetOutputInferDataType(node, index);
  TypeId device_type_id = AnfAlgo::GetOutputDeviceDataType(node, index);
  auto shape = common::AnfAlgo::GetOutputInferShape(node, index);
  ShapeVector temp_shape;
  tensor::TensorPtr tensor;
  bool is_internal_output = kernel_graph->IsInternalOutput(node, index);
  (void)temp_shape.insert(temp_shape.end(), shape.begin(), shape.end());
  if (is_internal_output) {
    tensor = kernel_graph->GetInternalOutputTensor(node, index);
    if (tensor == nullptr) {
      size_t type_size = GetTypeByte(TypeIdToType(device_type_id));
      if (type_size == 0) {
        MS_LOG(EXCEPTION) << "Invalid type_size " << type_size;
      }
      size_t tensor_size = std::accumulate(temp_shape.begin(), temp_shape.end(), type_size, std::multiplies<size_t>());
      if (tensor_size < address->size_) {
        temp_shape.clear();
        (void)temp_shape.emplace_back(address->size_ / type_size);
      }
      tensor = std::make_shared<tensor::Tensor>(infer_type_id, temp_shape);
    }
    kernel_graph->AddInternalOutputTensor(node, index, tensor);
  } else {
    tensor = std::make_shared<tensor::Tensor>(infer_type_id, temp_shape);
  }
  tensor->set_device_address(address);
  tensor->set_sync_status(kNeedSyncDeviceToHostImmediately);
  if (bound_addresses->find(address) == bound_addresses->end()) {
    if (infer_type_id != device_type_id) {
      size_t type_size = GetTypeByte(TypeIdToType(device_type_id));
      ShapeVector data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(), type_size, std::multiplies<size_t>());
      address->ptr_ = static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(tensor_size);
      address->size_ = tensor_size;
      address->type_id_ = device_type_id;
    } else {
      tensor->set_sync_status(kNoNeedSync);
    }
    (void)bound_addresses->insert(address);
  }
  tensor->SetNeedWait(true);
  tensor->SetIsGraphOutput();
  return tensor;
}

BaseRef CPUKernelRuntime::GetOrCreateTensorForOutput(
  session::KernelGraph *kernel_graph, const session::KernelWithIndex &kernel_with_index,
  std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
  std::map<AnfNodePtr, tensor::TensorPtr> *input_param_tensor_map, std::set<DeviceAddressPtr> *bound_addresses) {
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  MS_EXCEPTION_IF_NULL(input_param_tensor_map);
  auto &input_node = kernel_with_index.first;
  auto index = kernel_with_index.second;
  MS_EXCEPTION_IF_NULL(input_node);

  if (input_node->isa<CNode>()) {
    auto node = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    if (common::AnfAlgo::GetCNodeName(input_node) == prim::kPrimMakeTuple->name()) {
      VectorRef ret;
      for (size_t i = 1; i < node->inputs().size(); i++) {
        auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(node->input(i), 0);
        auto out = GetOrCreateTensorForOutput(kernel_graph, item_with_index, tensor_to_node, input_param_tensor_map,
                                              bound_addresses);
        ret.push_back(out);
      }
      return ret;
    }
    auto tensor = CreateTensorForOutput(kernel_graph, node, index, bound_addresses);
    (*tensor_to_node)[tensor] = kernel_with_index;
    return tensor;
  } else if (input_node->isa<Parameter>()) {
    auto iter = input_param_tensor_map->find(input_node);
    if (iter != input_param_tensor_map->end()) {
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
  auto &input_nodes = kernel_graph->input_nodes();
  if (input_nodes.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Input size " << inputs.size() << " is not equal to input node size " << input_nodes.size();
  }

  std::map<AnfNodePtr, tensor::TensorPtr> input_param_tensor_map;
  size_t input_idx = 0;
  for (auto &item : input_nodes) {
    MS_EXCEPTION_IF_NULL(item);
    input_param_tensor_map[item] = inputs[input_idx];
    input_idx++;
  }

  std::set<DeviceAddressPtr> bound_addresses;
  auto output_nodes = kernel_graph->outputs();
  for (const auto &item : output_nodes) {
    auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(item, 0, false);
    auto out = GetOrCreateTensorForOutput(kernel_graph, item_with_index, tensor_to_node, &input_param_tensor_map,
                                          &bound_addresses);
    outputs->push_back(std::move(out));
  }
}

void CPUKernelRuntime::BindInputTensorAddressPtr(const session::KernelGraph &kernel_graph,
                                                 const std::vector<tensor::TensorPtr> &inputs) {
  auto &input_nodes = kernel_graph.input_nodes();
  if (input_nodes.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Input size" << inputs.size() << " is not equal to input node size " << input_nodes.size();
  }
  for (size_t input_idx = 0; input_idx < input_nodes.size(); ++input_idx) {
    auto &item = input_nodes[input_idx];
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>() || HasAbstractMonad(item)) {
      continue;
    }
    auto address = AnfAlgo::GetMutableOutputAddr(item, 0);
    auto tensor = inputs[input_idx];
    MS_EXCEPTION_IF_NULL(address);
    MS_EXCEPTION_IF_NULL(tensor);
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode) {
      auto tensor_address = tensor->device_address();
      if (common::AnfAlgo::IsParameterWeight(item->cast<ParameterPtr>()) && tensor_address != nullptr &&
          tensor_address != address) {
        tensor->data_sync();
      }
    }
    if (GetTypeByte(TypeIdToType(tensor->data_type())) == GetTypeByte(TypeIdToType(address->type_id_))) {
      address->ptr_ = tensor->data_c();
    } else {
      ShapeVector data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(),
                                           GetTypeByte(TypeIdToType(address->type_id_)), std::multiplies<size_t>());
      if (address->ptr_ == nullptr || address->size_ != tensor_size) {
        address->ptr_ = static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(tensor_size);
        address->size_ = tensor_size;
      }
      if (!address->SyncHostToDevice(data_shape, LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                     tensor->data_c())) {
        MS_LOG(EXCEPTION) << "Parameter node sync host to device failed!";
      }
    }
    auto input_param = item->cast<ParameterPtr>();
    if (input_param != nullptr && input_param->IsUsedByRealKernelInGraph(kernel_graph.graph_id())) {
      auto tensor_shape = tensor->shape();
      common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(item, 0)}, {tensor_shape},
                                                  item.get());
    }
    address->ref_count_ = INIT_NODE_REF;
    if (common::AnfAlgo::IsParameterWeight(input_param)) {
      tensor->set_device_address(address);
    }
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
      if (address_ptr->type_id_ == tensor->data_type_c() && tensor->sync_status() == kNoNeedSync) {
        address_ptr->ptr_ = tensor->data_c();
      }
      address_ptr->ref_count_ = INIT_NODE_REF;
    }
  }
}

void CPUKernelRuntime::BindInputOutput(session::KernelGraph *kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                                       VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
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

void CPUKernelRuntime::GetRuntimeAddressFromNode(const AnfNodePtr &node, std::vector<kernel::AddressPtr> *inputs,
                                                 std::vector<kernel::AddressPtr> *outputs,
                                                 std::vector<kernel::AddressPtr> *workspaces) {
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(workspaces);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_num; ++i) {
    auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(node, i).get();
    MS_EXCEPTION_IF_NULL(device_address);
    AddRuntimeAddress(device_address, inputs);
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; ++i) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(node, i).get();
    MS_EXCEPTION_IF_NULL(device_address);
    AddRuntimeAddress(device_address, outputs);
  }
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetWorkspaceAddr(node, i);
    MS_EXCEPTION_IF_NULL(device_address);
    AddRuntimeAddress(device_address, workspaces);
  }
}

bool CPUKernelRuntime::Run(const session::KernelGraph &kernel_graph, bool) {
  static_cast<CPUMemoryManager *>(mem_manager_.get())->IncreaseAddressRefCount(&kernel_graph);

  auto kernels = kernel_graph.execution_order();

#ifndef ENABLE_SECURITY
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool iter_dump_flag = dump_json_parser.GetIterDumpFlag();
  uint32_t graph_id = kernel_graph.graph_id();
#endif
#ifdef ENABLE_DUMP_IR
  std::string name = "mem_address_list";
  (void)mindspore::RDR::RecordMemAddressInfo(SubModuleId::SM_KERNEL, name);
#endif
  for (const auto &kernel : kernels) {
#ifdef ENABLE_PROFILE
    double start_time = GetTime();
#endif
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    // akg kernel do not support dynamic shape by now
    kernel::NativeCpuKernelMod *cpu_kernel = nullptr;
    if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) != KernelType::AKG_KERNEL) {
      cpu_kernel = dynamic_cast<kernel::NativeCpuKernelMod *>(kernel_mod);
      MS_EXCEPTION_IF_NULL(cpu_kernel);
    }
    if (common::AnfAlgo::IsDynamicShape(kernel)) {
      AnfAlgo::InferShape(kernel);
      auto args = kernel::GetArgsFromCNode(kernel);
      if (cpu_kernel != nullptr && cpu_kernel->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map) ==
                                     static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
        MS_LOG(EXCEPTION) << "Node " << kernel->fullname_with_scope() << " Resize failed!";
      }
    }
    std::vector<kernel::AddressPtr> kernel_inputs;
    std::vector<kernel::AddressPtr> kernel_workspaces;
    std::vector<kernel::AddressPtr> kernel_outputs;
    GetRuntimeAddressFromNode(kernel, &kernel_inputs, &kernel_outputs, &kernel_workspaces);
    bool ret = true;
#ifndef ENABLE_SECURITY
    auto profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
    MS_EXCEPTION_IF_NULL(profiler_inst);
    if (profiler_inst->GetEnableFlag()) {
      uint32_t pid = getpid();
      profiler_inst->OpDataProducerBegin(kernel->fullname_with_scope(), pid);
    }
#endif
#ifdef ENABLE_DUMP_IR
    kernel::KernelLaunchInfo mem_info = {kernel_inputs, kernel_workspaces, kernel_outputs};
    std::string op_name = kernel->fullname_with_scope();
    (void)mindspore::RDR::UpdateMemAddress(SubModuleId::SM_KERNEL, name, op_name, mem_info);
#endif
    try {
      ret = kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, nullptr);
    } catch (std::exception &e) {
      MS_LOG(EXCEPTION) << e.what() << trace::DumpSourceLines(kernel);
    }
#ifndef ENABLE_SECURITY
    if (iter_dump_flag) {
      CPUE2eDump::DumpCNodeData(kernel, graph_id);
    }
    if (profiler_inst->GetEnableFlag()) {
      profiler_inst->OpDataProducerEnd();
    }
#endif
    if (!ret) {
#ifdef ENABLE_DUMP_IR
      mindspore::RDR::TriggerAll();
#endif
      MS_LOG(EXCEPTION) << "Launch kernel failed." << trace::DumpSourceLines(kernel);
    }
    static_cast<CPUMemoryManager *>(mem_manager_.get())->DecreaseAddressRefCount(kernel);
#ifdef ENABLE_PROFILE
    double cost_time = GetTime() - start_time;
    MS_LOG(INFO) << "cpu kernel: " << kernel->fullname_with_scope() << "  costs " << cost_time * 1e6 << " us";
#endif
  }
#ifndef ENABLE_SECURITY
  if (iter_dump_flag) {
    CPUE2eDump::DumpParameters(&kernel_graph, graph_id);
    CPUE2eDump::DumpConstants(&kernel_graph, graph_id);
  }
  if (graph_id == 0) {
    dump_json_parser.UpdateDumpIter();
  }
#endif
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
