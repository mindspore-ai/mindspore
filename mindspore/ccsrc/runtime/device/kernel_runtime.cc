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

#include "runtime/device/kernel_runtime.h"
#include <functional>
#include <utility>
#include <vector>
#include <set>
#include "backend/optimizer/common/helper.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "backend/session/kernel_graph.h"
#include "common/trans.h"
#include "debug/data_dump/dump_json_parser.h"
#include "frontend/operator/ops.h"
#include "ir/value.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/shape_utils.h"
#include "utils/utils.h"
#include "frontend/parallel/context.h"
#include "debug/env_config_parser.h"
#include "pipeline/pynative/pynative_profiling.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "ps/ps_cache/ps_cache_manager.h"
#endif

using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;

namespace mindspore {
namespace device {
constexpr float kMaxMemReuseFactor = 0.8;
constexpr float kMinMemReuseFactor = 0.5;
constexpr float kRetryFactor = 0.1;
constexpr size_t kAtomicCleanInputSize = 2;
namespace {
std::vector<AnfNodePtr> GetGraphInputs(const session::KernelGraph &graph) {
  auto graph_inputs = graph.inputs();
  std::vector<AnfNodePtr> result(graph_inputs.begin(), graph_inputs.end());
  std::set<AnfNodePtr> inputs_set(graph_inputs.begin(), graph_inputs.end());
  auto kernels = graph.execution_order();
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto input_num = AnfAlgo::GetInputTensorNum(kernel);
    for (size_t i = 0; i < input_num; ++i) {
      auto input_node = kernel->input(i + 1);
      auto input_real_node = AnfAlgo::VisitKernelWithReturnType(input_node, 0).first;
      MS_EXCEPTION_IF_NULL(input_real_node);
      if (input_real_node->isa<Parameter>() && inputs_set.find(input_real_node) == inputs_set.end()) {
        (void)inputs_set.insert(input_real_node);
        (void)result.emplace_back(input_real_node);
      }
    }
  }
  return result;
}
}  // namespace
constexpr size_t kMinInputSize = 2;
KernelRuntime::~KernelRuntime() {
  stream_ = nullptr;
  independent_stream_ = nullptr;
  communication_stream_ = nullptr;
}

bool KernelRuntime::Load(const session::KernelGraph &, bool) {
  MS_LOG(INFO) << "Call default load.";
  return true;
}

bool KernelRuntime::LoadData(const session::KernelGraph &) {
  MS_LOG(INFO) << "Call default load data.";
  return false;
}

bool KernelRuntime::NodeOutputDeviceAddressExist(const AnfNodePtr &kernel, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel);
  if (AnfAlgo::OutputAddrExist(kernel, index)) {
    const auto &address = AnfAlgo::GetOutputAddr(kernel, index);
    MS_EXCEPTION_IF_NULL(address);
    return address->DeviceType() == GetTargetDeviceAddressType();
  }
  return false;
}

void KernelRuntime::AssignMemory(const session::KernelGraph &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto enable_mem_scheduler = context_ptr->get_param<bool>(MS_CTX_ENABLE_MEM_SCHEDULER);
  if (enable_mem_scheduler) {
    AssignStaticMemoryValueNode(graph);
    ResetNodeAddress(graph);
  } else {
    MS_EXCEPTION_IF_NULL(mem_manager_);
    mem_manager_->ResetDynamicMemory();
    AssignStaticMemory(graph);
    AssignDynamicMemory(graph);
  }
  UpdateRefNodeOutputMem(graph);
}

void KernelRuntime::RunOpGetCommunicationInputInfo(const AnfNodePtr &node, size_t *total_size,
                                                   std::vector<DeviceAddressPtr> *address_list,
                                                   std::vector<size_t> *align_size_list) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(total_size);
  MS_EXCEPTION_IF_NULL(address_list);
  MS_EXCEPTION_IF_NULL(align_size_list);
  size_t input_num = AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(node, i, true);
    auto input_node = input_node_with_index.first;
    MS_EXCEPTION_IF_NULL(input_node);
    DeviceAddressPtr address = nullptr;
    if (AnfAlgo::OutputAddrExist(input_node, input_node_with_index.second)) {
      address = AnfAlgo::GetMutableOutputAddr(input_node, input_node_with_index.second);
    } else {
      if (input_node->isa<CNode>()) {
        address = PreAssignCNodeMemory(input_node, input_node_with_index.second);
      } else {
        MS_LOG(EXCEPTION) << "Communication node inputs only support CNode";
      }
    }
    MS_EXCEPTION_IF_NULL(address);
    auto align_size = MemoryManager::GetCommonAlignSize(address->size());
    *total_size += align_size;
    address_list->emplace_back(address);
    align_size_list->emplace_back(align_size);
  }
}

void KernelRuntime::RunOpAssignCommunicationInput(const AnfNodePtr &node) const {
  if (!AnfAlgo::IsCommunicationOp(node)) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  size_t total_size = 0;
  std::vector<DeviceAddressPtr> address_list;
  std::vector<size_t> align_size_list;
  RunOpGetCommunicationInputInfo(node, &total_size, &address_list, &align_size_list);
  if (address_list.empty()) {
    return;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() < kMinInputSize) {
    MS_LOG(ERROR) << "No inputs for " << cnode->fullname_with_scope();
    return;
  }

  if (!mem_manager_->MallocContinuousMemFromMemPool(address_list, total_size, align_size_list)) {
    MS_LOG(EXCEPTION) << "Allocate continuous memory failed, totol_size:" << total_size;
  }
}

void KernelRuntime::RunOpGetCommunicationOutputInfo(const AnfNodePtr &node, size_t *total_size,
                                                    std::vector<size_t> *align_size_list,
                                                    std::vector<DeviceAddressPtr> *device_address_list) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(total_size);
  MS_EXCEPTION_IF_NULL(align_size_list);
  MS_EXCEPTION_IF_NULL(device_address_list);
  auto runtime_info = node->user_data<session::OpRuntimeInfo>();
  auto output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; ++i) {
    MS_EXCEPTION_IF_NULL(runtime_info);
    DeviceAddressPtr address = nullptr;
    if (AnfAlgo::OutputAddrExist(node, i)) {
      address = AnfAlgo::GetMutableOutputAddr(node, i);
    } else {
      std::string output_format = runtime_info->output_format(i);
      auto output_type = runtime_info->output_type(i);
      address =
        CreateDeviceAddress(nullptr, runtime_info->output_tensor_size(i), output_format, output_type, {node, i});
    }
    MS_EXCEPTION_IF_NULL(address);
    auto align_size = MemoryManager::GetCommonAlignSize(address->size());
    *total_size += align_size;
    align_size_list->emplace_back(align_size);
    device_address_list->emplace_back(address);
  }
}

void KernelRuntime::RunOpAssignCommunicationOutput(const AnfNodePtr &node) const {
  if (!AnfAlgo::IsCommunicationOp(node)) {
    return;
  }

  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);

  size_t total_size = 0;
  std::vector<size_t> align_size_list;
  std::vector<DeviceAddressPtr> device_address_list;
  RunOpGetCommunicationOutputInfo(node, &total_size, &align_size_list, &device_address_list);

  if (align_size_list.empty()) {
    return;
  }

  if (!mem_manager_->MallocContinuousMemFromMemPool(device_address_list, total_size, align_size_list)) {
    MS_LOG(EXCEPTION) << "Allocate continuous memory failed, totol_size:" << total_size;
  }
}

void KernelRuntime::RunOpMallocPre(const session::KernelGraph &graph,
                                   const std::vector<tensor::TensorPtr> &input_tensors) {
  const auto &nodes = graph.execution_order();
  // Malloc for Node output
  for (const auto &node : nodes) {
    auto output_num = AnfAlgo::GetOutputTensorNum(node);
    for (size_t i = 0; i < output_num; ++i) {
      MS_EXCEPTION_IF_NULL(node);
      auto runtime_info = node->user_data<session::OpRuntimeInfo>();
      MS_EXCEPTION_IF_NULL(runtime_info);
      auto const &output_format = runtime_info->output_format(i);
      auto output_type = runtime_info->output_type(i);
      auto tensor_size = runtime_info->output_tensor_size(i);
      // Create DeviceAddress without ptr.
      // Get real device ptr after KernelBuild finish.
      auto device_address = CreateDeviceAddress(nullptr, tensor_size, output_format, output_type);
      device_address->set_host_shape(trans::GetRuntimePaddingShape(node, i));
      AnfAlgo::SetOutputAddr(device_address, i, node.get());
    }
  }

  // Malloc for graph input
  if (input_tensors.size() != graph.inputs().size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size()
                      << " should be equal to graph input parameter size " << graph.inputs().size();
  }
  for (size_t input_index = 0; input_index < graph.inputs().size(); ++input_index) {
    auto item = graph.inputs()[input_index];
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>()) {
      continue;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      auto current_tensor = input_tensors[input_index];
      MS_EXCEPTION_IF_NULL(current_tensor);
      auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(current_tensor->device_address());
      if (output_address != nullptr && output_address->DeviceType() == GetTargetDeviceAddressType()) {
        AnfAlgo::SetOutputAddr(output_address, index, item.get());
        continue;
      }
      auto op_runtime_info = item->user_data<session::OpRuntimeInfo>();
      MS_EXCEPTION_IF_NULL(op_runtime_info);
      TypeId output_type_id = op_runtime_info->output_type(index);
      auto output_tensor_size = op_runtime_info->output_tensor_size(index);
      auto output_format = op_runtime_info->output_format(index);
      auto device_address =
        CreateDeviceAddress(nullptr, output_tensor_size, output_format, output_type_id, {item, index});
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
      current_tensor->set_device_address(device_address);
      current_tensor->set_sync_status(kNeedSyncHostToDevice);
    }
  }
}

void KernelRuntime::ResetNodeAddress(const session::KernelGraph &kernel_graph) {
  auto kernels = kernel_graph.execution_order();
  for (auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel);
    for (size_t j = 0; j < input_num; ++j) {
      auto input_index = AnfAlgo::GetRealInputIndex(kernel, j);
      KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(kernel, input_index, true);
      auto index = kernel_with_index.second;
      auto &input_node = kernel_with_index.first;
      if (NodeOutputDeviceAddressExist(input_node, index)) {
        continue;
      }
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(input_node, index);
      if (output_type_id == kTypeUnknown) {
        MS_LOG(WARNING) << "It is not suggested to use a lonely weight parameter as the output of graph";
        continue;
      }
      auto tensor_size = AnfAlgo::GetOutputTensorMemSize(input_node, index);
      auto device_address = CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(input_node, index),
                                                output_type_id, {input_node, index});
      AnfAlgo::SetOutputAddr(device_address, index, input_node.get());
    }

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

void KernelRuntime::RunOpAssignMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                      const session::KernelGraph &graph,
                                      const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetDynamicMemory();

  for (const auto &node : graph.execution_order()) {
    RunOpAssignCommunicationOutput(node);
    RunOpAssignCommunicationInput(node);
  }

  RunOpAssignInputMemory(input_tensors, graph);
  AssignStaticMemoryValueNode(graph);
  for (const auto &node : graph.execution_order()) {
    RunOpAssignOutputMemory(node, tensor_to_node);
    RunOpAssignWorkSpaceMemory(node);
  }
  UpdateRefNodeOutputMem(graph);
}

void KernelRuntime::RunOpClearMemory(const session::KernelGraph &graph) const {
  // clear input parameter memory resource
  for (const auto &input_node : graph.inputs()) {
    MS_EXCEPTION_IF_NULL(input_node);
    AnfAlgo::SetOutputAddr(nullptr, 0, input_node.get());
  }
  // clear input value node memory resource
  for (const auto &value_node : graph.graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    AnfAlgo::SetOutputAddr(nullptr, 0, value_node.get());
  }
  for (const auto &cnode : graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    // clear output memory resource
    size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t index = 0; index < output_num; ++index) {
      AnfAlgo::SetOutputAddr(nullptr, index, cnode.get());
    }
    // clear workspace memory resource
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t index = 0; index < workspace_lists.size(); ++index) {
      AnfAlgo::SetWorkspaceAddr(nullptr, index, cnode.get());
    }
  }
}

#ifdef ENABLE_DEBUGGER
bool KernelRuntime::DumpDataEnabled() {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  return dump_json_parser.e2e_dump_enabled();
}

bool KernelRuntime::DumpDataEnabledIteration() {
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.e2e_dump_enabled()) {
    return false;
  }

  auto cur_iter = dump_json_parser.cur_dump_iter();
  if (dump_json_parser.IsDumpIter(cur_iter)) {
    return true;
  }
  return false;
}
#endif

void KernelRuntime::AssignStaticMemory(const session::KernelGraph &graph) {
  AssignStaticMemoryInput(graph);
  AssignStaticMemoryValueNode(graph);
  AssignStaticMemoryOutput(graph);
}

void KernelRuntime::RunOpAssignInputMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                           const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (input_tensors.size() != graph.inputs().size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size()
                      << " should be equal to graph input parameter size " << graph.inputs().size();
  }

  for (size_t input_index = 0; input_index < graph.inputs().size(); ++input_index) {
    auto item = graph.inputs()[input_index];
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>()) {
      continue;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      auto current_tensor = input_tensors[input_index];
      MS_EXCEPTION_IF_NULL(current_tensor);
      auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(current_tensor->device_address());
      if (output_address != nullptr && output_address->DeviceType() == GetTargetDeviceAddressType()) {
        if (output_address->ptr_ == nullptr) {
          if (!mem_manager_->MallocMemFromMemPool(output_address, output_address->size())) {
            MS_LOG(EXCEPTION) << "Allocate memory failed, size:" << output_address->size();
          }
        }

        AnfAlgo::SetOutputAddr(output_address, index, item.get());
        continue;
      }
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      if (output_type_id == kTypeUnknown) {
        output_type_id = AnfAlgo::GetOutputInferDataType(item, index);
      }
      auto tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
      auto device_address =
        CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id, {item, index});
      MS_EXCEPTION_IF_NULL(device_address);
      MS_EXCEPTION_IF_NULL(mem_manager_);
      auto ret = mem_manager_->MallocMemFromMemPool(device_address, tensor_size);
      if (!ret) {
        MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << tensor_size;
      }
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
}

void KernelRuntime::RunOpAssignOutputMemory(
  const AnfNodePtr &kernel, const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    return;
  }

  // Use device_address Allocated in RunOpMallocPre.
  for (auto &iter : tensor_to_node) {
    auto device_address = iter.first->device_address();
    AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(device_address), iter.second.second,
                           iter.second.first.get());
  }

  for (size_t i = 0; i < output_sizes.size(); ++i) {
    if (AnfAlgo::OutputAddrExist(kernel, i, false)) {
      auto address = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
      MS_EXCEPTION_IF_NULL(address);
      if (address->ptr() == nullptr) {
        MS_EXCEPTION_IF_NULL(mem_manager_);
        if (!mem_manager_->MallocMemFromMemPool(address, address->size())) {
          MS_LOG(EXCEPTION) << "Allocate memory failed, size:" << address->size();
        }
      }
      continue;
    }
    if (AnfAlgo::GetCNodeName(kernel) == kApplyMomentumOpName) {
      auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
      AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
      continue;
    }
    std::string output_format = AnfAlgo::GetOutputFormat(kernel, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
    auto device_address = CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type, {kernel, i});
    device_address->set_host_shape(trans::GetRuntimePaddingShape(kernel, i));
    MS_EXCEPTION_IF_NULL(device_address);
    auto ret = mem_manager_->MallocMemFromMemPool(device_address, output_sizes[i]);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << output_sizes[i];
    }
    AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
  }
}

void KernelRuntime::RunOpAssignWorkSpaceMemory(const AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (kernel->isa<CNode>()) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_lists.size(); ++i) {
      auto device_address = CreateDeviceAddress(nullptr, workspace_lists[i], "", kTypeUnknown);
      MS_EXCEPTION_IF_NULL(device_address);
      auto ret = mem_manager_->MallocMemFromMemPool(device_address, workspace_lists[i]);
      if (!ret) {
        MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << workspace_lists[i];
      }
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());
    }
  }
}

void KernelRuntime::RunOpAssignOutputNodeMemory(const ValuePtr &pre_output_value, const session::KernelGraph &graph) {
  if (pre_output_value == nullptr) {
    return;
  }
  std::vector<tensor::TensorPtr> pre_output_tensors;
  TensorValueToTensor(pre_output_value, &pre_output_tensors);
  auto output_nodes = graph.outputs();
  if (pre_output_tensors.size() != output_nodes.size()) {
    MS_LOG(EXCEPTION) << "The size of pre output tensors [" << pre_output_tensors.size()
                      << "] is not equal to the size of output nodes of graph [" << output_nodes.size() << "]";
  }
  // share output address with pre output tensors
  for (size_t i = 0; i < output_nodes.size(); ++i) {
    auto output_node_with_index = AnfAlgo::VisitKernel(output_nodes[i], 0);
    auto output_node = output_node_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    if (!output_node->isa<CNode>()) {
      if (output_node->isa<Parameter>()) {
        auto param = output_node->cast<ParameterPtr>();
        if (param != nullptr && !param->has_default()) {
          MS_LOG(EXCEPTION) << "The output parameter should be real parameter!";
        }
      }
      continue;
    }
    auto real_output_cnode = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(real_output_cnode);
    MS_EXCEPTION_IF_NULL(pre_output_tensors[i]);
    if (pre_output_tensors[i]->device_address() == nullptr) {
      MS_LOG(INFO) << "The address of pre output tensor [" << i << "] is a nullptr!";
      continue;
    }
    if (opt::IsNopNode(real_output_cnode)) {
      if (real_output_cnode->inputs().size() < kMinInputSize) {
        MS_LOG(EXCEPTION) << "The input size of output node: " << real_output_cnode->DebugString()
                          << " should large than one!";
      }
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(pre_output_tensors[i]->device_address()),
                             output_node_with_index.second, real_output_cnode->input(1).get());
    } else {
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(pre_output_tensors[i]->device_address()),
                             output_node_with_index.second, output_node_with_index.first.get());
    }
  }
}

void KernelRuntime::AssignStaticMemoryInput(const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  MS_LOG(INFO) << "AssignStaticMemoryInput start for graph " << graph.graph_id();
  auto graph_inputs = GetGraphInputs(graph);
  auto graph_valid_input = graph.valid_inputs();
  graph_inputs.insert(graph_inputs.end(), graph.child_graph_result().begin(), graph.child_graph_result().end());
  std::vector<AnfNodePtr> need_alloc_nodes;
  auto add_need_alloc_nodes = [&need_alloc_nodes, graph, this](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<Parameter>()) {
      return;
    }
    if (NodeOutputDeviceAddressExist(node, 0)) {
      return;
    }
    auto input_param = node->cast<ParameterPtr>();
    if (input_param != nullptr && !input_param->IsUsedByRealKernelInGraph(graph.graph_id())) {
      return;
    }
    need_alloc_nodes.push_back(node);
  };

  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    auto input_node = graph_inputs[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (i < graph_valid_input.size() && !graph_valid_input[i]) {
      continue;
    }
    if (AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
      auto outs = AnfAlgo::GetAllOutput(input_node);
      for (auto &out : outs) {
        MS_EXCEPTION_IF_NULL(out);
        add_need_alloc_nodes(out);
      }
    }
    add_need_alloc_nodes(input_node);
  }
#if ((defined ENABLE_CPU) && (!defined _WIN32))
  bool ps_cache_check = false;
#endif
  for (auto &item : need_alloc_nodes) {
    MS_EXCEPTION_IF_NULL(item);
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      // if graph output is a weight and doesn't link to any cnode, it's data type will be unknown
      if (output_type_id == kTypeUnknown) {
        MS_LOG(WARNING) << "It is not suggested to use a lonely weight parameter as the output of graph";
        continue;
      }
      DeviceAddressPtr device_address = nullptr;
#if ((defined ENABLE_CPU) && (!defined _WIN32))
      const std::string &param_name = item->fullname_with_scope();
      if (ps::ps_cache_instance.IsHashTable(param_name)) {
        MS_LOG(INFO) << "Parameter(" << param_name << ")"
                     << " enables the embeddingLookup cache in parameter server training mode.";
        // PS embeddingLookup cache check.
        if (!ps_cache_check) {
          CheckIfSupportPSEmbeddingCache(graph);
          ps_cache_check = true;
        }
        const auto &address = ps::ps_cache_instance.QueryHashTableAddr(param_name);
        MS_EXCEPTION_IF_NULL(address.addr);
        device_address = CreateDeviceAddress(address.addr, address.size, AnfAlgo::GetOutputFormat(item, index),
                                             output_type_id, {item, index});
        AnfAlgo::SetOutputAddr(device_address, index, item.get());
        continue;
      }
#endif
      auto tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
      device_address =
        CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id, {item, index});
      MS_LOG(INFO) << "Assign Static Memory for Input node, size:" << tensor_size
                   << " node:" << item->fullname_with_scope() << " index: " << index;
      if (mem_manager_->MallocMem(kStaticMem, tensor_size, device_address, graph.graph_id()) == nullptr) {
        MS_LOG(EXCEPTION) << "Cannot alloc address when flag is: " << kStaticMem << ", tensor size is: " << tensor_size;
      }
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
  MS_LOG(INFO) << "AssignStaticMemoryInput end";
}

void KernelRuntime::AssignStaticMemoryOutput(const session::KernelGraph &graph) {
  MS_LOG(INFO) << "AssignStaticMemoryOutput start for graph " << graph.graph_id();
  auto nodes = AnfAlgo::GetAllOutput(graph.output(), {prim::kPrimTupleGetItem});
  std::vector<session::KernelWithIndex> non_communication_op;
  // Assign Communicate Op Memory firstly.
  for (const auto &node : nodes) {
    auto kernel_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    if (!kernel_with_index.first->isa<CNode>() || !AnfAlgo::IsRealKernel(kernel_with_index.first)) {
      continue;
    }
    if (AnfAlgo::IsCommunicationOp(kernel_with_index.first)) {
      AssignCommunicationNodeMem(kStaticMem, kernel_with_index.first);
    } else {
      non_communication_op.emplace_back(kernel_with_index);
    }
  }

  for (const auto &item_with_index : non_communication_op) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    MS_LOG(DEBUG) << "AssignNodeOutputMem for " << item_with_index.first->fullname_with_scope();
    AssignNodeOutputMem(kStaticMem, item_with_index.first, SizeToInt(item_with_index.second));
  }
  MS_LOG(INFO) << "AssignStaticMemoryOutput end";
}

void KernelRuntime::UpdateRefNodeOutputMem(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto output_num = AnfAlgo::GetOutputTensorNum(kernel);
    if (output_num == 0) {
      MS_LOG(DEBUG) << "This kernel has no output size.";
      continue;
    }
    for (size_t i = 0; i < output_num; ++i) {
      session::AnfWithOutIndex out_pair(kernel, i);
      if (graph.IsInRefOutputMap(out_pair)) {
        auto origin_pair = graph.GetRefCorrespondOutput(out_pair);
        MS_EXCEPTION_IF_NULL(origin_pair.first);
        auto origin_node_output_addr = AnfAlgo::GetMutableOutputAddr(origin_pair.first, origin_pair.second);
        MS_EXCEPTION_IF_NULL(origin_node_output_addr);
        auto cur_node_output_addr = AnfAlgo::GetMutableOutputAddr(kernel, i);
        if (origin_node_output_addr.get() != cur_node_output_addr.get()) {
          MS_LOG(DEBUG) << "REF address is not same, ref node output need address update";
          MS_LOG(DEBUG) << "REF origin op is " << origin_pair.first->DebugString() << ", output index is "
                        << origin_pair.second << ", cur op is " << kernel->DebugString() << ", out index is " << i;
          AnfAlgo::SetOutputAddr(origin_node_output_addr, i, kernel.get());
        }
      }
    }
  }
}

void KernelRuntime::AssignCommunicationNodeMem(MemType type, const AnfNodePtr &node) {
  AssignCommunicationNodeInputMem(type, node);
  AssignCommunicationNodeOutputMem(type, node);
  AssignWorkSpaceMem(type, node);
}

void KernelRuntime::GenKernelEvents(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  if (kernels.empty() || graph_kernel_events_map_.find(graph.graph_id()) != graph_kernel_events_map_.end()) {
    return;
  }
  auto kernel_events =
    std::pair<std::vector<std::vector<std::function<void()>>>, std::vector<std::vector<std::function<void()>>>>();
  auto &kernel_pre_run_events = kernel_events.first;
  auto &kernel_post_run_events = kernel_events.second;
  kernel_pre_run_events.resize(kernels.size());
  kernel_post_run_events.resize(kernels.size());
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto &kernel = kernels[i];
    if (!AnfAlgo::IsCommunicationOp(kernel)) {
      continue;
    }
    auto pre_event = CreateDeviceEvent();
    auto post_event = CreateDeviceEvent();
    MS_EXCEPTION_IF_NULL(pre_event);
    MS_EXCEPTION_IF_NULL(post_event);
    pre_event->set_wait_stream(communication_stream_);
    pre_event->set_record_stream(stream_);
    post_event->set_wait_stream(stream_);
    post_event->set_record_stream(communication_stream_);
    kernel_pre_run_events[i].emplace_back([pre_event]() {
      pre_event->RecordEvent();
      pre_event->WaitEvent();
    });
    kernel_post_run_events[i].emplace_back([post_event]() { post_event->RecordEvent(); });
    bool found_nearest_child = false;
    for (size_t j = i + 1; j < kernels.size(); ++j) {
      auto &child = kernels[j];
      MS_EXCEPTION_IF_NULL(child);
      if (AnfAlgo::IsCommunicationOp(child)) {
        continue;
      }
      auto input_size = child->inputs().size() - 1;
      for (size_t k = 0; k < input_size; ++k) {
        auto kernel_index = AnfAlgo::VisitKernelWithReturnType(AnfAlgo::GetInputNode(child, k), 0, true);
        if (kernel_index.first == kernel) {
          found_nearest_child = true;
          break;
        }
      }
      if (found_nearest_child) {
        kernel_pre_run_events[j].emplace_back([post_event]() { post_event->WaitEvent(); });
        break;
      }
    }
    if (!found_nearest_child) {
      kernel_post_run_events[i].emplace_back([post_event]() { post_event->WaitEvent(); });
    }
  }
  graph_kernel_events_map_[graph.graph_id()] = std::move(kernel_events);
}

void KernelRuntime::AssignCommunicationNodeOutputMem(MemType type, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
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
  size_t output_index = 0;
  std::vector<size_t> align_size_list;
  for (uint64_t mem_size : output_sizes) {
    if (AnfAlgo::OutputAddrExist(node, output_index++)) {
      MS_LOG(INFO) << "Communication op " << node->fullname_with_scope() << " has output device address";
      return;
    }
    if (context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
      mem_size = MemoryManager::GetCommonAlignSize(mem_size);
    }
    total_size += mem_size;
    align_size_list.emplace_back(mem_size);
  }

  if (align_size_list.empty()) {
    return;
  }

  if (type == kSomasReuseDynamicMem) {
    bool not_reuse = KernelMemNotReuse(node);
    if (not_reuse) {
      type = kDynamicMem;
      MS_LOG(INFO) << "Disable Memory Reuse for " << node->fullname_with_scope() << "'s output.";
    }
  }

  uint8_t *output_ptr = nullptr;
  for (size_t j = 0; j < align_size_list.size(); ++j) {
    std::string output_format = AnfAlgo::GetOutputFormat(node, j);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(node, j);
    auto address = CreateDeviceAddress(nullptr, output_sizes[j], output_format, output_type, {node, j});
    MS_EXCEPTION_IF_NULL(address);
    if (output_ptr == nullptr) {
      output_ptr = mem_manager_->MallocOutputMem(node, 0, type, total_size, address, true);
      MS_EXCEPTION_IF_NULL(output_ptr);
    } else {
      address->set_ptr(output_ptr);
    }
    AnfAlgo::SetOutputAddr(address, j, node.get());
    output_ptr += align_size_list[j];
  }
}
bool KernelRuntime::KernelMemNotReuse(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return false;
}

DeviceAddressPtr KernelRuntime::PreAssignCNodeMemory(const AnfNodePtr &anf_node, size_t index) const {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!anf_node->isa<CNode>()) {
    MS_LOG(EXCEPTION) << "anf_node should be a cnode";
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (opt::IsNopNode(cnode)) {
    const size_t kNopNodeInputSize = 2;
    if (cnode->size() != kNopNodeInputSize) {
      MS_LOG(EXCEPTION) << cnode->fullname_with_scope() << " has invalid input size: " << cnode->size();
    }
    auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(anf_node, index);
    return PreAssignCNodeMemory(input_node_with_index.first, input_node_with_index.second);
  }
  auto kernel_mod = AnfAlgo::GetKernelMod(anf_node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.size() <= index) {
    MS_LOG(EXCEPTION) << "Previous node output size " << output_sizes.size() << " <= node index " << index;
  }
  std::string output_format = AnfAlgo::GetOutputFormat(anf_node, index);
  auto output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, index);
  auto address = CreateDeviceAddress(nullptr, output_sizes[index], output_format, output_type, {anf_node, index});
  AnfAlgo::SetOutputAddr(address, index, anf_node.get());
  return address;
}

void KernelRuntime::AssignCommunicationNodeInputMem(MemType type, const AnfNodePtr &node) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  size_t total_size = 0;
  std::vector<std::pair<DeviceAddressPtr, size_t>> addr_size;
  size_t input_num = AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node_with_index = AnfAlgo::GetPrevNodeOutput(node, i, true);
    auto input_node = input_node_with_index.first;
    MS_EXCEPTION_IF_NULL(input_node);
    if (AnfAlgo::OutputAddrExist(input_node, input_node_with_index.second)) {
      MS_LOG(INFO) << "Communication op " << input_node->fullname_with_scope() << " has input device address";
      return;
    }
    DeviceAddressPtr address = nullptr;
    if (input_node->isa<CNode>()) {
      address = PreAssignCNodeMemory(input_node, input_node_with_index.second);
    } else {
      MS_LOG(EXCEPTION) << "Communication node inputs only support CNode";
    }
    MS_EXCEPTION_IF_NULL(address);
    auto mem_size = MemoryManager::GetCommonAlignSize(address->size());
    total_size += mem_size;
    addr_size.emplace_back(address, mem_size);
  }
  if (addr_size.empty()) {
    return;
  }
  if (type == kSomasReuseDynamicMem) {
    bool not_reuse = KernelMemNotReuse(node);
    if (not_reuse) {
      type = kDynamicMem;
      MS_LOG(INFO) << "Disable Memory Reuse for " << node->fullname_with_scope() << "'s input.";
    }
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() < kMinInputSize) {
    // communication node's input should contain itself and at least on input
    MS_LOG(ERROR) << "No inputs for " << cnode->fullname_with_scope();
    return;
  }
  auto first_input_node = cnode->input(1);
  auto prenode_index = AnfAlgo::VisitKernelWithReturnType(first_input_node, 0, true);
  uint8_t *input_ptr = mem_manager_->MallocOutputMem(prenode_index.first, prenode_index.second, type, total_size,
                                                     addr_size[0].first, true);
  for (const auto &iter : addr_size) {
    MS_EXCEPTION_IF_NULL(iter.first);
    iter.first->set_ptr(input_ptr);
    input_ptr += iter.second;
  }
}

void KernelRuntime::AssignNodeOutputMem(MemType type, const AnfNodePtr &node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);

  if (type == kSomasReuseDynamicMem) {
    bool not_reuse = KernelMemNotReuse(node);
    if (not_reuse) {
      type = kDynamicMem;
      MS_LOG(INFO) << "Disable Memory Reuse for " << node->fullname_with_scope() << "'s output.";
    }
  }

  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    return;
  }
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    if ((kGetAllOuts != index) && (SizeToInt(i) != index)) {
      continue;
    }
    if (NodeOutputDeviceAddressExist(node, i)) {
      MS_LOG(INFO) << "Already malloc index:" << i;
      continue;
    }
    MS_LOG(DEBUG) << "Assign Node:" << node->fullname_with_scope() << " output memory size:" << output_sizes[i];
    if (type == kStaticMem) {
      MS_LOG(INFO) << "Assign Static Memory for Output node, size:" << output_sizes[i]
                   << " node:" << node->fullname_with_scope();
    }
    std::string output_format = AnfAlgo::GetOutputFormat(node, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(node, i);
    auto device_address = CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type, {node, i});
    MS_EXCEPTION_IF_NULL(device_address);
    uint8_t *ptr = mem_manager_->MallocOutputMem(node, i, type, output_sizes[i], device_address, false);
    MS_EXCEPTION_IF_NULL(ptr);
    device_address->set_host_shape(trans::GetRuntimePaddingShape(node, i));
    AnfAlgo::SetOutputAddr(device_address, i, node.get());
  }
}

DeviceAddressPtr KernelRuntime::AssignExtraStaticMem(const TensorPtr &tensor, const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto tensor_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_LOG(DEBUG) << "Assign Node:" << node->fullname_with_scope()
                << "Assign Static Memory for Output node, size:" << tensor_address->size();
  auto device_address = CreateDeviceAddress(nullptr, tensor_address->size(), tensor_address->format(),
                                            tensor_address->type_id(), {node, index});
  MS_EXCEPTION_IF_NULL(device_address);
  uint8_t *ptr = mem_manager_->MallocOutputMem(node, index, kStaticMem, tensor_address->size(), device_address, false);
  MS_EXCEPTION_IF_NULL(ptr);
  return device_address;
}

void KernelRuntime::AssignValueNodeTensor(const ValueNodePtr &value_node, const ValuePtr &node_value,
                                          size_t output_idx) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);
  // Graph id should be passed to record static memory if profiling is enabled.
  auto kernel_info = dynamic_cast<device::KernelInfo *>(value_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  uint32_t graph_id = kernel_info->graph_id();
  for (const auto &tensor : tensors) {
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (output_address != nullptr && output_address->DeviceType() == GetTargetDeviceAddressType()) {
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), output_idx++,
                             value_node.get());
      continue;
    }
    size_t tensor_size = LongToSize(tensor->data().nbytes());
    auto node_size = AnfAlgo::GetOutputTensorMemSize(value_node, output_idx);
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
    if (output_type_id == kTypeUnknown) {
      output_type_id = AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
    auto output_format = AnfAlgo::GetOutputFormat(value_node, output_idx);
    DeviceAddressPtr address =
      CreateDeviceAddress(nullptr, node_size, output_format, output_type_id, {value_node, output_idx});
    MS_EXCEPTION_IF_NULL(address);
    if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER) &&
        !mem_manager_->MallocMemFromMemPool(address, node_size)) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << node_size;
    } else {
      MS_LOG(INFO) << "Assign Static Memory for Value node, size:" << node_size
                   << " node:" << value_node->fullname_with_scope();
      if (mem_manager_->MallocMem(kStaticMem, node_size, address, graph_id) == nullptr) {
        MS_LOG(EXCEPTION) << "Cannot alloc address when flag is: " << kStaticMem << ", tensor size is: " << node_size;
      }
    }
    AnfAlgo::SetOutputAddr(address, output_idx, value_node.get());
    if (!address->SyncHostToDevice(trans::GetRuntimePaddingShape(value_node, 0), tensor_size, tensor->data_type(),
                                   tensor->data_c(), tensor->device_info().host_format_)) {
      MS_EXCEPTION(NotExistsError) << "ValueNode SyncHostToDevice fail!" << value_node->DebugString()
                                   << "node format is" << AnfAlgo::GetOutputFormat(value_node, output_idx)
                                   << "node dtype is " << AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
  }
}

void KernelRuntime::AssignStaticMemoryValueNode(const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  MS_LOG(DEBUG) << "AssignStaticMemoryValueNode start for graph " << graph.graph_id();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // order the value nodes
  std::map<std::string, ValueNodePtr> value_nodes_map;
  for (auto &node : graph.graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    value_nodes_map[node->fullname_with_scope()] = node;
  }

  for (auto &item : value_nodes_map) {
    auto value_node = item.second;
    MS_EXCEPTION_IF_NULL(value_node);
    if (NodeOutputDeviceAddressExist(value_node, 0)) {
      MS_LOG(DEBUG) << "value_node[" << value_node->DebugString() << "] address already exist";
      auto device_address = AnfAlgo::GetMutableOutputAddr(value_node, 0);
      if (device_address->ptr_ == nullptr) {
        if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
          if (!mem_manager_->MallocMemFromMemPool(device_address, device_address->size_)) {
            MS_LOG(EXCEPTION) << "MallocMemFromMemPool failed";
          }
        } else {
          if (mem_manager_->MallocMem(kStaticMem, device_address->size_, device_address, graph.graph_id())) {
            MS_LOG(EXCEPTION) << "MallocMem kStaticMem failed";
          }
        }
      }
      continue;
    }
    auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    MS_LOG(DEBUG) << "Malloc memory for " << value_node->fullname_with_scope();
    if (node_value->isa<Tensor>() || node_value->isa<ValueTuple>()) {
      AssignValueNodeTensor(value_node, node_value, 0);
    } else if (node_value->isa<StringImm>()) {
      auto value = GetValue<std::string>(node_value);
      size_t tensor_size = value.size();
      DeviceAddressPtr address = nullptr;
      address = CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, kNumberTypeUInt8);
      MS_EXCEPTION_IF_NULL(address);
      if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER) &&
          !mem_manager_->MallocMemFromMemPool(address, tensor_size)) {
        MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << tensor_size;
      } else {
        MS_LOG(INFO) << "Assign Static Memory for Value node, size:" << tensor_size
                     << " node:" << value_node->fullname_with_scope();
        if (mem_manager_->MallocMem(kStaticMem, tensor_size, address, graph.graph_id()) == nullptr) {
          MS_LOG(EXCEPTION) << "Cannot alloc address when flag is: " << kStaticMem
                            << ", tensor size is: " << tensor_size;
        }
      }
      AnfAlgo::SetOutputAddr(address, 0, value_node.get());
      ShapeVector shape = {1, SizeToLong(tensor_size)};
      if (!address->SyncHostToDevice(shape, tensor_size, kNumberTypeUInt8, value.data())) {
        MS_LOG(EXCEPTION) << "kValueNode SyncHostToDevice fail!";
      }
    }
  }
  MS_LOG(DEBUG) << "AssignStaticMemoryValueNode end";
}

void KernelRuntime::AssignDynamicMemory(const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_enable_mem_reuse = EnvConfigParser::GetInstance().GetSysMemreuse();
  auto mem_type = kDynamicMem;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.e2e_dump_enabled() && dump_json_parser.dump_mode() == 0) {
    mindspore::EnvConfigParser::GetInstance().SetSysMemreuse(false);
    is_enable_mem_reuse = false;
    MS_LOG(INFO) << "Disable Memory Reuse when e2e dump is enable and dump mode is set to dump all kernels";
  }

  if (is_enable_mem_reuse) {
    MS_LOG(INFO) << "Memory Reuse is enable...";
    mem_manager_->MallocSomasDynamicMem(graph);
    mem_type = kSomasReuseDynamicMem;
  } else {
    MS_LOG(INFO) << "Memory Reuse is disable...";
  }
  auto &execution_nodes = graph.execution_order();
  std::vector<CNodePtr> compute_nodes;
  // communication nodes first
  for (auto &node : execution_nodes) {
    if (AnfAlgo::IsCommunicationOp(node)) {
      // skip if the memory is already allocated
      AssignCommunicationNodeMem(mem_type, node);
    } else {
      compute_nodes.emplace_back(node);
    }
  }

  // then compute nodes
  for (auto &node : compute_nodes) {
    AssignNodeOutputMem(mem_type, node, kGetAllOuts);
    AssignWorkSpaceMem(mem_type, node);
  }
}

void KernelRuntime::AssignWorkSpaceMem(MemType type, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  size_t index = 0;
  for (auto &size : kernel_mod->GetWorkspaceSizeList()) {
    if (AnfAlgo::WorkspaceAddrExist(node, index)) {
      MS_LOG(INFO) << "Op " << node->fullname_with_scope() << " has workspace device address";
      return;
    }
    auto ptr = mem_manager_->MallocWorkSpaceMem(node, index, type, size);
    AnfAlgo::SetWorkspaceAddr(CreateDeviceAddress(ptr, size, "", kTypeUnknown), index, node.get());
    index++;
  }
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
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto visit_nop_node = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel);
  for (size_t i = 0; i < input_num; ++i) {
    auto op_name = AnfAlgo::GetCNodeName(cnode);
    constexpr auto none_placeholder_index = 3;
    if (op_name == kDynamicRNNOpName && i == none_placeholder_index) {
      continue;
    }
    if (op_name == kDynamicGRUV2OpName) {
      auto none_index = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, "placeholder_index");
      auto item = std::find(none_index.begin(), none_index.end(), i);
      if (item != none_index.end()) {
        continue;
      }
    }
    auto real_input = AnfAlgo::GetRealInputIndex(kernel, i);
    auto device_address = AnfAlgo::GetPrevNodeOutputAddr(kernel, real_input, visit_nop_node);
    MS_EXCEPTION_IF_NULL(device_address);
    kernel::AddressPtr input = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(input);
    input->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(input->addr);
    input->size = device_address->size_;
    kernel_inputs->emplace_back(input);
  }

  for (size_t i = 0; i < kernel_mod.GetOutputSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetOutputAddr(kernel, i, visit_nop_node);
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

void KernelRuntime::GenAddrCleanLaunchArgs(const CNodePtr &cnode, AddressPtrList *kernel_inputs,
                                           const std::shared_ptr<MemScheduler> &mem_scheduler) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  if (cnode->inputs().size() != kAtomicCleanInputSize) {
    MS_LOG(EXCEPTION) << "Atomic Addr clean Node Input nodes not equal 2.";
  }
  MS_EXCEPTION_IF_NULL(cnode->inputs()[1]);
  auto pre_node = (cnode->inputs()[1])->cast<CNodePtr>();
  // set clean output address
  if (AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
#if defined(__APPLE__)
    auto clean_output_indexes = AnfAlgo::GetNodeAttr<std::vector<int>>(pre_node, kAttrAtomicOutputIndexs);
#else
    auto clean_output_indexes = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
#endif
    for (auto index : clean_output_indexes) {
      auto device_address = AnfAlgo::GetOutputAddr(pre_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      if (mem_scheduler != nullptr) {
        GetOrMallocAddress(mem_scheduler, device_address, input);
      } else {
        input->addr = device_address->ptr_;
        MS_EXCEPTION_IF_NULL(input->addr);
      }
      input->size = device_address->size_;
      kernel_inputs->emplace_back(input);
    }
    MS_LOG(DEBUG) << "AtomicAddClean clean output size:" << clean_output_indexes.size();
  }
  // set clean workspace address
  if (AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
#if defined(__APPLE__)
    auto clean_workspaces_indexes = AnfAlgo::GetNodeAttr<std::vector<int>>(pre_node, kAttrAtomicWorkspaceIndexs);
#else
    auto clean_workspaces_indexes = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
#endif
    for (const auto &index : clean_workspaces_indexes) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(pre_node, index);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      if (mem_scheduler != nullptr) {
        GetOrMallocAddress(mem_scheduler, device_address, workspace);
      } else {
        workspace->addr = device_address->ptr_;
        MS_EXCEPTION_IF_NULL(workspace->addr);
      }
      workspace->size = device_address->size_;
      kernel_inputs->emplace_back(workspace);
    }
  }
}

void KernelRuntime::LaunchKernelEvent(const std::vector<std::vector<std::function<void()>>> &kernel_events,
                                      size_t index) const {
  if (index >= kernel_events.size()) {
    return;
  }
  for (auto &event : kernel_events[index]) {
    event();
  }
}

bool KernelRuntime::LaunchKernelWithPynativeProfiling(kernel::KernelMod *kernel_mod, const std::string &op_name,
                                                      const std::vector<AddressPtr> &inputs,
                                                      const std::vector<AddressPtr> &workspace,
                                                      const std::vector<AddressPtr> &outputs, void *stream) {
  MS_EXCEPTION_IF_NULL(kernel_mod);
  MS_EXCEPTION_IF_NULL(stream);
  float cost_time = 0;
  auto start = CreateDeviceTimeEvent();
  auto end = CreateDeviceTimeEvent();
  MS_EXCEPTION_IF_NULL(start);
  MS_EXCEPTION_IF_NULL(end);
  start->set_record_stream(stream);
  end->set_record_stream(stream);
  start->RecordEvent();
  bool ret = kernel_mod->Launch(inputs, workspace, outputs, stream);
  end->RecordEvent();
  start->SyncEvent();
  end->SyncEvent();
  start->ElapsedTime(&cost_time, end.get());
  auto launch_end_time = GetTime();
  double launch_start_time = launch_end_time - cost_time / kBasicTimeTransferUnit;
  auto op_launch_start_time_end_time = std::make_pair(launch_start_time, launch_end_time);
  PynativeProfiler::SetDeviceOpNameAndLaunchTimePoint(std::make_pair(op_name, op_launch_start_time_end_time));
  PynativeProfiler::SetDeviceOpNameAndLaunchCostTime(std::make_pair(op_name, cost_time / kBasicTimeTransferUnit));
  if (!ret) {
    MS_LOG(EXCEPTION) << "Launch kernel failed, kernel name is : " << op_name;
  }
  return ret;
}

void KernelRuntime::DebugStreamSync(const CNodePtr &kernel) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_sync_run = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
  if (enable_sync_run) {
    if (!SyncStream()) {
      MS_LOG(EXCEPTION) << "Op " << kernel->fullname_with_scope() << " run failed!";
    }
  }
}

void KernelRuntime::GetOrMallocAddress(const std::shared_ptr<MemScheduler> &mem_scheduler,
                                       const DeviceAddress *device_address, const kernel::AddressPtr &kernel_addr) {
  if (device_address->ptr_ != nullptr) {
    kernel_addr->addr = device_address->ptr_;
  } else {
    kernel_addr->addr = mem_scheduler->GetOrMalloc(device_address, device_address->size_);
    if (mem_scheduler->IsHighPriorityMem(device_address)) {
      device_address->ptr_ = kernel_addr->addr;
    }
  }
}

void KernelRuntime::AssignKernelAddress(const std::shared_ptr<MemScheduler> &mem_scheduler, const AnfNodePtr &kernel,
                                        AddressPtrList *kernel_inputs, AddressPtrList *kernel_workspaces,
                                        AddressPtrList *kernel_outputs) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  MS_EXCEPTION_IF_NULL(kernel_workspaces);
  MS_EXCEPTION_IF_NULL(kernel_outputs);
  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetCNodeName(cnode) == kAtomicAddrCleanOpName) {
    return GenAddrCleanLaunchArgs(cnode, kernel_inputs, mem_scheduler);
  }
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  size_t input_num = AnfAlgo::GetInputTensorNum(kernel);
  for (size_t j = 0; j < input_num; ++j) {
    auto real_input = AnfAlgo::GetRealInputIndex(kernel, j);
    auto kernel_with_index = AnfAlgo::GetPrevNodeOutput(kernel, real_input, true);
    auto index = kernel_with_index.second;
    auto &input_node = kernel_with_index.first;
    auto device_address = AnfAlgo::GetOutputAddr(input_node, index, true);
    MS_EXCEPTION_IF_NULL(device_address);
    kernel::AddressPtr input = std::make_shared<kernel::Address>();
    GetOrMallocAddress(mem_scheduler, device_address, input);
    input->size = device_address->size_;
    kernel_inputs->emplace_back(input);
  }

  for (size_t j = 0; j < kernel_mod->GetOutputSizeList().size(); ++j) {
    auto device_address = AnfAlgo::GetOutputAddr(kernel, j, true);
    kernel::AddressPtr output = std::make_shared<kernel::Address>();
    GetOrMallocAddress(mem_scheduler, device_address, output);
    output->size = device_address->size_;
    kernel_outputs->emplace_back(output);
  }

  for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetWorkspaceAddr(kernel, i);
    kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
    GetOrMallocAddress(mem_scheduler, device_address, workspace);
    workspace->size = device_address->size_;
    kernel_workspaces->emplace_back(workspace);
  }
}

void KernelRuntime::SyncNodeOutputTensors(const std::shared_ptr<MemScheduler> &mem_scheduler,
                                          const session::KernelGraph &graph, const AnfNodePtr &kernel, bool mock) {
  MS_EXCEPTION_IF_NULL(mem_scheduler);
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  for (size_t j = 0; j < kernel_mod->GetOutputSizeList().size(); ++j) {
    auto tensor = graph.GetNodeOutputTensor(std::make_pair(kernel, j));
    auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, j, true);
    if (mock) {
      if (graph.IsInternalOutput(kernel, j) && device_address != nullptr) {
        mem_scheduler->SetMemPriority(device_address.get(), kMemPriorityHigh);
      }
      continue;
    }
    if (tensor != nullptr) {
      if (device_address == nullptr) {
        tensor->data_sync(false);
        tensor->set_device_address(nullptr);
        tensor->set_sync_status(kNeedSyncHostToDevice);
        continue;
      }
      if (!SyncStream()) {
        MS_LOG(ERROR) << "SyncStream failed";
      }
      auto origin_ptr = device_address->ptr_;
      if (origin_ptr == nullptr) {
        device_address->ptr_ = mem_scheduler->GetOrMalloc(device_address.get(), device_address->size_);
      }
      tensor->set_device_address(device_address);
      tensor->data_sync(false);
      tensor->set_device_address(nullptr);
      if (origin_ptr == nullptr) {
        device_address->ptr_ = nullptr;
      }
      tensor->set_sync_status(kNeedSyncHostToDevice);
    }
  }
}

void KernelRuntime::InitGraphInputTensors(const std::shared_ptr<MemScheduler> &mem_scheduler,
                                          const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_scheduler);
  auto &input_nodes = graph.input_nodes();
  auto &input_tensors = graph.input_tensors();
  if (input_tensors.size() != input_nodes.size()) {
    MS_LOG_EXCEPTION << "Invalid input tensor size:" << input_tensors.size() << " vs node size:" << input_nodes.size();
  }
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto tensor = input_tensors[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto input_node = input_nodes[i];
    if (!input_node->isa<Parameter>()) {
      continue;
    }
    if (AnfAlgo::OutputAddrExist(input_node, 0)) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
      MS_EXCEPTION_IF_NULL(tensor);
      MemPriority priority = kMemPriorityHigh;
      auto tensor_address = tensor->device_address();
      if (tensor_address != nullptr && tensor_address != device_address) {
        tensor->data_sync(false);
        priority = kMemPriorityLow;
      }
      auto tensor_size = LongToSize(tensor->data().nbytes());
      mem_scheduler->Init(device_address.get(), tensor->data_c(), tensor_size, priority);
    }
  }
}

bool KernelRuntime::LaunchKernel(const session::KernelGraph &graph, const AnfNodePtr &kernel,
                                 const std::shared_ptr<MemScheduler> &mem_scheduler, bool mock) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  AddressPtrList kernel_inputs;
  AddressPtrList kernel_workspaces;
  AddressPtrList kernel_outputs;
  auto stream = kernel_mod->GetStream();
  if (stream == nullptr) {
    if (AnfAlgo::IsCommunicationOp(kernel)) {
      stream = communication_stream_;
    } else {
      stream = stream_;
    }
  }
  bool ret = true;
  if (mem_scheduler != nullptr) {
    ret = mem_scheduler->PreCompute(stream);
    if (!ret) {
      return ret;
    }
    AssignKernelAddress(mem_scheduler, kernel, &kernel_inputs, &kernel_workspaces, &kernel_outputs);
  } else if (!kernel_mod->GetInputsAddr().empty() || !kernel_mod->GetOutputsAddr().empty()) {
    kernel_inputs = kernel_mod->GetInputsAddr();
    kernel_outputs = kernel_mod->GetOutputsAddr();
    kernel_workspaces = kernel_mod->GetWorkSpacesAddr();
  } else {
    GenLaunchArgs(*kernel_mod, kernel, &kernel_inputs, &kernel_workspaces, &kernel_outputs);
  }
  if (!mock) {
    if (pynative_mode_profiling_flag_) {
      ret = LaunchKernelWithPynativeProfiling(kernel_mod, kernel->fullname_with_scope(), kernel_inputs,
                                              kernel_workspaces, kernel_outputs, stream);
    } else {
      ret = kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream);
    }
  }
  if (mem_scheduler != nullptr) {
    SyncNodeOutputTensors(mem_scheduler, graph, kernel, mock);
    ret = mem_scheduler->PostCompute(stream);
    if (!ret) {
      return ret;
    }
  }
  return ret;
}

bool KernelRuntime::LaunchKernelMod(const session::KernelGraph &graph, bool mock) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::shared_ptr<MemScheduler> mem_scheduler = nullptr;
  auto enable_mem_scheduler = context_ptr->get_param<bool>(MS_CTX_ENABLE_MEM_SCHEDULER);
  if (enable_mem_scheduler) {
    mem_scheduler = mem_scheduler_manager_.GetOrCreateMemScheduler(graph.graph_id());
    MS_EXCEPTION_IF_NULL(mem_scheduler);
    mem_scheduler->SetMemHandler(mem_manager_);
    mem_scheduler->RecordMemUsage();
    InitGraphInputTensors(mem_scheduler, graph);
  }
  const auto &kernels = graph.execution_order();
  std::vector<DynamicKernelPtr> dynamic_kernel_list;
  auto iter = graph_dynamic_kernel_map_.find(graph.graph_id());
  if (iter != graph_dynamic_kernel_map_.end()) {
    dynamic_kernel_list = iter->second;
  }
  if (!dynamic_kernel_list.empty() && dynamic_kernel_list.size() != kernels.size()) {
    MS_LOG(EXCEPTION) << "The size of dynamic kernels " << dynamic_kernel_list.size()
                      << " should be equal to the size of kernels " << kernels.size();
  }
  std::vector<std::vector<std::function<void()>>> kernel_pre_run_events;
  std::vector<std::vector<std::function<void()>>> kernel_post_run_events;
  auto events_iter = graph_kernel_events_map_.find(graph.graph_id());
  if (events_iter != graph_kernel_events_map_.end()) {
    kernel_pre_run_events = events_iter->second.first;
    kernel_post_run_events = events_iter->second.second;
  }
  for (size_t i = 0; i < kernels.size(); ++i) {
    LaunchKernelEvent(kernel_pre_run_events, i);
    if (!dynamic_kernel_list.empty() && dynamic_kernel_list[i] != nullptr &&
        dynamic_kernel_list[i]->is_dynamic_shape()) {
      dynamic_kernel_list[i]->InferShape();
      dynamic_kernel_list[i]->UpdateArgs();
      dynamic_kernel_list[i]->Execute();
      if (!SyncStream()) {
        MS_LOG(ERROR) << "SyncStream failed";
        return false;
      }
      dynamic_kernel_list[i]->PostExecute();
    } else {
      auto &kernel = kernels[i];
      MS_EXCEPTION_IF_NULL(kernel);

      // Skip transpose kernel with "nop_op" attr which is not hidden or removed in PyNative infer scenario. Transpose
      // kernel, which is not supposed to be executed, is generated in TransDataSplit to support specific Transdata.
      // And hard code here should be removed after new Transdata programme is implemented in the foreseeable future.
      if (AnfAlgo::HasNodeAttr("nop_op", kernel)) {
        for (size_t idx = 0; idx < AnfAlgo::GetOutputTensorNum(kernel); idx += 1) {
          auto real_input = AnfAlgo::GetRealInputIndex(kernel, idx);
          auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, real_input);
          AnfAlgo::SetOutputAddr(device_address, idx, kernel.get());
        }
        continue;
      }
      auto ret = LaunchKernel(graph, kernel, mem_scheduler, mock);
      if (!ret) {
        MS_LOG(ERROR) << "Launch kernel failed.";
        return false;
      }
      KernelLaunchProfiling(kernel->fullname_with_scope());
      DebugStreamSync(kernel);
    }
    LaunchKernelEvent(kernel_post_run_events, i);
  }
  if (mem_scheduler != nullptr) {
    mem_scheduler->OptMemUsage();
  }
  return true;
}

void KernelRuntime::UseMemSchedulerIfNeeded(const session::KernelGraph &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto enable_mem_scheduler = context_ptr->get_param<bool>(MS_CTX_ENABLE_MEM_SCHEDULER);
  if (enable_mem_scheduler) {
    auto mem_scheduler = mem_scheduler_manager_.GetOrCreateMemScheduler(graph.graph_id());
    if (mem_scheduler->need_record_event()) {
      (void)LaunchKernelMod(graph, true);
    }
    float mem_used_factor = kMaxMemReuseFactor;
    while (!mem_scheduler->optimized() && mem_used_factor >= kMinMemReuseFactor) {
      mem_scheduler->SetMemUsedFactor(mem_used_factor);
      bool ret = LaunchKernelMod(graph, true);
      if (ret) {
        mem_scheduler->SetOptimized(true);
      } else {
        mem_used_factor -= kRetryFactor;
      }
    }
  }
}

bool KernelRuntime::LaunchKernels(const session::KernelGraph &graph) {
  UseMemSchedulerIfNeeded(graph);
  if (!LaunchKernelMod(graph)) {
    MS_LOG(ERROR) << "LaunchKernelMod failed!";
    return false;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    if (!SyncStream()) {
      MS_LOG(ERROR) << "SyncStream failed";
      return false;
    }
  }
  return true;
}

void KernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id) {
  MS_LOG(INFO) << "Clear graph:" << graph_id << " runtime resource";
}

#if ((defined ENABLE_CPU) && (!defined _WIN32))
void KernelRuntime::GetFirstPSEmbeddingCache(const session::KernelGraph &graph,
                                             AnfNodePtr *const first_cache_input_index,
                                             size_t *const first_cache_size) {
  for (const auto &kernel : graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_name = AnfAlgo::GetCNodeName(kernel);
    if (kernel_name != kGatherV2OpName && kernel_name != kSparseGatherV2OpName) {
      continue;
    }
    auto input_param = AnfAlgo::GetPrevNodeOutput(kernel, 0, true);
    auto input_index = AnfAlgo::GetPrevNodeOutput(kernel, 1, true);
    MS_EXCEPTION_IF_NULL(input_param.first);
    MS_EXCEPTION_IF_NULL(input_index.first);
    auto param_name = input_param.first->fullname_with_scope();
    if (!ps::ps_cache_instance.IsHashTable(param_name)) {
      continue;
    }
    auto size = ps::ps_cache_instance.QueryHashTableSize(param_name);
    while (input_index.first->isa<CNode>() && (AnfAlgo::GetCNodeName(input_index.first) == kCastOpName)) {
      input_index = AnfAlgo::GetPrevNodeOutput(input_index.first, 0, true);
      MS_EXCEPTION_IF_NULL(input_index.first);
    }
    auto cnode =
      AnfAlgo::IsGraphKernel(input_index.first) ? AnfAlgo::GetOutputOfGraphkernel(input_index) : input_index.first;
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->isa<CNode>()) {
      MS_LOG(EXCEPTION) << "The embeddingLookup whose input index should be a CNode but got "
                        << cnode->fullname_with_scope();
    }
    auto input_index_node_name = AnfAlgo::GetCNodeName(cnode);
    if (input_index_node_name != kGetNextOpName) {
      bool full_batch = parallel::ParallelContext::GetInstance()->full_batch();
      if ((!full_batch && (input_index_node_name != kUniqueOpName)) ||
          (full_batch && (input_index_node_name != kMinimumOpName))) {
        MS_LOG(ERROR) << "The input index of the embeddingLookup(" << kernel->fullname_with_scope()
                      << ") cache is from " << cnode->fullname_with_scope();
        MS_LOG(EXCEPTION) << "The embeddingLookup whose input index isn't from dataset doesn't support cache in "
                             "parameter server training mode.";
      }
    }
    *first_cache_input_index = cnode;
    *first_cache_size = size;
    MS_LOG(INFO) << "The input index of the first embeddingLookup cache is from " << cnode->fullname_with_scope()
                 << ", the cache size is " << size;
    return;
  }
}

void KernelRuntime::CheckSparsePSEmbeddingCache(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto pre_node = AnfAlgo::GetPrevNodeOutput(node, 1, true);
  MS_EXCEPTION_IF_NULL(pre_node.first);
  while (pre_node.first->isa<CNode>() && (AnfAlgo::GetCNodeName(pre_node.first) != kUniqueOpName)) {
    pre_node = AnfAlgo::GetPrevNodeOutput(pre_node.first, 0, true);
    MS_EXCEPTION_IF_NULL(pre_node.first);
  }
  if (!(pre_node.first->isa<CNode>()) || (AnfAlgo::GetCNodeName(pre_node.first) != kUniqueOpName)) {
    MS_LOG(EXCEPTION) << "The input_indices of kernel[SparseGatherV2] must be unique in parameter server cache mode";
  }

  pre_node = AnfAlgo::GetPrevNodeOutput(pre_node.first, 0, true);
  MS_EXCEPTION_IF_NULL(pre_node.first);
  while (pre_node.first->isa<CNode>() && (AnfAlgo::GetCNodeName(pre_node.first) == kCastOpName)) {
    pre_node = AnfAlgo::GetPrevNodeOutput(pre_node.first, 0, true);
    MS_EXCEPTION_IF_NULL(pre_node.first);
  }
  if (!(pre_node.first->isa<CNode>()) || (AnfAlgo::GetCNodeName(pre_node.first) != kGetNextOpName)) {
    MS_LOG(EXCEPTION) << "The input indices of kernel[Unique] must be produced from dataset directly and the indices "
                         "value can not be changed before delivering to kernel[Unique] in parameter server cache mode.";
  }
}

void KernelRuntime::CheckIfSupportPSEmbeddingCache(const session::KernelGraph &graph) {
  AnfNodePtr first_cache_input_index = nullptr;
  size_t first_cache_size = 0;
  GetFirstPSEmbeddingCache(graph, &first_cache_input_index, &first_cache_size);
  MS_EXCEPTION_IF_NULL(first_cache_input_index);
  for (const auto &kernel : graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_name = AnfAlgo::GetCNodeName(kernel);
    if (kernel_name != kGatherV2OpName && kernel_name != kSparseGatherV2OpName) {
      continue;
    }
    auto input_param = AnfAlgo::GetPrevNodeOutput(kernel, 0, true);
    auto input_index = AnfAlgo::GetPrevNodeOutput(kernel, 1, true);
    MS_EXCEPTION_IF_NULL(input_param.first);
    MS_EXCEPTION_IF_NULL(input_index.first);
    if (!input_param.first->isa<Parameter>()) {
      continue;
    }
    auto param_name = input_param.first->fullname_with_scope();
    if (ps::ps_cache_instance.IsHashTable(param_name) && (kernel_name == kSparseGatherV2OpName)) {
      CheckSparsePSEmbeddingCache(kernel);
    }
    while (input_index.first->isa<CNode>() && (AnfAlgo::GetCNodeName(input_index.first) == kCastOpName)) {
      input_index = AnfAlgo::GetPrevNodeOutput(input_index.first, 0, true);
      MS_EXCEPTION_IF_NULL(input_index.first);
    }
    auto cnode =
      AnfAlgo::IsGraphKernel(input_index.first) ? AnfAlgo::GetOutputOfGraphkernel(input_index) : input_index.first;
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode == first_cache_input_index) {
      if (!ps::ps_cache_instance.IsHashTable(param_name)) {
        MS_LOG(ERROR) << "The embeddingLookup(" << kernel->fullname_with_scope() << ") doesn't enable cache.";
        MS_LOG(EXCEPTION) << "All the embeddingLookups whose input indices are from dataset must enable cache at the "
                             "same time when one of them enables cache in parameter server training mode.";
      }
      auto size = ps::ps_cache_instance.QueryHashTableSize(param_name);
      if (size != first_cache_size) {
        MS_LOG(ERROR) << "The cache size(" << size << ") of embeddingLookup(" << kernel->fullname_with_scope()
                      << ") is not the same as other embeddingLookup cache size(" << first_cache_size << ").";
        MS_LOG(EXCEPTION) << "The cache sizes of embeddingLookups are not the same in parameter server training mode.";
      }
    } else if (ps::ps_cache_instance.IsHashTable(param_name)) {
      MS_LOG(ERROR) << "The input index of the embeddingLookup(" << kernel->fullname_with_scope() << ") cache is from "
                    << cnode->fullname_with_scope();
      MS_LOG(EXCEPTION) << "The embeddingLookup whose input index isn't from dataset doesn't support cache in "
                           "parameter server training mode.";
    } else if (cnode->isa<CNode>() && (AnfAlgo::GetCNodeName(cnode) == kGetNextOpName)) {
      MS_LOG(ERROR) << "The EmbeddingLookup kernel(" << kernel->fullname_with_scope() << ") doesn't enable cache.";
      MS_LOG(EXCEPTION) << "All EmbeddingLookup kernels whose input indices are from dataset must enable cache at "
                           "the same time and parameter 'sparse' must be equal to the value of 'enable_sparse' in "
                           "context setting in parameter server training mode.";
    }
  }
}
#endif
}  // namespace device
}  // namespace mindspore
