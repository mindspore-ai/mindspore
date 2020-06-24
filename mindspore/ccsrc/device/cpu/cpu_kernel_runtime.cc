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
#include "device/cpu/cpu_kernel_runtime.h"
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <utility>
#include <functional>
#include <unordered_map>
#include <set>
#include "kernel/kernel.h"
#include "device/cpu/cpu_device_address.h"
#include "utils/context/ms_context.h"
#include "utils/config_manager.h"
#include "utils/profile.h"
#include "common/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "session/session_basic.h"
#include "operator/ops.h"

namespace mindspore {
namespace device {
namespace cpu {
const size_t INIT_NODE_REF = 1;
namespace {
TypeId GetCPUSupportOutputTypeId(const TypeId type_id) {
  TypeId support_type_id = type_id;
  if (type_id == kNumberTypeUInt32) {
    support_type_id = kNumberTypeInt32;
  }
  if (type_id == kNumberTypeFloat || type_id == kNumberTypeFloat16 || type_id == kNumberTypeFloat32 ||
      type_id == kNumberTypeFloat64) {
    support_type_id = kNumberTypeFloat32;
  }
  if (support_type_id != kNumberTypeInt32 && support_type_id != kNumberTypeFloat32) {
    MS_LOG(EXCEPTION) << "Check output type failed.";
  }
  return support_type_id;
}
}  // namespace

void CPUKernelRuntime::AssignKernelAddress(session::KernelGraph *kernel_graph) {
  AssignValueNodeAddress(kernel_graph);
  AssignInputNodeAddress(kernel_graph);
  AssignKernelOutputAddress(kernel_graph);
  resource_manager_.MemPlan(kernel_graph);
  resource_manager_.MemMalloc(kernel_graph);
}

void CPUKernelRuntime::AssignValueNodeAddress(session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  size_t type_size = sizeof(float);
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
      std::vector<int> data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(), type_size, std::multiplies<size_t>());
      DeviceAddressPtr address = CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, kNumberTypeFloat32);
      MS_EXCEPTION_IF_NULL(address);
      if (tensor->data_type() == kNumberTypeFloat32 || tensor->data_type() == kNumberTypeInt32) {
        address->ptr_ = tensor->data_c();
      } else {
        address->ptr_ = resource_manager_.MemMalloc(tensor_size);
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
  size_t type_size = sizeof(float);
  for (auto &item : kernel_graph->inputs()) {
    MS_EXCEPTION_IF_NULL(item);
    if (item->isa<Parameter>()) {
      auto output_num = AnfAlgo::GetOutputTensorNum(item);
      for (size_t index = 0; index < output_num; index++) {
        TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
        std::vector<size_t> fmt_shape = AnfAlgo::GetOutputDeviceShape(item, index);
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

BaseRef CPUKernelRuntime::CreatTensorForOutput(const session::KernelWithIndex &kernel_with_index,
                                               const std::unordered_map<AnfNode *, tensor::TensorPtr> &input_map,
                                               std::set<DeviceAddressPtr> *bound_addresses,
                                               std::vector<tensor::TensorPtr> *need_sync_outputs) {
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
        auto out = CreatTensorForOutput(item_with_index, input_map, bound_addresses, need_sync_outputs);
        ret.push_back(out);
      }
      return ret;
    }
    size_t output_size = AnfAlgo::GetOutputTensorNum(node);
    if (index >= output_size) {
      MS_LOG(EXCEPTION) << "Invalid input index " << index;
    }
    auto address = AnfAlgo::GetMutableOutputAddr(node, index);
    MS_EXCEPTION_IF_NULL(address);
    auto shape = AnfAlgo::GetOutputInferShape(node, index);
    std::vector<int> temp_shape;
    (void)temp_shape.insert(temp_shape.end(), shape.begin(), shape.end());
    TypeId type_id = AnfAlgo::GetOutputInferDataType(node, index);
    type_id = GetCPUSupportOutputTypeId(type_id);
    tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
    MS_EXCEPTION_IF_NULL(tensor);
    if (bound_addresses->find(address) != bound_addresses->end()) {
      tensor->set_device_address(address);
      need_sync_outputs->emplace_back(tensor);
    } else {
      address->ptr_ = tensor->data_c();
      address->ref_count_ = INIT_NODE_REF;
      (void)bound_addresses->insert(address);
    }
    tensor->set_dirty(false);
    return tensor;
  } else if (input_node->isa<Parameter>() || input_node->isa<ValueNode>()) {
    auto iter = input_map.find(input_node.get());
    if (iter != input_map.end()) {
      return iter->second;
    }
  }
  return BaseRef();
}

void CPUKernelRuntime::BindInputOutput(const session::KernelGraph *kernel_graph,
                                       const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs,
                                       std::vector<tensor::TensorPtr> *need_sync_outputs) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  // bind input ptr
  auto &input_nodes = kernel_graph->inputs();
  if (input_nodes.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Input size not equal to input node size!";
  }
  std::unordered_map<AnfNode *, tensor::TensorPtr> input_map;
  size_t input_idx = 0;
  for (auto &item : input_nodes) {
    MS_EXCEPTION_IF_NULL(item);
    input_map[item.get()] = inputs[input_idx];
    if (item->isa<Parameter>()) {
      auto address = AnfAlgo::GetMutableOutputAddr(item, 0);
      auto tensor = inputs[input_idx];
      auto tensor_address = tensor->device_address();
      MS_EXCEPTION_IF_NULL(address);
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor_address != nullptr && tensor_address != address) {
        (void)tensor->data_sync();
      }
      std::vector<int> data_shape = tensor->shape();
      size_t tensor_size =
        std::accumulate(data_shape.begin(), data_shape.end(), sizeof(float), std::multiplies<size_t>());
      if (tensor->data_type() == kNumberTypeFloat32 || tensor->data_type() == kNumberTypeInt32) {
        address->ptr_ = tensor->data_c();
      } else {
        address->ptr_ = resource_manager_.MemMalloc(tensor_size);
        if (!address->SyncHostToDevice(data_shape, LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                       tensor->data_c())) {
          MS_LOG(EXCEPTION) << "Parameter node sync host to device failed!";
        }
        tensor->set_dirty(true);
      }
      address->ref_count_ = INIT_NODE_REF;
      tensor->set_device_address(address);
    }
    input_idx++;
  }
  // new output and bind ptr
  std::set<DeviceAddressPtr> bound_addresses;
  auto output_nodes = kernel_graph->outputs();
  for (const auto &item : output_nodes) {
    auto item_with_index = AnfAlgo::VisitKernelWithReturnType(item, 0, true);
    auto out = CreatTensorForOutput(item_with_index, input_map, &bound_addresses, need_sync_outputs);
    outputs->push_back(std::move(out));
  }
}

void CPUKernelRuntime::AddRuntimeAddress(DeviceAddress *address, std::vector<kernel::AddressPtr> *input_list) {
  MS_EXCEPTION_IF_NULL(address);
  kernel::AddressPtr input = std::make_shared<kernel::Address>();
  MS_EXCEPTION_IF_NULL(input);
  if (address->ptr_ == nullptr) {
    address->ptr_ = resource_manager_.MemMalloc(address->size_);
  }
  MS_EXCEPTION_IF_NULL(address->ptr_);
  input->addr = address->ptr_;
  input->size = address->size_;
  input_list->push_back(input);
}

void CPUKernelRuntime::IncreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs) {
  resource_manager_.IncreaseSummaryRefCount(summary_outputs);
}

void CPUKernelRuntime::DecreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs) {
  resource_manager_.DecreaseSummaryRefCount(summary_outputs);
}

bool CPUKernelRuntime::Run(session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  resource_manager_.IncreaseAddressRefCount(kernel_graph);

  auto kernels = kernel_graph->execution_order();
  for (const auto &kernel : kernels) {
#ifdef ENABLE_PROFILE
    double start_time = GetTime();
#endif
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
    auto ret = kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, 0);
    resource_manager_.DecreaseAddressRefCount(kernel);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Launch kernel failed.";
    }
#ifdef ENABLE_PROFILE
    double cost_time = GetTime() - start_time;
    MS_LOG(INFO) << "cpu kernel: " << kernel->fullname_with_scope() << "  costs " << cost_time * 1e6 << " us";
#endif
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
