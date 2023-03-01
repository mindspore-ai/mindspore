/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/pynative/run_op_helper.h"

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include "utils/log_adapter.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "include/common/utils/convert_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/pynative/op_runtime_info.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "kernel/common_utils.h"
#ifndef ENABLE_SECURITY
#include "profiler/device/profiling.h"
using mindspore::profiler::ProfilerManager;
#endif

namespace mindspore::runtime {
namespace {
// 1. Device type is different in heterogeneous scenes.
// 2. The device address format is different.
void UpdateInputTensorFromDevice(const std::vector<AnfNodePtr> &input_nodes,
                                 const std::vector<tensor::TensorPtr> &input_tensors,
                                 const device::DeviceContext *device_context) {
  MS_LOG(DEBUG) << "Start";
  auto input_size = input_nodes.size();
  for (size_t i = 0; i < input_size; ++i) {
    auto &tensor = input_tensors[i];
    auto &input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    auto node_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
    // node_address can't be null
    MS_EXCEPTION_IF_NULL(node_address);
    if (tensor_address != nullptr) {
      if (tensor_address->GetDeviceType() != device_context->GetDeviceType() ||
          tensor_address->format() != node_address->format()) {
        // Need wait for OpExecutor task finish
        tensor->data_sync();
        // If tensor address is null, we will set Parameter address to the Tensor.
        tensor->set_device_address(nullptr);
      }
    }
  }
  MS_LOG(DEBUG) << "End";
}

void UpdateParameterShapeFromInputTensor(const AnfNodePtr &input_node, const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_tensor == nullptr || !input_node->isa<Parameter>()) {
    return;
  }

  auto input_param = input_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(input_param);
  if (!input_param->has_dynamic_shape()) {
    return;
  }

  auto shape = input_tensor->shape();
  MS_LOG(DEBUG) << "Update input node shape to:" << shape;
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(input_node, 0)}, {shape},
                                              input_node.get());
}

void SetDeviceAddress(const AnfNodePtr &input_node, const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto tensor_address = std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
  auto node_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);

  UpdateParameterShapeFromInputTensor(input_node, input_tensor);

  MS_EXCEPTION_IF_NULL(node_address);
  if (tensor_address == nullptr) {
    input_tensor->set_device_address(node_address);
    input_tensor->set_sync_status(kNeedSyncHostToDeviceImmediately);
    input_tensor->set_lazy_callback([]() { runtime::OpExecutor::GetInstance().WaitAll(); });
    node_address->set_from_persistent_mem(input_tensor->is_parameter());
    node_address->SetNodeIndex(input_node, 0);
  }

  // The DeviceType and format of DeviceAddress is always the same after UpdateInputTensor
  if (tensor_address != nullptr && tensor_address != node_address) {
    AnfAlgo::SetOutputAddr(tensor_address, 0, input_node.get());
  }
}

void UpdateInputNodeDeviceAddress(const std::vector<AnfNodePtr> &input_nodes,
                                  const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_LOG(DEBUG) << "Start";
  auto input_size = input_nodes.size();
  auto tensor_size = input_tensors.size();
  if (input_size != tensor_size) {
    MS_LOG(EXCEPTION) << "input node size:" << input_size << " not equal to tensors size:" << tensor_size;
  }
  for (size_t i = 0; i < input_size; ++i) {
    auto &input_node = input_nodes[i];
    auto &input_tensor = input_tensors[i];
    MS_EXCEPTION_IF_NULL(input_tensor);
    if (input_tensor->isa<tensor::MapTensor>()) {
      auto map_tensor = input_tensor->cast<tensor::MapTensorPtr>();
      SetDeviceAddress(input_node, map_tensor);
      SetDeviceAddress(input_node, map_tensor->key_tensor());
      SetDeviceAddress(input_node, map_tensor->value_tensor());
      SetDeviceAddress(input_node, map_tensor->status_tensor());
    } else {
      SetDeviceAddress(input_node, input_tensor);
    }
  }
  MS_LOG(DEBUG) << "End";
}

void UpdateRefNodeOutputDeviceAddress(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ref_node_map = graph->GetRefMap();
  for (const auto &iter : ref_node_map) {
    auto &output_pair = iter.first;
    auto &input_pair = iter.second;
    auto &ref_node = output_pair.first;
    auto output_index = output_pair.second;
    auto &input_node = input_pair.first;
    auto input_node_output_index = input_pair.second;
    if (!AnfAlgo::OutputAddrExist(input_node, input_node_output_index, false)) {
      MS_LOG(WARNING) << "Output address not exist, node " << input_node->fullname_with_scope() << " index "
                      << input_node_output_index;
      continue;
    }
    auto input_addr = AnfAlgo::GetMutableOutputAddr(input_node, input_node_output_index, false);
    AnfAlgo::SetOutputAddr(input_addr, output_index, ref_node.get());
  }
}

void CopyTensorDataToDevice(const tensor::TensorPtr &tensor, const AnfNodePtr &node,
                            const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  if ((device_address->GetPtr() == nullptr) &&
      (!device_context->device_res_manager_->AllocateMemory(device_address.get()))) {
    MS_LOG(EXCEPTION) << "Allocate memory failed";
  }
  // Copy data from host tensor to device.
  auto tensor_size = LongToSize(tensor->data().nbytes());
  auto tensor_type = tensor->data_type();
  MS_LOG(DEBUG) << "Copy to device, node:" << common::AnfAlgo::GetNodeDebugString(node);
  if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0), tensor_size, tensor_type,
                                        tensor->data_c(), tensor->device_info().host_format_)) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed";
  }
}

void CopyValueNodeTensorToDevice(const ValueNodePtr &node, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);

  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);

  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);
  for (size_t i = 0; i < tensors.size(); i++) {
    const auto &tensor = tensors[i];
    MS_EXCEPTION_IF_NULL(tensor);

    const auto &node_address = AnfAlgo::GetMutableOutputAddr(node, i, false);
    MS_EXCEPTION_IF_NULL(node_address);
    node_address->SetNodeIndex(node, i);
    if (node_address->GetPtr() != nullptr) {
      return;
    }
    tensor->set_device_address(node_address);
    CopyTensorDataToDevice(tensor, node, device_context);
  }
}

void CopyValueNodeStringToDevice(const ValueNodePtr &node, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  const auto &node_address = AnfAlgo::GetMutableOutputAddr(node, 0, false);
  MS_EXCEPTION_IF_NULL(node_address);
  if (node_address->GetPtr() != nullptr) {
    return;
  }

  if (!device_context->device_res_manager_->AllocateMemory(node_address.get())) {
    MS_LOG(EXCEPTION) << "Allocate memory failed";
  }

  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);
  // Copy data to device.
  auto value = GetValue<std::string>(node_value);
  size_t tensor_size = value.size();
  ShapeVector shape = {1, SizeToLong(tensor_size)};
  if (!node_address->SyncHostToDevice(shape, tensor_size, kNumberTypeUInt8, value.data())) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed";
  }
}

void CopyValueNodeDataToDevice(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Start";
  const auto &value_nodes = graph->graph_value_nodes();
  for (const auto &value_node : value_nodes) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueTuple>()) {
      CopyValueNodeTensorToDevice(value_node, device_context);
    } else if (node_value->isa<StringImm>()) {
      CopyValueNodeStringToDevice(value_node, device_context);
    } else {
      MS_LOG(INFO) << "Unknown value node type:" << value_node->DebugString();
    }
  }
  MS_LOG(DEBUG) << "End";
}

void UpdateAddressSizeForDynamicShapeTensor(const tensor::TensorPtr &input_tensor) {
  if (input_tensor->base_shape_ptr() != nullptr) {
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(input_tensor->device_address());
    MS_EXCEPTION_IF_NULL(device_address);
    auto tensor_size = LongToSize(input_tensor->data().nbytes());
    if (tensor_size != device_address->GetSize()) {
      device_address->SetSize(tensor_size);
    }
  }
}

void CopyMapTensorDataToDevice(const tensor::MapTensorPtr &map_tensor, const AnfNodePtr &input_node,
                               const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(map_tensor);
  auto key_tensor = map_tensor->key_tensor();
  MS_EXCEPTION_IF_NULL(key_tensor);
  UpdateAddressSizeForDynamicShapeTensor(key_tensor);
  CopyTensorDataToDevice(key_tensor, input_node, device_context);
  key_tensor->set_sync_status(kNoNeedSync);
  auto value_tensor = map_tensor->value_tensor();
  MS_EXCEPTION_IF_NULL(value_tensor);
  UpdateAddressSizeForDynamicShapeTensor(value_tensor);
  CopyTensorDataToDevice(value_tensor, input_node, device_context);
  value_tensor->set_sync_status(kNoNeedSync);
  auto status_tensor = map_tensor->status_tensor();
  MS_EXCEPTION_IF_NULL(status_tensor);
  UpdateAddressSizeForDynamicShapeTensor(status_tensor);
  CopyTensorDataToDevice(status_tensor, input_node, device_context);
  status_tensor->set_sync_status(kNoNeedSync);
}

void CopyParameterDataToDevice(const std::vector<AnfNodePtr> &input_nodes,
                               const std::vector<tensor::TensorPtr> &input_tensors,
                               const device::DeviceContext *device_context) {
  MS_LOG(DEBUG) << "Start";
  auto input_size = input_nodes.size();
  if (input_size > input_tensors.size()) {
    MS_LOG(EXCEPTION) << "input_size is bigger than input_tensors size, input_size:" << input_size
                      << ", input_tensors size:" << input_tensors.size();
  }
  for (size_t i = 0; i < input_size; ++i) {
    MS_EXCEPTION_IF_NULL(input_tensors[i]);
    if (input_tensors[i]->NeedSyncHostToDeviceImmediately()) {
      // First op in dynamic shape scenario(feed mode)
      if (input_tensors[i]->isa<tensor::MapTensor>()) {
        auto map_tensor = input_tensors[i]->cast<tensor::MapTensorPtr>();
        CopyMapTensorDataToDevice(map_tensor, input_nodes[i], device_context);
      } else {
        UpdateAddressSizeForDynamicShapeTensor(input_tensors[i]);
        CopyTensorDataToDevice(input_tensors[i], input_nodes[i], device_context);
        input_tensors[i]->set_sync_status(kNoNeedSync);
      }
    }
  }
  MS_LOG(DEBUG) << "End";
}

void UpdateOutputAddrSize(const AnfNodePtr &node, const std::shared_ptr<OpRuntimeInfo> &runtime_info) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  auto output_size = runtime_info->GetOutputSize();
  for (size_t i = 0; i < output_size; ++i) {
    auto output_address = runtime_info->GetOutputDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(output_address);
    auto output_addr_size = AnfAlgo::GetOutputTensorMemSize(node, i);
    if (output_addr_size != output_address->GetSize()) {
      output_address->SetSize(output_addr_size);
    }
    output_address->set_host_shape(trans::GetRuntimePaddingShape(node, 0));
  }
}

bool MallocForKernelInput(const std::shared_ptr<OpRuntimeInfo> &runtime_info,
                          const device::DeviceContext *device_context, const CNodePtr &node) {
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(runtime_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto input_size = runtime_info->GetInputSize();
  for (size_t i = 0; i < input_size; ++i) {
    if (common::AnfAlgo::IsNoneInput(node, i)) {
      MS_LOG(DEBUG) << "Input [" << i << "] of " << node->fullname_with_scope() << " is None.";
      continue;
    }
    auto input_address = runtime_info->GetInputDeviceAddress(i);
    kernel_mod->set_input_user_data(input_address->user_data().get(), i);
    MS_EXCEPTION_IF_NULL(input_address);
    if (input_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(input_address.get())) {
      return false;
    }
  }
  return true;
}

bool MallocForKernelOutput(const std::shared_ptr<OpRuntimeInfo> &runtime_info, const AnfNodePtr &node,
                           const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_size = runtime_info->GetOutputSize();
  auto kernel_out_size_list = kernel_mod->GetOutputSizeList();
  if (kernel_out_size_list.size() != output_size) {
    MS_LOG(ERROR) << "Node " << node->fullname_with_scope() << " output num is:" << output_size
                  << " but kernel_mod output num:" << kernel_out_size_list.size();
    return false;
  }
  for (size_t i = 0; i < output_size; ++i) {
    auto device_address = runtime_info->GetOutputDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(device_address);
    kernel_mod->set_output_user_data(device_address->user_data().get(), i);
    // For example, we need to call cudnnGetRNNTrainingReserveSize to get real output size in LstmGpuKernelMod!
    if (kernel_out_size_list[i] != device_address->GetSize()) {
      // If the format of the DeviceAddress is different, then the size is originally different.
      // Such as NCHW(1,1,1,3) and NC1HWC0(1,1,1,1,16). So we don't need to update the size.
      if (AnfAlgo::GetOutputFormat(node, i) == device_address->format()) {
        if (device_address->GetPtr() != nullptr) {
          MS_LOG(ERROR) << "kernel mod output " << i << " size:" << kernel_out_size_list[i]
                        << " not equal to device_address size:" << device_address->GetSize()
                        << ", but the device address is already have ptr";
          return false;
        }
        device_address->SetSize(kernel_out_size_list[i]);
      }
    }
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(ERROR) << "Allocate output memory failed, node:" << node->fullname_with_scope();
      return false;
    }
  }
  return true;
}

kernel::AddressPtrList CreateKernelInputAddress(const std::shared_ptr<OpRuntimeInfo> &runtime_info,
                                                const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  auto input_size = runtime_info->GetInputSize();
  kernel::AddressPtrList inputs;
  for (size_t i = 0; i < input_size; ++i) {
    if (common::AnfAlgo::IsNoneInput(node, i)) {
      (void)inputs.emplace_back(std::make_shared<kernel::Address>());
      MS_LOG(DEBUG) << "Input[" << i << "]:"
                    << " is None Input";
      continue;
    }
    auto device_address = runtime_info->GetInputDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(device_address);
    (void)inputs.emplace_back(
      std::make_shared<kernel::Address>(device_address->GetMutablePtr(), device_address->GetSize()));
    MS_LOG(DEBUG) << "input[" << i << "]:" << inputs.back()->addr << " size:" << inputs.back()->size;
  }
  return inputs;
}

kernel::AddressPtrList CreateKernelWorkspaceAddress(const std::shared_ptr<OpRuntimeInfo> &runtime_info,
                                                    const device::DeviceContext *device_context, const CNodePtr &kernel,
                                                    bool is_dynamic_shape) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto workspace_size = runtime_info->GetWorkspaceSize();
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();

  std::vector<device::DeviceAddressPtr> add_workspaces;
  if (is_dynamic_shape) {
    // Resize of workspaces, because of the dynamic size of workspace.
    if (workspace_size < workspace_sizes.size()) {
      for (size_t i = workspace_size; i < workspace_sizes.size(); ++i) {
        auto device_address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, workspace_sizes[i], "",
                                                                                       kTypeUnknown, ShapeVector());
        MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(kernel)
                      << " addr:" << device_address;
        AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());  // set to kernel_info
        MS_EXCEPTION_IF_NULL(device_address);
        (void)add_workspaces.emplace_back(device_address);
      }
    }
  }

  // Set workspace address new size
  for (size_t i = 0; i < workspace_size && i < workspace_sizes.size(); ++i) {
    auto device_address = runtime_info->GetWorkspaceDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->SetSize(workspace_sizes[i]);
  }

  kernel::AddressPtrList workspaces;
  for (size_t i = 0; i < workspace_size && i < workspace_sizes.size(); ++i) {
    auto device_address = runtime_info->GetWorkspaceDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    (void)workspaces.emplace_back(
      std::make_shared<kernel::Address>(device_address->GetMutablePtr(), device_address->GetSize()));
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->addr << " size:" << workspaces.back()->size;
  }

  for (size_t i = workspace_size; i < workspace_sizes.size(); ++i) {
    auto device_address = add_workspaces[i];
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    (void)workspaces.emplace_back(
      std::make_shared<kernel::Address>(device_address->GetMutablePtr(), device_address->GetSize()));
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->addr << " size:" << workspaces.back()->size;
  }
  return workspaces;
}

kernel::AddressPtrList CreateKernelWorkspaceAddressDynamic(const std::shared_ptr<OpRuntimeInfo> &runtime_info,
                                                           const device::DeviceContext *device_context,
                                                           const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
  if (workspace_sizes.size() != runtime_info->GetWorkspaceSize()) {
    MS_LOG(EXCEPTION) << "Check workspace size failed, kernel_mode workspace num " << workspace_sizes.size()
                      << " kernel_info workspace num " << runtime_info->GetWorkspaceSize();
  }

  kernel::AddressPtrList workspaces;
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto device_address = runtime_info->GetWorkspaceDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    (void)workspaces.emplace_back(
      std::make_shared<kernel::Address>(device_address->GetMutablePtr(), device_address->GetSize()));
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->addr << " size:" << workspaces.back()->size;
  }
  return workspaces;
}

kernel::AddressPtrList CreateKernelOutputAddress(const std::shared_ptr<OpRuntimeInfo> &runtime_info) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  auto output_size = runtime_info->GetOutputSize();
  kernel::AddressPtrList outputs;
  for (size_t i = 0; i < output_size; ++i) {
    auto device_address = runtime_info->GetOutputDeviceAddress(i);
    (void)outputs.emplace_back(
      std::make_shared<kernel::Address>(device_address->GetMutablePtr(), device_address->GetSize()));
    MS_LOG(DEBUG) << "output[" << i << "]:" << outputs.back()->addr << " size:" << outputs.back()->size;
  }
  return outputs;
}

// Host to Device or Device to Host
void CopyDataToDevice(const KernelGraphPtr &graph, const std::vector<tensor::TensorPtr> &input_tensors,
                      const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  CopyValueNodeDataToDevice(graph, device_context);
  CopyParameterDataToDevice(graph->input_nodes(), input_tensors, device_context);
}

void InferNodeRealShape(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) == KernelType::AKG_KERNEL) {
    MS_LOG(EXCEPTION) << "Akg kernel do not support dynamic shape: " << kernel->fullname_with_scope();
  }
  opt::dynamic_shape::InferOp(kernel);
}

void ResizeNodeInput(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto args = kernel::GetArgsFromCNode(kernel);
  if (kernel_mod->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map) ==
      static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
    MS_LOG(EXCEPTION) << "Node " << kernel->fullname_with_scope() << " Resize failed.";
  }
}

void UpdateKernelWorkspace(const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto workspace_num = kernel_mod->GetWorkspaceSizeList().size();
  auto kernel_info = dynamic_cast<device::KernelInfo *>(kernel->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  kernel_info->set_workspace_address_list(std::vector<device::DeviceAddressPtr>(workspace_num, nullptr));
}

void UpdateGraphInputsCache(const KernelGraphPtr &graph) {
  const auto &inputs = graph->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<Parameter>()) {
      continue;
    }
    auto op_runtime_info = input->user_data<runtime::OpRuntimeInfo>();
    if (op_runtime_info != nullptr) {
      op_runtime_info->Resize(input);
    }
  }
}

// kernel_mode launch
void LaunchKernelsDynamic(const KernelGraphPtr &graph, const device::DeviceContext *device_context,
                          bool is_gradient_out) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(DEBUG) << "Start";
  UpdateGraphInputsCache(graph);

  // Get device address from OpRuntimeInfo
  const auto &execution_order = graph->execution_order();
  for (auto const &node : execution_order) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start launch kernel " << node->fullname_with_scope() << " kernel type "
                  << AnfAlgo::GetKernelType(node);
    auto is_dynamic_shape = common::AnfAlgo::IsDynamicShape(node);
    auto runtime_info = node->user_data<runtime::OpRuntimeInfo>();
    MS_EXCEPTION_IF_NULL(runtime_info);

    if (!MallocForKernelInput(runtime_info, device_context, node)) {
      MS_LOG(EXCEPTION) << "Malloc for kernel input failed, Memory isn't enough, node:" << node->fullname_with_scope();
    }
    auto inputs = CreateKernelInputAddress(runtime_info, node);

    InferNodeRealShape(node);

    runtime::DeviceAddressUtils::CreateKernelOutputDeviceAddress(device_context, graph, is_gradient_out);
    runtime::DeviceAddressUtils::UpdateDeviceAddressForInplaceNode(graph);
    runtime::DeviceAddressUtils::UpdateDeviceAddressForRefNode(graph);

    ResizeNodeInput(node);
    UpdateKernelWorkspace(node);

    runtime::DeviceAddressUtils::CreateKernelWorkspaceDeviceAddress(device_context, graph);
    auto workspaces = CreateKernelWorkspaceAddressDynamic(runtime_info, device_context, node);

    if (!MallocForKernelOutput(runtime_info, node, device_context)) {
      MS_LOG(EXCEPTION) << "Malloc for kernel output failed, node:" << node->fullname_with_scope();
    }
    auto outputs = CreateKernelOutputAddress(runtime_info);
    const size_t stream_id = AnfAlgo::GetStreamId(node);
    if (!device_context->kernel_executor_->LaunchKernel(node, inputs, workspaces, outputs, stream_id)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, name:" << node->fullname_with_scope();
    }

    if (is_dynamic_shape) {
      kernel::UpdateNodeShape(node);
      UpdateOutputAddrSize(node, runtime_info);
#ifndef ENABLE_SECURITY
      if (common::AnfAlgo::GetCNodeName(node) != kGetNextOpName) {
        ProfilerManager::GetInstance()->SetNetDynamicShapeStatus();
      }
#endif
    }
  }
  MS_LOG(DEBUG) << "End";
}

void LaunchKernels(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(DEBUG) << "Start";

  // Get device address from OpRuntimeInfo
  const auto &execution_order = graph->execution_order();
  for (auto const &node : execution_order) {
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "Start launch kernel " << node->fullname_with_scope() << " kernel type "
                  << AnfAlgo::GetKernelType(node);
    auto is_dynamic_shape = common::AnfAlgo::IsDynamicShape(node);
    auto runtime_info = node->user_data<runtime::OpRuntimeInfo>();
    MS_EXCEPTION_IF_NULL(runtime_info);

    if (!MallocForKernelInput(runtime_info, device_context, node)) {
      MS_LOG(EXCEPTION) << "Malloc for kernel input failed, Memory isn't enough, node:" << node->fullname_with_scope();
    }
    auto inputs = CreateKernelInputAddress(runtime_info, node);
    if (is_dynamic_shape) {
      InferNodeRealShape(node);
      ResizeNodeInput(node);
#ifndef ENABLE_SECURITY
      if (common::AnfAlgo::GetCNodeName(node) != kGetNextOpName) {
        ProfilerManager::GetInstance()->SetNetDynamicShapeStatus();
      }
#endif
    }

    auto workspaces = CreateKernelWorkspaceAddress(runtime_info, device_context, node, is_dynamic_shape);

    if (!MallocForKernelOutput(runtime_info, node, device_context)) {
      MS_LOG(EXCEPTION) << "Malloc for kernel output failed, Memory isn't enough, node:" << node->fullname_with_scope();
    }
    auto outputs = CreateKernelOutputAddress(runtime_info);
    const size_t stream_id = AnfAlgo::GetStreamId(node);
    if (!device_context->kernel_executor_->LaunchKernel(node, inputs, workspaces, outputs, stream_id)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, name:" << node->fullname_with_scope();
    }

    if (is_dynamic_shape) {
      kernel::UpdateNodeShape(node);
      UpdateOutputAddrSize(node, runtime_info);
    }
  }
  MS_LOG(DEBUG) << "End";
}

void WaitCommunicationFinish(const std::vector<tensor::TensorPtr> &input_tensors) {
  for (auto &input_tensor : input_tensors) {
    MS_EXCEPTION_IF_NULL(input_tensor);
    if (input_tensor->NeedWaitDevice()) {
      input_tensor->WaitDevice();
    }
  }
}

void ReleaseKernelResource(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (IsOneOfCacheBlackList(common::AnfAlgo::GetCNodeName(kernel))) {
      auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
      if (kernel_mod) {
        kernel_mod->ReleaseResource();
      }
    }
  }
}
}  // namespace

// Determine the address of the graph and do not change the address in subsequent executions
void UpdateDeviceAddress(const KernelGraphPtr &graph, const std::vector<tensor::TensorPtr> &tensors_without_value_mask,
                         const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Start";
  const auto &input_nodes = graph->input_nodes();
  UpdateInputTensorFromDevice(input_nodes, tensors_without_value_mask, device_context);
  UpdateInputNodeDeviceAddress(input_nodes, tensors_without_value_mask);
  UpdateRefNodeOutputDeviceAddress(graph);
  MS_LOG(DEBUG) << "End";
}

void RunSingleOpGraph(const KernelGraphPtr &graph, const std::vector<tensor::TensorPtr> &input_tensors,
                      const device::DeviceContext *device_context) {
  WaitCommunicationFinish(input_tensors);
  CopyDataToDevice(graph, input_tensors, device_context);
  LaunchKernels(graph, device_context);
  ReleaseKernelResource(graph);
}

void RunSingleOpGraphDynamic(const KernelGraphPtr &graph, const std::vector<tensor::TensorPtr> &input_tensors,
                             const device::DeviceContext *device_context, bool is_gradient_out) {
  WaitCommunicationFinish(input_tensors);
  CopyDataToDevice(graph, input_tensors, device_context);
  LaunchKernelsDynamic(graph, device_context, is_gradient_out);
  ReleaseKernelResource(graph);
}
}  // namespace mindspore::runtime
