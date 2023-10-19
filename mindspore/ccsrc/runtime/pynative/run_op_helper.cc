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
#include <map>
#include <algorithm>
#include "ops/structure_op_name.h"
#include "utils/log_adapter.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/convert_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/pynative/op_runtime_info.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "kernel/framework_utils.h"
#ifndef ENABLE_SECURITY
#include "include/backend/debug/profiler/profiling.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"

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
    MS_EXCEPTION_IF_NULL(device_context);
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
      MS_EXCEPTION_IF_NULL(map_tensor);
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
      MS_EXCEPTION_IF_NULL(input_node);
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
  MS_EXCEPTION_IF_CHECK_FAIL(device_address != nullptr, "Tensor device address is nullptr, id is " + tensor->id());
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
  MS_EXCEPTION_IF_NULL(input_tensor);
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
    output_address->set_host_shape(trans::GetRuntimePaddingShape(node, i));
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
      MS_EXCEPTION_IF_NULL(node);
      MS_LOG(DEBUG) << "Input [" << i << "] of " << node->fullname_with_scope() << " is None.";
      continue;
    }
    auto input_address = runtime_info->GetInputDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(kernel_mod);
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

kernel::AddressPtrList CreateKernelWorkspaceAddressDynamic(
  const device::DeviceContext *device_context, const CNodePtr &kernel,
  std::vector<device::DeviceAddressPtr> *workspace_device_address) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();

  kernel::AddressPtrList workspaces;
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(nullptr, workspace_sizes[i], "",
                                                                                   kTypeUnknown, ShapeVector());
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate dynamic workspace memory failed";
    }
    MS_EXCEPTION_IF_NULL(workspace_device_address);
    (void)workspace_device_address->emplace_back(device_address);
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
    MS_EXCEPTION_IF_NULL(device_address);
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
  opt::InferOp(kernel);
}

kernel::KernelArgs InferNodeRealShape(const CNodePtr &kernel, const pynative::ExecuteKernelInfo &execute_kernel,
                                      const std::vector<tensor::TensorPtr> &input_tensors) {
  MS_EXCEPTION_IF_NULL(kernel);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelInfer,
                                     kernel->fullname_with_scope(), false);
  return opt::dynamic_shape::InferOp(kernel, execute_kernel, input_tensors);
}

void ResizeNodeInput(const CNodePtr &kernel, const kernel::KernelArgs &args) {
  MS_EXCEPTION_IF_NULL(kernel);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelResize,
                                     kernel->fullname_with_scope(), false);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  kernel_mod->set_use_kernel_tensor(true);
  if (kernel_mod->Resize(args.inputs, args.outputs, args.depend_tensor_map) ==
      static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
    MS_LOG(EXCEPTION) << "Node " << kernel->fullname_with_scope() << " Resize failed.";
  }
}

kernel::AddressPtrList MallocInputMemoryForDeviceAddress(
  const std::vector<device::DeviceAddressPtr> &device_addressess) {
  kernel::AddressPtrList ret;
  for (auto &device_address : device_addressess) {
    if (device_address == nullptr) {
      (void)ret.emplace_back(std::make_shared<kernel::Address>());
      continue;
    }
    if (device_address->GetPtr() == nullptr) {
      (void)ret.emplace_back(std::make_shared<kernel::Address>(nullptr, 0));
    } else {
      (void)ret.emplace_back(
        std::make_shared<kernel::Address>(device_address->GetMutablePtr(), device_address->GetSize()));
    }
  }
  return ret;
}

kernel::AddressPtrList MallocOutputMemoryForDeviceAddress(
  const std::vector<device::DeviceAddressPtr> &device_addressess, const device::DeviceContext *device_context,
  std::vector<device::DeviceAddressPtr> *alloc_output_device_address) {
  kernel::AddressPtrList ret;
  for (auto &device_address : device_addressess) {
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr) {
      MS_EXCEPTION_IF_NULL(alloc_output_device_address);
      (void)alloc_output_device_address->emplace_back(device_address);
      MS_EXCEPTION_IF_NULL(device_context);
      MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
      if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
        MS_LOG(EXCEPTION) << "Allocate device memory failed!";
      }
    }
    (void)ret.emplace_back(
      std::make_shared<kernel::Address>(device_address->GetMutablePtr(), device_address->GetSize()));
  }
  return ret;
}

device::DeviceAddressPtr CreateTensorDeviceAddressWithTensorAndCachedInfo(
  const OpCompilerInfoPtr &op_compiler_info, const TensorPtr &tensor,
  const device::DeviceAddressPtr &cached_device_address, const AnfNodePtr &node, bool skip_sync) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(cached_device_address);
  auto &device_context = op_compiler_info->device_context_;
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto format = cached_device_address->format();
  auto dtype = cached_device_address->type_id();
  auto shape = tensor->shape();
  size_t tensor_size = GetTensorDeviceSize(device_context, node, shape, format, dtype, 0);

  auto new_device_address =
    device_context->device_res_manager_->CreateDeviceAddress(nullptr, tensor_size, format, dtype, tensor->shape());
  MS_EXCEPTION_IF_NULL(new_device_address);
  new_device_address->set_from_persistent_mem(tensor->is_parameter());
  if (!skip_sync) {
    if (!device_context->device_res_manager_->AllocateMemory(new_device_address.get())) {
      MS_LOG(EXCEPTION) << "Device(id:" << device_context->device_context_key().device_id_
                        << ") memory isn't enough and alloc failed, kernel name: " << node->DebugString()
                        << ", alloc size: " << new_device_address->GetSize() << "B.";
    }
    if (!new_device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0), tensor_size, dtype,
                                              tensor->data_c(), tensor->device_info().host_format_)) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed";
    }
  }

  cached_device_address->set_ptr(new_device_address->GetMutablePtr());
  cached_device_address->set_host_shape(new_device_address->host_shape());
  cached_device_address->SetSize(new_device_address->GetSize());
  return new_device_address;
}

void UpdateTensorCache(const DeviceContext *device_context, const device::DeviceAddressPtr &input_device_address,
                       const device::DeviceAddressPtr &cached_device_address, const TensorPtr &tensor,
                       const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(cached_device_address);
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(input_device_address);
  auto format = cached_device_address->format();
  auto size = GetTensorDeviceSize(device_context, node, tensor->shape(), format, cached_device_address->type_id(), 0);
  cached_device_address->SetSize(size);
  cached_device_address->set_host_shape(tensor->shape());
  cached_device_address->set_ptr(input_device_address->GetMutablePtr());
}

void UpdateOutputDeviceInfo(const std::vector<device::DeviceAddressPtr> &device_addressess, const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_size_list = kernel_mod->GetOutputSizeList();
  if (device_addressess.size() != output_size_list.size()) {
    MS_LOG(EXCEPTION) << "Output device address's size " << device_addressess.size()
                      << " is not equal output_size_list's size " << output_size_list.size();
  }
  auto output_num = device_addressess.size();
  for (size_t i = 0; i < output_num; i++) {
    MS_EXCEPTION_IF_NULL(device_addressess[i]);
    auto out_abstract = kernel->abstract();
    MS_EXCEPTION_IF_NULL(out_abstract);
    if (out_abstract->isa<abstract::AbstractTuple>()) {
      auto abstract_tuple = out_abstract->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(abstract_tuple);
      out_abstract = abstract_tuple->elements()[i];
    }
    auto shape = out_abstract->BuildShape();
    device_addressess[i]->set_host_shape(BaseShapeToShape(shape));
    device_addressess[i]->SetSize(output_size_list[i]);
  }
}

void UpdateInputTensorForHeterogeneous(const DeviceContext *device_context, const TensorPtr input_tensor,
                                       const device::DeviceAddressPtr &cached_device_address) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(cached_device_address);
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto device_sync = input_tensor->device_address();
  if (device_sync == nullptr) {
    return;
  }
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->GetDeviceType() != device_context->GetDeviceType() ||
      device_address->format() != cached_device_address->format()) {
    // Need wait for OpExecutor task finish
    input_tensor->data_sync();
    // If tensor address is null, we will set Parameter address to the Tensor.
    input_tensor->set_device_address(nullptr);
  }
}

void UpdateInputInCompileInfo(const OpCompilerInfoPtr &op_compiler_info, const std::vector<TensorPtr> &input_tensors,
                              std::map<device::DeviceAddressPtr, tensor::TensorPtr> *address_map_to_tensor) {
  MS_EXCEPTION_IF_NULL(op_compiler_info->graph_);
  auto input_tensors_num = input_tensors.size();
  auto cached_input_num = op_compiler_info->inputs_.size();
  if (input_tensors_num != cached_input_num) {
    MS_LOG(EXCEPTION) << "Real input tensor's number " << input_tensors_num
                      << " is not equal cached input tensor's number " << cached_input_num << " !";
  }
  const auto &device_context = op_compiler_info->device_context_;
  for (size_t i = 0; i < input_tensors_num; ++i) {
    auto &input_tensor = input_tensors[i];
    MS_EXCEPTION_IF_NULL(input_tensor);
    UpdateInputTensorForHeterogeneous(device_context, input_tensor, op_compiler_info->inputs_[i]);
    auto device_sync = input_tensor->device_address();
    auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
    auto skip_sync = true;
    auto &ignore_list = op_compiler_info->ignore_host_to_device_inputs_;
    if (ignore_list.empty() || ignore_list.find(op_compiler_info->inputs_[i]) == ignore_list.end()) {
      skip_sync = false;
    }
    common::AnfAlgo::SetOutputInferTypeAndShape({input_tensor->data_type()}, {input_tensor->shape()},
                                                op_compiler_info->graph_->inputs()[i].get());
    if (device_address != nullptr) {
      // Update cached input info by input tensor info
      UpdateTensorCache(device_context, device_address, op_compiler_info->inputs_[i], input_tensor,
                        op_compiler_info->graph_->inputs()[i]);
    } else {
      // Create new device address using tensor and cached device address
      auto new_device_address = CreateTensorDeviceAddressWithTensorAndCachedInfo(
        op_compiler_info, input_tensor, op_compiler_info->inputs_[i], op_compiler_info->graph_->inputs()[i], skip_sync);
      if (!skip_sync) {
        input_tensor->set_device_address(new_device_address);
      }
      (*address_map_to_tensor)[op_compiler_info->inputs_[i]] = input_tensor;
    }
  }
}

void SetOutputDeviceAddressFlag(const pynative::OpCompilerInfoPtr &op_compiler_info,
                                const session::BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  MS_EXCEPTION_IF_NULL(op_run_info);
  size_t output_size = op_compiler_info->outputs_.size();
  if (op_run_info->is_gradient_out) {
    for (auto &index : op_run_info->base_op_run_info.output_indexes) {
      if (index < output_size) {
        MS_EXCEPTION_IF_NULL(op_compiler_info->outputs_[index]);
        op_compiler_info->outputs_[index]->set_from_persistent_mem(true);
      }
    }
  }
}

void ClearAllocOutputDeviceAddress(const std::vector<device::DeviceAddressPtr> &device_address_list,
                                   const std::vector<device::DeviceAddressPtr> &alloc_output_device_address) {
  MS_LOG(DEBUG) << "Clear Alloc-OutputDeviceAddress Start";
  for (auto alloc_output_address : alloc_output_device_address) {
    MS_EXCEPTION_IF_NULL(alloc_output_address);
    auto skip_clear = false;
    for (auto device_address : device_address_list) {
      MS_EXCEPTION_IF_NULL(device_address);
      if (device_address->GetPtr() == alloc_output_address->GetPtr()) {
        skip_clear = true;
        break;
      }
    }
    if (!skip_clear) {
      alloc_output_address->ClearDeviceMemory();
    }
  }
  MS_LOG(DEBUG) << "Clear Alloc-OutputDeviceAddress End";
}

void ReleaseCacheInfo(const pynative::OpCompilerInfoPtr &op_compiler_info,
                      const std::vector<device::DeviceAddressPtr> &ref_node) {
  MS_LOG(DEBUG) << "Release CacheInfo Start";
  for (auto &execute_kernel : op_compiler_info->execute_kernel_list_) {
    auto kernel = execute_kernel.kernel_;
    for (auto &input_address : execute_kernel.inputs_device_address_) {
      auto iter = std::find(ref_node.begin(), ref_node.end(), input_address);
      if (input_address == nullptr || iter != ref_node.end()) {
        MS_LOG(DEBUG) << "Release skip input_address:" << kernel->fullname_with_scope();
        continue;
      }
      if (op_compiler_info->value_map_to_tensor_.find(input_address) == op_compiler_info->value_map_to_tensor_.end()) {
        input_address->set_ptr(nullptr);
      }
    }
    for (auto &output_address : execute_kernel.outputs_device_address_) {
      MS_EXCEPTION_IF_NULL(output_address);
      auto iter = std::find(ref_node.begin(), ref_node.end(), output_address);
      if (iter != ref_node.end()) {
        continue;
      }
      output_address->set_ptr(nullptr);
    }
  }
  MS_LOG(DEBUG) << "Release CacheInfo End";
}

void SyncCacheInfoToOutput(const pynative::OpCompilerInfoPtr &op_compiler_info,
                           vector<device::DeviceAddressPtr> *device_address_list) {
  MS_EXCEPTION_IF_NULL(device_address_list);
  auto outputs_num = device_address_list->size();
  auto &device_context = op_compiler_info->device_context_;
  auto cached_output_num = op_compiler_info->outputs_.size();
  if (outputs_num != cached_output_num) {
    MS_LOG(EXCEPTION) << "Real output tensor's number " << outputs_num << " is not equal cached output tensor's number "
                      << cached_output_num << " !";
  }
  for (size_t j = 0; j < outputs_num; ++j) {
    auto output_address = (*device_address_list)[j];
    MS_EXCEPTION_IF_NULL(op_compiler_info->outputs_[j]);
    if (output_address == nullptr) {
      output_address =
        runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(op_compiler_info->outputs_[j], device_context);
      MS_EXCEPTION_IF_NULL(output_address);
      output_address->set_ptr(op_compiler_info->outputs_[j]->GetMutablePtr());
      output_address->set_from_mem_pool(op_compiler_info->outputs_[j]->from_mem_pool());
      output_address->set_from_persistent_mem(op_compiler_info->outputs_[j]->from_persistent_mem());
      (*device_address_list)[j] = output_address;
    } else {
      if (output_address->GetPtr() == nullptr) {
        output_address->set_original_ref_count(op_compiler_info->outputs_[j]->original_ref_count());
        output_address->ResetRefCount();
        output_address->set_ptr(op_compiler_info->outputs_[j]->GetMutablePtr());
        output_address->set_from_mem_pool(op_compiler_info->outputs_[j]->from_mem_pool());
        output_address->set_from_persistent_mem(op_compiler_info->outputs_[j]->from_persistent_mem());
      }
    }
  }
}

void MallocForConstValue(const pynative::OpCompilerInfoPtr &op_compiler_info) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  const auto &device_context = op_compiler_info->device_context_;
  const auto &graph = op_compiler_info->graph_;
  CopyValueNodeDataToDevice(graph, device_context);
}

void UpdateOutputAddressForRef(const OpCompilerInfoPtr &op_compiler_info,
                               vector<device::DeviceAddressPtr> *device_address_list,
                               std::vector<device::DeviceAddressPtr> *ref_node_cache) {
  MS_EXCEPTION_IF_NULL(ref_node_cache);
  const auto &graph = op_compiler_info->graph_;
  MS_EXCEPTION_IF_NULL(graph);
  const auto &ref_node_map = graph->GetRefMap();
  for (const auto &iter : ref_node_map) {
    auto &output_pair = iter.first;
    auto &input_pair = iter.second;
    auto &ref_node = output_pair.first;
    auto output_index = output_pair.second;
    auto input_address = GetInputAddressForRef(input_pair.first, op_compiler_info);
    if (input_address == nullptr) {
      continue;
    }
    auto output_address = GetOutputAddressForRef(ref_node, op_compiler_info, output_index);
    MS_EXCEPTION_IF_NULL(output_address);
    output_address->set_ptr(input_address->GetMutablePtr());
    output_address->set_from_mem_pool(input_address->from_mem_pool());
    output_address->set_from_persistent_mem(input_address->from_persistent_mem());
    output_address->set_host_shape(input_address->host_shape());
    for (size_t index = 0; index < op_compiler_info->outputs_.size(); ++index) {
      if (output_address == op_compiler_info->outputs_[index]) {
        if ((*device_address_list)[index] == nullptr) {
          (*device_address_list)[index] = input_address;
        }
      }
    }

    // Ref
    (void)ref_node_cache->emplace_back(input_address);
  }
}

std::vector<tensor::TensorPtr> GetAllInputTensor(
  const std::vector<device::DeviceAddressPtr> &device_address_list,
  const std::map<device::DeviceAddressPtr, tensor::TensorPtr> &address_map_to_tensor,
  const std::map<device::DeviceAddressPtr, tensor::TensorPtr> &value_map_to_tensor) {
  std::vector<tensor::TensorPtr> ret;
  for (auto &device_address : device_address_list) {
    auto iter = address_map_to_tensor.find(device_address);
    if (iter != address_map_to_tensor.end()) {
      (void)ret.emplace_back(iter->second);
      continue;
    }

    iter = value_map_to_tensor.find(device_address);
    if (iter != value_map_to_tensor.end()) {
      (void)ret.emplace_back(iter->second);
      continue;
    }

    (void)ret.emplace_back(nullptr);
  }

  return ret;
}

void UpdateOutputShapeForCompileInfo(const std::vector<device::DeviceAddressPtr> &outputs_device_address,
                                     const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto out_abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(out_abstract);
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto need_retrieve = kernel_mod->IsNeedRetrieveOutputShape();
  auto update_address = [&node, &need_retrieve, &outputs_device_address](const BaseShapePtr &shape_ptr, size_t index) {
    const auto &shape_vector = BaseShapeToShape(shape_ptr);
    MS_EXCEPTION_IF_NULL(outputs_device_address[index]);
    outputs_device_address[index]->set_host_shape(shape_vector);
    if (need_retrieve) {
      auto output_size = AnfAlgo::GetOutputTensorMemSize(node, index, shape_vector);
      outputs_device_address[index]->SetSize(output_size);
    }
  };

  if (out_abstract->isa<abstract::AbstractTuple>()) {
    auto abstract_tuple = out_abstract->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abstract_tuple);
    auto num = abstract_tuple->elements().size();
    for (size_t output_index = 0; output_index < num; ++output_index) {
      auto &real_abstract = abstract_tuple->elements()[output_index];
      MS_EXCEPTION_IF_NULL(real_abstract);
      auto shape_ptr = real_abstract->BuildShape();
      update_address(shape_ptr, output_index);
    }
  } else {
    auto shape_ptr = out_abstract->BuildShape();
    update_address(shape_ptr, 0);
  }
}

// launch dynamic kernel with input and output tensors
void LaunchKernelsDynamic(const pynative::OpCompilerInfoPtr &op_compiler_info,
                          const session::BackendOpRunInfoPtr &op_run_info,
                          std::vector<device::DeviceAddressPtr> *device_address_list) {
  MS_LOG(DEBUG) << "Start";
  // Get input tensors without const value
  auto input_tensors = GetTensorWithoutValueMask(op_run_info);
  // Update input tensors to op compiler info cache
  std::map<device::DeviceAddressPtr, tensor::TensorPtr> address_map_to_tensor;
  UpdateInputInCompileInfo(op_compiler_info, input_tensors, &address_map_to_tensor);
  // Alloc const value memory
  MallocForConstValue(op_compiler_info);

  // Ref
  MS_EXCEPTION_IF_NULL(device_address_list);
  if (device_address_list->empty()) {
    for (size_t i = 0; i < op_compiler_info->outputs_.size(); ++i) {
      device_address_list->push_back(nullptr);
    }
  }
  std::vector<device::DeviceAddressPtr> ref_node;
  UpdateOutputAddressForRef(op_compiler_info, device_address_list, &ref_node);
  auto &device_context = op_compiler_info->device_context_;
  auto &execute_kernel_list = op_compiler_info->execute_kernel_list_;
  size_t size = op_compiler_info->execute_kernel_list_.size();

  // Check if need to infer again
  bool is_need_infer = false;
  MS_EXCEPTION_IF_NULL(op_run_info->base_op_run_info.abstract);
  if (size > 1 || op_run_info->base_op_run_info.abstract->BuildShape()->IsDynamic()) {
    is_need_infer = true;
  }

  // Set whether output device memory is malloced from persistent area
  SetOutputDeviceAddressFlag(op_compiler_info, op_run_info);

  std::vector<device::DeviceAddressPtr> alloc_output_device_address;
  // Execute all kernels
  for (size_t i = 0; i < size; ++i) {
    auto &execute_kernel = execute_kernel_list[i];
    const CNodePtr &kernel = execute_kernel.kernel_;
    MS_EXCEPTION_IF_NULL(kernel);

    // Check if need infer shape
    std::vector<tensor::TensorPtr> tensors = GetAllInputTensor(
      execute_kernel.inputs_device_address_, address_map_to_tensor, op_compiler_info->value_map_to_tensor_);
    kernel::KernelArgs args;
    if (is_need_infer) {
      args = InferNodeRealShape(kernel, execute_kernel, tensors);
    } else {
      kernel->set_abstract(op_run_info->base_op_run_info.abstract);
      args = opt::dynamic_shape::SetOpArgs(kernel, execute_kernel, tensors);
    }

    // Resize
    ResizeNodeInput(kernel, args);

    // Malloc input tensor memory
    auto inputs = MallocInputMemoryForDeviceAddress(execute_kernel.inputs_device_address_);

    // Malloc workspace memory
    std::vector<device::DeviceAddressPtr> workspace_device_address;
    auto workspaces = CreateKernelWorkspaceAddressDynamic(device_context, kernel, &workspace_device_address);

    // Update output tensor shape
    UpdateOutputDeviceInfo(execute_kernel.outputs_device_address_, kernel);

    // Malloc output tensor memory
    auto outputs = MallocOutputMemoryForDeviceAddress(execute_kernel.outputs_device_address_, device_context,
                                                      &alloc_output_device_address);

    // Launch kernel
    const size_t stream_id = AnfAlgo::GetStreamId(kernel);
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetKernelExecutor(true));
    if (!device_context->GetKernelExecutor(true)->LaunchKernel(kernel, inputs, workspaces, outputs, stream_id)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, name:" << kernel->fullname_with_scope();
    }

    if (is_need_infer) {
      kernel::UpdateNodeShape(kernel);
      UpdateOutputShapeForCompileInfo(execute_kernel.outputs_device_address_, kernel);
    }
  }

  // Sync cached device info to output device info
  SyncCacheInfoToOutput(op_compiler_info, device_address_list);

  // Clear Alloc Output DeviceAddress without in device_address_list
  ClearAllocOutputDeviceAddress(*device_address_list, alloc_output_device_address);

  // Release all device memory in cache
  ReleaseCacheInfo(op_compiler_info, ref_node);
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
      auto args = kernel::GetArgsFromCNode(node);
      ResizeNodeInput(node, *args);
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
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetKernelExecutor(true));
    if (!device_context->GetKernelExecutor(false)->LaunchKernel(node, inputs, workspaces, outputs, stream_id)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, name:" << node->fullname_with_scope();
    }

    if (is_dynamic_shape) {
      kernel::UpdateNodeShape(node);
      UpdateOutputAddrSize(node, runtime_info);
    }
  }
  MS_LOG(DEBUG) << "End";
}
}  // namespace

std::vector<tensor::TensorPtr> GetTensorWithoutValueMask(const session::BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  std::vector<tensor::TensorPtr> tensors_without_value_node;
  const auto &input_tensors = op_run_info->base_op_run_info.input_tensor;
  const auto &tensors_mask = op_run_info->base_op_run_info.input_mask;
  if (input_tensors.size() != tensors_mask.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size() << " should be equal to tensors mask size "
                      << tensors_mask.size();
  }
  for (size_t index = 0; index < tensors_mask.size(); ++index) {
    if (tensors_mask.at(index) != kValueNodeTensorMask) {
      (void)tensors_without_value_node.emplace_back(input_tensors.at(index));
    }
  }
  return tensors_without_value_node;
}

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
  CopyDataToDevice(graph, input_tensors, device_context);
  LaunchKernels(graph, device_context);
}

void RunSingleOpDynamic(const session::BackendOpRunInfoPtr &op_run_info, const OpCompilerInfoPtr &op_compiler_info,
                        vector<device::DeviceAddressPtr> *device_address_list) {
  LaunchKernelsDynamic(op_compiler_info, op_run_info, device_address_list);
}

void LaunchKernelTask(const pynative::KernelTaskType &task_type, DeviceContext *device_context,
                      const device::DeviceAddressPtrList &input_addr_list,
                      const TensorStorageInfoPtrList &input_storage_list,
                      const device::DeviceAddressPtrList &output_addr_list) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(DEBUG) << "Start, task_type:" << task_type;
  if (!device_context->GetKernelExecutor(false)->ExecuteKernelTask(task_type, input_addr_list, input_storage_list,
                                                                   output_addr_list)) {
    MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type:" << task_type;
  }
  MS_LOG(DEBUG) << "End";
}
}  // namespace mindspore::runtime
