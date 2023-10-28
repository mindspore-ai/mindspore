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
  // Break copy data to device address if has the device_address has flag ignore.
  if (TEST_FLAG(device_address->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
    MS_LOG(DEBUG) << "Node " << node->DebugString() << " with address " << device_address
                  << " has flag ignore device address, so skip copy tensor to device";
    return;
  }
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

void CopyNodeValueToDevice(const device::DeviceAddressPtr &device_address, const AnfNodePtr &node,
                           const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_address);
  // Break copy data to device address if has the device_address has flag ignore.
  if (TEST_FLAG(device_address->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
    MS_LOG(DEBUG) << "Node " << node->DebugString() << " with address " << device_address
                  << " has flag ignore device address, so skip copy tensor to device";
    return;
  }

  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  if ((device_address->GetPtr() == nullptr) &&
      (!device_context->device_res_manager_->AllocateMemory(device_address.get()))) {
    MS_LOG(EXCEPTION) << "Allocate memory failed";
  }

  // Copy data from host to device.
  const auto &kernel_tensor = device_address->kernel_tensor();
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto data_size = kernel_tensor->size();
  if (data_size == 0) {
    MS_LOG(INFO) << "Node " << node->DebugString() << " is empty.";
    return;
  }
  const void *node_value = kernel_tensor->GetValuePtr();
  MS_EXCEPTION_IF_NULL(node_value);
  auto data_type_id = kernel_tensor->dtype_id();
  auto format = kernel_tensor->GetStringFormat();
  MS_LOG(DEBUG) << "Copy to device, node:" << common::AnfAlgo::GetNodeDebugString(node);
  if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0), data_size, data_type_id, node_value,
                                        format)) {
    MS_LOG(EXCEPTION) << "SyncHostToDevice failed";
  }
}

void CopyValueNodeDataToDevice(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Start";
  const auto &value_nodes = graph->graph_value_nodes();
  for (const auto &value_node : value_nodes) {
    MS_EXCEPTION_IF_NULL(value_node);
    const auto &node_address = AnfAlgo::GetMutableOutputAddr(value_node, 0, false);
    MS_EXCEPTION_IF_NULL(node_address);
    node_address->SetNodeIndex(value_node, 0);
    if (node_address->GetPtr() != nullptr) {
      continue;
    }

    CopyNodeValueToDevice(node_address, value_node, device_context);
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
    if (TEST_FLAG(input_address->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
      MS_LOG(DEBUG) << "Node " << node->DebugString() << " input[" << i << "] with address " << input_address
                    << " has flag ignore device address, so skip malloc device address";
      continue;
    }
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

std::vector<kernel::KernelTensor *> GetInputKernelTensors(const std::shared_ptr<OpRuntimeInfo> &runtime_info,
                                                          const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  auto input_size = runtime_info->GetInputSize();
  std::vector<kernel::KernelTensor *> inputs;
  for (size_t i = 0; i < input_size; ++i) {
    if (common::AnfAlgo::IsNoneInput(node, i)) {
      (void)inputs.emplace_back(nullptr);
      MS_LOG(DEBUG) << "Input[" << i << "]:"
                    << " is None Input";
      continue;
    }
    auto device_address = runtime_info->GetInputDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(device_address);
    (void)inputs.emplace_back(device_address->kernel_tensor().get());
    MS_EXCEPTION_IF_NULL(inputs.back());
    MS_LOG(DEBUG) << "input[" << i << "]:" << inputs.back()->device_ptr() << " size:" << inputs.back()->size();
  }
  return inputs;
}

std::vector<kernel::KernelTensor *> GetWorkspaceKernelTensors(const std::shared_ptr<OpRuntimeInfo> &runtime_info,
                                                              const device::DeviceContext *device_context,
                                                              const CNodePtr &kernel, bool is_dynamic_shape,
                                                              bool is_dynamic_value) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto workspace_size = runtime_info->GetWorkspaceSize();
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();

  std::vector<device::DeviceAddressPtr> add_workspaces;
  if (is_dynamic_shape || is_dynamic_value) {
    // Resize of workspaces, because of the dynamic size of workspace.
    if (workspace_size < workspace_sizes.size()) {
      for (size_t i = workspace_size; i < workspace_sizes.size(); ++i) {
        auto kernel_tensor = std::make_shared<KernelTensor>(
          nullptr, workspace_sizes[i], "", kTypeUnknown, ShapeVector(),
          device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
        auto device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
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

  std::vector<kernel::KernelTensor *> workspaces;
  for (size_t i = 0; i < workspace_size && i < workspace_sizes.size(); ++i) {
    auto device_address = runtime_info->GetWorkspaceDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    (void)workspaces.emplace_back(device_address->kernel_tensor().get());
    MS_EXCEPTION_IF_NULL(workspaces.back());
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->device_ptr()
                  << " size:" << workspaces.back()->size();
  }

  for (size_t i = workspace_size; i < workspace_sizes.size(); ++i) {
    auto device_address = add_workspaces[i];
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed";
    }
    (void)workspaces.emplace_back(device_address->kernel_tensor().get());
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->device_ptr()
                  << " size:" << workspaces.back()->size();
  }
  return workspaces;
}

std::vector<kernel::KernelTensor *> GetWorkspaceKernelTensorsDynamic(
  const device::DeviceContext *device_context, const CNodePtr &kernel,
  std::vector<device::DeviceAddressPtr> *workspace_device_address) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();

  std::vector<kernel::KernelTensor *> workspaces;
  workspaces.reserve(workspace_sizes.size());
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto kernel_tensor = std::make_shared<KernelTensor>(nullptr, workspace_sizes[i], "", kTypeUnknown, ShapeVector(),
                                                        device_context->device_context_key().device_name_,
                                                        device_context->device_context_key().device_id_);
    auto device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate dynamic workspace memory failed";
    }
    MS_EXCEPTION_IF_NULL(workspace_device_address);
    (void)workspace_device_address->emplace_back(device_address);
    (void)workspaces.emplace_back(device_address->kernel_tensor().get());
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->device_ptr()
                  << " size:" << workspaces.back()->size();
  }
  return workspaces;
}

std::vector<kernel::KernelTensor *> GetOutputKernelTensors(const std::shared_ptr<OpRuntimeInfo> &runtime_info) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  auto output_size = runtime_info->GetOutputSize();
  std::vector<kernel::KernelTensor *> outputs;
  for (size_t i = 0; i < output_size; ++i) {
    auto device_address = runtime_info->GetOutputDeviceAddress(i);
    MS_EXCEPTION_IF_NULL(device_address);
    (void)outputs.emplace_back(device_address->kernel_tensor().get());
    MS_LOG(DEBUG) << "output[" << i << "]:" << outputs.back()->device_ptr() << " size:" << outputs.back()->size();
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

BaseShapePtr InferNodeRealShape(const CNodePtr &kernel, const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(kernel);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelInfer,
                                     kernel->fullname_with_scope(), false);
  auto *kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  return opt::dynamic_shape::InferShape(kernel_mod->primitive(), input_args);
}

void ResizeKernelMod(const CNodePtr &kernel, const std::vector<kernel::KernelTensor *> &inputs,
                     const std::vector<kernel::KernelTensor *> &outputs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelResize,
                                     kernel->fullname_with_scope(), false);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  kernel_mod->set_use_kernel_tensor(true);

  int ret = kernel_mod->Resize(inputs, outputs);
  if (ret != kernel::KRET_OK) {
    MS_LOG(EXCEPTION) << "Resize failed for kernel: " << kernel->fullname_with_scope();
  }
}

void AllocateOutputMemory(const std::vector<device::DeviceAddressPtr> &device_addressess,
                          const device::DeviceContext *device_context,
                          std::vector<device::DeviceAddressPtr> *alloc_output_device_address) {
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
  }
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
  const auto &shape = tensor->shape();
  size_t tensor_size = GetTensorDeviceSize(device_context, node, shape, format, dtype, 0);

  // Update shape and size for cached device address.
  cached_device_address->set_host_shape(shape);
  cached_device_address->kernel_tensor()->SetShapeVector(shape);
  cached_device_address->SetSize(tensor_size);

  // Create new device address from cached device device address.
  const auto &kernel_tensor = cached_device_address->kernel_tensor();
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto new_kernel_tensor = kernel_tensor->Clone();
  MS_EXCEPTION_IF_NULL(new_kernel_tensor);
  new_kernel_tensor->set_device_ptr(nullptr);
  auto new_device_address = device_context->device_res_manager_->CreateDeviceAddress(new_kernel_tensor);
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
  cached_device_address->kernel_tensor()->SetShapeVector(tensor->shape());
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
    const auto &kernel_tensor = device_addressess[i]->kernel_tensor();
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    device_addressess[i]->set_host_shape(kernel_tensor->GetShapeVector());
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
    output_address->kernel_tensor()->SetShapeVector(input_address->host_shape());

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

void UpdateOutputShape(const std::vector<device::DeviceAddressPtr> &outputs_device_address) {
  for (size_t i = 0; i < outputs_device_address.size(); i++) {
    const auto &device_address = outputs_device_address[i];
    MS_EXCEPTION_IF_NULL(device_address);
    const auto &kernel_tensor = device_address->kernel_tensor();
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    device_address->set_host_shape(kernel_tensor->GetShapeVector());
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
  const auto &execute_kernel_list = op_compiler_info->execute_kernel_list_;
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
  std::vector<kernel::KernelTensor *> input_kernel_tensors;
  std::vector<abstract::AbstractBasePtr> input_kernel_tensors_for_infer;
  std::vector<kernel::KernelTensor *> output_kernel_tensors;
  // Execute all kernels
  for (size_t i = 0; i < size; ++i) {
    auto &execute_kernel = execute_kernel_list[i];
    const CNodePtr &kernel = execute_kernel.kernel_;
    MS_EXCEPTION_IF_NULL(kernel);

    // Fetch input kernel tensor.
    const auto &input_device_addresses = execute_kernel.inputs_device_address_;
    size_t input_size = input_device_addresses.size();
    input_kernel_tensors.resize(input_size);
    input_kernel_tensors_for_infer.resize(input_size);
    for (size_t j = 0; j < input_size; j++) {
      MS_EXCEPTION_IF_NULL(input_device_addresses[j]);
      input_kernel_tensors[j] = input_device_addresses[j]->kernel_tensor().get();
      input_kernel_tensors_for_infer[j] = input_device_addresses[j]->kernel_tensor();
    }

    // Fetch output kernel tensor.
    const auto &output_device_addresses = execute_kernel.outputs_device_address_;
    size_t output_size = output_device_addresses.size();
    output_kernel_tensors.resize(output_size);
    for (size_t j = 0; j < output_size; j++) {
      MS_EXCEPTION_IF_NULL(output_device_addresses[j]);
      output_kernel_tensors[j] = output_device_addresses[j]->kernel_tensor().get();
    }

    BaseShapePtr out_shape;
    if (is_need_infer) {
      out_shape = InferNodeRealShape(kernel, input_kernel_tensors_for_infer);
    } else {
      kernel->set_abstract(op_run_info->base_op_run_info.abstract);
      out_shape = op_run_info->base_op_run_info.abstract->GetShape();
    }
    // Update output kernel tensor.
    opt::dynamic_shape::UpdateKernelTensorShape(out_shape, output_kernel_tensors);

    // Resize
    ResizeKernelMod(kernel, input_kernel_tensors, output_kernel_tensors);

    // Malloc workspace memory
    std::vector<device::DeviceAddressPtr> workspace_device_address;
    auto workspace_kernel_tensors = GetWorkspaceKernelTensorsDynamic(device_context, kernel, &workspace_device_address);

    // Update output tensor shape
    UpdateOutputDeviceInfo(execute_kernel.outputs_device_address_, kernel);

    // Malloc output tensor memory
    AllocateOutputMemory(execute_kernel.outputs_device_address_, device_context, &alloc_output_device_address);

    // Launch kernel
    const size_t stream_id = AnfAlgo::GetStreamId(kernel);
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetKernelExecutor(true));
    if (!device_context->GetKernelExecutor(true)->LaunchKernel(kernel, input_kernel_tensors, workspace_kernel_tensors,
                                                               output_kernel_tensors, stream_id)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, name:" << kernel->fullname_with_scope();
    }

    if (is_need_infer) {
      auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      if (kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
        kernel_mod->UpdateOutputShapeAndSize(input_kernel_tensors, output_kernel_tensors);
        UpdateOutputShape(execute_kernel.outputs_device_address_);
      }
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
    auto inputs = GetInputKernelTensors(runtime_info, node);
    auto outputs = GetOutputKernelTensors(runtime_info);
    bool is_dynamic_value = common::AnfAlgo::IsDynamicValue(node);
    if (is_dynamic_value) {
      auto kernel_mod = runtime_info->GetKernelMod();
      MS_EXCEPTION_IF_NULL(kernel_mod);
      if (kernel_mod->Resize(inputs, outputs) != static_cast<int>(kernel::KRET_OK)) {
        MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " resize failed";
      }
    }
    auto workspaces = GetWorkspaceKernelTensors(runtime_info, device_context, node, is_dynamic_shape, is_dynamic_value);

    if (!MallocForKernelOutput(runtime_info, node, device_context)) {
      MS_LOG(EXCEPTION) << "Malloc for kernel output failed, Memory isn't enough, node:" << node->fullname_with_scope();
    }

    const size_t stream_id = AnfAlgo::GetStreamId(node);
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetKernelExecutor(true));
    if (!device_context->GetKernelExecutor(false)->LaunchKernel(node, inputs, workspaces, outputs, stream_id)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, name:" << node->fullname_with_scope();
    }
  }
  MS_LOG(DEBUG) << "End";
}
}  // namespace

std::vector<tensor::TensorPtr> GetTensorWithoutValueMask(const session::BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  std::vector<tensor::TensorPtr> tensors_without_value_node;
  const auto &input_values = op_run_info->base_op_run_info.expanded_input_values;
  const auto &input_masks = op_run_info->base_op_run_info.input_masks;
  if (input_values.size() != input_masks.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_values.size() << " should be equal to tensors mask size "
                      << input_masks.size();
  }
  for (size_t index = 0; index < input_masks.size(); ++index) {
    if (input_masks.at(index) != kValueNodeMask) {
      if (!input_values[index]->isa<tensor::Tensor>()) {
        MS_LOG(EXCEPTION) << "The " << index << "' input shoulde be a Tensor, but got "
                          << input_values[index]->ToString();
      }
      (void)tensors_without_value_node.emplace_back(input_values.at(index)->cast<tensor::TensorPtr>());
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
