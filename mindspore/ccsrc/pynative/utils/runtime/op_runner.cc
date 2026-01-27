/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#include "include/pynative/utils/runtime/op_runner.h"

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include <array>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "utils/log_adapter.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/backend/common/exec_order/execute_order_tracker.h"
#include "include/backend/common/pass_manager/helper.h"
#include "device_address/device_type.h"
#include "ir/map_tensor.h"
#include "include/backend/common/device_address_utils.h"
#include "include/pynative/utils/runtime/op_runtime_info.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "include/pynative/utils/runtime/op_compiler.h"
#include "include/backend/common/exec_order/kernel_cache.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "tools/profiler/profiling.h"
#include "include/backend/common/pass_manager/dynamic_shape_helper.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pynative/utils/runtime/ir_converter.h"
#include "utils/stream_guard.h"
#include "primitive/structure_op_name.h"
#include "primitive/framework_op_name.h"
using mindspore::profiler::ProfilerManager;
using EdgePtr = mindspore::pynative::EdgePtr;

namespace mindspore::runtime {
namespace {
const std::set<std::string> no_dyn_need_update_ops = {kDynamicGetNextV2OpName, kDynamicGetNextAscendOpName,
                                                      kGetNextOpName, kGetNextFromQueueOpName, kReceiveOpName};
constexpr size_t kContextSize = 5;
std::unique_ptr<std::mutex> kDeviceContextMutex = std::make_unique<std::mutex>();
std::array<DeviceContext *, kContextSize> kDeviceContexts = {nullptr, nullptr, nullptr, nullptr, nullptr};

// 1. Device type is different in heterogeneous scenes.
// 2. The device address format is different.
void UpdateInputTensorFromDevice(const std::vector<AnfNodePtr> &input_nodes,
                                 const std::vector<tensor::TensorPtr> &input_tensors,
                                 const device::DeviceContext *device_context, size_t stream_id) {
  MS_LOG(DEBUG) << "Start";
  auto input_size = input_nodes.size();
  for (size_t i = 0; i < input_size; ++i) {
    auto &tensor = input_tensors[i];
    auto &input_node = input_nodes[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_address = tensor->device_address();

    auto node_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
    // node_address can't be null
    MS_EXCEPTION_IF_NULL(node_address);
    MS_EXCEPTION_IF_NULL(device_context);

    CheckAutoH2D(device_context, tensor);

    if (tensor_address != nullptr) {
      auto node_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0);
      if (tensor_address->GetDeviceType() != device_context->GetDeviceType() ||
          tensor->format() != node_kernel_tensor->format()) {
        runtime::Pipeline::Get().WaitForward();
        // malloc memory in a contiguous block.
        node_address->set_from_persistent_mem(tensor->is_parameter());
        node_address->SetNodeIndex(input_node, 0);

        tensor->set_sync_status(kNeedSyncHostToDeviceImmediately);
        tensor->set_need_pipeline_sync(true);
        tensor->set_implicit_copy_address(tensor_address);
        tensor->set_implicit_copy_format(tensor->format());
        tensor->set_implicit_copy_storage_info(tensor->storage_info());
        tensor->set_device_address(node_address);
        tensor->set_format(node_kernel_tensor->format());
        tensor->set_storage_info(node_kernel_tensor->tensor_storage_info());
        MS_LOG(DEBUG) << "Set to_device callback for tensor " << tensor->ToString() << " id " << tensor->id();
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

void SetDeviceAddress(const AnfNodePtr &input_node, const tensor::TensorPtr &input_tensor,
                      const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto tensor_address = input_tensor->device_address();
  auto node_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);

  UpdateParameterShapeFromInputTensor(input_node, input_tensor);

  // The DeviceType and format of DeviceAddress is always the same after UpdateInputTensor
  if (tensor_address != nullptr && tensor_address != node_address) {
    auto address = tensor_address;
    if (input_tensor->storage_info() != nullptr) {
      MS_EXCEPTION(RuntimeError) << "Not support view tensor.";
    }
    AnfAlgo::SetOutputAddr(address, input_tensor->storage_info(), 0, input_node);
  }
}

void UpdateInputNodeDeviceAddress(const std::vector<AnfNodePtr> &input_nodes,
                                  const std::vector<tensor::TensorPtr> &input_tensors,
                                  const device::DeviceContext *device_context) {
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
      SetDeviceAddress(input_node, map_tensor, device_context);
      SetDeviceAddress(input_node, map_tensor->key_tensor(), device_context);
      SetDeviceAddress(input_node, map_tensor->value_tensor(), device_context);
      SetDeviceAddress(input_node, map_tensor->status_tensor(), device_context);
    } else {
      SetDeviceAddress(input_node, input_tensor, device_context);
    }
  }
  MS_LOG(DEBUG) << "End";
}

void CopyTensorDataToDevice(const tensor::TensorPtr &tensor, const AnfNodePtr &node,
                            const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto device_address = tensor->device_address();
  MS_EXCEPTION_IF_CHECK_FAIL(device_address != nullptr,
                             "Tensor device address is nullptr, id is " + std::to_string(tensor->id()));
  // Break copy data to device address if has the device_address has flag ignore.
  auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, 0, false);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (TEST_FLAG(kernel_tensor->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
    MS_LOG(DEBUG) << "Node " << node->DebugString() << " with address " << device_address
                  << " has flag ignore device address, so skip copy tensor to device";
    return;
  }

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "CopyTensorData", "");
  auto mem_type =
    tensor->is_parameter() ? memory::mem_pool::MemType::kWeight : memory::mem_pool::MemType::kPyNativeInput;
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", mem_type, device_address->GetSize(),
                                                 device_address.get());
  if ((device_address->GetPtr() == nullptr) &&
      (!device_context->device_res_manager_->AllocateMemory(device_address.get(), CurrentStream::id()))) {
    MS_LOG(EXCEPTION) << "Allocate memory failed, alloc size " << device_address->GetSize() << "B";
  }

  MS_LOG(DEBUG) << "Copy to device, node:" << common::AnfAlgo::GetNodeDebugString(node) << " tensor "
                << tensor->ToString() << " id:" << tensor->id();

  MS_LOG(DEBUG) << "Start lazy copy for tensor " << tensor->ToString();
  runtime::DeviceAddressUtils::LazyCopy(tensor, CurrentStream::id());

  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
    MarkTensorAsOutput, "PyNative", device::GetDeviceNameByType(device_address->GetDeviceType()),
    device_address->GetPtr(), tensor->data_type(), tensor->shape(), tensor->storage_info());
}

void CopyValueNodeDataToDevice(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Start";
  const auto &value_nodes = graph->graph_value_nodes();
  for (const auto &value_node : value_nodes) {
    MS_EXCEPTION_IF_NULL(value_node);
    const auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    if (!node_value->isa<tensor::Tensor>() && !node_value->isa<ValueTuple>() && !node_value->isa<Scalar>() &&
        !node_value->isa<StringImm>()) {
      MS_LOG(INFO) << "Unknown value node type:" << value_node->DebugString();
      continue;
    }

    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(value_node, 0, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto node_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(node_address);
    node_address->SetNodeIndex(value_node, 0);
    if (node_address->GetPtr() != nullptr) {
      continue;
    }
    auto shape = AnfAlgo::GetRuntimePaddingShape(value_node, 0);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "CopyValueNodeData", "");
    runtime::DeviceAddressUtils::CopyNoneTensorDataToDevice(device_context, kernel_tensor, shape);
  }
  MS_LOG(DEBUG) << "End";
}

void UpdateAddressSizeForDynamicShapeTensor(const tensor::TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  if (input_tensor->base_shape_ptr() != nullptr) {
    auto device_address = input_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    auto tensor_size = input_tensor->DataNBytes();
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
      MS_LOG(DEBUG) << "Input [" << i << "] of " << node->fullname_with_scope() << " is None, no need to allocate.";
      continue;
    }
    auto input_kernel_tensor = runtime_info->GetInputKernelTensor(i);
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    auto input_address = input_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(kernel_mod);
    MS_EXCEPTION_IF_NULL(input_address);
    if (TEST_FLAG(input_kernel_tensor->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
      MS_LOG(DEBUG) << "Node " << node->DebugString() << " input[" << i << "] with address " << input_address
                    << " has flag ignore device address, so skip malloc device address";
      continue;
    }
    if (input_address->GetPtr() == nullptr) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                     input_address->GetSize(), input_address.get());
      if (!device_context->device_res_manager_->AllocateMemory(input_address.get(), CurrentStream::id())) {
        MS_LOG(EXCEPTION) << "Allocate memory failed, alloc size " << input_address->GetSize() << "B";
      }
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
    auto kernel_tensor = runtime_info->GetOutputKernelTensor(i);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    // For example, we need to call cudnnGetRNNTrainingReserveSize to get real output size in LstmGpuKernelMod!
    if (kernel_out_size_list[i] != device_address->GetSize() &&
        AnfAlgo::GetOutputFormat(node, i) == kernel_tensor->GetStringFormat()) {
      // If the format of the DeviceAddress is different, then the size is originally different.
      // Such as NCHW(1,1,1,3) and NC1HWC0(1,1,1,1,16). So we don't need to update the size.
      if (device_address->GetPtr() != nullptr) {
        MS_LOG(ERROR) << "kernel mod output " << i << " size:" << kernel_out_size_list[i]
                      << " not equal to device_address size:" << device_address->GetSize()
                      << ", but the device address is already have ptr";
        return false;
      }
      device_address->SetSize(kernel_out_size_list[i]);
    }
    if (device_address->GetPtr() == nullptr) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                     device_address->GetSize(), device_address.get());
      if (!device_context->device_res_manager_->AllocateMemory(device_address.get(), CurrentStream::id())) {
        MS_LOG(EXCEPTION) << "Allocate output memory failed, alloc node:" << node->fullname_with_scope()
                          << " alloc size:" << device_address->GetSize() << "B";
      }
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
    auto kernel_tensor = runtime_info->GetInputKernelTensor(i);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    (void)inputs.emplace_back(kernel_tensor.get());
    MS_EXCEPTION_IF_NULL(inputs.back());
    MS_LOG(DEBUG) << "input[" << i << "]:" << inputs.back()->device_ptr() << " size:" << inputs.back()->size();
  }
  return inputs;
}

std::vector<abstract::AbstractBasePtr> GetInputKernelTensorsForInfer(const std::shared_ptr<OpRuntimeInfo> &runtime_info,
                                                                     const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  auto input_size = runtime_info->GetInputSize();
  std::vector<abstract::AbstractBasePtr> inputs;
  for (size_t i = 0; i < input_size; ++i) {
    auto kernel_tensor = runtime_info->GetInputKernelTensor(i);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    (void)inputs.emplace_back(kernel_tensor);
    MS_EXCEPTION_IF_NULL(inputs.back());
  }
  return inputs;
}

std::vector<kernel::KernelTensor *> GetWorkspaceKernelTensors(const std::shared_ptr<OpRuntimeInfo> &runtime_info,
                                                              const device::DeviceContext *device_context,
                                                              size_t workspace_size, size_t workspace_sizes) {
  std::vector<kernel::KernelTensor *> workspaces;
  for (size_t i = 0; i < workspace_size && i < workspace_sizes; ++i) {
    auto kernel_tensor = runtime_info->GetWorkspaceKernelTensor(i);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kWorkSpace,
                                                   device_address->GetSize(), device_address.get());
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get(), CurrentStream::id())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed, alloc size:" << device_address->GetSize() << "B";
    }
    (void)workspaces.emplace_back(kernel_tensor.get());
    MS_EXCEPTION_IF_NULL(workspaces.back());
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->device_ptr()
                  << " size:" << workspaces.back()->size();
  }
  return workspaces;
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

  std::vector<KernelTensorPtr> add_workspaces;
  if (is_dynamic_shape || is_dynamic_value) {
    // Resize of workspaces, because of the dynamic size of workspace.
    if (workspace_size < workspace_sizes.size()) {
      for (size_t i = workspace_size; i < workspace_sizes.size(); ++i) {
        auto kernel_tensor =
          AnfAlgo::CreateKernelTensor(nullptr, workspace_sizes[i], Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
                                      device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                      device_context->device_context_key().device_id_);
        MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(kernel)
                      << " kernel tensor:" << kernel_tensor;
        AnfAlgo::SetWorkspaceKernelTensor(kernel_tensor, i, kernel.get());  // set to kernel_info
        (void)add_workspaces.emplace_back(kernel_tensor);
      }
    }
  }

  // Set workspace address new size
  for (size_t i = 0; i < workspace_size && i < workspace_sizes.size(); ++i) {
    auto kernel_tensor = runtime_info->GetWorkspaceKernelTensor(i);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->SetSize(workspace_sizes[i]);
  }

  std::vector<kernel::KernelTensor *> workspaces =
    GetWorkspaceKernelTensors(runtime_info, device_context, workspace_size, workspace_sizes.size());
  for (size_t i = workspace_size; i < workspace_sizes.size(); ++i) {
    auto kernel_tensor = add_workspaces[i];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    auto device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kWorkSpace,
                                                   device_address->GetSize(), device_address.get());
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get(), CurrentStream::id())) {
      MS_LOG(EXCEPTION) << "Allocate workspace memory failed, alloc size:" << device_address->GetSize() << "B";
    }
    (void)workspaces.emplace_back(kernel_tensor.get());
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->device_ptr()
                  << " size:" << workspaces.back()->size();
  }
  return workspaces;
}

std::vector<kernel::KernelTensor *> GetWorkspaceKernelTensorsDynamic(
  const device::DeviceContext *device_context, const CNodePtr &kernel,
  std::vector<kernel::KernelTensorPtr> *workspace_kts) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();

  std::vector<kernel::KernelTensor *> workspaces;
  workspaces.reserve(workspace_sizes.size());
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    auto kernel_tensor =
      AnfAlgo::CreateKernelTensor(nullptr, workspace_sizes[i], Format::DEFAULT_FORMAT, kTypeUnknown, ShapeVector(),
                                  device::GetDeviceNameByType(device_context->device_context_key().device_type_),
                                  device_context->device_context_key().device_id_);
    auto device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kWorkSpace,
                                                   device_address->GetSize(), device_address.get());
    if (device_address->GetPtr() == nullptr &&
        !device_context->device_res_manager_->AllocateMemory(device_address.get(), CurrentStream::id())) {
      MS_LOG(EXCEPTION) << "Allocate dynamic workspace memory failed, alloc size:" << device_address->GetSize() << "B";
    }
    MS_EXCEPTION_IF_NULL(workspace_kts);
    (void)workspace_kts->emplace_back(kernel_tensor);
    (void)workspaces.emplace_back(kernel_tensor.get());
    MS_LOG(DEBUG) << "workspace[" << i << "]:" << workspaces.back()->device_ptr()
                  << " size:" << workspaces.back()->size() << " kernel tensor:" << kernel_tensor->ToString();
  }
  return workspaces;
}

std::vector<kernel::KernelTensor *> GetOutputKernelTensors(const std::shared_ptr<OpRuntimeInfo> &runtime_info) {
  MS_EXCEPTION_IF_NULL(runtime_info);
  auto output_size = runtime_info->GetOutputSize();
  std::vector<kernel::KernelTensor *> outputs;
  for (size_t i = 0; i < output_size; ++i) {
    auto kernel_tensor = runtime_info->GetOutputKernelTensor(i);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    (void)outputs.emplace_back(kernel_tensor.get());
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

void SetOutputDeviceAddressFlag(const pynative::OpCompilerInfoPtr &op_compiler_info,
                                const session::BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  MS_EXCEPTION_IF_NULL(op_run_info);
  const auto &simple_graph = op_compiler_info->simple_graph_;
  size_t output_size = simple_graph->outputs_.size();
  // Reset grad output flag.
  const auto &outputs = simple_graph->outputs_;
  for (const auto &output : outputs) {
    output->is_grad_ = false;
  }

  if (op_run_info->is_gradient_out) {
    const auto &output_indexes = op_run_info->base_op_run_info.output_indexes;
    for (auto index : output_indexes) {
      if (index >= output_size) {
        MS_LOG(EXCEPTION) << "Gradient output index " << index << " >= graph output size " << output_size;
      }
      const auto &output = outputs[index];
      MS_EXCEPTION_IF_NULL(output);
      output->is_grad_ = true;
      MS_LOG(DEBUG) << "Set grad flag for op " << op_run_info->base_op_run_info.op_name << " index " << index;
    }
  }
}

void MallocForConstValue(const pynative::OpCompilerInfoPtr &op_compiler_info) {
  MS_EXCEPTION_IF_NULL(op_compiler_info);
  const auto &device_context = op_compiler_info->device_context_;
  const auto &graph = op_compiler_info->graph_;
  CopyValueNodeDataToDevice(graph, device_context);
}

void TrackerACLMemory(const std::vector<kernel::KernelTensor *> &input_tensors,
                      const std::vector<kernel::KernelTensor *> &output_tensors) {
  if (!device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    return;
  }
  for (auto &kernel_tensor : input_tensors) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
      MarkTensorAsInput, "PyNative", device::GetDeviceNameByType(kernel_tensor->GetDeviceType()),
      kernel_tensor->device_ptr(), kernel_tensor->dtype_id(), kernel_tensor->GetDeviceShapeVector(),
      kernel_tensor->tensor_storage_info());
  }
  for (auto &kernel_tensor : output_tensors) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
      MarkTensorAsOutput, "PyNative", device::GetDeviceNameByType(kernel_tensor->GetDeviceType()),
      kernel_tensor->device_ptr(), kernel_tensor->dtype_id(), kernel_tensor->GetDeviceShapeVector(),
      kernel_tensor->tensor_storage_info());
  }
}

void LaunchKernels(const KernelGraphPtr &graph, const device::DeviceContext *device_context,
                   const session::BackendOpRunInfoPtr &op_run_info,
                   const std::vector<tensor::TensorPtr> &input_tensors) {
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
    bool is_dynamic_value = common::AnfAlgo::IsDynamicValue(node);
    auto runtime_info = node->user_data<runtime::OpRuntimeInfo>();
    MS_EXCEPTION_IF_NULL(runtime_info);

    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", node->fullname_with_scope(),
                                                   op_run_info->base_op_run_info.op_name);
    auto &cache = KernelCache::GetInstance();
    if (cache.need_add) {
      cache.Add(node);
    }

    if (EnableExecuteOrderDump()) {
      auto &execute_order_tracker = ExecuteOrderTracker::GetInstance();
      execute_order_tracker.ProcessNode(node);
    }
    if (!MallocForKernelInput(runtime_info, device_context, node)) {
      MS_LOG(EXCEPTION) << "Malloc for kernel input failed, Memory isn't enough, node:" << node->fullname_with_scope();
    }

    auto inputs = GetInputKernelTensors(runtime_info, node);
    auto outputs = GetOutputKernelTensors(runtime_info);
    if (is_dynamic_shape) {
      auto input_kernel_tensors_for_infer = GetInputKernelTensorsForInfer(runtime_info, node);
      auto out_shape = InferNodeRealShape(node, input_kernel_tensors_for_infer);
      opt::dynamic_shape::UpdateKernelTensorShape(out_shape, outputs);
      ResizeKernelMod(node, inputs, outputs);
    } else if (is_dynamic_value) {
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

    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetKernelExecutor());
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    const size_t stream_id = op_run_info->base_op_run_info.stream_id;
    auto stream = device_context->device_res_manager_->GetStream(stream_id);
    static auto no_simu = !common::IsCompileSimulation();
    TrackerACLMemory(inputs, outputs);
    if (no_simu &&
        !device_context->GetKernelExecutor()->LaunchKernel(node, inputs, workspaces, outputs, kernel_mod, stream)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, name:" << node->fullname_with_scope();
    }
    runtime::DeviceAddressUtils::ProcessCrossStreamAddress(op_run_info->base_op_run_info.op_name, device_context,
                                                           stream_id, inputs, outputs);
  }
  MS_LOG(DEBUG) << "End";
}

void AllocateOutputMemory(const std::vector<EdgePtr> &output_edges, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  for (const auto &edge : output_edges) {
    MS_EXCEPTION_IF_NULL(edge);
    const auto &kernel_tensor = edge->kernel_tensor_;
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    const auto &device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() == nullptr) {
      if (edge->is_grad_) {
        device_address->set_from_persistent_mem(true);
      }
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                     device_address->GetSize(), device_address.get());
      MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
      if (!device_context->device_res_manager_->AllocateMemory(device_address.get(), CurrentStream::id())) {
        MS_LOG(EXCEPTION) << "Allocate device memory failed, alloc size:" << device_address->GetSize() << "B";
      }
    }
  }
}

void UpdateOutputDeviceInfo(const std::vector<EdgePtr> &edges, const CNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_size_list = kernel_mod->GetOutputSizeList();
  if (edges.size() != output_size_list.size()) {
    MS_LOG(EXCEPTION) << "Output device address's size " << edges.size() << " is not equal output_size_list's size "
                      << output_size_list.size();
  }

  auto output_num = edges.size();
  for (size_t i = 0; i < output_num; ++i) {
    const auto &edge = edges[i];
    MS_EXCEPTION_IF_NULL(edge);
    const auto &kernel_tensor = edge->kernel_tensor_;
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    const auto &device_address = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->SetSize(output_size_list[i]);
  }
}

void UpdateAddressInfoByInputTensor(const OpCompilerInfoPtr &op_compiler_info, const tensor::TensorPtr &tensor,
                                    const EdgePtr &edge, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(node);
  auto &device_context = op_compiler_info->device_context_;
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  const auto &kernel_tensor = edge->origin_kernel_tensor_;
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto origin_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(origin_address);

  const auto &format = kernel_tensor->GetStringFormat();
  const auto dtype = kernel_tensor->dtype_id();
  const auto &shape = tensor->shape();
  size_t tensor_size = DeviceAddressUtils::GetTensorDeviceSize(device_context, node, shape, format, dtype, 0);

  auto new_kernel_tensor = kernel_tensor->CloneKernelTensor();
  MS_EXCEPTION_IF_NULL(new_kernel_tensor);

  new_kernel_tensor->SetShapeVector(shape);
  new_kernel_tensor->set_device_ptr(nullptr);
  auto new_device_address = new_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(new_device_address);
  new_device_address->SetSize(tensor_size);
  new_device_address->set_from_persistent_mem(tensor->is_parameter());

  MS_LOG(DEBUG) << "Update input edge kernel tensor from "
                << (edge->kernel_tensor_ != nullptr ? edge->kernel_tensor_->ToString() : "None") << " to "
                << new_kernel_tensor->ToString();
  edge->kernel_tensor_ = new_kernel_tensor;
}

std::vector<kernel::KernelTensor *> GetInputKernelTensors(const std::vector<EdgePtr> &edges) {
  std::vector<kernel::KernelTensor *> input_kernel_tensors;
  input_kernel_tensors.reserve(edges.size());
  (void)std::transform(edges.begin(), edges.end(), std::back_inserter(input_kernel_tensors), [](const EdgePtr &edge) {
    MS_EXCEPTION_IF_NULL(edge->kernel_tensor_);
    return edge->kernel_tensor_.get();
  });
  return input_kernel_tensors;
}

std::vector<abstract::AbstractBasePtr> GetInputInferAbstract(const std::vector<EdgePtr> &edges) {
  std::vector<abstract::AbstractBasePtr> input_abstracts;
  input_abstracts.reserve(edges.size());
  (void)std::transform(edges.begin(), edges.end(), std::back_inserter(input_abstracts), [](const EdgePtr &edge) {
    MS_EXCEPTION_IF_NULL(edge->kernel_tensor_);
    return edge->kernel_tensor_;
  });
  return input_abstracts;
}

std::vector<kernel::KernelTensor *> GetOutputKernelTensors(const std::vector<EdgePtr> &edges,
                                                           const DeviceContext *device_context) {
  std::vector<kernel::KernelTensor *> output_kernel_tensors;
  output_kernel_tensors.reserve(edges.size());
  for (const auto &edge : edges) {
    // For example, output is dynamic or the output is between two ops.
    if (edge->kernel_tensor_ == nullptr) {
      edge->kernel_tensor_ =
        runtime::DeviceAddressUtils::CloneEmptyKernelTensor(edge->origin_kernel_tensor_, device_context);
    }
    const auto &kernel_tensor = edge->kernel_tensor_;
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    output_kernel_tensors.push_back(kernel_tensor.get());
  }
  return output_kernel_tensors;
}

void CheckInputTensorStream(const TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  const auto &address = tensor->device_address();
  if (address == nullptr) {
    MS_LOG(INFO) << "Input tensor address is null, " << tensor->ToString();
    return;
  }
  if (address->stream_id() != CurrentStream::id()) {
    MS_LOG(INFO) << "Input " << tensor->ToString() << " stream id:" << address->stream_id()
                 << " current stream id:" << CurrentStream::id();
  }
}
}  // namespace

std::vector<tensor::TensorPtr> OpRunner::GetTensorWithoutValueMask(const session::BackendOpRunInfoPtr &op_run_info) {
  MS_EXCEPTION_IF_NULL(op_run_info);
  std::vector<tensor::TensorPtr> tensors_without_value_node;
  const auto &input_values = op_run_info->base_op_run_info.expanded_input_values;
  const auto &input_masks = op_run_info->base_op_run_info.input_types;
  if (input_values.size() != input_masks.size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_values.size() << " should be equal to tensors mask size "
                      << input_masks.size();
  }
  for (size_t index = 0; index < input_masks.size(); ++index) {
    if (input_masks.at(index) != InputType::kConstant) {
      if (!input_values[index]->isa<tensor::Tensor>()) {
        MS_LOG(EXCEPTION) << "The " << index << "' input should be a Tensor, but got "
                          << input_values[index]->ToString();
      }
      auto tensor = input_values.at(index)->cast<tensor::TensorPtr>();
      CheckInputTensorStream(tensor);
      (void)tensors_without_value_node.emplace_back(tensor);
    }
  }
  return tensors_without_value_node;
}

// Determine the address of the graph and do not change the address in subsequent executions
void OpRunner::UpdateDeviceAddress(const KernelGraphPtr &graph,
                                   const std::vector<tensor::TensorPtr> &tensors_without_value_mask,
                                   const device::DeviceContext *device_context, bool is_sync, size_t stream_id) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Start";
  const auto &input_nodes = graph->input_nodes();
  UpdateInputTensorFromDevice(input_nodes, tensors_without_value_mask, device_context, stream_id);
  UpdateInputNodeDeviceAddress(input_nodes, tensors_without_value_mask, device_context);
  pynative::OpCompiler::UpdateRefNodeOutputDeviceAddress(graph);
  MS_LOG(DEBUG) << "End";
}

void OpRunner::RunSingleOpGraph(const session::BackendOpRunInfoPtr &op_run_info,
                                const OpCompilerInfoPtr &op_compiler_info,
                                const std::vector<tensor::TensorPtr> &input_tensors) {
  CopyDataToDevice(op_compiler_info->graph_, input_tensors, op_compiler_info->device_context_);
  LaunchKernels(op_compiler_info->graph_, op_compiler_info->device_context_, op_run_info, input_tensors);
}

void OpRunner::LaunchKernelTask(const runtime::KernelTaskType &task_type, DeviceContext *device_context,
                                const tensor::TensorPtrList &input_tensors, const tensor::TensorPtrList &output_tensors,
                                size_t stream_id) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(DEBUG) << "Start, task_type:" << task_type;
  static auto no_simu = !common::IsCompileSimulation();
  if (no_simu &&
      !device_context->GetKernelExecutor()->ExecuteKernelTask(task_type, input_tensors, output_tensors, stream_id)) {
    MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type:" << task_type;
  }
  MS_LOG(DEBUG) << "End";
}

DeviceContext *OpRunner::GetDeviceContext(device::DeviceType device_type) {
  auto index = static_cast<size_t>(device_type);
  auto cached_device_context = kDeviceContexts[index];

  if (cached_device_context != nullptr) {
    return cached_device_context;
  }

  GilReleaseWithCheck release_gil;
  std::unique_lock<std::mutex> lock(*kDeviceContextMutex);
  // double-checked locking
  cached_device_context = kDeviceContexts[index];
  if (cached_device_context != nullptr) {
    return cached_device_context;
  }

  auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_type, device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->Initialize();

  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  (void)device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  kDeviceContexts[index] = device_context;
  MS_LOG(DEBUG) << "Get device context of " << device_type << " id " << device_id;
  return device_context;
}

void OpRunner::ChildAfterFork() {
  kDeviceContexts.fill(nullptr);
  kDeviceContextMutex = std::make_unique<std::mutex>();
}

void DynamicOpRunner::RunSingleOpGraph(const session::BackendOpRunInfoPtr &op_run_info,
                                       const OpCompilerInfoPtr &op_compiler_info,
                                       const std::vector<tensor::TensorPtr> &input_tensors) {
  DynamicOpRunner::CopyHostToDevice(op_compiler_info, input_tensors);
  MallocForConstValue(op_compiler_info);

  const auto &simple_graph = op_compiler_info->simple_graph_;
  const auto &single_ops = simple_graph->single_ops_;
  bool is_need_infer = false;
  auto op_num = single_ops.size();
  MS_EXCEPTION_IF_NULL(op_run_info->base_op_run_info.abstract);
  if (op_num > 1 || op_run_info->base_op_run_info.abstract->BuildShape()->IsDynamic()) {
    is_need_infer = true;
  }

  SetOutputDeviceAddressFlag(op_compiler_info, op_run_info);

  const auto *device_context = op_compiler_info->device_context_;
  // Execute all kernels
  for (size_t i = 0; i < op_num; ++i) {
    const auto &single_op = single_ops[i];
    const CNodePtr &kernel = single_op->kernel_;
    MS_EXCEPTION_IF_NULL(kernel);

    auto &cache = KernelCache::GetInstance();
    if (cache.need_add) {
      cache.Add(kernel);
    }

    if (EnableExecuteOrderDump()) {
      auto &execute_order_tracker = ExecuteOrderTracker::GetInstance();
      execute_order_tracker.ProcessNode(kernel);
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", kernel->fullname_with_scope(),
                                                   op_run_info->base_op_run_info.op_name);

    // Fetch input kernel tensor.
    const auto &input_edges = single_op->inputs_;
    const auto &output_edges = single_op->outputs_;

    const auto &input_kernel_tensors = GetInputKernelTensors(input_edges);
    const auto &input_abstracts = GetInputInferAbstract(input_edges);
    const auto &output_kernel_tensors = GetOutputKernelTensors(output_edges, device_context);

    BaseShapePtr out_shape;
    static auto no_simu = !common::IsCompileSimulation();
    if (is_need_infer) {
      if (!no_simu) {
        bool is_dyn_shape = common::AnfAlgo::IsDynamicShape(kernel) || common::AnfAlgo::IsDynamicSequence(kernel);
        bool is_dyn_value = common::AnfAlgo::IsDynamicValue(kernel);
        bool is_dyn_type = common::AnfAlgo::IsAnyTypeOutput(kernel);
        if (is_dyn_value && (is_dyn_shape || is_dyn_type)) {
          MS_LOG(EXCEPTION) << "For " << kernel->fullname_with_scope()
                            << ", the output shape depends on the actual execution, and it will affect the accuracy of "
                               "memory in dryrun mode.";
        }
      }
      out_shape = InferNodeRealShape(kernel, input_abstracts);
    } else {
      kernel->set_abstract(op_run_info->base_op_run_info.abstract);
      out_shape = op_run_info->base_op_run_info.abstract->GetShape();
    }
    // Update output kernel tensor.
    opt::dynamic_shape::UpdateKernelTensorShape(out_shape, output_kernel_tensors);

    // Resize
    ResizeKernelMod(kernel, input_kernel_tensors, output_kernel_tensors);

    // Malloc workspace memory
    std::vector<kernel::KernelTensorPtr> workspace_kts;
    auto workspace_kernel_tensors = GetWorkspaceKernelTensorsDynamic(device_context, kernel, &workspace_kts);

    // Update output tensor shape
    UpdateOutputDeviceInfo(output_edges, kernel);

    // Malloc output tensor memory
    AllocateOutputMemory(output_edges, device_context);

    // Launch kernel
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetKernelExecutor());
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    const size_t stream_id = op_run_info->base_op_run_info.stream_id;
    auto stream = device_context->device_res_manager_->GetStream(stream_id);
    TrackerACLMemory(input_kernel_tensors, output_kernel_tensors);

    if (no_simu &&
        !device_context->GetKernelExecutor()->LaunchKernel(kernel, input_kernel_tensors, workspace_kernel_tensors,
                                                           output_kernel_tensors, kernel_mod, stream)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, name:" << kernel->fullname_with_scope();
    }

    if (is_need_infer) {
      if (kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
        if (!no_simu && no_dyn_need_update_ops.find(kernel_mod->kernel_name()) == no_dyn_need_update_ops.end()) {
          MS_LOG(EXCEPTION) << "For " << kernel->fullname_with_scope()
                            << ", the output shape depends on the actual execution, and it will affect the accuracy of "
                               "memory in dryrun mode.";
        }
        kernel_mod->UpdateOutputShapeAndSize(input_kernel_tensors, output_kernel_tensors);
      }
    }
    runtime::DeviceAddressUtils::ProcessCrossStreamAddress(op_run_info->base_op_run_info.op_name, device_context,
                                                           stream_id, input_kernel_tensors, output_kernel_tensors);
  }
}

void DynamicOpRunner::UpdateInputDeviceAddress(const OpCompilerInfoPtr &op_compiler_info,
                                               const std::vector<tensor::TensorPtr> &input_tensors, bool is_sync,
                                               size_t stream_id) {
  MS_LOG(DEBUG) << "Start update input device address for " << op_compiler_info->graph_info_;
  const auto &simple_graph = op_compiler_info->simple_graph_;
  auto input_tensors_num = input_tensors.size();
  auto op_input_num = simple_graph->inputs_.size();
  if (input_tensors_num != op_input_num) {
    MS_LOG(EXCEPTION) << "Real input tensor's num " << input_tensors_num << " is not equal to op input num"
                      << op_input_num << " !";
  }
  const auto &inputs = simple_graph->inputs_;
  for (size_t i = 0; i < input_tensors_num; ++i) {
    const auto &input_tensor = input_tensors[i];
    MS_EXCEPTION_IF_NULL(input_tensor);
    const auto &input_edge = inputs[i];

    auto device_address = input_tensor->device_address();
    if (input_tensor->storage_info() != nullptr) {
      MS_EXCEPTION(RuntimeError) << "Not support view tensor for " << op_compiler_info->graph_info_;
    }

    const auto &input_node = input_edge->node_with_index_.first;
    common::AnfAlgo::SetOutputInferTypeAndShape({input_tensor->data_type()}, {input_tensor->shape()}, input_node.get());

    UpdateAddressInfoByInputTensor(op_compiler_info, input_tensor, input_edge, input_node);

    auto new_device_address = input_edge->kernel_tensor_->device_address();
    if (device_address->GetDeviceType() != new_device_address->GetDeviceType()) {
      runtime::Pipeline::Get().WaitForward();
      MS_LOG(DEBUG) << "Input tensor device address type:"
                    << device::GetDeviceNameByType(device_address->GetDeviceType()) << " but ir device address type:"
                    << device::GetDeviceNameByType(new_device_address->GetDeviceType());
      input_tensor->set_implicit_copy_address(device_address);
      input_tensor->set_implicit_copy_format(input_tensor->format());
      input_tensor->set_implicit_copy_storage_info(input_tensor->storage_info());
      input_tensor->set_device_address(new_device_address);
      input_tensor->set_format(input_edge->kernel_tensor_->format());
      input_tensor->set_storage_info(input_edge->kernel_tensor_->tensor_storage_info());
    } else {
      MS_LOG(DEBUG) << "Input device type is same, set tensor device address to ir input.";
      input_edge->kernel_tensor_->set_device_address(device_address);
      input_edge->kernel_tensor_->set_format(input_tensor->format());
      input_edge->kernel_tensor_->set_tensor_storage_info(input_tensor->storage_info());
    }

    if (device_address->GetDeviceType() == device::DeviceType::kCPU && input_edge->ignore_h2d_) {
      input_edge->kernel_tensor_->SetValue(input_tensor);
    }
  }
  MS_LOG(DEBUG) << "End update input device address for " << op_compiler_info->graph_info_;
}

void DynamicOpRunner::CopyHostToDevice(const OpCompilerInfoPtr &op_compiler_info,
                                       const std::vector<tensor::TensorPtr> &input_tensors) {
  const auto &input_edges = op_compiler_info->simple_graph_->inputs_;
  auto input_tensors_num = input_tensors.size();
  auto input_edge_num = input_edges.size();
  if (input_tensors_num != input_edge_num) {
    MS_LOG(EXCEPTION) << "Real input tensor's number " << input_tensors_num << " is not equal to input edges number "
                      << input_edge_num << " !";
  }

  const auto &device_context = op_compiler_info->device_context_;
  for (size_t i = 0; i < input_tensors_num; ++i) {
    const auto &input_tensor = input_tensors[i];
    MS_EXCEPTION_IF_NULL(input_tensor);
    auto device_address = input_tensor->device_address();

    const auto &input_edge = input_edges[i];
    if (input_edge->ignore_h2d_) {
      continue;
    }

    const auto &input_node = input_edge->node_with_index_.first;
    MS_EXCEPTION_IF_NULL(input_node);
    common::AnfAlgo::SetOutputInferTypeAndShape({input_tensor->data_type()}, {input_tensor->shape()}, input_node.get());

    if (device_address == nullptr) {
      MS_LOG(EXCEPTION) << "Input DeviceAddress cannot be null before copy host to device, op name "
                        << op_compiler_info->graph_info_;
    }

    if (device_address->GetMutablePtr() != nullptr) {
      continue;
    }

    auto mem_type =
      input_tensor->is_parameter() ? memory::mem_pool::MemType::kWeight : memory::mem_pool::MemType::kPyNativeInput;
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", mem_type, device_address->GetSize(),
                                                   device_address.get());
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get(), CurrentStream::id())) {
      MS_LOG(EXCEPTION) << "Device(id:" << device_context->device_context_key().device_id_
                        << ") memory isn't enough and alloc failed, kernel name: " << input_node->DebugString()
                        << ", alloc size: " << device_address->GetSize() << "B.";
    }

    MS_LOG(DEBUG) << "Start lazy copy for input " << i << " tensor " << input_tensor->ToString();
    runtime::DeviceAddressUtils::LazyCopy(input_tensor, CurrentStream::id());
    MS_LOG(DEBUG) << "Copy host tensor to device for op " << op_compiler_info->graph_info_ << " input " << i;
  }
}
}  // namespace mindspore::runtime
