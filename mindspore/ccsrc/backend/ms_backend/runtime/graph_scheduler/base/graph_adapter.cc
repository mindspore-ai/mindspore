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

#include "backend/ms_backend/runtime/graph_scheduler/base/graph_adapter.h"

#include <string>
#include <memory>
#include <vector>
#include "ir/tensor.h"
#include "ir/dtype/tensor_type.h"
#include "utils/convert_utils.h"
#include "include/utils/anfalgo.h"
#include "include/utils/parallel_context.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/runtime/hardware_abstract/kernel_base/device_tensor_store.h"
#include "backend/ms_backend/runtime/actors/base/actor_common.h"
#include "backend/ms_backend/runtime/graph_scheduler/base/scheduler_helper.h"
#include "include/backend/common/device_address_utils.h"

namespace mindspore::pynative {
namespace {
constexpr auto kAttrBpropValueNodeRefCount = "bprop_value_node_ref_count";
constexpr auto kAttrValueNodeForwardOuputFlags = "value_node_forward_output_flags";

tensor::TensorPtr GetTensorFromValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    return nullptr;
  }
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  // ValueTuple is already expanded into tensors in backend.
  if (!value->isa<tensor::Tensor>()) {
    MS_LOG(DEBUG) << "Only need to process forward output tensor. value:" << value->ToString();
    return nullptr;
  }

  auto tensor = value->cast<tensor::TensorPtr>();
  return tensor;
}

kernel::KernelTensorPtr CreateValueNodeKernelTensor(const ValueNodePtr &value_node,
                                                    const device::DeviceContext *device_context) {
  size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(value_node, 0);
  TypeId data_type = AnfAlgo::GetOutputDeviceDataType(value_node, 0);
  if (data_type == kTypeUnknown) {
    data_type = common::AnfAlgo::GetOutputInferDataType(value_node, 0);
  }
  auto output_format = AnfAlgo::GetOutputFormat(value_node, 0);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {value_node, 0}, nullptr, tensor_size, output_format, data_type, AnfAlgo::GetRuntimePaddingShape(value_node, 0),
    device::GetDeviceNameByType(device_context->device_context_key().device_type_),
    device_context->device_context_key().device_id_);
  AnfAlgo::SetOutputKernelTensor(kernel_tensor, 0, value_node.get());
  return kernel_tensor;
}

bool CopyTensorData(const tensor::TensorPtr &tensor, const kernel::KernelTensorPtr &kernel_tensor,
                    const AnfNodePtr &node, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_address = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  if (device_address->GetPtr() == nullptr) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "CopyTensorData", "CopyTensorData", "", false);
    auto mem_type =
      tensor->is_parameter() ? memory::mem_pool::MemType::kWeight : memory::mem_pool::MemType::kPyNativeInput;
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "CopyTensorData", mem_type, device_address->GetSize(),
                                                   device_address.get());
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(ERROR) << "Allocate memory failed, allocate size " << device_address->GetSize();
      return false;
    }
  }

  // Copy data from host tensor to device.
  auto host_tensor_size = tensor->DataNBytes();
  auto host_tensor_type = tensor->data_type();
  if (!AsyncCopy(kernel_tensor.get(), tensor.get(), device_address->stream_id())) {
    std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope() +
                             ", tensor size: " + std::to_string(host_tensor_size) +
                             ", tensor type: " + std::to_string(static_cast<int>(host_tensor_type)) +
                             ", device address size: " + std::to_string(device_address->GetSize());
    MS_LOG(ERROR) << error_info;
    return false;
  }
  return true;
}

tensor::TensorPtr HandleAddressForHeterogeneous(const tensor::TensorPtr &tensor, const ValueNodePtr &value_node,
                                                const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(device_context);
  auto device_address = tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->GetDeviceType() != device_context->GetDeviceType()) {
    auto new_kernel_tensor = CreateValueNodeKernelTensor(value_node, device_context);
    MS_EXCEPTION_IF_NULL(new_kernel_tensor);
    if (!CopyTensorData(tensor, new_kernel_tensor, value_node, device_context)) {
      MS_LOG(EXCEPTION) << "CopyTensorData failed, value_node " << value_node->DebugString();
    }

    return std::make_shared<tensor::Tensor>(new_kernel_tensor->dtype_id(), new_kernel_tensor->GetShapeVector(),
                                            new_kernel_tensor->device_address());
  }
  return std::make_shared<tensor::Tensor>(tensor->data_type(), tensor->shape(), device_address);
}
}  // namespace

void GraphAdapter::GenerateBackoffValueNodeOwners(const KernelGraphPtr &graph) {
  for (auto &kernel : graph->execution_order()) {
    if (!AnfAlgo::IsKernelSelectBackoffOp(kernel)) {
      continue;
    }
    for (size_t j = 0; j < common::AnfAlgo::GetInputTensorNum(kernel); ++j) {
      const auto &input_node = common::AnfAlgo::GetInputNode(kernel, j);
      const auto &real_input_node = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, false).first;
      MS_EXCEPTION_IF_NULL(real_input_node);
      if (real_input_node->isa<ValueNode>()) {
        (void)node_to_backoff_kernels_[real_input_node.get()].insert(kernel);
        MS_LOG(DEBUG) << "Generate backoff ValueNode " << real_input_node->DebugString() << " with kernel "
                      << kernel->DebugString();
      }
    }
  }
}

void GraphAdapter::HandleBackoffValueNode(const ValueNodePtr &value_node, const AnfNodePtr &front_node,
                                          const DeviceContext *device_context) const {
  auto iter = node_to_backoff_kernels_.find(value_node.get());
  if (iter == node_to_backoff_kernels_.end()) {
    return;
  }

  MS_LOG(DEBUG) << "Backoff ValueNode " << value_node->ToString();
  const auto &kernels = iter->second;
  for (const auto &kernel : kernels) {
    const auto &real_device_context = device::FetchRealDeviceContext(kernel, device_context);
    MS_EXCEPTION_IF_NULL(real_device_context);

    if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
      MS_LOG(EXCEPTION) << "The device address is not exist: " << value_node->ToString();
    }
    auto old_kernel_tensor = AnfAlgo::GetOutputKernelTensor(value_node, 0, false);
    MS_EXCEPTION_IF_NULL(old_kernel_tensor);
    auto device_tensor = old_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_tensor);

    auto kernel_tensor =
      AnfAlgo::CreateKernelTensor(nullptr, device_tensor->GetSize(), old_kernel_tensor->format(),
                                  old_kernel_tensor->dtype_id(), old_kernel_tensor->GetShapeVector(),
                                  device::GetDeviceNameByType(real_device_context->device_context_key().device_type_),
                                  real_device_context->device_context_key().device_id_);

    kernel_tensor->SetHostInfo(std::make_shared<abstract::TensorShape>(old_kernel_tensor->GetShapeVector()),
                               std::make_shared<TensorType>(TypeIdToType(old_kernel_tensor->dtype_id())), nullptr);

    kernel_tensor->set_stream_id(device_tensor->stream_id());
    const auto &new_device_tensor = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(new_device_tensor);
    new_device_tensor->SetNodeIndex(value_node, 0);
    new_device_tensor->set_from_persistent_mem(true);
    MS_LOG(DEBUG) << "Create backoff kernel tensor:" << kernel_tensor->ToString() << " for ValueNode "
                  << value_node->ToString();
    runtime::SchedulerHelper::AddDeviceTensorStore(front_node, kernel_tensor);
  }
}

void GraphAdapter::UpdateForwardOutputInBpropGraph(const KernelGraphPtr &graph,
                                                   const device::DeviceContext *device_context, bool no_control_flow) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Update start";
  auto value_node_ref_counts = GetValue<std::vector<uint32_t>>(graph->get_attr(kAttrBpropValueNodeRefCount));
  auto value_node_forward_output_flags = GetValue<std::vector<bool>>(graph->get_attr(kAttrValueNodeForwardOuputFlags));
  size_t value_node_size = graph->graph_value_nodes().size();
  if (value_node_ref_counts.size() != value_node_size || value_node_forward_output_flags.size() != value_node_size) {
    MS_LOG(EXCEPTION) << "value_node_ref_count.size " << value_node_ref_counts.size()
                      << " value_node_forward_output_flags.size " << value_node_forward_output_flags.size()
                      << " not equal to " << value_node_size;
  }

  size_t value_node_index = 0;
  HashMap<device::DeviceAddressPtr, size_t> address_ref_count;
  // Update ValueNode device address
  for (auto &value_node : graph->graph_value_nodes()) {
    auto is_forward_output = value_node_forward_output_flags[value_node_index];
    if (!is_forward_output) {
      value_node_index++;
      continue;
    }
    uint32_t value_node_ref_count = value_node_ref_counts[value_node_index++];
    auto tensor = GetTensorFromValueNode(value_node);
    MS_EXCEPTION_IF_NULL(tensor);

    auto input_tensor = HandleAddressForHeterogeneous(tensor, value_node, device_context);
    auto device_address = runtime::DeviceAddressUtils::ConvertContiguousDeviceAddress(nullptr, input_tensor);

    auto abs = tensor->ToAbstract()->Broaden();
    MS_EXCEPTION_IF_NULL(abs);
    auto shape = abs->GetShape();
    auto type = abs->GetType();
    auto value = abs->GetValue();
    const auto &kernel_tensor = std::make_shared<kernel::KernelTensor>(shape, type, value);
    kernel_tensor->set_device_address(device_address);
    kernel_tensor->set_tensor_storage_info(nullptr);
    kernel_tensor->SetShapeVector(tensor->shape());
    tensor->set_device_address(device_address);
    tensor->set_storage_info(nullptr);
    auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
    MS_EXCEPTION_IF_NULL(front_node);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetDeviceType() != device::DeviceType::kCPU && no_control_flow) {
      address_ref_count[device_address] += value_node_ref_count;
      device_address->AddHeldByNode(front_node->cast<ValueNodePtr>());
    }
    runtime::DeviceTensorStore::GetInstance().Insert(front_node.get(), kernel_tensor);
    HandleBackoffValueNode(value_node, front_node, device_context);
  }
  MS_LOG(DEBUG) << "Update end";
}

void GraphAdapter::HandleHeterogeneousTensors(const std::vector<std::vector<tensor::TensorPtr>> &input_tensors,
                                              const std::vector<device::DeviceContext *> &device_contexts,
                                              ActorSet *actor_set) {
  if (input_tensors.size() < device_contexts.size()) {
    MS_LOG(EXCEPTION) << "Invalid input_tensors size " << input_tensors.size() << " device_contexts size "
                      << device_contexts.size();
  }
  for (size_t i = 0; i < device_contexts.size(); ++i) {
    auto tensors = input_tensors[i];
    auto device_context = device_contexts[i];
    MS_EXCEPTION_IF_NULL(device_context);
    for (auto &tensor : tensors) {
      if (tensor != nullptr && tensor->device_address() != nullptr) {
        auto device_address = tensor->device_address();
        MS_EXCEPTION_IF_NULL(device_address);
        if (device_address->GetDeviceType() != device_context->GetDeviceType()) {
          actor_set->data_prepare_actor_->set_heter_weights(true);
        }
      }
    }
  }
}

bool GraphAdapter::IsAutoParallel() {
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  return parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel;
}

bool GraphAdapter::IsPynativeGeGraphSink(const GraphCompilerInfo &graph_compiler_info) {
  bool is_sink = std::any_of(graph_compiler_info.graphs_.begin(), graph_compiler_info.graphs_.end(),
                             [](const KernelGraphPtr &graph) { return GraphAdapter::IsPynativeGeGraphSink(graph); });
  return is_sink;
}

bool GraphAdapter::IsPynativeGeGraphSink(const FuncGraphPtr &func_graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->backend_policy() != "ge" || !context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(func_graph);
  return true;
}

void UpdateValueNodeAbstractFromTensor(const ValueNodePtr &value_node, const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(tensor);
  auto real_shape = tensor->shape();
  auto old_abs = value_node->abstract();
  auto old_abs_tensor = dyn_cast<abstract::AbstractTensor>(old_abs);
  MS_EXCEPTION_IF_NULL(old_abs_tensor);
  auto new_abs = std::make_shared<abstract::AbstractTensor>(old_abs_tensor->element(),
                                                            std::make_shared<abstract::Shape>(real_shape));
  value_node->set_abstract(new_abs);
  MS_LOG(DEBUG) << "Change bprop ValueNode abstract from " << old_abs->ToString() << " to " << new_abs->ToString();
}

void GraphAdapter::SensTensorToDevice(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->is_dynamic_shape()) {
    return;
  }
  const auto &value_nodes = graph->graph_value_nodes();
  for (const auto &value_node : value_nodes) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    std::vector<tensor::TensorPtr> tensors;
    TensorValueToTensor(value, &tensors);
    for (const auto &tensor : tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      if (!tensor->has_user_data(kTensorUserDataIsSensTensor)) {
        continue;
      }
      const auto &device_address = tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_address);
      if (device_address->GetDeviceType() != device_context->GetDeviceType()) {
        UpdateValueNodeAbstractFromTensor(value_node, tensor);
        auto node_kernel_tensor = CreateValueNodeKernelTensor(value_node, device_context);
        MS_EXCEPTION_IF_NULL(node_kernel_tensor);
        auto node_address = node_kernel_tensor->device_address();
        MS_EXCEPTION_IF_NULL(node_address);
        AnfAlgo::SetOutputAddr(node_address, node_kernel_tensor->tensor_storage_info(), 0, value_node);
        MS_LOG(DEBUG) << "Start to copy sens tensor to device";
        if (!CopyTensorData(tensor, node_kernel_tensor, value_node, device_context)) {
          MS_LOG(EXCEPTION) << "ValueNode host to device copy failed";
        }
        tensor->set_device_address(node_address);
        tensor->set_storage_info(node_kernel_tensor->tensor_storage_info());
      }
    }
  }
}
}  // namespace mindspore::pynative
