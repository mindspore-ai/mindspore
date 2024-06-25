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

#include "runtime/pynative/graph_adapter.h"

#include <string>
#include <memory>
#include <vector>
#include "ir/tensor.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/pyboost_utils.h"

namespace mindspore::pynative {
namespace {
constexpr auto kAttrBpropValueNodeRefCount = "bprop_value_node_ref_count";
constexpr auto kAttrValueNodeForwardOuputFlags = "value_node_forward_output_flags";

tensor::BaseTensorPtr GetTensorFromValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    return nullptr;
  }
  auto value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  // ValueTuple is already expanded into tensors in backend.
  if (!value->isa<tensor::BaseTensor>()) {
    MS_LOG(DEBUG) << "Only need to process forward output tensor. value:" << value->ToString();
    return nullptr;
  }

  auto tensor = value->cast<tensor::BaseTensorPtr>();
  return tensor;
}

HashMap<ValueNodePtr, size_t> GetGraphValueNodeRefCounts(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  HashMap<ValueNodePtr, size_t> value_node_ref_counts;
  // For example:
  //   %1 MakeTuple(V1, V2)
  //   %2 TupleGetItem(0, %1)
  //   %3 Kernel(%2)
  // V2 is not used by kernel. Need to remove.
  auto execution_nodes = graph->execution_order();
  for (auto &node : execution_nodes) {
    std::vector<session::KernelWithIndex> real_inputs;
    common::AnfAlgo::GetRealInputs(node, &real_inputs);
    for (auto &real_input : real_inputs) {
      auto input = real_input.first;
      MS_EXCEPTION_IF_NULL(input);
      if (input->isa<ValueNode>()) {
        auto value_node = input->cast<ValueNodePtr>();
        value_node_ref_counts[value_node] += 1;
      }
    }
  }

  // ValueNodes as graph outputs
  auto outputs = common::AnfAlgo::GetAllOutput(graph->output());
  for (auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output);
    if (output->isa<ValueNode>()) {
      auto value_node = output->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      value_node_ref_counts[value_node] += 1;
    }
  }

  return value_node_ref_counts;
}

device::DeviceAddressPtr CreateValueNodeAddress(const ValueNodePtr &value_node,
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
    {value_node, 0}, nullptr, tensor_size, output_format, data_type, trans::GetRuntimePaddingShape(value_node, 0),
    device_context->device_context_key().device_name_, device_context->device_context_key().device_id_);
  return device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
}

bool CopyTensorData(const tensor::BaseTensorPtr &tensor, const device::DeviceAddressPtr &device_address,
                    const AnfNodePtr &node, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  device::DynamicMemAllocatorDebugInfo::SetDebugInfo(node->fullname_with_scope(), device::AllocatorType::kConstantValue,
                                                     0);
  if (device_address->GetPtr() == nullptr) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "CopyTensorData", "CopyTensorData", "");
    auto mem_type =
      tensor->is_parameter() ? device::tracker::MemType::kWeight : device::tracker::MemType::kPyNativeInput;
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "CopyTensorData", mem_type, device_address->GetSize(),
                                                   device_address.get());
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(ERROR) << "Allocate memory failed, allocate size " << device_address->GetSize();
      return false;
    }
  }

  // Copy data from host tensor to device.
  auto host_tensor_size = LongToSize(tensor->data().nbytes());
  auto host_tensor_type = tensor->data_type();
  if (!device_address->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0), host_tensor_size, host_tensor_type,
                                        kOpFormat_DEFAULT, tensor->data_ptr())) {
    std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope() +
                             ", tensor size: " + std::to_string(host_tensor_size) +
                             ", tensor type: " + std::to_string(static_cast<int>(host_tensor_type)) +
                             ", device address size: " + std::to_string(device_address->GetSize());
    MS_LOG(ERROR) << error_info;
    return false;
  }
  return true;
}

device::DeviceAddressPtr HandleAddressForHeterogeneous(const tensor::BaseTensorPtr &tensor,
                                                       const ValueNodePtr &value_node,
                                                       const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(device_context);
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  if (device_address == nullptr) {
    MS_LOG(INFO) << "Forward output " << tensor->ToString() << " device address is null";
    device_address = CreateValueNodeAddress(value_node, device_context);
    if (!CopyTensorData(tensor, device_address, value_node, device_context)) {
      MS_LOG(EXCEPTION) << "CopyTensorData failed, value_node " << value_node->DebugString();
    }
  }
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->GetDeviceType() != device_context->GetDeviceType()) {
    tensor->data_sync();
    auto new_device_address = CreateValueNodeAddress(value_node, device_context);
    MS_EXCEPTION_IF_NULL(new_device_address);
    if (!CopyTensorData(tensor, new_device_address, value_node, device_context)) {
      MS_LOG(EXCEPTION) << "CopyTensorData failed, value_node " << value_node->DebugString();
    }
    return new_device_address;
  }
  return device_address;
}
}  // namespace

void GraphAdapter::RemoveUnusedValueNodes(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto value_node_ref_counts = GetGraphValueNodeRefCounts(graph);
  for (const auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto iter = value_node_ref_counts.find(value_node);
    if (iter == value_node_ref_counts.end()) {
      MS_LOG(DEBUG) << "Remove unused ValueNode " << value_node->DebugString();
      graph->RemoveNodeFromGraph(value_node);
    }
  }
}

void GraphAdapter::ClearForwardOutputValueNodeDeviceAddress(const KernelGraphPtr &graph,
                                                            const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<tensor::BaseTensor>()) {
      auto tensor = value->cast<tensor::BaseTensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (!tensor->is_forward_output()) {
        continue;
      }

      if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
        MS_LOG(DEBUG) << "Output addr is not exist for ValueNode " << value_node->ToString();
        continue;
      }
      const auto &device_address = AnfAlgo::GetMutableOutputAddr(value_node, 0);
      auto new_device_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(device_address, device_context);
      AnfAlgo::SetOutputAddr(new_device_address, 0, value_node.get());
    }
  }
}

// The device address of graph value node need to release
// if the value node is output of forward_graph in PyNative mode.
void GraphAdapter::GenerateRefCountForBpropValueNode(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  HashMap<std::string, size_t> tensor_counts;
  HashMap<ValueNodePtr, size_t> value_node_ref_counts = GetGraphValueNodeRefCounts(graph);

  std::vector<size_t> value_node_ref_count_list;
  std::vector<bool> value_node_forward_output_flags;
  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    auto tensor = GetTensorFromValueNode(value_node);
    if (tensor == nullptr || !tensor->is_forward_output()) {
      (void)value_node_ref_count_list.emplace_back(SIZE_MAX);
      (void)value_node_forward_output_flags.emplace_back(false);
      continue;
    }

    auto iter = value_node_ref_counts.find(value_node);
    if (iter == value_node_ref_counts.end()) {
      // The value_node is in bp graph but not used.
      // e.g. %1-MakeTuple(T1, T2) -> TupleGetItem(%1, 0). T2 is not used.
      MS_LOG(DEBUG) << "ValueNode " << value_node->ToString() << " is not used in graph";
      (void)value_node_ref_count_list.emplace_back(SIZE_MAX);
      (void)value_node_forward_output_flags.emplace_back(false);
      continue;
    }

    (void)value_node_ref_count_list.emplace_back(iter->second);
    (void)value_node_forward_output_flags.emplace_back(true);
    MS_LOG(DEBUG) << "ValueNode " << value_node->DebugString() << " ref_count " << iter->second;
  }
  graph->set_attr(kAttrBpropValueNodeRefCount, MakeValue(value_node_ref_count_list));
  graph->set_attr(kAttrValueNodeForwardOuputFlags, MakeValue(value_node_forward_output_flags));
}

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
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(value_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_tensor);

    auto kernel_tensor = std::make_shared<kernel::KernelTensor>(
      nullptr, device_tensor->GetSize(), device_tensor->kernel_tensor()->format(), device_tensor->type_id(),
      device_tensor->host_shape(), device_context->device_context_key().device_name_,
      device_context->device_context_key().device_id_);

    kernel_tensor->SetHostInfo(
      std::make_shared<abstract::TensorShape>(device_tensor->kernel_tensor()->GetShapeVector()),
      std::make_shared<TensorType>(TypeIdToType(device_tensor->kernel_tensor()->dtype_id())), nullptr);

    kernel_tensor->set_stream_id(device_tensor->stream_id());
    auto new_device_tensor = real_device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
    MS_EXCEPTION_IF_NULL(new_device_tensor);
    new_device_tensor->SetNodeIndex(value_node, 0);
    new_device_tensor->set_from_persistent_mem(true);
    MS_LOG(DEBUG) << "Create backoff device tensor:" << new_device_tensor << " type:" << new_device_tensor->type_id()
                  << " for ValueNode " << value_node->ToString();
    runtime::SchedulerHelper::AddDeviceTensorStore(front_node.get(), new_device_tensor);
  }
}

void GraphAdapter::UpdateForwardOutputInBpropGraph(const KernelGraphPtr &graph,
                                                   const device::DeviceContext *device_context, bool no_control_flow) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Update start";
  auto value_node_ref_counts = GetValue<std::vector<size_t>>(graph->get_attr(kAttrBpropValueNodeRefCount));
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
    size_t value_node_ref_count = value_node_ref_counts[value_node_index++];
    auto tensor = GetTensorFromValueNode(value_node);
    MS_EXCEPTION_IF_NULL(tensor);

    auto device_address = HandleAddressForHeterogeneous(tensor, value_node, device_context);
    device_address = std::dynamic_pointer_cast<device::DeviceAddress>(
      kernel::pyboost::PyBoostUtils::ContiguousByDeviceAddress(device_address));
    runtime::DeviceAddressUtils::CreateKernelTensor(device_address, tensor);
    tensor->set_device_address(device_address);
    auto front_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
    MS_EXCEPTION_IF_NULL(front_node);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetDeviceType() != device::DeviceType::kCPU && no_control_flow) {
      address_ref_count[device_address] += value_node_ref_count;
      device_address->AddHeldByNode(front_node->cast<ValueNodePtr>());
    }
    runtime::DeviceTensorStore::GetInstance().Insert(front_node.get(), device_address);
    HandleBackoffValueNode(value_node, front_node, device_context);
  }

  for (auto &[address, ref_count] : address_ref_count) {
    MS_EXCEPTION_IF_NULL(address);
    address->set_original_ref_count(ref_count);
    address->ResetRefCount();
    MS_LOG(DEBUG) << "device_address " << address.get() << " ref_count " << address->ref_count();
  }
  MS_LOG(DEBUG) << "Update end";
}

void GraphAdapter::HandleHeterogeneousTensors(const std::vector<std::vector<tensor::TensorPtr>> &input_tensors,
                                              const std::vector<device::DeviceContext *> &device_contexts) {
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
        auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
        MS_EXCEPTION_IF_NULL(device_address);
        if (device_address->GetDeviceType() != device_context->GetDeviceType()) {
          tensor->data_sync();
          tensor->set_device_address(nullptr);
        }
      }
    }
  }
}

void GraphAdapter::ReplaceGraphParameterProperties(const KernelGraphPtr &graph,
                                                   const std::vector<tensor::TensorPtr> &input_tensors,
                                                   const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(graph);
  size_t index = 0;
  for (const auto &input_node : graph->input_nodes()) {
    auto parameters = common::AnfAlgo::GetAllOutput(input_node);
    for (const auto &parameter : parameters) {
      MS_EXCEPTION_IF_NULL(parameter);
      if (index >= input_tensors.size()) {
        MS_LOG(EXCEPTION) << "Parameter size out of range. Parameter index: " << index
                          << ", input size: " << input_tensors.size();
      }
      const auto &input_tensor = input_tensors[index++];
      MS_EXCEPTION_IF_NULL(input_tensor);
      const auto &tensor_address = input_tensor->device_address();
      auto address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor_address);
      if (address == nullptr || address->GetDeviceType() != device_context->GetDeviceType()) {
        // Need to discard input tensor properties in heterogeneous scenarios.
        // For example, the format of device_address in input_tensor is 5D format,
        // and it's invalid for CPU graph parameter.
        continue;
      }

      auto kernel_build_info_builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
      MS_EXCEPTION_IF_NULL(kernel_build_info_builder);
      kernel_build_info_builder->SetOutputsFormat(std::vector<std::string>{address->format()});
      kernel_build_info_builder->SetOutputsDeviceType(std::vector<TypeId>{address->type_id()});
      kernel_build_info_builder->SetOutputsReshapeType({address->padding_type()});
      AnfAlgo::SetOutputAddr(address, 0, parameter.get());
      AnfAlgo::SetSelectKernelBuildInfo(kernel_build_info_builder->Build(), parameter.get());

      auto abstract = parameter->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      auto shape = abstract->BuildShape();
      auto new_abs = std::make_shared<abstract::AbstractTensor>(TypeIdToType(address->type_id()), shape);
      parameter->set_abstract(new_abs);
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
  if (func_graph->has_flag(kFlagEnableRunGraphBySingleOp)) {
    return false;
  }

  return true;
}

bool GraphAdapter::PyNativeEnableTaskSink(const FuncGraphPtr &func_graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool pynative_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
  if (!pynative_mode) {
    return true;
  }

  MS_EXCEPTION_IF_NULL(func_graph);
  if (GraphAdapter::IsPynativeGeGraphSink(func_graph)) {
    MS_LOG(DEBUG) << "Enable graph sink for PyNative";
    return true;
  }

  if (!func_graph->has_attr(kAttrJitLevel)) {
    MS_LOG(EXCEPTION) << "Not jit_level set to func_graph";
  }
  auto jit_level_value = func_graph->get_attr(kAttrJitLevel);
  auto jit_level = GetValue<std::string>(jit_level_value);
  if (jit_level != kAttrJitLevelO2) {
    MS_LOG(INFO) << "jit_level is " << jit_level << ", task sink is disabled";
    return false;
  }

  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  auto is_cut_graph = std::any_of(node_list.begin(), node_list.end(), [](const AnfNodePtr &node) {
    return common::AnfAlgo::IsBpropCutOpExecInBackend(node);
  });

  auto has_comm_op = std::any_of(node_list.begin(), node_list.end(),
                                 [](const AnfNodePtr &node) { return common::AnfAlgo::IsCommunicationOp(node); });

  auto is_auto_parallel = IsAutoParallel();

  MS_LOG(INFO) << "JitLevel is " << jit_level << " is_auto_parallel " << is_auto_parallel << " has_comm_op "
               << has_comm_op << " is_cut_graph " << is_cut_graph;

  return !is_auto_parallel && !has_comm_op && !is_cut_graph;
}

void UpdateValueNodeAbstractFromTensor(const ValueNodePtr &value_node, const tensor::BaseTensorPtr &tensor) {
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

void GraphAdapter::UpdateDynamicValueNodeAbstract(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->is_dynamic_shape()) {
    return;
  }
  MS_LOG(INFO) << "Update dynamic shape value node for graph " << graph->graph_id();
  const auto &value_nodes = graph->graph_value_nodes();
  for (auto &value_node : value_nodes) {
    MS_EXCEPTION_IF_NULL(value_node);
    const auto &value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<tensor::BaseTensor>()) {
      auto tensor = value->cast<tensor::BaseTensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor->is_forward_output()) {
        UpdateValueNodeAbstractFromTensor(value_node, tensor);
      }
    }
  }
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
    std::vector<tensor::BaseTensorPtr> tensors;
    TensorValueToTensor(value, &tensors);
    for (const auto &tensor : tensors) {
      MS_EXCEPTION_IF_NULL(tensor);
      if (!tensor->has_user_data(kTensorUserDataIsSensTensor)) {
        continue;
      }
      const auto &device_address = tensor->device_address();
      if (device_address == nullptr) {
        UpdateValueNodeAbstractFromTensor(value_node, tensor);
        auto node_address = CreateValueNodeAddress(value_node, device_context);
        MS_EXCEPTION_IF_NULL(node_address);
        tensor->set_device_address(node_address);
        AnfAlgo::SetOutputAddr(node_address, 0, value_node.get());
        MS_LOG(DEBUG) << "Start to copy sens tensor to device";
        if (!CopyTensorData(tensor, node_address, value_node, device_context)) {
          MS_LOG(EXCEPTION) << "ValueNode host to device copy failed";
        }
      }
    }
  }
}
}  // namespace mindspore::pynative
