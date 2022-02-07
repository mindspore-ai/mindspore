/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <algorithm>

#include "runtime/framework/actor/data_prepare_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/actor/kernel_actor.h"
#include "runtime/framework/actor/loop_count_actor.h"
#include "runtime/framework/actor/debug_actor.h"
#include "runtime/hardware/device_context_manager.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"

namespace mindspore {
namespace runtime {
namespace {
void SyncTensorData(const TensorPtr &host_tensor, const DeviceTensorPtr &device_tensor, const AnfNodePtr &node,
                    const DeviceContext *device_context, OpContext<DeviceTensor> *const context,
                    GraphExecutionStrategy strategy) {
  MS_EXCEPTION_IF_NULL(host_tensor);
  MS_EXCEPTION_IF_NULL(device_tensor);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  if ((device_tensor->GetPtr() == nullptr) &&
      (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize()))) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy, *context, *device_context, node->fullname_with_scope(),
                                                device_tensor->GetSize());
  }

  // Copy data from host tensor to device.
  auto host_tensor_size = LongToSize(host_tensor->data().nbytes());
  auto host_tensor_type = host_tensor->data_type();
  if (!device_tensor->SyncHostToDevice(trans::GetRuntimePaddingShape(node, 0), host_tensor_size, host_tensor_type,
                                       host_tensor->data_c(), host_tensor->device_info().host_format_)) {
    std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope() +
                             ", host tensor size: " + std::to_string(host_tensor_size) +
                             ", host tensor type: " + std::to_string(static_cast<int>(host_tensor_type)) +
                             ", device tensor size: " + std::to_string(device_tensor->GetSize());
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy, (*context), error_info);
  }
}

void FetchContinuousMemoryInfo(const CNodePtr &node, std::vector<DeviceTensorPtr> *const addr_list,
                               std::vector<size_t> *const size_list, size_t *const total_size, bool is_input) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(addr_list);
  MS_EXCEPTION_IF_NULL(size_list);
  MS_EXCEPTION_IF_NULL(total_size);

  const auto &kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  (*addr_list).clear();
  (*size_list).clear();
  *total_size = 0;

  if (is_input) {
    const auto &intput_sizes = kernel_mod->GetInputSizeList();
    for (size_t i = 0; i < intput_sizes.size(); ++i) {
      const auto &device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(node, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      *total_size += intput_sizes[i];
      (void)size_list->emplace_back(intput_sizes[i]);
      (void)addr_list->emplace_back(device_tensor);
    }
  } else {
    const auto &output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, i, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      *total_size += output_sizes[i];
      (void)size_list->emplace_back(output_sizes[i]);
      (void)addr_list->emplace_back(device_tensor);
    }
  }
}

void ValueTupleToValue(const ValuePtr &value, std::vector<ValuePtr> *const values) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(values);
  if (value->isa<ValueTuple>()) {
    auto value_tuple = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(value_tuple);
    for (size_t i = 0; i < value_tuple->size(); ++i) {
      ValuePtr element = value_tuple->value()[i];
      MS_EXCEPTION_IF_NULL(element);

      if (element->isa<ValueTuple>()) {
        ValueTupleToValue(element, values);
      } else {
        (void)values->emplace_back(element);
      }
    }
  } else {
    (void)values->emplace_back(value);
  }
}

void PrepareDataForValue(const ValuePtr &value, const KernelWithIndex &node_with_index,
                         const DeviceContext *device_context, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(value);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  const auto &node = node_with_index.first;
  MS_EXCEPTION_IF_NULL(node);
  size_t index = node_with_index.second;
  MS_LOG(INFO) << "Prepare data for value node" << node->DebugString() << " index:" << index;

  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, index, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (device_tensor->GetPtr() != nullptr) {
    return;
  }

  if (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize())) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, *device_context,
                                                node->fullname_with_scope(), device_tensor->GetSize());
  }

  TypeId type = kNumberTypeBegin;
  auto host_addr = std::make_unique<char[]>(device_tensor->GetSize());
  MS_EXCEPTION_IF_NULL(host_addr);
  if (value->isa<BoolImm>()) {
    type = kNumberTypeBool;
    (reinterpret_cast<bool *>(host_addr.get()))[0] = GetValue<bool>(value);
  } else if (value->isa<Int64Imm>()) {
    type = kNumberTypeInt64;
    (reinterpret_cast<int64_t *>(host_addr.get()))[0] = GetValue<int64_t>(value);
  } else if (value->isa<Int32Imm>()) {
    type = kNumberTypeInt32;
    (reinterpret_cast<int32_t *>(host_addr.get()))[0] = GetValue<int32_t>(value);
  } else if (value->isa<Monad>()) {
    return;
  } else {
    std::string error_info = "Invalid value:" + value->ToString();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  auto type_size = GetTypeByte(TypeIdToType(type));
  if (type_size > device_tensor->GetSize()) {
    std::string error_info = "Invalid device tensor size:" + std::to_string(device_tensor->GetSize()) +
                             " for type:" + std::to_string(type) + " type size:" + std::to_string(type_size) +
                             " for value:" + value->ToString() + " in node:" + node->DebugString() +
                             " index:" + std::to_string(index);
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  if (!device_tensor->SyncHostToDevice({}, type_size, type, host_addr.get())) {
    std::string error_info = "SyncHostToDevice failed, node name: " + node->DebugString();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
}

void UpdateRefNodeOutputDeviceAddress(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ref_node_map = graph->GetRefMap();
  for (auto iter : ref_node_map) {
    auto &output_pair = iter.first;
    auto &input_pair = iter.second;
    auto &ref_node = output_pair.first;
    auto output_index = output_pair.second;
    auto &input_node = input_pair.first;
    auto input_node_output_index = input_pair.second;

    auto input_addr = AnfAlgo::GetMutableOutputAddr(input_node, input_node_output_index);
    auto ref_node_output_addr = AnfAlgo::GetMutableOutputAddr(ref_node, output_index);
    // Just compare shared_ptr of two DeviceAddress.
    // The ptr of DeviceAddress may still be nullptr.
    if (input_addr != ref_node_output_addr) {
      // AnfAlgo::SetOutputAddr cannot update the device_address of frontend Tensor
      // if the output of RefNode is used by subsequent nodes.
      // Because the frontend Tensor is copied from backend Tensor and the shared_ptr of Tensor is different.
      if (input_addr->GetMutablePtr() == nullptr) {
        AnfAlgo::SetOutputAddr(input_addr, output_index, ref_node.get());
      } else {
        ref_node_output_addr->set_ptr(input_addr->GetMutablePtr());
      }
    }
  }
}

void UpdateGraphsRefNodeAddress(const std::vector<KernelGraphPtr> &graphs) {
  for (const auto &graph : graphs) {
    // The DeviceAddress of the graph parameter has been updated.
    // The output address of RefNode needs to be consistent with the address of parameter.
    if (!graph->is_executing_sink()) {
      UpdateRefNodeOutputDeviceAddress(graph);
    }
  }
}
}  // namespace
void DataPrepareActor::Init() {
  MS_EXCEPTION_IF_NULL(graph_compiler_info_);
  strategy_ = graph_compiler_info_->strategy_;
  if (graph_compiler_info_->graphs_.size() != graph_compiler_info_->device_contexts_.size()) {
    MS_LOG(EXCEPTION) << "The number of graphs is not equal to the number of device contexts.";
  }

  for (auto &iter : continuous_memory_nodes_) {
    size_t total_size = 0;
    std::vector<size_t> size_list;
    std::vector<DeviceTensorPtr> addr_list;
    // Inputs need continuous memory.
    if (iter.second.first == true) {
      FetchContinuousMemoryInfo(iter.first.first, &addr_list, &size_list, &total_size, true);
      (void)continuous_memory_alloc_list_list_.emplace_back(addr_list);
      (void)size_list_list_.emplace_back(size_list);
      (void)total_size_list_.emplace_back(total_size);
      (void)continuous_memory_device_contexts_.emplace_back(iter.first.second);
    }

    // Outputs need continuous memory.
    if (iter.second.second == true) {
      FetchContinuousMemoryInfo(iter.first.first, &addr_list, &size_list, &total_size, false);
      (void)continuous_memory_alloc_list_list_.emplace_back(addr_list);
      (void)size_list_list_.emplace_back(size_list);
      (void)total_size_list_.emplace_back(total_size);
      (void)continuous_memory_device_contexts_.emplace_back(iter.first.second);
    }
  }
}

void DataPrepareActor::UpdateDynamicShape(const AnfNodePtr &input_node, const TensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_tensor == nullptr) {
    return;
  }

  if (!input_node->isa<Parameter>()) {
    return;
  }

  auto input_param = input_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(input_param);
  if (!input_param->has_dynamic_shape()) {
    return;
  }

  auto shape = input_tensor->shape();
  std::vector<size_t> shape_tmp;
  std::transform(shape.begin(), shape.end(), std::back_inserter(shape_tmp), IntToSize);
  AnfAlgo::SetOutputInferTypeAndShape({AnfAlgo::GetOutputInferDataType(input_node, 0)}, {shape_tmp}, input_node.get());
}

void DataPrepareActor::PrepareData(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                   OpContext<DeviceTensor> *const context, GraphExecutionStrategy real_strategy) {
  MS_EXCEPTION_IF_NULL(context);
  MS_LOG(DEBUG) << "Data prepare actor(" << GetAID().Name() << ") prepares data.";

  real_strategy_ = real_strategy;
  // Convert actor running data from input tensors.
  if (input_tensors.size() > 0) {
    try {
      PrepareDataForDeviceTensorStore(input_tensors, context);
      if (strategy_ == GraphExecutionStrategy::kPipeline) {
        PrepareDataForHostTensorQueue(input_tensors, context);
      } else if (strategy_ == GraphExecutionStrategy::kStep) {
        PrepareDataForStepMode(input_tensors, context);
      }

      UpdateGraphsRefNodeAddress(graph_compiler_info_->graphs_);

      // Debug actor is blocked, must wait debug actor callback message to process continue.
      if (debug_aid_ != nullptr && strategy_ == GraphExecutionStrategy::kPipeline) {
        SendDebugReq(context);
        return;
      }
    } catch (const std::exception &e) {
      std::string error_info = e.what();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
  }

  // Allocate continuous memory and send output to trigger the step running.
  if (continuous_memory_alloc_list_list_.size() > 0) {
    SendMemoryAllocReq(context);
  } else {
    PostRun(context);
  }
}

void DataPrepareActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  ActorDispatcher::Send(*debug_aid_, &DebugActor::DebugOnStepBegin, graph_compiler_info_->graphs_,
                        graph_compiler_info_->device_contexts_, context, &GetAID());
}

void DataPrepareActor::OnDebugFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (continuous_memory_alloc_list_list_.size() > 0) {
    SendMemoryAllocReq(context);
  } else {
    PostRun(context);
  }
}

void DataPrepareActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  // Allocate continuous memory in the begin of the step running.
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateContinuousMemory,
                        &continuous_memory_alloc_list_list_, &size_list_list_, &total_size_list_,
                        &continuous_memory_device_contexts_, context, GetAID());
}

void DataPrepareActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  PostRun(context);
}

void DataPrepareActor::PrepareDataForDeviceTensorStore(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                                       OpContext<DeviceTensor> *const context) {
  const auto &parser = graph_compiler_info_->control_node_parser_;
  MS_EXCEPTION_IF_NULL(parser);
  for (size_t i = 0; i < graph_compiler_info_->graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info_->graphs_[i];
    const auto &device_context = graph_compiler_info_->device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    // Prepare the data of device tensor store(value nodes of graph).
    for (const auto &value_node : graph->graph_value_nodes()) {
      if (AnfAlgo::OutputAddrExist(value_node, 0)) {
        PrepareDataForValueNode(value_node, device_context, context);
      }
    }

    // Prepare the data of device tensor store(weights of graph).
    const auto &input_nodes = graph->input_nodes();
    const auto &tensors = input_tensors[i];
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      const auto &input_tensor = tensors[j];
      MS_EXCEPTION_IF_NULL(input_node);
      const auto front_node = FetchFrontNodeByBackendNode(input_node, graph);
      if (IsPersistentDeviceTensor(input_node) && parser->IsRootGraphPersistentDeviceTensor(front_node)) {
        PrepareDataForWeightNode(input_node, front_node, input_tensor, device_context, context);
      }
    }
  }

  PrepareDeviceTensorStoreForControlNode(graph_compiler_info_->control_node_parser_, input_tensors.back(), context);
}

void DataPrepareActor::PrepareDataForHostTensorQueue(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                                     OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if ((host_data_source_actor_ == nullptr) || (host_tensor_queue_ == nullptr)) {
    return;
  }

  std::vector<TensorPtr> host_tensors;
  host_tensors.resize(host_data_source_actor_->data_nodes().size());
  // Fill host tensors.
  for (size_t i = 0; i < graph_compiler_info_->graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info_->graphs_[i];
    MS_EXCEPTION_IF_NULL(graph);

    const auto &input_nodes = graph->input_nodes();
    const auto &tensors = input_tensors[i];
    if (input_nodes.size() != tensors.size()) {
      std::string error_info = "Invalid tensor size:" + std::to_string(tensors.size()) +
                               " and input node size:" + std::to_string(input_nodes.size()) +
                               " for kernel graph:" + graph->ToString();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      const auto &input_tensor = tensors[j];
      MS_EXCEPTION_IF_NULL(input_node);
      if (!IsHostQueueDSActor(input_node, graph, graph_compiler_info_->origin_parameters_order_, strategy_) ||
          input_tensor == nullptr) {
        continue;
      }

      UpdateDynamicShape(input_node, input_tensor);

      auto tensor_position = host_data_source_actor_->FetchNodePosition(input_node);
      if (tensor_position >= host_tensors.size()) {
        std::string error_info = "The position of tensor is out of range: " + std::to_string(tensor_position);
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
      }
      host_tensors[tensor_position] = input_tensor;

      auto tensor_address = std::dynamic_pointer_cast<DeviceTensor>(input_tensor->device_address());
      auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_address);
      if ((tensor_address != nullptr) && (tensor_address->DeviceType() == device_address->DeviceType()) &&
          !device_address->is_ptr_persisted()) {
        AnfAlgo::SetOutputAddr(tensor_address, 0, input_node.get());
        tensor_address->SetNodeIndex(input_node, 0);
      }
    }
  }

  PrepareHostTensorQueueForControlNode(input_tensors.back(), &host_tensors, context);

  host_tensor_queue_->Push(host_tensors);
}

void DataPrepareActor::PrepareDataForStepMode(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                              OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  std::vector<TensorPtr> host_tensors;
  if ((host_data_source_actor_ != nullptr) && (host_tensor_queue_ != nullptr)) {
    host_tensors.resize(host_data_source_actor_->data_nodes().size());
  }

  for (size_t i = 0; i < graph_compiler_info_->graphs_.size(); ++i) {
    const auto &graph = graph_compiler_info_->graphs_[i];
    const auto &device_context = graph_compiler_info_->device_contexts_[i];
    MS_EXCEPTION_IF_NULL(graph);
    MS_EXCEPTION_IF_NULL(device_context);

    const auto &input_nodes = graph->input_nodes();
    const auto &tensors = input_tensors[i];
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      const auto &input_tensor = tensors[j];
      MS_EXCEPTION_IF_NULL(input_node);
      MS_EXCEPTION_IF_NULL(input_tensor);
      if (IsPersistentDeviceTensor(input_node)) {
        continue;
      }

      UpdateDynamicShape(input_node, input_tensor);

      if ((host_data_source_actor_ != nullptr) && (host_tensor_queue_ != nullptr)) {
        auto tensor_position = host_data_source_actor_->FetchNodePosition(input_node);
        if (tensor_position >= host_tensors.size()) {
          std::string error_info = "The position of tensor is out of range: " + std::to_string(tensor_position);
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
        }
        host_tensors[tensor_position] = input_tensor;
      }

      auto host_tensor_address = std::dynamic_pointer_cast<DeviceTensor>(input_tensor->device_address());
      if (host_tensor_address != nullptr) {
        if (host_tensor_address->DeviceType() != device_context->GetDeviceAddressType()) {
          input_tensor->data_sync();
          input_tensor->set_device_address(nullptr);
        } else {
          AnfAlgo::SetOutputAddr(host_tensor_address, 0, input_node.get());
          host_tensor_address->SetNodeIndex(input_node, 0);
          continue;
        }
      }

      if (!AnfAlgo::OutputAddrExist(input_node, 0, false)) {
        TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(input_node, 0);
        if (output_type_id == kTypeUnknown) {
          output_type_id = AnfAlgo::GetOutputInferDataType(input_node, 0);
        }
        size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(input_node, 0);
        auto device_address = device_context->CreateDeviceAddress(
          nullptr, tensor_size, AnfAlgo::GetOutputFormat(input_node, 0), output_type_id);
        MS_EXCEPTION_IF_NULL(device_address);
        AnfAlgo::SetOutputAddr(device_address, 0, input_node.get());
        device_address->SetNodeIndex(input_node, 0);
      }
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
      input_tensor->set_device_address(device_tensor);
      UpdateRefCount(device_tensor.get(), true);

      SyncTensorData(input_tensor, device_tensor, input_node, device_context, context, real_strategy_);
    }
  }

  if ((host_data_source_actor_ != nullptr) && (host_tensor_queue_ != nullptr)) {
    host_tensor_queue_->Push(host_tensors);
  }
}

//  The branch processing of PrepareDataForValueNode that value type is tensor.
void DataPrepareActor::PrepareDataForValueNodeTensor(const ValueNodePtr &node, const ValuePtr &node_value,
                                                     const DeviceContext *device_context,
                                                     OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);

  std::vector<TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);
  for (size_t i = 0; i < tensors.size(); i++) {
    const auto &tensor = tensors[i];
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }

    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, i, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
    if (device_tensor->GetPtr() != nullptr) {
      return;
    }
    MS_LOG(INFO) << "Prepare device data for value node: " << node->fullname_with_scope() << ", output index: " << i;
    tensor->set_device_address(device_tensor);
    UpdateRefCount(device_tensor.get(), true);

    SyncTensorData(tensor, device_tensor, node, device_context, context, real_strategy_);
  }
}

void DataPrepareActor::PrepareDataForControlValueNode(const KernelWithIndex &node_with_index,
                                                      const DeviceContext *device_context,
                                                      OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  if (!node_with_index.first->isa<ValueNode>()) {
    return;
  }

  const auto &node = node_with_index.first->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(node);
  size_t index = node_with_index.second;
  const auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);
  std::vector<ValuePtr> values;
  ValueTupleToValue(node_value, &values);

  if (node_with_index.second >= values.size()) {
    std::string error_info =
      "Invalid index:" + std::to_string(node_with_index.second) + " for value node:" + node->DebugString();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
  const auto &value = values[index];
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, index, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->GetPtr() != nullptr) {
      return;
    }

    MS_LOG(INFO) << "Prepare device data for control value node: " << node->DebugString()
                 << ", output index: " << index;
    tensor->set_device_address(device_tensor);
    UpdateRefCount(device_tensor.get(), true);

    if (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize())) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, *device_context,
                                                  node->fullname_with_scope(), device_tensor->GetSize());
    }

    auto host_tensor_size = LongToSize(tensor->data().nbytes());
    auto host_tensor_type = tensor->data_type();
    auto shape = tensor->shape();
    if (!device_tensor->SyncHostToDevice(shape, host_tensor_size, host_tensor_type, tensor->data_c(),
                                         tensor->device_info().host_format_)) {
      std::string error_info = "Sync host to device failed for node:" + node->DebugString();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  } else {
    PrepareDataForValue(value, node_with_index, device_context, context);
  }
}

// Prepare the device data for persistent device tensor of value node.
void DataPrepareActor::PrepareDataForValueNode(const ValueNodePtr &node, const DeviceContext *device_context,
                                               OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  auto &node_value = node->value();
  MS_EXCEPTION_IF_NULL(node_value);

  if (node_value->isa<tensor::Tensor>() || node_value->isa<ValueTuple>()) {
    //  The branch processing that value type is tensor.
    PrepareDataForValueNodeTensor(node, node_value, device_context, context);
  } else if (node_value->isa<StringImm>()) {
    const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(node, 0, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
    if (device_tensor->GetPtr() != nullptr) {
      return;
    }
    MS_LOG(INFO) << "Prepare device data for value node: " << node->fullname_with_scope();

    if (!device_context->AllocateMemory(device_tensor.get(), device_tensor->GetSize())) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, *device_context,
                                                  node->fullname_with_scope(), device_tensor->GetSize());
    }

    // Copy data from value to device.
    auto value = GetValue<std::string>(node_value);
    size_t tensor_size = value.size();
    ShapeVector shape = {1, SizeToLong(tensor_size)};
    if (!device_tensor->SyncHostToDevice(shape, tensor_size, kNumberTypeUInt8, value.data())) {
      std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
  }
}

void DataPrepareActor::CopyDataFromHostToOtherDevice(const AnfNodePtr &front_node, const AnfNodePtr &backend_node,
                                                     const device::DeviceAddressPtr &host_tensor_address,
                                                     const DeviceContext *device_context,
                                                     OpContext<DeviceTensor> *context) const {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  const auto &device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
  if (device_tensors.size() > 1) {
    auto another_device_tensor = (device_tensors[0] == host_tensor_address) ? device_tensors[1] : device_tensors[0];
    MS_EXCEPTION_IF_NULL(another_device_tensor);
    auto another_device_type = another_device_tensor->DeviceType();
    const auto &another_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device::kDeviceTypeToName.at(another_device_type), device_context->device_context_key().device_id_});
    MS_EXCEPTION_IF_NULL(another_device_context);
    if ((another_device_tensor->GetPtr() == nullptr) &&
        (!another_device_context->AllocateMemory(another_device_tensor.get(), another_device_tensor->GetSize()))) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(real_strategy_, *context, *another_device_context,
                                                  backend_node->fullname_with_scope(),
                                                  another_device_tensor->GetSize());
    }

    MS_LOG(INFO) << "Prepare device data for weight node:" << backend_node->fullname_with_scope()
                 << ", device type:" << another_device_type;
    if (!Copy(another_device_tensor.get(), host_tensor_address.get())) {
      std::string error_info = "Sync data error.";
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
  }
}

// Prepare the device data for persistent device tensor of weight node from host tensor.
void DataPrepareActor::PrepareDataForWeightNode(const AnfNodePtr &backend_node, const AnfNodePtr &front_node,
                                                const TensorPtr &tensor, const DeviceContext *device_context,
                                                OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(front_node);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);
  if (tensor == nullptr) {
    return;
  }

  auto device_tensor = AnfAlgo::GetMutableOutputAddr(backend_node, 0, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  auto host_tensor_address = std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address());
  // Use the device address of host tensor to set device tensor.
  bool is_need_sync = false;
  if (host_tensor_address != device_tensor) {
    if (host_tensor_address == nullptr) {
      // The step mode can't reuse the device tensor, because other actors may use the device tensor in step mode.
      if ((strategy_ == GraphExecutionStrategy::kStep) ||
          (device_tensor->DeviceType() != device_context->GetDeviceAddressType())) {
        host_tensor_address = device_context->CreateDeviceAddress(nullptr, device_tensor->GetSize(),
                                                                  device_tensor->format(), device_tensor->type_id());
        host_tensor_address->set_from_persistent_mem(tensor->is_parameter());
      } else {
        host_tensor_address = device_tensor;
      }
      is_need_sync = true;
      tensor->set_device_address(host_tensor_address);
      UpdateRefCount(host_tensor_address.get(), true);
    }
    MS_EXCEPTION_IF_NULL(host_tensor_address);

    if (host_tensor_address->DeviceType() == device_tensor->DeviceType() &&
        !(host_tensor_address->format() != device_tensor->format() && strategy_ == GraphExecutionStrategy::kStep)) {
      // In the scenario of training + inference , the device address of the weight node can not be changed when
      // multi-graphs sink mode is set.
      if (device_tensor->is_ptr_persisted() && (host_tensor_address != device_tensor)) {
        if (!Copy(device_tensor.get(), host_tensor_address.get())) {
          std::string error_info = "Sync data error.";
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
        }
        host_tensor_address = device_tensor;
      } else {
        AnfAlgo::SetOutputAddr(host_tensor_address, 0, backend_node.get());
        host_tensor_address->SetNodeIndex(backend_node, 0);
      }
    } else {
      MS_LOG(INFO) << "The device type or format is not equal, host tensor type:" << host_tensor_address->DeviceType()
                   << " format:" << host_tensor_address->format()
                   << ", device tensor type:" << device_tensor->DeviceType() << " format:" << device_tensor->format();
      if (strategy_ == GraphExecutionStrategy::kStep) {
        tensor->data_sync();
        host_tensor_address = device_tensor;
        tensor->set_device_address(host_tensor_address);
        is_need_sync = true;
      }
    }
  }
  // Maybe the same host_tensor_address corresponds to the different front_node in shared weight scene,
  // so need update the device tensor store always.
  host_tensor_address->SetNodeIndex(backend_node, 0);
  DeviceTensorStore::GetInstance().Insert(front_node.get(), host_tensor_address);

  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  MS_EXCEPTION_IF_NULL(host_tensor_address);
  if (is_need_sync || (host_tensor_address->GetPtr() == nullptr)) {
    MS_LOG(INFO) << "Prepare device data for weight node:" << backend_node->DebugString()
                 << ", device type:" << host_tensor_address->DeviceType();
    SyncTensorData(tensor, host_tensor_address, backend_node, device_context, context, real_strategy_);
  }

  // Allocate another device memory and copy data from host tensor to another device(if exist).
  CopyDataFromHostToOtherDevice(front_node, backend_node, host_tensor_address, device_context, context);
}

void DataPrepareActor::PrepareDeviceTensorStoreForControlNode(const ControlNodeParserPtr &control_node_parser,
                                                              const std::vector<TensorPtr> &tensors,
                                                              OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(control_node_parser);
  if (!control_node_parser->IsInited()) {
    return;
  }

  for (const auto &value_node_with_context : control_node_parser->front_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node_with_context.first.first);
    if (AnfAlgo::OutputAddrExist(value_node_with_context.first.first, 0)) {
      PrepareDataForControlValueNode(value_node_with_context.first, value_node_with_context.second, context);
    }
  }

  const auto &control_node_parameters = control_node_parser->control_node_parameters();
  if (control_node_parameters.size() != tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Invalid tensor size.");
  }
  for (size_t i = 0; i < control_node_parameters.size(); ++i) {
    auto &front_parameter = control_node_parameters[i];
    auto &tensor = tensors[i];
    if (tensor == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(front_parameter);
    if (!control_node_parser->IsRootGraphPersistentDeviceTensor(front_parameter)) {
      continue;
    }

    auto device_tensors = DeviceTensorStore::GetInstance().Fetch(front_parameter.get());
    if (device_tensors.empty()) {
      MS_LOG(WARNING) << "Failed to get device tensor for front node:" << front_parameter->DebugString();
      continue;
    }
    MS_EXCEPTION_IF_NULL(device_tensors[0]);
    auto host_tensor_address = std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address());
    if ((device_tensors[0] == host_tensor_address) || (device_tensors[0]->GetPtr() != nullptr)) {
      continue;
    }

    auto node = (device_tensors[0]->GetNodeIndex()).first;
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(INFO) << "Prepare device data for weight node by root graph parameter:"
                 << front_parameter->fullname_with_scope() << ", backend node:" << node->DebugString()
                 << ", device type:" << device_tensors[0]->DeviceType();
    if (host_tensor_address == nullptr) {
      tensor->set_device_address(device_tensors[0]);
      auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device_tensors[0]->device_name(), device_tensors[0]->device_id()});
      SyncTensorData(tensor, device_tensors[0], node, device_context, context, GraphExecutionStrategy::kPipeline);
    } else {
      if (host_tensor_address->GetSize() != device_tensors[0]->GetSize()) {
        MS_LOG(WARNING) << "Please check the size of parameter:" << front_parameter->fullname_with_scope()
                        << ", host tensor size:" << host_tensor_address->GetSize()
                        << ", device tensor size:" << device_tensors[0]->GetSize();
      }
      host_tensor_address->SetNodeIndex(node, 0);
      UpdateRefCount(host_tensor_address.get(), true);
      DeviceTensorStore::GetInstance().Remove(front_parameter.get());
      DeviceTensorStore::GetInstance().Insert(front_parameter.get(), host_tensor_address);
    }
  }
}

void DataPrepareActor::PrepareHostTensorQueueForControlNode(const std::vector<TensorPtr> &tensors,
                                                            std::vector<TensorPtr> *const host_tensors,
                                                            OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(graph_compiler_info_->control_node_parser_);
  MS_EXCEPTION_IF_NULL(host_data_source_actor_);
  MS_EXCEPTION_IF_NULL(host_tensors);

  const auto &control_node_parameters = graph_compiler_info_->control_node_parser_->control_node_parameters();
  for (size_t i = 0; i < control_node_parameters.size(); ++i) {
    const auto &input_node = control_node_parameters[i];
    const auto &input_tensor = tensors[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPersistentDeviceTensor(input_node)) {
      continue;
    }

    if (find(graph_compiler_info_->origin_parameters_order_.begin(),
             graph_compiler_info_->origin_parameters_order_.end(),
             input_node) == graph_compiler_info_->origin_parameters_order_.end()) {
      continue;
    }

    auto tensor_position = host_data_source_actor_->FetchNodePosition(input_node);
    if (tensor_position >= host_tensors->size()) {
      std::string error_info = "The position of tensor is out of range: " + std::to_string(tensor_position);
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(real_strategy_, (*context), error_info);
    }
    (*host_tensors)[tensor_position] = input_tensor;

    const AnfNodePtr &backend_node = host_data_source_actor_->FetchNode(tensor_position);
    auto tensor_address = std::dynamic_pointer_cast<DeviceTensor>(input_tensor->device_address());
    auto device_address = AnfAlgo::GetMutableOutputAddr(backend_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_address);
    if ((tensor_address != nullptr) && (tensor_address->DeviceType() == device_address->DeviceType()) &&
        !device_address->is_ptr_persisted()) {
      AnfAlgo::SetOutputAddr(tensor_address, 0, backend_node.get());
      tensor_address->SetNodeIndex(backend_node, 0);
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
