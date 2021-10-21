/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/framework/actor/data_prepare_actor.h"
#include "runtime/framework/actor/memory_manager_actor.h"
#include "runtime/framework/actor/kernel_actor.h"
#include "runtime/framework/actor/loop_count_actor.h"
#include "runtime/framework/actor/debug_actor.h"
#include "runtime/hardware/device_context_manager.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"
#include "utils/convert_utils.h"
#include "common/trans.h"

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

void DataPrepareActor::PrepareData(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                   OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  // Convert actor running data from input tensors.
  if (input_tensors.size() > 0) {
    PrepareDataForDeviceTensorStore(input_tensors, context);
    if (strategy_ == GraphExecutionStrategy::kPipeline) {
      PrepareDataForHostTensorQueue(input_tensors, context);
    } else if (strategy_ == GraphExecutionStrategy::kStep) {
      PrepareDataForStepMode(input_tensors, context);
    }

    // Debug actor is blocked, must wait debug actor callback message to process continue.
    if (debug_aid_ != nullptr && strategy_ == GraphExecutionStrategy::kPipeline) {
      SendDebugReq(context);
      return;
    }
  }

  // Allocate continuous memory and send output to trigger the step running.
  if (continuous_memory_alloc_list_list_.size() > 0) {
    SendMemoryAllocReq(context);
  } else {
    SendOutput(context);
  }
}

void DataPrepareActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  Async(*debug_aid_, &DebugActor::DebugOnStepBegin, graph_compiler_info_->graphs_,
        graph_compiler_info_->device_contexts_, context, &GetAID());
}

void DataPrepareActor::OnDebugFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (continuous_memory_alloc_list_list_.size() > 0) {
    SendMemoryAllocReq(context);
  } else {
    SendOutput(context);
  }
}

void DataPrepareActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  // Allocate continuous memory in the begin of the step running.
  Async(memory_manager_aid_, &MemoryManagerActor::AllocateContinuousMemory, &continuous_memory_alloc_list_list_,
        &size_list_list_, &total_size_list_, &continuous_memory_device_contexts_, context, GetAID());
}

void DataPrepareActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  SendOutput(context);
}

void DataPrepareActor::SendOutput(OpContext<DeviceTensor> *const context) {
  for (auto &data_source_aid : data_source_aids_) {
    Async(data_source_aid, &DataSourceActor::FetchData, context);
  }

  auto source_aid = const_cast<AID *>(&GetAID());
  for (auto &kernel_aid : no_input_kernel_aids_) {
    Async(kernel_aid, &KernelActor::RunOpControl, source_aid, context);
  }

  // Trigger loop count actor running when there are no data source actor and kernel actor.
  if ((data_source_aids_.size() + no_input_kernel_aids_.size() == 0) && (loop_count_aid_ != nullptr)) {
    Async(*loop_count_aid_, &LoopCountActor::RunOpControl, source_aid, context);
  }
}

void DataPrepareActor::PrepareDataForDeviceTensorStore(const std::vector<std::vector<TensorPtr>> &input_tensors,
                                                       OpContext<DeviceTensor> *const context) {
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
      if (!IsPersistentDeviceTensor(input_node)) {
        continue;
      }
      const auto front_node = FetchFrontNodeByBackendNode(input_node, graph);
      PrepareDataForWeightNode(input_node, front_node, input_tensor, device_context, context);
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
    for (size_t j = 0; j < input_nodes.size(); ++j) {
      const auto &input_node = input_nodes[j];
      const auto &input_tensor = tensors[j];
      MS_EXCEPTION_IF_NULL(input_node);
      if (!IsHostQueueDSActor(input_node, graph, graph_compiler_info_->origin_parameters_order_, strategy_)) {
        continue;
      }
      auto tensor_position = host_data_source_actor_->FetchNodePosition(input_node);
      if (tensor_position >= host_tensors.size()) {
        std::string error_info = "The position of tensor is out of range: " + std::to_string(tensor_position);
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
      }
      host_tensors[tensor_position] = input_tensor;

      auto tensor_address = std::dynamic_pointer_cast<DeviceTensor>(input_tensor->device_address());
      auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_address);
      if ((tensor_address != nullptr) && (tensor_address->DeviceType() == device_address->DeviceType())) {
        AnfAlgo::SetOutputAddr(tensor_address, 0, input_node.get());
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

      if ((host_data_source_actor_ != nullptr) && (host_tensor_queue_ != nullptr)) {
        auto tensor_position = host_data_source_actor_->FetchNodePosition(input_node);
        if (tensor_position >= host_tensors.size()) {
          std::string error_info = "The position of tensor is out of range: " + std::to_string(tensor_position);
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
        }
        host_tensors[tensor_position] = input_tensor;
      }

      auto host_tensor_address = std::dynamic_pointer_cast<DeviceTensor>(input_tensor->device_address());
      if (host_tensor_address != nullptr) {
        AnfAlgo::SetOutputAddr(host_tensor_address, 0, input_node.get());
        continue;
      }

      if (!AnfAlgo::OutputAddrExist(input_node, 0, false)) {
        TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(input_node, 0);
        if (output_type_id == kTypeUnknown) {
          output_type_id = AnfAlgo::GetOutputInferDataType(input_node, 0);
        }
        size_t tensor_size = AnfAlgo::GetOutputTensorMemSize(input_node, 0);
        auto device_address = device_context->CreateDeviceAddress(
          nullptr, tensor_size, AnfAlgo::GetOutputFormat(input_node, 0), output_type_id);
        AnfAlgo::SetOutputAddr(device_address, 0, input_node.get());
      }
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
      input_tensor->set_device_address(device_tensor);
      UpdateRefCount(device_tensor.get(), true);

      SyncTensorData(input_tensor, device_tensor, input_node, device_context, context, strategy_);
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

    SyncTensorData(tensor, device_tensor, node, device_context, context, strategy_);
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
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy_, *context, *device_context, node->fullname_with_scope(),
                                                  device_tensor->GetSize());
    }

    // Copy data from value to device.
    auto value = GetValue<std::string>(node_value);
    size_t tensor_size = value.size();
    ShapeVector shape = {1, SizeToLong(tensor_size)};
    if (!device_tensor->SyncHostToDevice(shape, tensor_size, kNumberTypeUInt8, value.data())) {
      std::string error_info = "SyncHostToDevice failed, node name: " + node->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
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
  if (host_tensor_address != device_tensor) {
    if (host_tensor_address == nullptr) {
      host_tensor_address = device_context->CreateDeviceAddress(nullptr, device_tensor->GetSize(),
                                                                device_tensor->format(), device_tensor->type_id());
      tensor->set_device_address(host_tensor_address);
      UpdateRefCount(host_tensor_address.get(), true);
    }
    MS_EXCEPTION_IF_NULL(host_tensor_address);
    if (host_tensor_address->DeviceType() == device_tensor->DeviceType()) {
      AnfAlgo::SetOutputAddr(host_tensor_address, 0, backend_node.get());
    } else {
      MS_LOG(INFO) << "The device type is not equal, host tensor type:" << host_tensor_address->DeviceType()
                   << ", device tensor type:" << device_tensor->DeviceType();
    }
  }
  // Maybe the same host_tensor_address corresponds to the different front_node in shared weight scene,
  // so need update the device tensor store always.
  DeviceTensorStore::GetInstance().Insert(front_node.get(), host_tensor_address);

  // If the ptr of device tensor is not nullptr, it indicates that the device data has been prepared.
  MS_EXCEPTION_IF_NULL(host_tensor_address);
  if (host_tensor_address->GetPtr() == nullptr) {
    MS_LOG(INFO) << "Prepare device data for weight node:" << backend_node->fullname_with_scope()
                 << ", device type:" << host_tensor_address->DeviceType();
    SyncTensorData(tensor, host_tensor_address, backend_node, device_context, context, strategy_);
  }

  // Allocate another device memory and copy data from host tensor to another device(if exist).
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
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy_, *context, *another_device_context,
                                                  backend_node->fullname_with_scope(),
                                                  another_device_tensor->GetSize());
    }

    MS_LOG(INFO) << "Prepare device data for weight node:" << backend_node->fullname_with_scope()
                 << ", device type:" << another_device_type;
    if (!Copy(another_device_tensor.get(), host_tensor_address.get())) {
      std::string error_info = "Sync data error.";
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
    }
  }
}

// In control flow, all weight nodes associated with the host weight parameter need to use the same device tensor.
void DataPrepareActor::PrepareDataForControlWeightNode(
  const AnfNodePtr &node, const AnfNodePtr &front_node, const TensorPtr &tensor, const DeviceContext *device_context,
  const std::unordered_map<AnfNodePtr, std::vector<AnfNodePtr>> &host_parameter_to_weights,
  OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(front_node);
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context);

  auto device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
  bool need_update_device_tensor_store = (device_tensors.size() == 0) ? true : false;
  for (auto &device_tensor : device_tensors) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->GetPtr() == nullptr) {
      need_update_device_tensor_store = true;
      break;
    }
  }
  if (need_update_device_tensor_store) {
    PrepareDataForWeightNode(node, front_node, tensor, device_context, context);
  }

  const auto iter = host_parameter_to_weights.find(front_node);
  if (iter == host_parameter_to_weights.end()) {
    return;
  }

  // Fetch all the device tensors of host weight node and insert as the weight of other nodes.
  const auto &sub_front_nodes = host_parameter_to_weights.at(front_node);
  device_tensors = DeviceTensorStore::GetInstance().Fetch(front_node.get());
  for (const auto &sub_front_node : sub_front_nodes) {
    for (const auto &device_tensor : device_tensors) {
      MS_EXCEPTION_IF_NULL(sub_front_node);
      DeviceTensorStore::GetInstance().Insert(sub_front_node.get(), device_tensor);
    }
  }
}

void DataPrepareActor::PrepareDeviceTensorStoreForControlNode(const ControlNodeParserPtr &control_node_parser,
                                                              const std::vector<TensorPtr> &tensors,
                                                              OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(control_node_parser);
  for (const auto &value_node_with_context : control_node_parser->front_value_nodes()) {
    if (AnfAlgo::OutputAddrExist(value_node_with_context.first, 0)) {
      PrepareDataForValueNode(value_node_with_context.first->cast<ValueNodePtr>(), value_node_with_context.second,
                              context);
    }
  }

  const auto &control_node_parameters = control_node_parser->control_node_parameters();
  for (size_t i = 0; i < control_node_parameters.size(); ++i) {
    const auto &input_node = control_node_parameters[i];
    const auto &input_tensor = tensors[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPersistentDeviceTensor(input_node)) {
      const auto &front_to_backend_parameters = control_node_parser->front_to_backend_parameters();
      const auto &iter = front_to_backend_parameters.find(input_node);
      if (iter == front_to_backend_parameters.end()) {
        MS_LOG(EXCEPTION) << "Cannot find backend node for weight parameter:"
                          << AnfAlgo::GetNodeDebugString(input_node);
      }
      const auto &node_with_context = iter->second;
      PrepareDataForControlWeightNode(node_with_context.first, input_node, input_tensor, node_with_context.second,
                                      control_node_parser->host_parameter_to_weights(), context);
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
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
    }
    (*host_tensors)[tensor_position] = input_tensor;

    const AnfNodePtr &backend_node = host_data_source_actor_->FetchNode(tensor_position);
    auto tensor_address = std::dynamic_pointer_cast<DeviceTensor>(input_tensor->device_address());
    auto device_address = AnfAlgo::GetMutableOutputAddr(backend_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_address);
    if ((tensor_address != nullptr) && (tensor_address->DeviceType() == device_address->DeviceType())) {
      AnfAlgo::SetOutputAddr(tensor_address, 0, backend_node.get());
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
