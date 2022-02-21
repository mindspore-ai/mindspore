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

#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void OutputActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != output_nodes_.size()) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
  // Check outputs number.
  if (output_nodes_.size() != outputs_.size()) {
    MS_LOG(EXCEPTION) << "The outputs number is wrong.";
  }

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(outputs_num_ - device_tensor_store_keys_.size());
}

void OutputActor::RunOpControl(AID *const, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  ++current_count_;
  if (loop_count_ == current_count_) {
    if (current_outputs_num_ + device_tensor_store_keys_.size() != outputs_num_) {
      std::string error_info = "The outputs num is wrong, the total outputs num: " + std::to_string(outputs_num_) +
                               ", the current outputs num: " + std::to_string(current_outputs_num_) +
                               ", the device tensor store num: " + std::to_string(device_tensor_store_keys_.size());
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    // Because device tensor store can't send data, so fetch the output result of device tensor store in running end.
    for (const auto &device_tensor_store_key : device_tensor_store_keys_) {
      if (device_tensor_store_key.first >= outputs_.size()) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is of range.");
      }
      outputs_[device_tensor_store_key.first] =
        CreateOutputTensor(device_tensor_store_key.second, 0, device_tensor_store_key.first);
      if (outputs_[device_tensor_store_key.first] == nullptr) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Create output tensor failed.");
      }
      output_nodes_[device_tensor_store_key.first] = {device_tensor_store_key.second, 0};
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(device_tensor_store_key.second, 0, false);
      output_device_tensors_[device_tensor_store_key.first] = device_tensor.get();
    }

    // For dynamic_shape, UpdateOp maybe run after RunOpData, so it's needed to update shape of output tensor here
    // Check outputs number.
    if (output_nodes_.size() != outputs_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The outputs number is wrong.");
    }
    // Update output tensor's shape
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto shape = common::AnfAlgo::GetOutputInferShape(output_nodes_[i].first, output_nodes_[i].second);
      std::vector<int64_t> temp_shape;
      (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
      if (outputs_[i] == nullptr) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The outputs_[i] is nullptr.");
      }
      outputs_[i]->set_shape(temp_shape);
    }

    current_outputs_num_ = 0;
    current_count_ = 0;
    SET_OPCONTEXT_SUCCESS_RET((*context));
  }
}

void OutputActor::RunOpData(OpData<DeviceTensor> *const input_data, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  MS_EXCEPTION_IF_NULL(context);
  // Collect the output result in the last loop which is represented by "loop_count_ - current_count_ == 1".
  if (loop_count_ - current_count_ != 1) {
    return;
  }

  auto output_position = IntToSize(input_data->index_);
  if (output_position >= outputs_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is of range.");
  }

  auto node_with_index = input_data->data_->GetNodeIndex();
  auto tensor = CreateOutputTensor(node_with_index.first, node_with_index.second, output_position);
  if (tensor == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Create output tensor failed.");
  }
  tensor->set_need_release_device_mem(true);
  outputs_[output_position] = tensor;
  current_outputs_num_++;

  // Save the output nodes to clear the device tensor in the running end.
  output_nodes_[output_position] = node_with_index;
  output_device_tensors_[output_position] = input_data->data_;
}

TensorPtr OutputActor::CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index, size_t output_position) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_LOG(INFO) << "Create output tensor, output node: " << output_node->fullname_with_scope()
               << ", output index: " << output_index << ", output position: " << output_position;

  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto type_id = common::AnfAlgo::GetOutputInferDataType(output_node, output_index);
  std::vector<int64_t> temp_shape;
  auto shape = common::AnfAlgo::GetOutputInferShape(output_node, output_index);
  (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
  auto tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
  tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));

  if (output_position >= device_contexts_.size()) {
    MS_LOG(ERROR) << "The output position is of range: " << output_position;
    return nullptr;
  }
  auto &device_context = device_contexts_[output_position];
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(output_node, output_index, false);
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (device_context->GetDeviceAddressType() != device_tensor->DeviceType()) {
    auto old_device_context = device_context;
    device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_tensor->device_name(), device_tensor->device_id()});
    MS_LOG(INFO) << "Update device context from:" << old_device_context->GetDeviceAddressType()
                 << " to:" << device_context->GetDeviceAddressType();
  }

  // Create the device address and put it into host tensor.
  if (output_node_to_tensor_device_address_.count({output_node, output_index}) > 0) {
    tensor->set_device_address(output_node_to_tensor_device_address_[{output_node, output_index}]);
  } else {
    auto tensor_device_address = device_context->CreateDeviceAddress(nullptr, device_tensor->GetSize(),
                                                                     device_tensor->format(), device_tensor->type_id());
    MS_EXCEPTION_IF_NULL(tensor_device_address);
    tensor->set_device_address(tensor_device_address);
    output_node_to_tensor_device_address_[{output_node, output_index}] = tensor_device_address;
  }
  return tensor;
}

void OutputActor::UpdateOutputDeviceAddress() {
  // In the running end, when the device ptr of graph output node is set into host tensor, the graph output node
  // need be set new device ptr, to avoid that the device ptr context of host tensor be rewritten in the next
  // step or next loop. But the graph output nodes corresponding to device tensor store need to be skipped, because
  // they are fixed addresses and persistent.
  for (size_t i = 0; i < output_nodes_.size(); ++i) {
    auto &output_node = output_nodes_[i].first;
    if (i >= output_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid index:" << i << " current:" << output_device_tensors_.size();
    }
    auto device_tensor = output_device_tensors_[i];
    if (output_node == nullptr || device_tensor == nullptr) {
      MS_LOG(WARNING) << "The output node or device tensor is nullptr, need check whether affect the result.";
      continue;
    }

    auto &tensor = outputs_[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_device_address = std::dynamic_pointer_cast<DeviceTensor>(tensor->device_address());
    MS_EXCEPTION_IF_NULL(tensor_device_address);
    // Update tensor device address by device tensor of output node.
    tensor_device_address->set_original_ref_count(SIZE_MAX);
    tensor_device_address->ResetRefCount();
    tensor_device_address->set_dynamic_ref_count(INT32_MAX);
    auto node_with_index = device_tensor->GetNodeIndex();
    tensor_device_address->SetNodeIndex(node_with_index.first, node_with_index.second);
    tensor_device_address->set_from_persistent_mem(device_tensor->from_persistent_mem());
    tensor_device_address->set_host_shape(device_tensor->host_shape());
    // The outputs may have the same output node, so need skip when the node has been done.
    if (device_tensor->GetPtr() == nullptr) {
      continue;
    }

    // If the output node whose output address ptr can't be changed, then alloc the new device memory and copy the data:
    // 1.In the input as output scenario, the output device tensor may come from the input tensor and can't be replaced.
    // 2.The persisted address can't be replaced.
    if (output_node->isa<ValueNode>() || output_node->isa<Parameter>() || device_tensor->is_ptr_persisted()) {
      auto device_context = device_contexts_[i];
      MS_EXCEPTION_IF_NULL(device_context);
      if (!device_context->AllocateMemory(tensor_device_address.get(), tensor_device_address->GetSize())) {
        MS_LOG(EXCEPTION) << "Device(id:" << device_context->device_context_key().device_id_
                          << ") memory isn't enough and alloc failed, kernel name: "
                          << output_node->fullname_with_scope() << ", alloc size: " << tensor_device_address->GetSize()
                          << "B.";
      }
      if (!tensor_device_address->SyncDeviceToDevice(device_tensor)) {
        MS_LOG(EXCEPTION) << "Sync device to device failed, device type: " << tensor_device_address->DeviceType();
      }
    } else {
      // Move the device ptr from device_tensor to tensor_device_address.
      tensor_device_address->set_ptr(device_tensor->GetMutablePtr());
      tensor_device_address->set_from_mem_pool(device_tensor->from_mem_pool());
      device_tensor->set_ptr(nullptr);
      device_tensor->set_from_mem_pool(false);
    }
  }

  output_node_to_tensor_device_address_.clear();
  output_nodes_.clear();
  output_nodes_.resize(outputs_num_);
  output_device_tensors_.clear();
  output_device_tensors_.resize(outputs_num_);
}
}  // namespace runtime
}  // namespace mindspore
