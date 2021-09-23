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

#include "runtime/framework/actor/output_actor.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
namespace {
TensorPtr CreateOutputTensor(const AnfNodePtr &output_node, size_t output_index, size_t output_position) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_LOG(INFO) << "Create output tensor, output node: " << output_node->fullname_with_scope()
               << ", output index: " << output_index << ", output position: " << output_position;

  // Create host tensor, the output tensor should use the infer type, it will be handed correctly by tensor data sync
  // when infer type is not equal to device type.
  auto type_id = AnfAlgo::GetOutputInferDataType(output_node, output_index);
  std::vector<int64_t> temp_shape;
  auto shape = AnfAlgo::GetOutputInferShape(output_node, output_index);
  (void)std::copy(shape.begin(), shape.end(), std::back_inserter(temp_shape));
  auto tensor = std::make_shared<tensor::Tensor>(type_id, temp_shape);
  tensor->set_padding_type(AnfAlgo::GetOutputReshapeType(output_node, output_index));

  // Put device tensor into host tensor.
  const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(output_node, output_index, false);
  tensor->set_device_address(device_tensor);

  return tensor;
}
}  // namespace

void OutputActor::Init() {
  // Set the number of actor running dependent messages.
  if ((!need_loop_count_)) {
    running_dependent_msg_num_ = SizeToInt(outputs_num_ - device_tensor_store_keys_.size());
  }
}

void OutputActor::CollectLoopCount(size_t loop_count, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  current_count_ = loop_count;
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
    }

    current_outputs_num_ = 0;
    current_count_ = 0;
    SET_OPCONTEXT_SUCCESS_RET((*context));
  }
}

void OutputActor::UpdateOutputDeviceAddress() {
  // In the running end, when the device tensor of graph output node is set into host tensor, the graph output node
  // need be set new device tensor, to avoid that the device tensor context of host tensor be rewritten in the next
  // step or next loop. But the graph output nodes corresponding to device tensor store need to be skipped, because
  // they are fixed addresses and persistent.
  for (size_t i = 0; i < output_nodes_.size(); ++i) {
    auto &output_node = output_nodes_[i].first;
    auto output_index = output_nodes_[i].second;
    if ((output_node != nullptr) && (!IsPersistentDeviceTensor(output_node))) {
      const auto &device_tensor = AnfAlgo::GetMutableOutputAddr(output_node, output_index, false);
      // The outputs may have the same output node, so need skip when the node has been set to new device tensor.
      if ((device_tensor == nullptr) || (device_tensor->GetPtr() == nullptr)) {
        continue;
      }
      const auto &device_context = device_contexts_[i];
      MS_EXCEPTION_IF_NULL(device_context);
      auto new_device_tensor = device_context->CreateDeviceAddress(nullptr, device_tensor->GetSize(),
                                                                   device_tensor->format(), device_tensor->type_id());
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      new_device_tensor->set_original_ref_count(device_tensor->original_ref_count());
      new_device_tensor->ResetRefCount();
      AnfAlgo::SetOutputAddr(new_device_tensor, output_index, output_node.get());
    }
  }

  output_nodes_.clear();
  output_nodes_.resize(outputs_num_);
}

void OutputActor::CollectOutput(const AnfNodePtr &output_node, size_t output_index, size_t output_position,
                                OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(context);
  // Collect the output result in the last loop which is represented by "loop_count_ - current_count_ == 1".
  if (loop_count_ - current_count_ != 1) {
    return;
  }

  if (output_position >= outputs_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is of range.");
  }

  auto tensor = CreateOutputTensor(output_node, output_index, output_position);
  MS_EXCEPTION_IF_NULL(tensor);
  tensor->set_need_release_device_mem(true);
  outputs_[output_position] = tensor;
  current_outputs_num_++;

  // Save the output nodes to clear the device tensor in the running end.
  output_nodes_[output_position] = KernelWithIndex(output_node, output_index);

  // There is no loop count actor in step mode, need trigger call CollectLoopCount to replace old output device tensors.
  if (!need_loop_count_ && (current_outputs_num_ + device_tensor_store_keys_.size() == outputs_num_)) {
    CollectLoopCount(++current_count_, context);
  }
}
}  // namespace runtime
}  // namespace mindspore
