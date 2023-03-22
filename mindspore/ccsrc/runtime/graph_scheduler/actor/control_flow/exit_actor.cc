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

#include <algorithm>
#include "runtime/graph_scheduler/actor/control_flow/exit_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace runtime {
void ExitActor::Init() {
  // Init output data in base class.
  ControlActor::Init();

  // Init output data in each output branch.
  for (size_t i = 0; i < output_branch_data_arrows_.size(); ++i) {
    auto &output_branch_data_arrows = output_branch_data_arrows_[i];
    for (auto &data_arrow : output_branch_data_arrows) {
      MS_EXCEPTION_IF_NULL(data_arrow);
      auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
      (void)output_branch_data_[i].emplace_back(data_arrow->from_output_index_, std::move(data));
    }
  }

  // Check device contexts number.
  if (device_contexts_.size() != input_device_tensors_.size()) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
}

void ExitActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::FetchInput(context);
  CopyDeviceAddress(context);

  auto data_iter = output_branch_data_.find(output_branch_id_);
  if (data_iter != output_branch_data_.end()) {
    for (auto &output_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(output_data.second);
      if (output_data.first >= input_device_tensors_.size()) {
        MS_LOG(EXCEPTION) << "Invalid from index:" << output_data.first << " for actor:" << GetAID()
                          << " to actor:" << output_data.second->op_id_ << " to index:" << output_data.second->index_;
      }
      MS_EXCEPTION_IF_NULL(input_device_tensors_[output_data.first]);
      output_data.second->data_ = input_device_tensors_[output_data.first];
    }
  }
}

void ExitActor::SendOutput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // Before the exit actor sends output, it is necessary to ensure that all reference count calculations in the
  // graph are completed, otherwise the device tensor in the free memory list will be overwritten the next time
  // it is executed, resulting in multiple releases of ptr.
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::Wait, context, GetAID());
}

void ExitActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (IsRunningFailed(context)) {
    return;
  }

  // 1.Send output in base class.
  ControlActor::SendOutput(context);

  // 2.Send output data in output branch.
  const auto &branch_data_iter = output_branch_data_.find(output_branch_id_);
  if (branch_data_iter != output_branch_data_.end()) {
    for (const auto &output_data : branch_data_iter->second) {
      MS_EXCEPTION_IF_NULL(output_data.second);
      ActorDispatcher::Send(output_data.second->op_id_, &OpActor::RunOpData, output_data.second.get(), context);
    }
  }

  // 3.Send output control in output branch.
  const auto &control_iter = output_branch_control_arrows_.find(output_branch_id_);
  if (control_iter != output_branch_control_arrows_.end()) {
    auto source_aid = const_cast<AID *>(&GetAID());
    for (const auto &control_arrow : control_iter->second) {
      ActorDispatcher::Send(control_arrow, &OpActor::RunOpControl, source_aid, context);
    }
  }

  // 3.Send output partial in output branch.
  const auto &partial_iter = output_branch_partial_arrows_.find(output_branch_id_);
  if (partial_iter != output_branch_partial_arrows_.end()) {
    for (const auto &arrow : partial_iter->second) {
      MS_EXCEPTION_IF_NULL(arrow);
      if (IntToSize(arrow->from_output_index_) >= input_partials_.size()) {
        std::string error_info = "Invalid partial input:" + std::to_string(arrow->from_output_index_) +
                                 " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      auto output_partial = input_partials_[IntToSize(arrow->from_output_index_)];
      ActorDispatcher::Send(arrow->to_op_id_, &ControlActor::RunOpPartial, output_partial,
                            IntToSize(arrow->to_input_index_), context);
    }
  }
}

void ExitActor::IncreaseDynamicRefCounts(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::IncreaseDynamicRefCounts(context);

  // Increase dynamic ref count by the output data in output branch.
  if (output_branch_data_.count(output_branch_id_) > 0) {
    for (auto &output_data : output_branch_data_[output_branch_id_]) {
      MS_EXCEPTION_IF_NULL(output_data.second);
      IncreaseDynamicRefCount(output_data.second.get());
    }
  }

  // Increase dynamic ref count by the output partial in output branch.
  if (output_branch_partial_arrows_.count(output_branch_id_) > 0) {
    for (const auto &partial_arrow : output_branch_partial_arrows_[output_branch_id_]) {
      MS_EXCEPTION_IF_NULL(partial_arrow);
      if (IntToSize(partial_arrow->from_output_index_) >= input_partials_.size()) {
        std::string error_info = "Invalid partial input:" + std::to_string(partial_arrow->from_output_index_) +
                                 " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      auto output_partial = input_partials_[IntToSize(partial_arrow->from_output_index_)];
      IncreaseDynamicRefCount(output_partial);
    }
  }
  if (input_device_tensors_.size() != device_contexts_.size()) {
    MS_LOG(ERROR) << "Input device tensor size:" << input_device_tensors_.size()
                  << " is not equal to context size:" << device_contexts_.size() << " for actor:" << GetAID();
  }
  // The input device tensor may not have users and needs to free the memory.
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    if ((input_device_tensors_[i] != nullptr) && (input_device_tensors_[i]->dynamic_ref_count() == 0) &&
        (device_contexts_[i] != nullptr)) {
      MS_LOG(INFO) << GetAID().Name() << " input index:" << i << " has no user and free the memory.";
      // Update the real used device context by the input data.
      if (device_contexts_[i]->GetDeviceType() != input_device_tensors_[i]->GetDeviceType()) {
        device_contexts_[i] = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
          {input_device_tensors_[i]->device_name(), input_device_tensors_[i]->device_id()});
        MS_LOG(INFO) << "Update device context type to:" << device_contexts_[i]->GetDeviceType();
      }
      device_contexts_[i]->device_res_manager_->FreeMemory(input_device_tensors_[i]);
    }
  }
}

void ExitActor::CopyDeviceAddress(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // If node is not empty, it is the exit of funcgraph, no need to create device address.
  if (node_ != nullptr) {
    return;
  }
  if (input_device_tensors_.size() != is_need_copy_device_tensors_.size() ||
      input_device_tensors_.size() != is_dynamic_shapes_.size() ||
      input_device_tensors_.size() != device_contexts_.size()) {
    std::string error_info = "Invalid input device tensor size:" + std::to_string(input_device_tensors_.size()) +
                             " need tensor size:" + std::to_string(is_need_copy_device_tensors_.size()) +
                             " need dynamic shape size:" + std::to_string(is_dynamic_shapes_.size()) +
                             " need context size:" + std::to_string(device_contexts_.size()) +
                             " for actor:" + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  std::vector<DeviceTensor *> new_device_tensors;
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    auto &input_device_tensor = input_device_tensors_[i];
    if ((input_device_tensor == nullptr) || (!is_need_copy_device_tensors_[i])) {
      (void)new_device_tensors.emplace_back(input_device_tensor);
      continue;
    }

    // Update the real used device context by the input data.
    auto &device_context = device_contexts_[i];
    MS_EXCEPTION_IF_NULL(device_context);
    if (device_context->GetDeviceType() != input_device_tensor->GetDeviceType()) {
      device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {input_device_tensor->device_name(), input_device_tensor->device_id()});
      MS_LOG(INFO) << "Update device context type to:" << device_context->GetDeviceType();
    }

    const KernelWithIndex &node_with_index = input_device_tensor->GetNodeIndex();
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    // Create the new device tensor to take over the input_device_tensors which are the outputs of kernel graphs.
    DeviceTensorPtr new_device_tensor = nullptr;
    if (!is_dynamic_shapes_[i]) {
      new_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, input_device_tensor->GetSize(), input_device_tensor->format(), input_device_tensor->type_id(),
        input_device_tensor->host_shape());
    } else {
      // If there is a dynamic shape, the shape in the kernel should be used.
      MS_LOG(DEBUG) << "Update dynamic shape in kernel output:" << node_with_index.first->DebugString()
                    << " for actor:" << GetAID();
      const auto &host_shape = common::AnfAlgo::GetOutputInferShape(node_with_index.first, node_with_index.second);
      new_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, input_device_tensor->GetSize(), input_device_tensor->format(), input_device_tensor->type_id(),
        host_shape);
    }
    MS_EXCEPTION_IF_NULL(new_device_tensor);
    (void)created_device_tensors_.emplace_back(new_device_tensor);
    (void)new_device_tensors.emplace_back(new_device_tensor.get());
    new_device_tensor->SetNodeIndex(node_with_index.first, node_with_index.second);
    new_device_tensor->set_from_persistent_mem(input_device_tensor->from_persistent_mem());
    // The device address which is created by actor uses the dynamic ref count.
    new_device_tensor->set_dynamic_ref_count(0);
    new_device_tensor->set_original_ref_count(SIZE_MAX);
    new_device_tensor->ResetRefCount();

    // If the address ptr can't be changed, then alloc the new device memory and copy the data.
    if (input_device_tensor->is_ptr_persisted()) {
      device::DynamicMemAllocatorDebugInfo::SetDebugInfo(GetAID().Name(), device::AllocatorType::kOther);
      if (!device_context->device_res_manager_->AllocateMemory(new_device_tensor.get())) {
        SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, *device_context,
                                                    GetAID().Name(), new_device_tensor->GetSize());
      }
      if (!new_device_tensor->SyncDeviceToDevice(input_device_tensor)) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Sync device to device failed.");
      }
    } else {
      // Move the device ptr from input_device_tensor to new_device_tensor.
      input_device_tensor->Swap(new_device_tensor.get());
    }
    MS_LOG(DEBUG) << GetAID().Name() << " creates the dynamic ref device address:" << new_device_tensor.get()
                  << ", ptr:" << new_device_tensor->GetPtr()
                  << ", from node:" << node_with_index.first->fullname_with_scope()
                  << " with index:" << node_with_index.second;
  }
  input_device_tensors_.swap(new_device_tensors);

  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    if (output_data_by_output_index_[i].empty()) {
      continue;
    }

    const auto &device_tensor = input_device_tensors_[i];
    MS_EXCEPTION_IF_NULL(device_tensor);
    for (auto &output_data : output_data_by_output_index_[i]) {
      MS_EXCEPTION_IF_NULL(output_data);
      output_data->data_ = device_tensor;
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
