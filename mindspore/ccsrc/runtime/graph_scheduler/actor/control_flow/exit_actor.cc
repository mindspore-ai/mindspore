/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

      // Identify whether the output data flag is kOutputDataFlagToStack.
      bool is_to_stack = (data_arrow->to_op_id_.Name().find(kStackActorNameSuffix) != std::string::npos);
      size_t output_data_flag = is_to_stack ? kOutputDataFlagToStack : kOutputDataFlagInit;
      (void)output_branch_data_flag_[i].emplace_back(output_data_flag);
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

  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  CopyDeviceAddress(context);
  if (output_branch_dynamic_len_index_.find(output_branch_id_) == output_branch_dynamic_len_index_.end()) {
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
  } else {
    // The branch id need merge device address.
    MS_LOG(DEBUG) << "Exit actor:" << GetAID() << " merge output";
    MergeDynamiclenDeviceAddress(context);
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

  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kSendOutput, GetAID().Name());
  // 2.Send output data in output branch.
  const auto &branch_data_iter = output_branch_data_.find(output_branch_id_);
  if (branch_data_iter != output_branch_data_.end()) {
    MS_EXCEPTION_IF_CHECK_FAIL((output_branch_data_flag_.count(output_branch_id_) > 0),
                               "The output branch id is invalid.");
    const auto &output_data_flags = output_branch_data_flag_[output_branch_id_];
    MS_EXCEPTION_IF_CHECK_FAIL((output_data_flags.size() == branch_data_iter->second.size()),
                               "The output data flag size is wrong.");
    for (size_t i = 0; i < branch_data_iter->second.size(); ++i) {
      const auto &output_data = branch_data_iter->second[i];
      MS_EXCEPTION_IF_NULL(output_data.second);
      // Create a new op data for stack actor.
      if (TEST_FLAG(output_data_flags[i], kOutputDataFlagToStack)) {
        auto to_stack_data = std::make_unique<OpData<DeviceTensor>>(
          output_data.second->op_id_, output_data.second->data_, output_data.second->index_);
        (void)to_stack_data_.emplace_back(std::move(to_stack_data));
        ActorDispatcher::Send(output_data.second->op_id_, &OpActor::RunOpData, to_stack_data_.back().get(), context);
      } else {
        ActorDispatcher::Send(output_data.second->op_id_, &OpActor::RunOpData, output_data.second.get(), context);
      }
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

  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
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

void ExitActor::MergeDynamiclenDeviceAddress(OpContext<DeviceTensor> *const context) {
  if (output_branch_dynamic_len_index_.find(output_branch_id_) == output_branch_dynamic_len_index_.end()) {
    return;
  }
  auto real_indexes = output_branch_dynamic_len_index_[output_branch_id_];
  std::vector<OpPartialPtr> new_partials;
  std::vector<DeviceTensor *> new_device_tensors;
  // Collect the new output of actor, merge the device address for dynamic len.
  for (size_t i = 0; i < real_indexes.size(); ++i) {
    const auto &indexes = real_indexes[i].first;
    if (real_indexes[i].second) {
      std::vector<DeviceTensor *> addr_list;
      for (size_t index : indexes) {
        if (index > input_device_tensors_.size()) {
          std::string error_info = "Invalid real index:" + std::to_string(index) + " for index:" + std::to_string(i) +
                                   " total size:" + std::to_string(input_device_tensors_.size()) +
                                   " for actor:" + GetAID().Name();
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
        }
        if (input_device_tensors_[index] == nullptr) {
          std::string error_info =
            "Invalid input device address index:" + std::to_string(index) + " for index:" + std::to_string(i) +
            " total size:" + std::to_string(input_device_tensors_.size()) + " for actor:" + GetAID().Name();
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
        }
        addr_list.emplace_back(input_device_tensors_[index]);
      }
      DeviceTensor *new_device_tensor = nullptr;
      MergeDeviceAddress(context, addr_list, &new_device_tensor);
      new_device_tensors.emplace_back(new_device_tensor);
      new_partials.emplace_back(nullptr);
    } else if (indexes.empty() || indexes[0] >= input_partials_.size()) {
      std::string error_info = "Invalid index num:" + std::to_string(indexes.size()) +
                               " for index:" + std::to_string(i) + " for actor:" + GetAID().Name();
      MS_LOG(WARNING) << error_info;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    } else if (input_partials_[indexes[0]] != nullptr) {
      new_device_tensors.emplace_back(nullptr);
      new_partials.emplace_back(input_partials_[indexes[0]]);
    } else if (input_device_tensors_[indexes[0]] != nullptr) {
      new_device_tensors.emplace_back(input_device_tensors_[indexes[0]]);
      new_partials.emplace_back(nullptr);
    } else {
      std::string error_info = "Failed to get input for real index:" + std::to_string(indexes[0]) +
                               " for index:" + std::to_string(i) + " for actor:" + GetAID().Name();
      MS_LOG(WARNING) << error_info;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }
  auto data_iter = output_branch_data_.find(output_branch_id_);
  if (data_iter != output_branch_data_.end()) {
    for (auto &output_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(output_data.second);
      if (output_data.first >= new_device_tensors.size()) {
        MS_EXCEPTION_IF_NULL(output_data.second);
        MS_LOG(EXCEPTION) << "Invalid from index:" << output_data.first << " for actor:" << GetAID()
                          << " to actor:" << output_data.second->op_id_ << " to index:" << output_data.second->index_;
      }
      MS_EXCEPTION_IF_NULL(new_device_tensors[output_data.first]);
      output_data.second->data_ = new_device_tensors[output_data.first];
    }
  }
}

namespace {
void SetFromMemPoolFlag(const DeviceTensorPtr &device_tensor, size_t to_index,
                        const std::vector<std::pair<AID, DataArrow *>> &input_data_arrow_aids) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  const auto &iter =
    std::find_if(input_data_arrow_aids.begin(), input_data_arrow_aids.end(), [to_index](const auto &pair) {
      return pair.second != nullptr && pair.second->to_input_index_ == SizeToInt(to_index);
    });
  if (iter != input_data_arrow_aids.end() &&
      iter->first.Name().find(kAnyTypeKernelActorNameSuffix) != std::string::npos) {
    MS_LOG(DEBUG) << "Set from memory pool flag for ptr:" << device_tensor->GetPtr()
                  << " in device address:" << device_tensor;
    device_tensor->set_from_mem_pool(true);
  }
}
}  // namespace

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
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
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
        input_device_tensor->host_shape(), input_device_tensor->user_data());
      MS_LOG(DEBUG) << "Create device tensor:" << new_device_tensor << " type:" << new_device_tensor->type_id();
    } else {
      // If there is a dynamic shape, the shape in the kernel should be used.
      MS_LOG(DEBUG) << "Update dynamic shape in kernel output:" << node_with_index.first->DebugString()
                    << " for actor:" << GetAID();
      const auto &host_shape = common::AnfAlgo::GetOutputInferShape(node_with_index.first, node_with_index.second);
      new_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(
        nullptr, input_device_tensor->GetSize(), input_device_tensor->format(), input_device_tensor->type_id(),
        host_shape, input_device_tensor->user_data());
      MS_LOG(DEBUG) << "Create device tensor:" << new_device_tensor << " type:" << new_device_tensor->type_id();
    }
    MS_EXCEPTION_IF_NULL(new_device_tensor);
    const auto &swap_manager = device_context->device_res_manager_->swap_manager();
    if (swap_manager != nullptr) {
      swap_manager->AddSwappableTensor(new_device_tensor);
    }
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
      if (!new_device_tensor->from_mem_pool()) {
        SetFromMemPoolFlag(new_device_tensor, i, input_data_arrow_aids_);
      }
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
