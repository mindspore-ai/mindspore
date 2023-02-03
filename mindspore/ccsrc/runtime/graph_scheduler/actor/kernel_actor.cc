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

#include "runtime/graph_scheduler/actor/kernel_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/recorder_actor.h"
#include "runtime/graph_scheduler/actor/debug_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"
#include "distributed/recovery/recovery_context.h"
#include "distributed/collective/collective_manager.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace runtime {
namespace {
bool IsSomasEnable(const SomasInfo *somas_info) {
  return ((somas_info != nullptr) && (somas_info->whole_block_size_ != 0));
}
}  // namespace

using distributed::collective::CollectiveManager;
using distributed::recovery::RecoveryContext;

void KernelActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

  MS_EXCEPTION_IF_NULL(kernel_);
  real_input_num_ = common::AnfAlgo::GetInputTensorNum(kernel_);
  kernel_info_ = dynamic_cast<KernelInfo *>(kernel_->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info_);
  is_dynamic_shape_ = common::AnfAlgo::IsDynamicShape(kernel_);
  if (is_dynamic_shape_ && IsSomasEnable(somas_info_)) {
    MS_LOG(EXCEPTION) << "Not support the somas for the dynamic shape: " << GetAID().Name();
  }

  // Init the device tensors and kernel launch info.
  InitInputInfo();
  InitOutputInfo();
  InitWorkspaceInfo();

  // Init the output data.
  InitOutputData();
  if (output_data_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "The output data size is wrong: " << GetAID().Name();
  }
  size_t output_data_index = 0;
  for (auto &data_arrow : output_data_arrows_) {
    auto data = output_data_[output_data_index].first.get();
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) >= output_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID().Name();
    }
    data->data_ = output_device_tensors_[IntToSize(data_arrow->from_output_index_)];
    ++output_data_index;
  }
}

void KernelActor::InitInputInfo() {
  for (size_t i = 0; i < real_input_num_; ++i) {
    const auto &input_device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel_, i, false);
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    (void)real_input_data_infos_.emplace_back(
      std::make_shared<InputDataInfo>(input_device_tensor->format(), input_device_tensor->host_shape(),
                                      input_device_tensor->GetSize(), input_device_tensor->type_id()));
  }

  copy_input_device_tensors_.resize(real_input_num_);
  input_device_tensors_.resize(real_input_num_);
  for (auto &input_address : input_device_tensors_) {
    (void)memory_free_list_.emplace_back(input_address);
    (void)launch_info_.inputs_.emplace_back(std::make_shared<Address>());
  }
}

void KernelActor::InitOutputInfo() {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  const auto &output_addresses = kernel_info_->output_address_list();
  const auto &somas_outputs = kernel_info_->somas_output_result();
  bool output_need_somas = false;
  for (size_t i = 0; i < output_addresses.size(); ++i) {
    auto &output_address = output_addresses[i];
    MS_EXCEPTION_IF_NULL(output_address);
    (void)output_device_tensors_.emplace_back(output_address.get());
    (void)launch_info_.outputs_.emplace_back(std::make_shared<Address>());

    // The output taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, i)) {
      // Somas outputs use the info of kernelMod, and output address use the info of device address.
      if (somas_outputs[i].second < output_address->GetSize()) {
        MS_LOG(INFO) << GetAID().Name() << " check somas size warning, output index:" << i
                     << " somas aligned size:" << somas_outputs[i].second
                     << " is smaller than address size:" << output_address->GetSize();
      }
      UpdateRefCount(output_address.get(), true);
      output_need_somas = true;
    } else {
      (void)memory_alloc_list_.emplace_back(output_address.get());
      (void)memory_free_list_.emplace_back(output_address.get());
    }
  }

  if (output_need_somas && (!IsSomasEnable(somas_info_))) {
    MS_LOG(EXCEPTION) << "The somas is not enable for: " << GetAID().Name();
  }

  for (auto &external_reference_tensor : external_reference_tensors_) {
    (void)memory_free_list_.emplace_back(external_reference_tensor);
  }
}

void KernelActor::InitWorkspaceInfo() {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  // The size of workspace maybe changed in dynamic shape, so put workspace_address in the end of memory_alloc_list_ and
  // memory_free_list_, for the operation of dynamic_shape condition in FetchWorkspaceDeviceTensor.
  const auto &workspace_addresses = kernel_info_->workspace_address_list();
  const auto &somas_workspace = kernel_info_->somas_workspace_result();
  bool workspace_need_somas = false;
  for (size_t i = 0; i < workspace_addresses.size(); ++i) {
    auto &workspace_address = workspace_addresses[i];
    MS_EXCEPTION_IF_NULL(workspace_address);
    (void)workspace_device_tensors_.emplace_back(workspace_address.get());
    (void)launch_info_.workspaces_.emplace_back(std::make_shared<Address>());

    // The workspace taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_workspace, i)) {
      if (somas_workspace[i].second < workspace_address->GetSize()) {
        MS_LOG(INFO) << GetAID().Name() << " check somas size warning, workspace index:" << i
                     << " somas aligned size:" << somas_workspace[i].second
                     << " is smaller than address size:" << workspace_address->GetSize();
      }
      UpdateRefCount(workspace_address.get(), true);
      workspace_need_somas = true;
    } else {
      (void)memory_alloc_list_.emplace_back(workspace_address.get());
      (void)memory_free_list_.emplace_back(workspace_address.get());
    }
  }

  if (workspace_need_somas && (!IsSomasEnable(somas_info_))) {
    MS_LOG(EXCEPTION) << "The somas is not enable for: " << GetAID().Name();
  }
}

void KernelActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  FetchInputDeviceTensor(context);
  FetchOutputDeviceTensor(context);
  if (is_dynamic_shape_) {
    FetchWorkspaceDeviceTensor();
  }

  // Set the memory address for the tensors which use the somas.
  SetSomasMemory(context);

  // Allocate the memory address for other tensors which don't use the somas.
  if (memory_alloc_list_.size() > 0) {
    SendMemoryAllocReq(context);
  } else {
    OnMemoryAllocFinish(context);
  }
}

void KernelActor::FetchWorkspaceDeviceTensor() {
  MS_LOG(DEBUG) << "Start FetchWorkspaceDeviceTensor.";
  MS_EXCEPTION_IF_NULL(kernel_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel_);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
  // Resize of workspace_device_tensors_, memory_alloc_list_, memory_free_list_ and launch_info_.workspaces_, because of
  // the dynamic size of workspace.
  if (launch_info_.workspaces_.size() > workspace_sizes.size()) {
    size_t size = launch_info_.workspaces_.size() - workspace_sizes.size();
    (void)workspace_device_tensors_.erase(workspace_device_tensors_.end() - size, workspace_device_tensors_.end());
    (void)launch_info_.workspaces_.erase(launch_info_.workspaces_.end() - size, launch_info_.workspaces_.end());

    MS_EXCEPTION_IF_CHECK_FAIL((memory_alloc_list_.size() >= size), "The memory alloc list size is wrong.");
    MS_EXCEPTION_IF_CHECK_FAIL((memory_free_list_.size() >= size), "The memory free list size is wrong.");
    (void)memory_alloc_list_.erase(memory_alloc_list_.end() - size, memory_alloc_list_.end());
    (void)memory_free_list_.erase(memory_free_list_.end() - size, memory_free_list_.end());
  } else if (launch_info_.workspaces_.size() < workspace_sizes.size()) {
    if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
      MS_LOG(ERROR) << "Invalid device context for kernel actor:" + GetAID().Name();
      return;
    }
    for (size_t i = launch_info_.workspaces_.size(); i < workspace_sizes.size(); ++i) {
      auto device_address = device_contexts_[0]->device_res_manager_->CreateDeviceAddress(
        nullptr, workspace_sizes[i], "", kTypeUnknown, ShapeVector());
      MS_LOG(DEBUG) << "Create addr for node:" << common::AnfAlgo::GetNodeDebugString(kernel_)
                    << " addr:" << device_address;
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel_.get());  // set to kernel_info
      MS_EXCEPTION_IF_NULL(device_address);
      (void)workspace_device_tensors_.emplace_back(device_address.get());
      (void)launch_info_.workspaces_.emplace_back(std::make_shared<Address>());
      (void)memory_alloc_list_.emplace_back(device_address.get());
      (void)memory_free_list_.emplace_back(device_address.get());
    }
  }
  // Set workspace address new size
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    workspace_device_tensors_[i]->SetSize(workspace_sizes[i]);
  }
}

namespace {
void AllocateMemory(const std::vector<DeviceTensor *> &alloc_list, const DeviceContext *device_context,
                    OpContext<DeviceTensor> *const context, const std::string &actor_name) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(context);

  for (auto &device_tensor : alloc_list) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if ((device_tensor->GetPtr() != nullptr) || (device_tensor->GetSize() == 0)) {
      continue;
    }
    // Allocate memory through the device context.
    if (!device_context->device_res_manager_->AllocateMemory(device_tensor)) {
      SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kStep, *context, *device_context, actor_name,
                                                  device_tensor->GetSize());
    }
  }
}

void FreeMemory(const std::vector<DeviceTensor *> &free_list, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  for (auto &device_tensor : free_list) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (device_tensor->original_ref_count() == SIZE_MAX) {
      continue;
    }
    // The reference count is decremented to zero to free memory, and reset to the original count.
    device_tensor->DecreaseRefCount();
    if (device_tensor->ref_count() == 0) {
      // Free memory through the device context.
      if (device_tensor->GetPtr() != nullptr) {
        device_context->device_res_manager_->FreeMemory(device_tensor);
      }
      device_tensor->ClearUserData();
      device_tensor->ResetRefCount();
    }
  }
}
}  // namespace

void KernelActor::SetSomasMemory(OpContext<DeviceTensor> *const context) const {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(kernel_info_);
  if (!IsSomasEnable(somas_info_)) {
    return;
  }

  // Set the memory address for the output tensors which use the somas.
  const auto &somas_outputs = kernel_info_->somas_output_result();
  MS_EXCEPTION_IF_CHECK_FAIL((output_device_tensors_.size() >= somas_outputs.size()), "The output num is wrong.");
  for (size_t i = 0; i < somas_outputs.size(); ++i) {
    if (somas_outputs[i].second > 0) {
      auto device_ptr = GetSomasDevicePtr(somas_outputs[i].first);
      if (device_ptr == nullptr) {
        std::string error_info = GetAID().Name() + " get nullptr somas device for output index: " + std::to_string(i);
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, error_info);
      }
      // In this scenario, the Init function can ensure that the pointer of the relevant operation is not nullptr.
      // In order to perform performance, the pointer validity is not checked here.
      output_device_tensors_[i]->set_ptr(device_ptr);
    }
  }

  // Set the memory address for the workspace tensors which use the somas.
  const auto &somas_workspace = kernel_info_->somas_workspace_result();
  MS_EXCEPTION_IF_CHECK_FAIL((workspace_device_tensors_.size() >= somas_workspace.size()), "The output num is wrong.");
  for (size_t i = 0; i < somas_workspace.size(); ++i) {
    if (somas_workspace[i].second > 0) {
      auto device_ptr = GetSomasDevicePtr(somas_workspace[i].first);
      if (device_ptr == nullptr) {
        std::string error_info = GetAID().Name() + " get nullptr somas device for workspace index:" + std::to_string(i);
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, error_info);
      }
      // In this scenario, the Init function can ensure that the pointer of the relevant operation is not nullptr.
      // In order to perform performance, the pointer validity is not checked here.
      workspace_device_tensors_[i]->set_ptr(device_ptr);
    }
  }
}

void *KernelActor::GetSomasDevicePtr(size_t offset) const {
  MS_EXCEPTION_IF_NULL(somas_info_);
  // Get the ptr from the whole block.
  if (somas_info_->base_address_ != nullptr) {
    return AddressOffset(somas_info_->base_address_, offset);
  }

  // Get the ptr from the merged blocks.
  auto iter = somas_info_->merged_base_addresses_.upper_bound(offset);
  if (iter == somas_info_->merged_base_addresses_.begin()) {
    MS_LOG(ERROR) << GetAID().Name() << " can't find the merged block for offset: " << offset;
    return nullptr;
  }
  --iter;
  MS_EXCEPTION_IF_CHECK_FAIL((offset >= iter->first), "The offset is smaller than the merged block offset.");
  size_t real_offset = offset - iter->first;
  void *real_base_address = iter->second;
  if (real_base_address == nullptr) {
    MS_LOG(ERROR) << GetAID().Name() << " doesn't allocate the merged block base address for offset: " << iter->first;
    return nullptr;
  }
  return AddressOffset(real_base_address, real_offset);
}

void KernelActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  running_dependent_msg_num_ = 1;
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context),
                                                  "Invalid device context for kernel actor:" + GetAID().Name());
  }

  if (strategy_ == GraphExecutionStrategy::kPipeline) {
    if (ActorDispatcher::is_memory_allocation_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_,
                                device_contexts_[0], context, GetAID());
      OnMemoryAllocFinish(context);
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_,
                            device_contexts_[0], context, GetAID());
    }
  } else {
    AllocateMemory(memory_alloc_list_, device_contexts_[0], context, GetAID().Name());
  }
}

void KernelActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    MS_LOG(EXCEPTION) << "Invalid device context for kernel actor:" << GetAID();
  }
  if (strategy_ == GraphExecutionStrategy::kPipeline) {
    if (ActorDispatcher::is_memory_free_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &memory_free_list_,
                                device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &memory_free_list_,
                            device_contexts_[0], context, GetAID());
    }
  } else {
    FreeMemory(memory_free_list_, device_contexts_[0]);
  }

  // Free the address that is the temp store for kernel input copy.
  for (auto &copy_input_device_tensor : copy_input_device_tensors_) {
    if ((copy_input_device_tensor != nullptr) && (copy_input_device_tensor->GetPtr() != nullptr)) {
      device_contexts_[0]->device_res_manager_->FreeMemory(copy_input_device_tensor.get());
    }
  }
}

void KernelActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(kernel_);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  if (IsRunningFailed(context)) {
    return;
  }
  PreLaunchKernel(context);

  try {
    if (RecoveryContext::GetInstance()->enable_recovery() && CollectiveManager::instance()->need_reinit()) {
      // In disaster recovery scenarios, run dag in this step failed, the rest operators of graph do not need launch,
      // especially the collective communication operators.
      MS_LOG(WARNING) << "Collective communication need reinitialize, skip launch kernel: "
                      << kernel_->fullname_with_scope();
    } else if (!IsSkippedLaunch(kernel_, nullptr)) {
      auto ret = LaunchKernel(context);
      if (!ret) {
        std::string error_info = "Launch kernel failed: " + kernel_->fullname_with_scope();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
      }
    }
  } catch (const std::exception &e) {
    if (strategy_ == GraphExecutionStrategy::kPipeline) {
      MsException::Instance().SetException();
    }
    std::string error_info = "Launch kernel exception: " + kernel_->fullname_with_scope();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
  }

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr && strategy_ == GraphExecutionStrategy::kPipeline) {
    SendDebugReq(context);
    return;
  }

  PostLaunchKernel(context);
}

void KernelActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  running_dependent_msg_num_ = 1;
  ActorDispatcher::SendSync(*debug_aid_, &DebugActor::Debug, kernel_, &launch_info_, device_contexts_[0], context,
                            &GetAID());
  OnDebugFinish(context);
}

void KernelActor::OnDebugFinish(OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  PostLaunchKernel(context);
}

void KernelActor::PushInputDeviceTensor(const std::vector<TensorPtr> *input_tensors) {
  MS_EXCEPTION_IF_NULL(input_tensors);
  if (input_tensors->size() != real_input_num_) {
    MS_LOG(ERROR) << "Input tensor number: " << input_tensors->size()
                  << " is not equal to kernel's input size: " << real_input_num_;
    return;
  }

  for (size_t input_index = 0; input_index < input_tensors->size(); input_index++) {
    const auto &input_tensor = (*input_tensors)[input_index];
    MS_EXCEPTION_IF_NULL(input_tensor);
    const auto &device_tensor = std::dynamic_pointer_cast<DeviceTensor>(input_tensor->device_address());
    if (device_tensor != nullptr) {
      input_device_tensors_[input_index] = device_tensor.get();
      memory_free_list_[input_index] = device_tensor.get();
    }
  }
}

void KernelActor::CopyInputDeviceTensor(const OpData<DeviceTensor> *input_data,
                                        OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(input_data);
  MS_EXCEPTION_IF_NULL(input_data->data_);
  MS_EXCEPTION_IF_NULL(context);
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context),
                                                  "Invalid device context for kernel actor:" + GetAID().Name());
  }
  size_t input_data_index = IntToSize(input_data->index_);
  if (input_data_index >= real_input_data_infos_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The input index is of range.");
  }
  auto &real_input_info = real_input_data_infos_[input_data_index];
  MS_EXCEPTION_IF_NULL(real_input_info);
  if ((input_data->data_->GetDeviceType() == device_contexts_[0]->GetDeviceType()) &&
      AnfAlgo::IsEquivalentFormat(input_data->data_->format(), real_input_info->format_)) {
    return;
  }

  if (inputs_continuous_memory_) {
    std::string error_info = GetAID().Name() + " inputs must be continuous memory and can't be copied for index " +
                             std::to_string(input_data_index);
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
  }
  if (input_data_index >= copy_input_device_tensors_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The input index is of range.");
  }
  if (copy_input_device_tensors_[input_data_index] == nullptr) {
    copy_input_device_tensors_[input_data_index] = device_contexts_[0]->device_res_manager_->CreateDeviceAddress(
      nullptr, real_input_info->size_, real_input_info->format_, real_input_info->type_id_, real_input_info->shape_);
  }
  auto &new_device_tensor = copy_input_device_tensors_[input_data_index];
  MS_EXCEPTION_IF_NULL(new_device_tensor);
  // Dynamic shape need update size.
  if (IsDynamic(real_input_info->shape_) || common::AnfAlgo::IsDynamicSequence(kernel_)) {
    new_device_tensor->SetSize(input_data->data_->GetSize());
  }
  // Update the input device tensor.
  input_device_tensors_[input_data_index] = new_device_tensor.get();

  device::DynamicMemAllocatorDebugInfo::SetDebugInfo(GetAID().Name(), device::AllocatorType::kKernelOutput,
                                                     input_data_index);
  if ((new_device_tensor->GetPtr() == nullptr) &&
      (!device_contexts_[0]->device_res_manager_->AllocateMemory(new_device_tensor.get()))) {
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy_, *context, *(device_contexts_[0]), GetAID().Name(),
                                                new_device_tensor->GetSize());
  }
  MS_LOG(INFO) << GetAID().Name() << " the input position:" << input_data_index
               << " copy from device address:" << input_data->data_ << ", type:" << input_data->data_->GetDeviceType()
               << ", format:" << input_data->data_->format() << " to device address:" << new_device_tensor.get()
               << ", type:" << new_device_tensor->GetDeviceType() << ", format:" << new_device_tensor->format();
  // Copy from the real parameter to formal parameter and insert the device tensor copy store.
  if (!Copy(new_device_tensor.get(), input_data->data_)) {
    std::string error_info = "Copy device tensor failed: " + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
  }
  if (modifiable_ref_input_indexes_.count(input_data->index_) > 0) {
    DeviceTensorCopyStore::GetInstance().Insert(new_device_tensor.get(), input_data->data_);
  }
}

void KernelActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context),
                                                  "Invalid device context for kernel actor:" + GetAID().Name());
  }

  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      size_t input_index = IntToSize(input_data->index_);
      if (input_index >= input_device_tensors_.size()) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), "The input index is out of range.");
      }

      if (input_device_tensors_[input_index] != input_data->data_) {
        input_device_tensors_[input_index] = input_data->data_;
        memory_free_list_[input_index] = input_data->data_;
      }
      CopyInputDeviceTensor(input_data, context);
    }
  }

  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto device_tensor = DeviceTensorStore::GetInstance()
                           .Fetch(device_tensor_store_key.second.get(), device_contexts_[0]->GetDeviceType())
                           .get();
    if (device_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
    }

    if (device_tensor_store_key.first >= input_device_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), "The input index is out of range.");
    }
    if (input_device_tensors_[device_tensor_store_key.first] != device_tensor) {
      input_device_tensors_[device_tensor_store_key.first] = device_tensor;
      memory_free_list_[device_tensor_store_key.first] = device_tensor;
    }
  }
}

void KernelActor::FetchOutputDeviceTensor(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  auto &output_addresses = kernel_info_->output_address_list();
  const auto &kernel_mod = kernel_info_->kernel_mod();
  MS_EXCEPTION_IF_NULL(kernel_mod);
  const auto &output_size_list = kernel_mod->GetOutputSizeList();

  // May exist in the kernel which does not support the dynamic shape.
  if (output_addresses.size() != output_size_list.size()) {
    std::string error_info = "The outputs number(" + std::to_string(output_size_list.size()) + ") is wrong, " +
                             GetAID().Name() + " may not support the dynamic shape, please check.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
  }

  const auto &somas_outputs = kernel_info_->somas_output_result();
  // Update the size of output device tensor.
  for (size_t i = 0; i < output_addresses.size(); ++i) {
    auto output_address = output_addresses[i].get();
    MS_EXCEPTION_IF_NULL(output_address);
    // The output device tensor can't be changed.
    if (output_device_tensors_[i] != output_address) {
      std::string error_info =
        "The device tensor can't be changed of " + GetAID().Name() + " with output index " + std::to_string(i);
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
    }

    if (output_size_list[i] == output_address->GetSize()) {
      continue;
    }

    // Somas doesn't support the variable size.
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, i) && (somas_outputs[i].second < output_size_list[i])) {
      std::string error_info =
        "Somas doesn't support variable size of " + GetAID().Name() + " with output index " + std::to_string(i) +
        ".  Suggest to turn off memory optimization by setting the context memory_optimize_level' to 'O0' ";
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
    }

    // 1. The size of output address may be changed in dynamic shape scenario.
    // 2. If the format of the DeviceAddress is different, then the size is originally different.
    //    Such as NCHW(1,1,1,3) and NC1HWC0(1,1,1,1,16). So we don't need to update the size.
    // 3. For example, we need to call cudnnGetRNNTrainingReserveSize to get real output size in LstmGpuKernelMod!
    if (AnfAlgo::GetOutputFormat(kernel_, i) == output_address->format()) {
      output_address->SetSize(output_size_list[i]);
    }
  }
}

void KernelActor::PreLaunchKernel(OpContext<DeviceTensor> *) {
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel_);
  MS_EXCEPTION_IF_NULL(kernel_mod);

  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(input_device_tensors_[i]);
    MS_EXCEPTION_IF_NULL(launch_info_.inputs_[i]);
    launch_info_.inputs_[i]->addr = input_device_tensors_[i]->GetValidPtr(kernel_info_->stream_id());
    launch_info_.inputs_[i]->size = input_device_tensors_[i]->GetSize();
    if (input_device_tensors_[i]->user_data() != nullptr) {
      kernel_mod->set_input_user_data(input_device_tensors_[i]->user_data().get(), i);
    }
  }

  for (size_t i = 0; i < output_device_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_device_tensors_[i]);
    MS_EXCEPTION_IF_NULL(launch_info_.outputs_[i]);
    launch_info_.outputs_[i]->addr = output_device_tensors_[i]->GetValidPtr(kernel_info_->stream_id());
    launch_info_.outputs_[i]->size = output_device_tensors_[i]->GetSize();
    if (output_device_tensors_[i]->user_data() != nullptr) {
      kernel_mod->set_output_user_data(output_device_tensors_[i]->user_data().get(), i);
    }
  }

  for (size_t i = 0; i < workspace_device_tensors_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(workspace_device_tensors_[i]);
    MS_EXCEPTION_IF_NULL(launch_info_.workspaces_[i]);
    launch_info_.workspaces_[i]->addr = workspace_device_tensors_[i]->GetValidPtr(kernel_info_->stream_id());
    launch_info_.workspaces_[i]->size = workspace_device_tensors_[i]->GetSize();
  }
}

bool KernelActor::LaunchKernel(OpContext<DeviceTensor> *const) {
  // Check the skipped launch condition.
  if (is_launch_skipped_) {
    MS_EXCEPTION_IF_CHECK_FAIL((launch_info_.inputs_.size() >= 1), "The inputs size is wrong.");
    MS_EXCEPTION_IF_CHECK_FAIL((launch_info_.outputs_.size() == 1), "The outputs size is wrong.");
    MS_EXCEPTION_IF_NULL(launch_info_.inputs_[0]);
    MS_EXCEPTION_IF_NULL(launch_info_.outputs_[0]);
    if (launch_info_.inputs_[0]->addr == launch_info_.outputs_[0]->addr) {
      return true;
    } else {
      MS_LOG(ERROR) << "Input address and output address are not equal of skipped launch actor: " << GetAID().Name();
      return false;
    }
  }

  // Check the address of ref node.
  for (const auto &ref : kernel_info_->out_in_ref_map()) {
    size_t input_index = ref.second;
    size_t output_index = ref.first;
    MS_EXCEPTION_IF_CHECK_FAIL((launch_info_.inputs_.size() > input_index), "The ref input index is out of range.");
    MS_EXCEPTION_IF_CHECK_FAIL((launch_info_.outputs_.size() > output_index), "The ref output index is out of range.");
    MS_EXCEPTION_IF_NULL(launch_info_.inputs_[input_index]);
    MS_EXCEPTION_IF_NULL(launch_info_.outputs_[output_index]);
    if (launch_info_.inputs_[input_index]->addr != launch_info_.outputs_[output_index]->addr) {
      // Ref node may not use the output addr, so only print the warning info.
      MS_LOG(WARNING) << "Input address and output address are not equal of ref kernel actor: " << GetAID().Name()
                      << ", ref input index: " << input_index << ", ref output index: " << output_index
                      << ". Please check whether the output address is used, which may cause problems.";
    }
  }

  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  MS_LOG(DEBUG) << "Begin launch kernel of actor: " << GetAID().Name();
  auto ret = device_contexts_[0]->kernel_executor_->LaunchKernel(
    kernel_, launch_info_.inputs_, launch_info_.workspaces_, launch_info_.outputs_, kernel_info_->stream_id());
  MS_LOG(DEBUG) << "End launch kernel of actor: " << GetAID().Name();
  return ret;
}

void KernelActor::PostLaunchKernel(OpContext<DeviceTensor> *const context) {
  if (is_dynamic_shape_) {
    try {
      MS_EXCEPTION_IF_NULL(kernel_);
      kernel::UpdateNodeShape(kernel_);
      AnfAlgo::UpdateOutputAddrSize(kernel_info_, kernel_);
      AnfAlgo::UpdateInternalParameterShape(internal_parameters_, kernel_);
    } catch (const std::exception &e) {
      if (strategy_ == GraphExecutionStrategy::kPipeline) {
        MsException::Instance().SetException();
      }
      std::string error_info = "Update output shape, size and internal parameter shape after launch exception: " +
                               kernel_->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, (*context), error_info);
    }
  }

  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

  if ((modifiable_ref_input_indexes_.size() != 0) || (modifiable_ref_output_indexes_.size() != 0)) {
    RefreshDeviceTensorCopyStore(context);
  }

  // The input is invalid and needs to be erased when finish kernel launch.
  EraseInput(context);

  // Note that SendMemoryFreeReq must be in front of SendOutput, because SendOutput will trigger SendMemoryAllocReq of
  // the next actor and the actor is asynchronous execution. So it is necessary to ensure that SendMemoryFreeReq of the
  // current actor is in front of SendMemoryAllocReq of the next actor. One is to reuse the memory more fully, the
  // other is to ensure the execution order and avoid the illegal memory timing problem.
  if (memory_free_list_.size() > 0) {
    SendMemoryFreeReq(context);
  }

  if (strategy_ == GraphExecutionStrategy::kPipeline) {
    SendOutput(context);
  }
}

void KernelActor::RefreshDeviceTensorCopyStore(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  for (auto &ref_input_index : modifiable_ref_input_indexes_) {
    if (ref_input_index >= input_device_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The input index is of range.");
    }
    auto &input_device_tensor = input_device_tensors_[ref_input_index];
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto need_refreshed_device_tensors = DeviceTensorCopyStore::GetInstance().Fetch(input_device_tensor);
    for (auto &new_device_tensor : need_refreshed_device_tensors) {
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the input position:" << ref_input_index
                   << " refresh from device address:" << input_device_tensor
                   << ", type:" << input_device_tensor->GetDeviceType() << ", format:" << input_device_tensor->format()
                   << " to device address:" << new_device_tensor << ", type:" << new_device_tensor->GetDeviceType()
                   << ", format:" << new_device_tensor->format();
      if (!Copy(new_device_tensor, input_device_tensor)) {
        std::string error_info = "Copy input device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }

  for (auto &ref_output_index : modifiable_ref_output_indexes_) {
    if (ref_output_index >= output_device_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, "The output index is of range.");
    }
    auto &output_device_tensor = output_device_tensors_[ref_output_index];
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    auto need_refreshed_device_tensors = DeviceTensorCopyStore::GetInstance().Fetch(output_device_tensor);
    for (auto &new_device_tensor : need_refreshed_device_tensors) {
      MS_EXCEPTION_IF_NULL(new_device_tensor);
      MS_LOG(INFO) << GetAID().Name() << " the output position:" << ref_output_index
                   << " refresh from device address:" << output_device_tensor
                   << ", type:" << output_device_tensor->GetDeviceType()
                   << ", format:" << output_device_tensor->format() << " to device address:" << new_device_tensor
                   << ", type:" << new_device_tensor->GetDeviceType() << ", format:" << new_device_tensor->format();
      if (!Copy(new_device_tensor, output_device_tensor)) {
        std::string error_info = "Copy output device tensor failed: " + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(strategy_, *context, error_info);
      }
    }
  }
}

void KernelActor::SendRecorderInfo(OpContext<DeviceTensor> *const context) const {
  if (recorder_aid_ != nullptr) {
    MS_EXCEPTION_IF_NULL(kernel_);
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordInfo, kernel_->fullname_with_scope(), &launch_info_,
                          device_contexts_[0], context);
  }
}
}  // namespace runtime
}  // namespace mindspore
