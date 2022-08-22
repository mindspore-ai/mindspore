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

#include "runtime/graph_scheduler/actor/memory/memory_alloc_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"

namespace mindspore {
namespace runtime {
void MemoryAllocActor::Init() {
  MS_EXCEPTION_IF_CHECK_FAIL((!device_contexts_.empty()), "The device context doesn't exist.");
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]->device_res_manager_);
  MS_EXCEPTION_IF_NULL(somas_info_);
  MS_EXCEPTION_IF_CHECK_FAIL((somas_info_->whole_block_size_ != 0), "The alloc size of somas info is zero.");

  created_device_tensor_ = device_contexts_[0]->device_res_manager_->CreateDeviceAddress(
    nullptr, somas_info_->whole_block_size_, "DefaultFormat", kNumberTypeFloat16, {});
  (void)memory_alloc_list_.emplace_back(created_device_tensor_.get());
}

void MemoryAllocActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(created_device_tensor_);
  created_device_tensor_->set_ptr(nullptr);
  if (ActorDispatcher::is_memory_allocation_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_,
                              device_contexts_[0], context, GetAID());
    OnMemoryAllocFinish(context);
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_,
                          device_contexts_[0], context, GetAID());
  }
}

void MemoryAllocActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(somas_info_);
  MS_EXCEPTION_IF_NULL(created_device_tensor_);
  if (IsRunningFailed(context)) {
    return;
  }

  // Set the base address of somas info using the alloc memory.
  if (somas_info_->base_address_ != nullptr) {
    std::string error_info = GetAID().Name() + " already has the base address.";
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
  somas_info_->base_address_ = created_device_tensor_->GetMutablePtr();
  MS_LOG(DEBUG) << GetAID().Name() << " alloc memory: " << somas_info_->base_address_;

  PostRun(context);
}
}  // namespace runtime
}  // namespace mindspore
