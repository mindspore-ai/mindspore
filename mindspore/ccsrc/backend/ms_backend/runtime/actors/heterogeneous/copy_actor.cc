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

#include "backend/ms_backend/runtime/actors/heterogeneous/copy_actor.h"
#include <string>
#include <utility>
#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"

namespace mindspore {
namespace runtime {
const size_t kInputDeviceContextIndex = 0;
const size_t kOutputDeviceContextIndex = 1;

void CopyActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != runtime::kDeviceContextsNumTwo) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  const size_t kDeviceTensorNum = 1;
  input_kernel_tensors_.resize(kDeviceTensorNum);
  output_kernel_tensors_.resize(kDeviceTensorNum);

  // Check output data index.
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) != 0) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID().Name();
    }
  }

  InitOutputData();
}

void CopyActor::Run(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), GetAID().Name(), "");
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Copy actor:" << GetAID() << " start run.";
  FetchKernelTensor(context);
  SendMemoryAllocReq(context);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Copy actor:" << GetAID() << " end run.";
}

void CopyActor::SendMemoryAllocReq(OpContext<KernelTensor> *const context) {
  if (ActorDispatcher::is_memory_allocation_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &output_kernel_tensors_,
                              device_contexts_[kOutputDeviceContextIndex], context, GetAID());
    OnMemoryAllocFinish(context);
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &output_kernel_tensors_,
                          device_contexts_[kOutputDeviceContextIndex], context, GetAID());
  }
}

void CopyActor::SendMemoryFreeReq(OpContext<KernelTensor> *const context) {
  if (ActorDispatcher::is_memory_free_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &input_kernel_tensors_,
                              device_contexts_[kInputDeviceContextIndex], context, GetAID());
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &output_kernel_tensors_,
                              device_contexts_[kOutputDeviceContextIndex], context, GetAID());
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &input_kernel_tensors_,
                          device_contexts_[kInputDeviceContextIndex], context, GetAID());
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &output_kernel_tensors_,
                          device_contexts_[kOutputDeviceContextIndex], context, GetAID());
  }
}

void CopyActor::OnMemoryAllocFinish(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(output_kernel_tensors_[0]);
  MS_EXCEPTION_IF_NULL(output_kernel_tensors_[0]->device_address());
  MS_EXCEPTION_IF_NULL(input_kernel_tensors_[0]);
  MS_EXCEPTION_IF_NULL(input_kernel_tensors_[0]->device_address());
  if (IsRunningFailed(context)) {
    return;
  }
  if (!IsContiguousStorage(input_kernel_tensors_[0]->tensor_storage_info())) {
    std::stringstream error_info;
    error_info << "Not support non-contiguous input: " << input_kernel_tensors_[0]->ToString()
               << " for actor:" << GetAID();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info.str());
  }
  if (input_kernel_tensors_[0]->device_address()->GetSize() != output_kernel_tensors_[0]->device_address()->GetSize()) {
    MS_LOG(WARNING) << GetAID().Name()
                    << " copy size is not equal, input device tensor:" << input_kernel_tensors_[0]->device_address()
                    << " size:" << input_kernel_tensors_[0]->device_address()->GetSize()
                    << ", output device tensor:" << output_kernel_tensors_[0]->device_address()
                    << "size:" << output_kernel_tensors_[0]->device_address()->GetSize();
  }

  {
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kCopyData, GetAID().Name());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Copy device tensor from kernel tensor:" << input_kernel_tensors_[0]->ToString() << " to "
      << output_kernel_tensors_[0]->ToString() << " for copy actor:" << GetAID();
    output_kernel_tensors_[0]->SetType(input_kernel_tensors_[0]->GetType());
    output_kernel_tensors_[0]->SetShape(input_kernel_tensors_[0]->GetShape());
    output_kernel_tensors_[0]->set_user_data(input_kernel_tensors_[0]->user_data());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Set user data:" << input_kernel_tensors_[0]->user_data()
                                                 << " shape:" << input_kernel_tensors_[0]->GetShape()->ToString()
                                                 << " from device tensor:" << input_kernel_tensors_[0]->device_address()
                                                 << " to:" << output_kernel_tensors_[0]->device_address();
    output_kernel_tensors_[0]->set_need_sync_user_data(input_kernel_tensors_[0]->need_sync_user_data());
    if (!SyncAllStreamForDeviceAddress(output_kernel_tensors_[0]->device_address(),
                                       input_kernel_tensors_[0]->device_address()) ||
        !SyncCopy(output_kernel_tensors_[0].get(), input_kernel_tensors_[0].get(), kDefaultStreamIndex)) {
      std::string error_info = "Copy device tensor failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Add device tensor copy store for kernel tensor:" << output_kernel_tensors_[0]->ToString() << " and "
      << input_kernel_tensors_[0]->ToString() << " for copy actor:" << GetAID();
    KernelTensorCopyStore::GetInstance().Insert(output_kernel_tensors_[0].get(), input_kernel_tensors_[0].get());
  }

  PostRun(context);
}

void CopyActor::FetchKernelTensor(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  const auto &input_device_context = device_contexts_[kInputDeviceContextIndex];
  const auto &output_device_context = device_contexts_[kOutputDeviceContextIndex];
  MS_EXCEPTION_IF_NULL(input_device_context);
  MS_EXCEPTION_IF_NULL(output_device_context);

  if (device_tensor_store_keys_.size() > 0) {
    const auto &device_tensor_store_node = device_tensor_store_keys_[0].second;
    MS_EXCEPTION_IF_NULL(device_tensor_store_node);
    input_kernel_tensors_[0] =
      DeviceTensorStore::GetInstance().Fetch(device_tensor_store_node.get(), input_device_context->GetDeviceType());
    if (input_kernel_tensors_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_node->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(input_device_context->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    output_kernel_tensors_[0] =
      DeviceTensorStore::GetInstance().Fetch(device_tensor_store_node.get(), output_device_context->GetDeviceType());
    if (output_kernel_tensors_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_node->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(output_device_context->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  } else {
    const auto &data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "No input data.");
    }
    const auto &input_data = data_iter->second[0];
    MS_EXCEPTION_IF_NULL(input_data);
    input_kernel_tensors_[0] = input_data->data_;

    MS_EXCEPTION_IF_NULL(output_);
    output_kernel_tensors_[0] = output_;
  }

  if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
    MS_LOG(INFO) << "Run failed and early stop.";
    return;
  }
  if (is_need_update_output_size_ && (input_kernel_tensors_[0]->device_address()->GetSize() !=
                                      output_kernel_tensors_[0]->device_address()->GetSize())) {
    MS_LOG(DEBUG) << GetAID().Name() << " update output size from "
                  << output_kernel_tensors_[0]->device_address()->GetSize() << " to "
                  << input_kernel_tensors_[0]->device_address()->GetSize();
    output_kernel_tensors_[0]->device_address()->SetSize(input_kernel_tensors_[0]->device_address()->GetSize());
    const auto &output_kernel_tensor = output_kernel_tensors_[0];
    const auto &input_kernel_tensor = input_kernel_tensors_[0];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    output_kernel_tensor->SetType(input_kernel_tensor->GetType()->Clone());
    output_kernel_tensor->SetShape(input_kernel_tensor->GetShape()->Clone());
  }
}

void CopyActor::IncreaseNewRefCounts(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_kernel_tensors_[0]);
  MS_EXCEPTION_IF_NULL(output_kernel_tensors_[0]->device_address());
  if (output_data_arrows_.size() < output_free_size_) {
    std::stringstream error_info;
    error_info << "Invalid output size:" << output_data_arrows_.size() << " and free size:" << output_free_size_
               << " for actor:" << GetAID();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info.str());
  }
  for (size_t i = 0; i < output_data_arrows_.size() - output_free_size_; ++i) {
    output_kernel_tensors_[0]->IncreaseNewRefCount(GetAID().Name());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Increase new ref count for kernel tensor:" << output_kernel_tensors_[0]->ToString()
      << " in actor:" << GetAID();
  }
}

void CopyActor::UpdateOutputData(OpData<KernelTensor> *const output_data, const DataArrowPtr &, const AnfNodePtr &,
                                 OpContext<KernelTensor> *const) {
  MS_EXCEPTION_IF_NULL(output_data);
  output_data->data_ = output_kernel_tensors_[0];
}
}  // namespace runtime
}  // namespace mindspore
