/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/runtime/actor/data_source_actor.h"
#include <vector>
#include <string>
#include "backend/ge_backend/runtime/actor/memory_manager_actor.h"
#include "backend/ge_backend/runtime/actor/output_actor.h"
#include "backend/ge_backend/runtime/actor/recorder_actor.h"
#include "backend/ge_backend/runtime/actor/debug_actor.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "utils/ms_context.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
void DataSourceActor::Init() { InitOutputData(); }

void DataSourceActor::FetchData(OpContext<KernelTensor> *const context) {
  MS_LOG(INFO) << "Data source actor(" << GetAID().Name() << ") fetches data.";
  MS_EXCEPTION_IF_NULL(context);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), GetAID().Name(), "");
  // Pop the data of last time.
  if (!buffers_.empty()) {
    buffers_.pop();
  }

  // Construct device tensors and fill to the buffers from member nodes.
  FillDataBuffer();
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Allocate memory for device tensors.
  SendMemoryAllocReq(context);
}

void DataSourceActor::UpdateOutputData(OpData<KernelTensor> *const output_data, const DataArrowPtr &data_arrow,
                                       const AnfNodePtr &output_node, OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_data);
  MS_EXCEPTION_IF_NULL(data_arrow);
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(context);

  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }
  const auto &output_kernel_tensors = buffers_.front();

  auto position = FetchNodePosition({output_node, data_arrow->from_output_index_});
  // Host data souruce actor uses the node position, device data source actor uses the output index.
  auto output_position = (position != 0) ? position : IntToSize(data_arrow->from_output_index_);
  if (output_position >= output_kernel_tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The output index is of range.");
  }
  output_data->data_ = output_kernel_tensors[output_position];
}

void HostQueueDataSourceActor::FillDataBuffer() {
  // Construct device tensors.
  std::vector<KernelTensorPtr> kernel_tensors;
  for (auto &node_with_index : data_node_with_indexs_) {
    MS_LOG(DEBUG) << "Node:" << node_with_index.first->DebugString() << " index:" << node_with_index.second;
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node_with_index.first, node_with_index.second, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    (void)kernel_tensors.emplace_back(kernel_tensor);
  }

  buffers_.push(kernel_tensors);
}

void HostQueueDataSourceActor::SendMemoryAllocReq(OpContext<KernelTensor> *const context) {
  auto &kernel_tensors = buffers_.back();
  if (ActorDispatcher::is_memory_allocation_sync()) {
    if (IsSameDeviceType()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &kernel_tensors, context,
                                GetAID());
    } else {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateBatchMemory, &kernel_tensors, context,
                                GetAID());
    }
    OnMemoryAllocFinish(context);
  } else {
    if (IsSameDeviceType()) {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &kernel_tensors, context,
                            GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateBatchMemory, &kernel_tensors, context,
                            GetAID());
    }
  }
}

void HostQueueDataSourceActor::SendMemoryFreeReq(OpContext<KernelTensor> *const context) {
  auto &kernel_tensors = buffers_.front();
  if (ActorDispatcher::is_memory_free_sync()) {
    if (IsSameDeviceType()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &kernel_tensors, context,
                                GetAID());
    } else {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeBatchMemory, &kernel_tensors, context,
                                GetAID());
    }
  } else {
    if (IsSameDeviceType()) {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &kernel_tensors, context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeBatchMemory, &kernel_tensors, context,
                            GetAID());
    }
  }
}

void HostQueueDataSourceActor::AddCopyDataCallBack(bool enable_async_copy,
                                                   const mindspore::tensor::TensorPtrList &host_tensors,
                                                   const std::vector<KernelTensorPtr> &kernel_tensors) {
  if (!enable_async_copy || kernel_tensors.empty()) {
    return;
  }

  std::function<void(void)> callback_func = [host_tensors]() {
    // Clear buffer automatically.
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  auto callback_ret = host_context->device_res_manager_->LaunchCallback(callback_func, kernel_tensors[0]->stream_id());
  if (!callback_ret) {
    MS_LOG(EXCEPTION) << "Async Copy memory launch callback failed";
  }
}

namespace {
void CopyHostTensorToKernelTensor(const tensor::TensorPtr &host_tensor, const kernel::KernelTensorPtr &kernel_tensor,
                                  bool enable_async_copy, const KernelWithIndex &node_index,
                                  OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(host_tensor);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(context);
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  const auto &device_storage_info = kernel_tensor->tensor_storage_info();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::DeviceContextKey host_key = {device::GetDeviceTypeByName(device_name), device_id};
  device::DeviceContext *host_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
  MS_EXCEPTION_IF_NULL(host_context);
  MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
  host_context->device_res_manager_->BindDeviceToCurrentThread(false);
  // No used device address need skip.
  if (TEST_FLAG(kernel_tensor->flag(), device::kDeviceAddressFlagNotUsed)) {
    MS_LOG(DEBUG) << "Data source actor input kernel tensor is not used:" << kernel_tensor->ToString();
    return;
  }
  auto tensor_device_address = host_tensor->device_address();
  // Sync data from host_tensor_device_address to device_tensor.
  if (tensor_device_address != nullptr) {
    if (tensor_device_address == device_tensor) {
      return;
    }
    if (!host_context->device_res_manager_->SyncAllStreams() ||
        !SyncCopy(kernel_tensor.get(), host_tensor.get(), kDefaultStreamIndex)) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Copy data failed.");
    }
    return;
  }
  if (host_tensor->device_address() == nullptr && device_tensor->GetSize() == 0) {
    MS_LOG(INFO) << "Empty tuple sync";
    return;
  }

  MS_EXCEPTION_IF_NULL(node_index.first);
  if (enable_async_copy) {
    MS_LOG(INFO) << "Node : " << node_index.first->DebugString();
    if (!AsyncCopy(kernel_tensor.get(), host_tensor.get(), kDefaultStreamIndex)) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "SyncHostToDevice failed.");
    }
  } else {
    if (!host_context->device_res_manager_->SyncAllStreams() ||
        !SyncCopy(kernel_tensor.get(), host_tensor.get(), kDefaultStreamIndex)) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "SyncHostToDevice failed.");
    }
  }

  if (IsDynamic(kernel_tensor->GetShapeVector())) {
    kernel_tensor->SetShapeVector(host_tensor->shape());
  }
}
}  // namespace

void HostQueueDataSourceActor::OnMemoryAllocFinish(OpContext<KernelTensor> *const context) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(context);
  if (IsRunningFailed(context)) {
    return;
  }
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Get host tensors from host queue and get device tensors from buffers.
  MS_EXCEPTION_IF_NULL(host_queue_);
  if (host_queue_->IsEmpty()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Host data queue is empty.");
  }
  auto &host_tensors = host_queue_->Pull();
  auto &kernel_tensors = buffers_.back();
  if (host_tensors.size() != kernel_tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context),
                                      "The length of host tensors is not equal to the length of device tensors.");
  }

  // Copy data from host tensor to device tensor.
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  auto enable_async_copy = ms_context->IsEnableInferBoost() || is_infer_phase_;
  try {
    KernelWithIndex empty_node{nullptr, 0};
    for (size_t i = 0; i < host_tensors.size(); ++i) {
      CopyHostTensorToKernelTensor(host_tensors[i], kernel_tensors[i], enable_async_copy,
                                   i < data_node_with_indexs_.size() ? data_node_with_indexs_[i] : empty_node, context);
    }
    AddCopyDataCallBack(enable_async_copy, host_tensors, kernel_tensors);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Host data source actor run exception.");
  }
  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kCopyData, GetAID().Name(), false);

  PostRun(context);
}

size_t HostQueueDataSourceActor::FetchNodePosition(const KernelWithIndex &data_node) const {
  MS_EXCEPTION_IF_NULL(data_node.first);
  const auto &iter = data_node_position_map_.find(data_node);
  if (iter == data_node_position_map_.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, data_node.first)
      << "Data node: " << data_node.first->DebugString() << " index:" << data_node.second << " is not exist.";
  }
  return iter->second;
}

KernelWithIndex HostQueueDataSourceActor::FetchNode(size_t node_position) const {
  if (node_position >= data_node_with_indexs_.size()) {
    MS_LOG(EXCEPTION) << "The position of node is out of range: " << node_position;
  }
  return data_node_with_indexs_[node_position];
}

bool HostQueueDataSourceActor::IsSameDeviceType() const { return true; }

void HostQueueDataSourceActor::ReleaseData() {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, "DataSourceActorReleaseData");
  // The step end need free the host queue tensor.
  MS_EXCEPTION_IF_NULL(host_queue_);
  host_queue_->Pop();

  // The step end need release data node address.
  for (auto &data_node_with_index : data_node_with_indexs_) {
    if (!AnfAlgo::OutputAddrExist(data_node_with_index.first, data_node_with_index.second)) {
      continue;
    }
    auto old_kernel_tensor = AnfAlgo::GetOutputKernelTensor(data_node_with_index.first, data_node_with_index.second);
    MS_EXCEPTION_IF_NULL(old_kernel_tensor);
    auto old_address = old_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(old_address);
    if (old_address->GetPtr() == nullptr) {
      // The Address memory is already freed.
      continue;
    }
    // If the address from input tensor and the address is not used by runtime.
    if (old_kernel_tensor->original_ref_count() == SIZE_MAX && !old_kernel_tensor->is_ptr_persisted()) {
      auto kernel_tensor =
        AnfAlgo::GetOutputKernelTensor(data_node_with_index.first, data_node_with_index.second, false);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto new_kernel_tensor = kernel_tensor->CloneKernelTensor();
      MS_EXCEPTION_IF_NULL(new_kernel_tensor);
      new_kernel_tensor->set_device_ptr(nullptr);

      auto new_address = new_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(new_address);
      MS_LOG(DEBUG) << "Create new kernel tensor:" << new_kernel_tensor->ToString();
      new_kernel_tensor->set_original_ref_count(old_kernel_tensor->original_ref_count());
      new_kernel_tensor->ResetRefCount();
      new_kernel_tensor->set_flag(old_kernel_tensor->flag());
      auto [node, index] = old_address->GetNodeIndex();
      new_address->SetNodeIndex(node, index);
      AnfAlgo::SetOutputKernelTensor(new_kernel_tensor, data_node_with_index.second, data_node_with_index.first.get());
      if (ref_kernel_tensors_.find(data_node_with_index) == ref_kernel_tensors_.end()) {
        continue;
      }
      for (const auto &ref_kernel_tensor : ref_kernel_tensors_[data_node_with_index]) {
        if (ref_kernel_tensor != nullptr) {
          MS_LOG(DEBUG) << "Set pointer ref count from kernel tensor:" << new_kernel_tensor->ToString()
                        << " to:" << ref_kernel_tensor->ToString()
                        << " for data source node:" << data_node_with_index.first->DebugString();
          ref_kernel_tensor->set_pointer_ref_count(new_kernel_tensor.get());
        }
      }
    }
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
