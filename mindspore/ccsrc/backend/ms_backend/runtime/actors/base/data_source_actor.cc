/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "backend/ms_backend/runtime/actors/base/data_source_actor.h"
#include <vector>
#include <string>
#include "backend/ms_backend/runtime/actors/base/kernel_actor.h"
#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"
#include "backend/ms_backend/runtime/actors/base/output_actor.h"
#include "backend/ms_backend/runtime/actors/base/recorder_actor.h"
#include "backend/ms_backend/runtime/actors/base/debug_actor.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "utils/ms_context.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace runtime {
void DataSourceActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() < runtime::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  InitOutputData();
}

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

void HostQueueDataSourceActor::IncreaseNewRefCounts(OpContext<KernelTensor> *const context) {
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The device data source actor data queue is empty.");
  }
  const auto &output_kernel_tensors = buffers_.front();
  if (output_data_arrows_.size() != output_data_nodes_.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR(
      (*context), "Invalid data arrow size:" + std::to_string(output_data_arrows_.size()) + " and data node size:" +
                    std::to_string(output_data_nodes_.size()) + " for host queue data source actor.");
  }
  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    auto &data_arrow = output_data_arrows_[i];
    auto output_node = output_data_nodes_[i];
    MS_EXCEPTION_IF_NULL(data_arrow);
    MS_EXCEPTION_IF_NULL(output_node);
    auto position = FetchNodePosition({output_node, data_arrow->from_output_index_});
    if (position >= output_kernel_tensors.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Invalid output index:" + std::to_string(position) +
                                                      " total size:" + std::to_string(output_kernel_tensors.size()) +
                                                      " for device queue data source actor.");
    }
    MS_EXCEPTION_IF_NULL(output_kernel_tensors[position]);
    MS_EXCEPTION_IF_NULL(output_kernel_tensors[position]->device_address());
    output_kernel_tensors[position]->IncreaseNewRefCount(GetAID().Name());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Increase new ref count for kernel tensor:" << output_kernel_tensors[position]->ToString()
      << " in actor:" << GetAID();
  }
}

void HostQueueDataSourceActor::FillDataBuffer() {
  // Construct device tensors.
  std::vector<KernelTensorPtr> kernel_tensors;
  for (auto &node_with_index : data_node_with_indexs_) {
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node_with_index.first, node_with_index.second, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Node:" << node_with_index.first->DebugString() << " index:" << node_with_index.second
      << " kernel tensor:" << kernel_tensor->ToString();
    (void)kernel_tensors.emplace_back(kernel_tensor);
  }

  for (const auto &pair : heter_index_pair_) {
    if (pair.first >= kernel_tensors.size() || pair.second >= kernel_tensors.size()) {
      MS_LOG(EXCEPTION) << "Invalid index:" << pair.first << " " << pair.second
                        << " device tensor size:" << kernel_tensors.size() << " for data source actor.";
    }
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Add device tensor copy store for kernel tensor:" << kernel_tensors[pair.second]->ToString() << " and "
      << kernel_tensors[pair.first]->ToString() << " for actor:" << GetAID();
    KernelTensorCopyStore::GetInstance().Insert(kernel_tensors[pair.second].get(), kernel_tensors[pair.first].get());
  }

  buffers_.push(kernel_tensors);
}

void HostQueueDataSourceActor::SendMemoryAllocReq(OpContext<KernelTensor> *const context) {
  if (device_contexts_.empty()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Empty device contexts in device data source actor.");
  }
  auto &kernel_tensors = buffers_.back();
  if (ActorDispatcher::is_memory_allocation_sync()) {
    if (IsSameDeviceType()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &kernel_tensors,
                                device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateBatchMemory, &kernel_tensors,
                                &device_contexts_, context, GetAID());
    }
    OnMemoryAllocFinish(context);
  } else {
    if (IsSameDeviceType()) {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &kernel_tensors,
                            device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateBatchMemory, &kernel_tensors,
                            &device_contexts_, context, GetAID());
    }
  }
}

void HostQueueDataSourceActor::SendMemoryFreeReq(OpContext<KernelTensor> *const context) {
  if (device_contexts_.empty()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Empty device contexts in device data source actor.");
  }
  auto &kernel_tensors = buffers_.front();
  if (ActorDispatcher::is_memory_free_sync()) {
    if (IsSameDeviceType()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &kernel_tensors,
                                device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeBatchMemory, &kernel_tensors,
                                &device_contexts_, context, GetAID());
    }
  } else {
    if (IsSameDeviceType()) {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &kernel_tensors, device_contexts_[0],
                            context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeBatchMemory, &kernel_tensors,
                            &device_contexts_, context, GetAID());
    }
  }
}

void HostQueueDataSourceActor::AddCopyDataCallBack(
  bool enable_async_copy, const mindspore::tensor::TensorPtrList &host_tensors,
  const std::vector<mindspore::runtime::KernelTensorPtr> &kernel_tensors) {
  if (!enable_async_copy || kernel_tensors.empty()) {
    return;
  }

  device::CallbackFunc callback_func = [host_tensors]() {
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
  auto callback_ret =
    host_context->device_res_manager_->LaunchCallback(callback_func, kernel_tensors[0]->device_address()->stream_id());
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
  const auto &device_storage_info = kernel_tensor->tensor_storage_info();
  auto device_tensor = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_tensor);
  // No used device address need skip.
  if (TEST_FLAG(kernel_tensor->flag(), device::kDeviceAddressFlagNotUsed)) {
    kernel_tensor->IncreaseNewRefCount("data source actor");
    MS_LOG(DEBUG) << "Data source actor input kernel tensor is not used:" << kernel_tensor->ToString();
    return;
  }
  auto tensor_device_address = host_tensor->device_address();
  // Sync data from host_tensor_device_address to device_tensor.
  if (tensor_device_address != nullptr) {
    // Already set same pointer ref count.
    if (tensor_device_address->GetPtr() == device_tensor->GetPtr()) {
      return;
    }
    if (!SyncAllStreamForDeviceAddress(device_tensor, tensor_device_address) ||
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
  if (common::AnfAlgo::HasAbstractRef(node_index.first)) {
    MS_LOG(DEBUG) << "Set device address:" << kernel_tensor->device_address()->ToString()
                  << " to host tensor:" << host_tensor->ToString()
                  << " by data node:" << node_index.first->DebugString();
    host_tensor->set_device_address(kernel_tensor->device_address());
    host_tensor->set_storage_info(kernel_tensor->tensor_storage_info());
    kernel_tensor->set_new_ref_count(SIZE_MAX);
  }
  if (enable_async_copy) {
    MS_LOG(INFO) << "Node : " << node_index.first->DebugString();
    if (!AsyncCopy(kernel_tensor.get(), host_tensor.get(), kDefaultStreamIndex)) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "SyncHostToDevice failed.");
    }
  } else {
    if (!SyncAllStreamForDeviceAddress(device_tensor, host_tensor->device_address()) ||
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
                                      "The length of host tensors is not equal to the length of kernel tensors.");
  }

  static bool sync_copy_input = runtime::IsEnableRuntimeConfig(runtime::kRuntimeSyncCopyInput);
  // Copy data from host tensor to device tensor.
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  auto enable_async_copy = (ms_context->IsEnableInferBoost() || is_infer_phase_) && !sync_copy_input;
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

bool HostQueueDataSourceActor::IsSameDeviceType() const {
  for (size_t i = 1; i < device_contexts_.size(); i++) {
    if (device_contexts_[i] != device_contexts_[0]) {
      return false;
    }
  }
  return true;
}

void HostQueueDataSourceActor::ReleaseData() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kOutputProcess,
                                     "DataSourceActorReleaseData");
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
    if (old_kernel_tensor->new_ref_count() == SIZE_MAX && !old_kernel_tensor->is_ptr_persisted()) {
      auto new_kernel_tensor = old_kernel_tensor->CloneKernelTensor();
      MS_EXCEPTION_IF_NULL(new_kernel_tensor);
      auto new_address = new_kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(new_address);
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Create kernel tensor:" << new_kernel_tensor->ToString();
      new_kernel_tensor->set_new_ref_count(old_kernel_tensor->new_ref_count());
      new_address->set_ptr(nullptr);
      auto [node, index] = old_address->GetNodeIndex();
      new_address->SetNodeIndex(node, index);
      AnfAlgo::SetOutputKernelTensor(new_kernel_tensor, data_node_with_index.second, data_node_with_index.first.get());
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
