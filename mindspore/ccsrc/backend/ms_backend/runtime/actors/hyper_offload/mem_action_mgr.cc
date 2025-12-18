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

#include "backend/ms_backend/runtime/actors/hyper_offload/mem_action_mgr.h"

#include "backend/ms_backend/runtime/actors/hyper_offload/mem_use_analyzer.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"
#include "backend/ms_backend/runtime/actors/base/memory_manager_actor.h"

namespace mindspore {
namespace runtime {
constexpr size_t KStandardRefCount = 1;

namespace {
device::DeviceContext *GetDeviceContextPtr(device::DeviceType device_type) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::DeviceContextKey device_key = {device_type, device_id};
  return device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(device_key);
}
}  // namespace

bool MemActionMgr::IsKernelTensorCanBeMoved(const KernelTensorPtr &kernel_tensor) {
  size_t ref_count = kernel_tensor->new_ref_count();
  if (ref_count > KStandardRefCount) {
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Can not move kernel tensor: " << kernel_tensor->ToString()
                                   << " , ref_count: " << ref_count;
    return false;
  }
  auto &src_addr = kernel_tensor->device_address();
  auto src_type = src_addr->GetDeviceType();
  if (src_type != device::DeviceType::kAscend) {
    return true;
  }

  auto *src_ctx = GetDeviceContextPtr(src_type);
  MS_EXCEPTION_IF_NULL(src_ctx);
  if (!src_ctx->device_res_manager_->IsNotEventUsedMemory(src_addr.get())) {
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Unable free: " << kernel_tensor->ToString() << " , src_addr: " << src_addr;
    return false;
  }
  return true;
}

KernelTensorPtrPairList MemActionMgr::CreateRemoteEvents(const RemoteActionPtrList &remote_actions,
                                                         const device::DeviceContext *device_context) {
  KernelTensorPtrPairList copy_create_kernel_tensors;
  for (const auto &remote_action : remote_actions) {
    const auto &event_type = remote_action->event_type;
    const auto &kernel_tensor = remote_action->kernel_tensor;
    MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Insert event: " << event_type;
    if (event_type == HyperOffloadEventType::kDeviceToHost || event_type == HyperOffloadEventType::kHostToDevice) {
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto new_kernel_tensor =
        InsertCopyDataTask(remote_action->src_stream_id, device_context, event_type, kernel_tensor);
      copy_create_kernel_tensors.emplace_back(kernel_tensor, new_kernel_tensor);
    } else if (event_type == HyperOffloadEventType::kRecordWithMemoryEvent) {
      InsertRecordWithMemoryEvent(remote_action->src_stream_id, device_context, kernel_tensor);
    } else if (event_type == HyperOffloadEventType::kWaitWithMemoryEvent) {
      InsertWaitWithMemoryEvent(remote_action->dst_stream_id, device_context, kernel_tensor);
    } else if (event_type == HyperOffloadEventType::kRecordWaitPairEvent) {
      InsertRecordWaitEvents(remote_action->src_stream_id, remote_action->dst_stream_id, device_context);
    } else {
      MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "Unsupported event type: " << event_type;
    }
  }
  return copy_create_kernel_tensors;
}

// Create CopyData task for kernel_tensor
kernel::KernelTensorPtr MemActionMgr::InsertCopyDataTask(uint32_t stream_id,
                                                         const device::DeviceContext *device_context,
                                                         HyperOffloadEventType event_type,
                                                         const kernel::KernelTensorPtr &kernel_tensor) {
  if (!IsKernelTensorCanBeMoved(kernel_tensor)) {
    return nullptr;
  }
  device::DeviceType src_ty;
  device::DeviceType dst_ty;
  if (event_type == HyperOffloadEventType::kDeviceToHost) {
    src_ty = device::DeviceType::kAscend;
    dst_ty = device::DeviceType::kCPU;
  } else if (event_type == HyperOffloadEventType::kHostToDevice) {
    src_ty = device::DeviceType::kCPU;
    dst_ty = device::DeviceType::kAscend;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported event type " << event_type;
  }

  auto &src_addr = kernel_tensor->device_address();
  if (src_addr->GetDeviceType() != src_ty) {
    MS_LOG(EXCEPTION) << "Device type error: " << src_addr->GetDeviceType();
  }

  auto *dst_ctx = GetDeviceContextPtr(dst_ty);
  MS_EXCEPTION_IF_NULL(dst_ctx);
  auto dst_addr = std::make_shared<device::DeviceAddress>(dst_ctx->GetDeviceType());
  auto new_tensor =
    std::make_shared<kernel::KernelTensor>(dst_addr, kernel_tensor->GetShape(), kernel_tensor->GetType(), nullptr);
  new_tensor->set_size(kernel_tensor->GetSize());
  new_tensor->set_device_address(dst_addr);
  if (!dst_ctx->device_res_manager_->AllocateMemory(dst_addr.get(), stream_id)) {
    MS_LOG(EXCEPTION) << "Allocate memory for copy data task failed, dst device: " << dst_ty
                      << ", src kernel tensor: " << kernel_tensor->ToString();
  }

  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Insert Copy data task for src kernel tensor: " << kernel_tensor->ToString() << "\n"
                                 << "dst kernel tensor: " << new_tensor->ToString();
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Insert Copy data task for src device address: " << src_addr << "\n"
                                 << "dst device address: " << dst_addr;
  if (!AsyncCopy(dst_addr, src_addr, stream_id)) {
    MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "Copy data failed";
    dst_ctx->device_res_manager_->FreeMemory(dst_addr.get());
    return nullptr;
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Launch copy task success, src kernel tensor: " << kernel_tensor
                                 << ", dst kernel tensor: " << new_tensor << ", target stream id: " << stream_id;
  return new_tensor;
}

// Check the corresponding event for kernel_tensor, if not found, return false
// Insert Wait event for kernel_tensor on stream_id, erase this kernel_tensor's event info
bool MemActionMgr::InsertWaitWithMemoryEvent(uint32_t stream_id, const device::DeviceContext *device_context,
                                             const kernel::KernelTensorPtr &kernel_tensor) {
  auto it = kernel_events_map.find(kernel_tensor);
  if (it == kernel_events_map.end()) {
    MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "Invalid kernel tensor for wait event, " << kernel_tensor->ToString();
    return false;
  }
  auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
    device_context->device_context_key().device_type_);
  if (!multi_stream_controller->DispatchWaitEvent(stream_id, it->second)) {
    MS_VLOG(VL_HYPER_OFFLOAD_WARNING) << "Insert wait event for: " << kernel_tensor << " failed";
    return false;
  }
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Insert wait event for: " << kernel_tensor << " on stream id: " << stream_id;
  kernel_events_map.erase(it);

  // Free src kernel tensor
  auto *src_ctx = GetDeviceContextPtr(kernel_tensor->GetDeviceType());
  MS_EXCEPTION_IF_NULL(src_ctx);
  auto &src_addr = kernel_tensor->device_address();
  src_ctx->device_res_manager_->FreeMemory(src_addr.get());
  return true;
}

// Insert Record event for kernel_tensor on stream_id, using kernel_tensor as key, manage this event
bool MemActionMgr::InsertRecordWithMemoryEvent(uint32_t stream_id, const device::DeviceContext *device_context,
                                               const kernel::KernelTensorPtr &kernel_tensor) {
  auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
    device_context->device_context_key().device_type_);
  auto event = multi_stream_controller->DispatchRecordEvent(stream_id);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Insert record event for: " << kernel_tensor << " on stream id: " << stream_id;
  kernel_events_map[kernel_tensor] = event;
  return true;
}

bool MemActionMgr::InsertRecordWaitEvents(uint32_t src_stream_id, uint32_t dst_stream_id,
                                          const device::DeviceContext *device_context) {
  // Insert record wait pair between src_stream_id and dst_stream_id
  if (src_stream_id == dst_stream_id) {
    MS_LOG(EXCEPTION) << "No need insert events";
  }
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto &multi_stream_controller = device::DeviceContextManager::GetInstance().GetMultiStreamController(
    device_context->device_context_key().device_type_);
  MS_EXCEPTION_IF_NULL(multi_stream_controller);
  bool result = multi_stream_controller->DispatchRecordWaitEvent(dst_stream_id, src_stream_id);
  auto src_task_id = multi_stream_controller->GetTaskIdOnStream(src_stream_id);
  auto dst_task_id = multi_stream_controller->GetTaskIdOnStream(dst_stream_id);
  MS_VLOG(VL_HYPER_OFFLOAD_INFO) << "Get current src task id: " << src_task_id << " , src stream id: " << src_stream_id
                                 << " , dst task id: " << dst_task_id << " , dst stream id: " << dst_stream_id;
  return result;
}

}  // namespace runtime
}  // namespace mindspore
