/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "pynative/hyper_offload/offload_executor.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/pynative/utils/pyboost/offload_copy_registry.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "utils/log_adapter.h"
#include "pynative/hyper_offload/offload_task.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

OffloadExecutor::OffloadExecutor(const hal::StreamPtr &copy_stream) : copy_stream_(copy_stream) {
  MS_EXCEPTION_IF_NULL(copy_stream_);

  // Create temporary event for compute->copy synchronization
  tmp_event_ = std::make_shared<hal::Event>(false, false);  // enable_timing=false, blocking=false

  MS_LOG(DEBUG) << "OffloadExecutor initialized. copy_stream_id=" << copy_stream_->stream_id();
}

tensor::TensorPtr OffloadExecutor::ToHost(const tensor::TensorPtr &device_tensor) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  MS_EXCEPTION_IF_NULL(copy_stream_);
  // Use CopyToHost via registry to perform D2H on copy_stream
  // This creates a pinned memory host tensor without blocking host
  return kernel::pyboost::OffloadCopyRegistry::GetInstance().CopyToHost(copy_stream_->stream_id(), device_tensor);
}

tensor::TensorPtr OffloadExecutor::ToDeviceTensor(const tensor::TensorPtr &host_tensor) {
  MS_EXCEPTION_IF_NULL(host_tensor);
  MS_EXCEPTION_IF_NULL(host_tensor->device_address());
  MS_EXCEPTION_IF_NULL(copy_stream_);

  // Use CopyToDevice via registry to perform H2D on copy_stream
  return kernel::pyboost::OffloadCopyRegistry::GetInstance().CopyToDevice(copy_stream_->stream_id(), host_tensor);
}

void OffloadExecutor::D2HWithEvent(size_t tensor_id, const tensor::TensorPtr &original_tensor) {
  MS_EXCEPTION_IF_NULL(original_tensor);
  if (original_tensor->storage_info() != nullptr) {
    MS_LOG(EXCEPTION) << "Try offload view's output, which is not supported! tensor id: " << tensor_id;
  }
  auto &data = offload_cache_[tensor_id];
  data.original_tensor = original_tensor;

  // Step 1: Get current compute stream dynamically
  auto compute_stream = hal::CurrentStream();
  MS_EXCEPTION_IF_NULL(compute_stream);

  // Step 2: Record event on compute stream (mark computation completion point)
  tmp_event_->Record(compute_stream);

  // Step 3: Wait on copy stream (ensure computation is done before copy)
  tmp_event_->Wait(copy_stream_);

  // Step 4: Execute D2H using to_device (runs on copy stream)
  // Only create host_tensor on first D2H, subsequent D2H will reuse it
  // (but in practice, if host_tensor exists, we won't call D2H again)
  if (data.host_tensor == nullptr) {
    data.host_tensor = ToHost(original_tensor);
    MS_LOG(DEBUG) << "D2HWithEvent[" << tensor_id << "]: Created host_tensor";
  }

  // Step 5: Record D2H completion event on copy stream
  if (data.d2h_complete_event == nullptr) {
    data.d2h_complete_event = std::make_shared<hal::Event>(false, false);
  }
  data.d2h_complete_event->Record(copy_stream_);

  MS_LOG(DEBUG) << "D2HWithEvent[" << tensor_id << "]: D2H queued, tensor shape=" << original_tensor->shape();
}

void OffloadExecutor::H2DWithEvent(size_t tensor_id) {
  auto it = offload_cache_.find(tensor_id);
  if (it == offload_cache_.end()) {
    MS_LOG(EXCEPTION) << "H2DWithEvent[" << tensor_id << "]: No cached data found. D2H must be called first.";
  }

  auto &data = it->second;
  MS_EXCEPTION_IF_NULL(data.host_tensor);
  auto compute_stream = hal::CurrentStream();
  MS_EXCEPTION_IF_NULL(compute_stream);
  tmp_event_->Record(compute_stream);
  tmp_event_->Wait(copy_stream_);
  // Step 1: Execute H2D using to_device (runs on copy stream)
  // Create new device tensor each time
  data.device_tensor = ToDeviceTensor(data.host_tensor);

  // Step 2: Record H2D completion event on copy stream
  if (data.h2d_complete_event == nullptr) {
    data.h2d_complete_event = std::make_shared<hal::Event>(false, false);
  }
  data.h2d_complete_event->Record(copy_stream_);

  MS_LOG(DEBUG) << "H2DWithEvent[" << tensor_id << "]: H2D queued";
}

void OffloadExecutor::WaitD2HAndSwitch(size_t tensor_id) {
  auto it = offload_cache_.find(tensor_id);
  if (it == offload_cache_.end()) {
    MS_LOG(EXCEPTION) << "WaitD2HAndSwitch[" << tensor_id << "]: No cached data found.";
  }

  auto &data = it->second;
  MS_EXCEPTION_IF_NULL(data.d2h_complete_event);
  MS_EXCEPTION_IF_NULL(data.original_tensor);
  MS_EXCEPTION_IF_NULL(data.host_tensor);

  // Step 1: Get current compute stream dynamically and wait for D2H completion
  auto compute_stream = hal::CurrentStream();
  MS_EXCEPTION_IF_NULL(compute_stream);
  data.d2h_complete_event->Wait(compute_stream);

  // Step 2: Switch address: original tensor now points to host memory
  // This allows the original device memory to be released
  // auto host_device_address = data.host_tensor->device_address();
  // MS_EXCEPTION_IF_NULL(host_device_address);
  OffloadAsync(data.original_tensor);

  MS_LOG(DEBUG) << "WaitD2HAndSwitch[" << tensor_id << "]: Address switched to host, device memory can be released";
}

void OffloadExecutor::WaitH2DAndSwitch(size_t tensor_id) {
  auto it = offload_cache_.find(tensor_id);
  if (it == offload_cache_.end()) {
    MS_LOG(EXCEPTION) << "WaitH2DAndSwitch[" << tensor_id << "]: No cached data found.";
  }

  auto &data = it->second;
  MS_EXCEPTION_IF_NULL(data.h2d_complete_event);
  MS_EXCEPTION_IF_NULL(data.original_tensor);
  MS_EXCEPTION_IF_NULL(data.device_tensor);

  // Step 1: Get current compute stream dynamically and wait for H2D completion
  auto compute_stream = hal::CurrentStream();
  MS_EXCEPTION_IF_NULL(compute_stream);
  data.h2d_complete_event->Wait(compute_stream);

  // Step 2: Switch address: original tensor now points to new device memory
  auto device_address = data.device_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_address);
  LoadAsync(data.original_tensor, device_address);

  // Note: host_tensor is NOT cleared, it will be reused for next D2H->H2D cycle

  MS_LOG(DEBUG) << "WaitH2DAndSwitch[" << tensor_id << "]: Address switched to device, device_tensor cleared";
}

void OffloadExecutor::WaitH2DAndFreeCache(size_t tensor_id) {
  WaitH2DAndSwitch(tensor_id);
  FreeCache(tensor_id);
}

void OffloadExecutor::FreeDevice(size_t tensor_id) {
  auto it = offload_cache_.find(tensor_id);
  if (it == offload_cache_.end()) {
    MS_LOG(EXCEPTION) << "FreeDevice[" << tensor_id << "]: No cached data found.";
  }
  auto &data = it->second;
  MS_EXCEPTION_IF_NULL(data.host_tensor);
  MS_EXCEPTION_IF_NULL(data.original_tensor);
  // Only perform OffloadAsync to free device memory.
  // The tensor data is already on host from a previous d2h cycle,
  // so we skip the redundant d2h and just release device memory.
  OffloadAsync(data.original_tensor);
  MS_LOG(DEBUG) << "FreeDevice[" << tensor_id << "]: OffloadAsync executed, device memory freed";
}

void OffloadExecutor::FreeCache(size_t tensor_id) {
  auto it = offload_cache_.find(tensor_id);
  if (it == offload_cache_.end()) {
    MS_LOG(EXCEPTION) << "FreeCache[" << tensor_id << "]: No cached data found.";
  }
  auto &data = it->second;
  data.original_tensor = nullptr;
  data.host_tensor = nullptr;
  data.device_tensor = nullptr;
  MS_LOG(DEBUG) << "FreeCache[" << tensor_id << "]: Cache cleared";
}

void OffloadExecutor::OffloadAsync(const tensor::TensorPtr &source_tensor) {
  auto task = std::make_shared<OffloadAsyncTask>(
    [source_tensor]() { source_tensor->device_address()->ClearDeviceMemory(); }, runtime::TaskType::kBackendTask);
  static bool need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph()) {
    runtime::Pipeline::Get().WaitForward();
    task->Run();
  } else {
    runtime::Pipeline::Get().backend_stage()->Push(task);
  }
}

void OffloadExecutor::LoadAsync(const tensor::TensorPtr &source_tensor,
                                const device::DeviceAddressPtr &device_address) {
  auto task = std::make_shared<OffloadAsyncTask>(
    [source_tensor, device_address]() {
      if (source_tensor->device_address()->GetDevicePtr() != nullptr) {
        MS_LOG(EXCEPTION) << "Source tensor data ptr should be nullptr, which should be offload! But got"
                          << source_tensor->device_address()->GetDevicePtr();
      }
      source_tensor->device_address()->SetDevicePtr(device_address->GetDevicePtr());
      device_address->set_from_mem_pool(false);
    },
    runtime::TaskType::kBackendTask);
  static bool need_sync = runtime::OpExecutor::NeedSync();
  if (need_sync && !runtime::OpExecutor::GetInstance().async_for_graph()) {
    runtime::Pipeline::Get().WaitForward();
    task->Run();
  } else {
    runtime::Pipeline::Get().backend_stage()->Push(task);
  }
}

OffloadTensorData *OffloadExecutor::GetCachedData(size_t tensor_id) {
  auto it = offload_cache_.find(tensor_id);
  if (it == offload_cache_.end()) {
    return nullptr;
  }
  return &it->second;
}

void OffloadExecutor::Reset() {
  offload_cache_.clear();
  MS_LOG(DEBUG) << "OffloadExecutor cache cleared";
}

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore
