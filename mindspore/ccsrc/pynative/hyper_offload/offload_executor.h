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

#ifndef MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_EXECUTOR_H_
#define MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_EXECUTOR_H_

#include <memory>
#include <unordered_map>
#include "ir/tensor.h"
#include "include/pynative/utils/runtime/event.h"
#include "include/pynative/utils/runtime/stream.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

// Cached data for each offloaded tensor
struct OffloadTensorData {
  // Original tensor (hold reference for address switching)
  tensor::TensorPtr original_tensor{nullptr};

  // Host tensor created by D2H (pinned memory, long-term cached for reuse)
  tensor::TensorPtr host_tensor{nullptr};

  // Device tensor created by H2D (temporary, set to nullptr after address switch)
  tensor::TensorPtr device_tensor{nullptr};

  // Event for D2H completion
  hal::EventPtr d2h_complete_event{nullptr};

  // Event for H2D completion
  hal::EventPtr h2d_complete_event{nullptr};
};

// OffloadExecutor handles the actual offload/load operations with event synchronization.
// It manages:
// - D2H (Device to Host) with event record/wait
// - H2D (Host to Device) with event record/wait
// - Event waiting for compute stream synchronization
class OffloadExecutor {
 public:
  // Constructor: only copy_stream is passed in, compute_stream is obtained dynamically
  explicit OffloadExecutor(const hal::StreamPtr &copy_stream);
  ~OffloadExecutor() = default;

  // ========== Core Operations ==========

  // D2H with event synchronization:
  // 1. compute_stream.Record(tmp_event)
  // 2. copy_stream.Wait(tmp_event)
  // 3. to_device(original -> host, pinned memory)
  // 4. copy_stream.Record(d2h_event)
  // Cache: original_tensor, host_tensor, d2h_event
  void D2HWithEvent(size_t tensor_id, const tensor::TensorPtr &original_tensor);

  // H2D with event synchronization:
  // 1. to_device(host -> device)
  // 2. copy_stream.Record(h2d_event)
  // Cache: device_tensor, h2d_event
  void H2DWithEvent(size_t tensor_id);

  // Wait for D2H completion on compute stream and switch address:
  // 1. compute_stream.Wait(d2h_event)
  // 2. original_tensor->set_device_address(host_tensor->device_address())
  void WaitD2HAndSwitch(size_t tensor_id);

  // Wait for H2D completion on compute stream and switch address:
  // 1. compute_stream.Wait(h2d_event)
  // 2. original_tensor->set_device_address(device_tensor->device_address())
  // 3. device_tensor = nullptr (release reference to avoid double memory)
  void WaitH2DAndSwitch(size_t tensor_id);

  // Wait for H2D completion on compute stream and free cache:
  // 1. compute_stream.Wait(h2d_event)
  // 2. free cache
  void WaitH2DAndFreeCache(size_t tensor_id);

  // Free device memory only (OffloadAsync), skip d2h for already-offloaded tensors.
  // Used when an output has multiple d2h/h2d cycles: from the second time onwards,
  // the data is already on host, so we only need to release device memory.
  void FreeDevice(size_t tensor_id);

  void OffloadAsync(const tensor::TensorPtr &source_tensor);

  void LoadAsync(const tensor::TensorPtr &source_tensor, const device::DeviceAddressPtr &device_address);
  // ========== Getters ==========

  hal::StreamPtr copy_stream() const { return copy_stream_; }

  // Get cached data for inspection
  OffloadTensorData *GetCachedData(size_t tensor_id);

  // Clear all cached data
  void Reset();

 private:
  // Create host tensor using to_device (D2H with pinned memory)
  tensor::TensorPtr ToHost(const tensor::TensorPtr &device_tensor);

  // Create device tensor using to_device (H2D)
  tensor::TensorPtr ToDeviceTensor(const tensor::TensorPtr &host_tensor);

  // Free cache
  void FreeCache(size_t tensor_id);

  // Copy stream (dedicated for D2H/H2D)
  hal::StreamPtr copy_stream_{nullptr};

  // Temporary event for compute->copy synchronization
  hal::EventPtr tmp_event_{nullptr};

  // Cache: tensor_id -> OffloadTensorData
  std::unordered_map<size_t, OffloadTensorData> offload_cache_;
};

using OffloadExecutorPtr = std::shared_ptr<OffloadExecutor>;

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OFFLOAD_EXECUTOR_H_
