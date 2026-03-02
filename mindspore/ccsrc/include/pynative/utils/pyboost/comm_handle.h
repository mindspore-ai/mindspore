/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_COMM_HANDLE_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_COMM_HANDLE_H_

#include <memory>
#include <utility>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <optional>
#include <type_traits>
#include <algorithm>
#include "include/utils/visible.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "runtime/hardware_abstract/event/device_event.h"
#include "ir/tensor.h"
#include "include/backend/common/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class PYBOOST_API CommHandle : public std::enable_shared_from_this<CommHandle> {
 public:
  CommHandle() = default;
  explicit CommHandle(const device::DeviceContext *device_ctx) : device_ctx_(device_ctx) {}
  ~CommHandle();

  DeviceEventPtr CreateEvent();

  void RecordEvent(size_t stream_id);

  void UpdateTaskId(size_t stream_id);

  // Update memory stream id from multiple inputs (variadic template)
  template <typename... T>
  void UpdateMemoryStreamId(const T &...args) {
    (GetMemoryStreamIdInfo(args), ...);
  }

  void WaitDeviceEvent(size_t cur_stream_id);

  void ReleaseMultiStreamEvent(size_t cur_stream_id);

  DeviceEventPtr event() const { return event_; }

  const device::DeviceContext *device_ctx() const { return device_ctx_; }

  void Wait();

  void SetOpName(const std::string &op_name) { op_name_ = op_name; }

  const std::string &GetOpName() const { return op_name_; }

  void SetWaitCalled() {
    std::lock_guard<std::mutex> lock(wait_called_mutex_);
    wait_called_.store(true);
    wait_called_cv_.notify_all();
  }
  bool IsWaitCalled() const { return wait_called_.load(); }
  void WaitForWaitCalled() const;
  const std::set<size_t> &memory_stream_id_set() const { return memory_stream_id_set_; }

 private:
  void AddMemoryStreamId(size_t stream_id);
  // Helper function to extract stream_id from a single tensor
  void GetMemoryStreamIdInfoFromInput(const tensor::TensorPtr &tensor);

  // Template function to handle optional input
  template <typename T>
  void GetMemoryStreamIdInfo(const std::optional<T> &opt) {
    if (opt.has_value()) {
      return GetMemoryStreamIdInfo(opt.value());
    }
  }

  // Template function to handle vector input
  template <typename T>
  void GetMemoryStreamIdInfo(const std::vector<T> &inputs) {
    if constexpr (!std::is_same_v<T, tensor::TensorPtr>) {
      return;
    }
    for_each(inputs.begin(), inputs.end(), [this](auto item) { GetMemoryStreamIdInfoFromInput(item); });
  }

  // Template function to handle single input (tensor)
  template <typename T, typename = typename std::enable_if_t<!is_vector<T>::value && !is_optional<T>::value, T>>
  void GetMemoryStreamIdInfo(const T &input) {
    if constexpr (std::is_same_v<T, tensor::TensorPtr>) {
      GetMemoryStreamIdInfoFromInput(input);
    }
  }

 private:
  DeviceEventPtr event_{nullptr};
  // Transfer between multi-stage pipelines
  std::shared_ptr<int64_t> task_id_on_stream_ = std::make_shared<int64_t>(0L);
  std::shared_ptr<size_t> record_stream_id_ = std::make_shared<size_t>(0);
  std::set<size_t> memory_stream_id_set_;
  const device::DeviceContext *device_ctx_;
  std::string op_name_;                             // Operation name for debugging
  mutable std::mutex wait_called_mutex_;            // Mutex for wait_called_ condition variable
  mutable std::condition_variable wait_called_cv_;  // Condition variable for wait_called_
  std::atomic<bool> wait_called_{false};            // Track if wait() has been called (thread-safe)
};

using CommHandlePtr = std::shared_ptr<CommHandle>;
void PYBOOST_API WaitTaskFunc(CommHandlePtr comm_handle);

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_PYBOOST_COMM_HANDLE_H_
