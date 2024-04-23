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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_MULTI_STREAM_CONTROLLER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_MULTI_STREAM_CONTROLLER_H_

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "utils/log_adapter.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "include/backend/visible.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
namespace device {
template <typename T>
struct AtomicWrapper {
  AtomicWrapper() : value_(0L) {}
  explicit AtomicWrapper(const std::atomic<T> &value) : value_(value.load()) {}
  AtomicWrapper(const AtomicWrapper &other) : value_(other.value_.load()) {}
  AtomicWrapper &operator=(const AtomicWrapper &other) { value_.store(other.value_.load()); }

  std::atomic<T> value_;
};

class BACKEND_EXPORT TaskIdOnStreamManager {
 public:
  TaskIdOnStreamManager() = default;

  void Resize(uint32_t stream_size);

  int64_t Query(uint32_t user_stream_id, uint32_t memory_stream_id);

  bool Update(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id);

  int64_t Launch(uint32_t stream_id);

  int64_t Get(uint32_t stream_id);

 private:
  std::mutex mutex_;
  bool initialized_{false};
  uint32_t initialize_size_{0};
  std::vector<AtomicWrapper<int64_t>> generator_;
  std::vector<std::vector<int64_t>> status_;
};

// Event pool recycled with ref count, pool will reuse event when cannot create more events.
class BACKEND_EXPORT EventPool {
 public:
  explicit EventPool(std::function<DeviceEventPtr(void)> event_creator);
  ~EventPool();

  EventPool() = delete;
  EventPool(const EventPool &) = delete;
  EventPool &operator=(const EventPool &) = delete;

  // Get event from pool, event was wrapper by shared_ptr.
  DeviceEventPtr Get();

 private:
  std::mutex mutex_;
  bool expired_{false};
  // Pool will just create event before reach core size, use half of size limits as core size.
  size_t core_size_{32768};
  size_t size_{0};
  std::function<DeviceEventPtr(void)> event_creator_;
  std::list<DeviceEvent *> events_;
  // cached_events_ hold shared ptr of event, since device res manager return a smart pointer.
  std::list<DeviceEventPtr> cached_events_;
};
using EventPoolPtr = std::shared_ptr<EventPool>;

class MultiStreamController;
using MultiStreamControllerPtr = std::shared_ptr<MultiStreamController>;

class BACKEND_EXPORT MultiStreamController {
 public:
  MultiStreamController() = default;
  MultiStreamController(const MultiStreamController &) = delete;
  MultiStreamController &operator=(const MultiStreamController &) = delete;
  ~MultiStreamController() = default;

  static MultiStreamControllerPtr &GetInstance();

  void Refresh(const DeviceContext *device_context);
  bool UpdateTaskIdOnStream(const DeviceContext *device_context, int64_t task_id_on_stream, uint32_t user_stream_id,
                            uint32_t memory_stream_id);
  int64_t QueryTaskIdOnStream(const DeviceContext *device_context, uint32_t user_stream_id, uint32_t memory_stream_id);
  int64_t LaunchTaskIdOnStream(const DeviceContext *device_context, uint32_t stream_id);
  int64_t GetTaskIdOnStream(const DeviceContext *device_context, uint32_t stream_id);

  std::mutex &GetStreamMutex(const DeviceContext *device_context, size_t stream_id);

  // memory_stream_addresses pair : memory_stream_id, address.
  bool RecordEvent(const DeviceContext *device_context, int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses);
  bool WaitEvent(const DeviceContext *device_context, int64_t task_id_on_stream, uint32_t user_stream_id,
                 uint32_t memory_stream_id);
  bool WaitEvent(const DeviceContext *device_context, int64_t task_id_on_stream, uint32_t user_stream_id);
  bool DispatchRecordWaitEvent(const DeviceContext *device_context, uint32_t user_stream_id, uint32_t memory_stream_id);

  bool SyncStream(const DeviceContext *device_context, size_t stream_id);
  bool SyncAllStreams(const DeviceContext *device_context);
  bool SyncNotDefaultStreams(const DeviceContext *device_context);

 private:
  std::unordered_map<const DeviceContext *, TaskIdOnStreamManager> task_id_on_stream_manager_;
  std::unordered_map<const DeviceContext *, std::unordered_map<uint32_t, std::mutex>> stream_mutexes_;
  std::unordered_map<const DeviceContext *, EventPoolPtr> event_pools_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_MULTI_STREAM_CONTROLLER_H_
