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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_HARDWARE_ABSTRACT_EVENT_EVENT_H
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_HARDWARE_ABSTRACT_EVENT_EVENT_H

#include <string>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "runtime/hardware_abstract/visible.h"
#include "include/runtime/hardware_abstract/stream/stream.h"
#include "runtime/hardware_abstract/event/device_event.h"

namespace mindspore {
namespace hal {
class Event;
using EventPtr = std::shared_ptr<Event>;

class RUNTIME_HARDWARE_EXPORT Event {
 public:
  Event() = default;
  explicit Event(bool enable_timing, bool blocking) : enable_timing_(enable_timing), blocking_(blocking) {}
  virtual ~Event();

  // Record this event by stream
  virtual void Record(const StreamPtr &stream);

  // Query if this event has completed
  bool Query();

  // Sync event
  void Synchronize();

  // The given stream wait for this event
  virtual void Wait(const StreamPtr &stream);

  // Return the time elapsed in milliseconds after this event was recorded and before the other_event was recorded.
  float ElapsedTime(const EventPtr &other_event);

  std::shared_ptr<DeviceEvent> event() { return event_; }

  std::string ToStringRepr() const;

  bool is_created() const { return is_created_; }

 protected:
  // Lazy init, event will be created when event is recording.
  void CreateEvent(const StreamPtr &stream);

  virtual void DispatchRecordEventTask(const StreamPtr &stream);

  virtual void DispatchWaitEventTask(const StreamPtr &stream);

  // Core logic for record event task, can be called from different dispatch contexts
  static void DoRecordEventTask(const StreamPtr &stream, const std::shared_ptr<DeviceEvent> &event,
                                size_t record_stream_id, const std::shared_ptr<int64_t> &task_id_on_stream);

  // Core logic for wait event task, can be called from different dispatch contexts
  static void DoWaitEventTask(const StreamPtr &stream, const std::shared_ptr<DeviceEvent> &event,
                              size_t record_stream_id, const std::shared_ptr<int64_t> &task_id_on_stream);

  bool enable_timing_{false};
  bool blocking_{false};
  // is_created_ will be true after event was recorded.
  bool is_created_{false};
  // Store event alloc from device.
  std::shared_ptr<DeviceEvent> event_{nullptr};
  // The stream object that helps create event_. We can use this to access device_res_manager_;
  StreamPtr creator_stream_{nullptr};
  // Transfer between multi-stage pipelines
  std::shared_ptr<int64_t> task_id_on_stream_ = std::make_shared<int64_t>(0L);
  size_t record_stream_id_{0};
  device::DeviceContext *device_ctx_{nullptr};
};

class RUNTIME_HARDWARE_EXPORT EventCnt {
 public:
  static void IncreaseUnrecordedCnt(const std::shared_ptr<DeviceEvent> &event) {
    std::lock_guard<std::mutex> lock(unrecorded_cnt_mtx_);
    unrecorded_cnt_[event]++;
    MS_LOG(DEBUG) << "unrecorded_cnt:" << unrecorded_cnt_[event];
  }

  static void DecreaseUnrecordedCnt(const std::shared_ptr<DeviceEvent> &event) {
    std::lock_guard<std::mutex> lock(unrecorded_cnt_mtx_);

    if (unrecorded_cnt_[event] <= 1) {
      unrecorded_cnt_.erase(event);
      MS_LOG(DEBUG) << "unrecorded_cnt:0";
    } else {
      unrecorded_cnt_[event]--;
      MS_LOG(DEBUG) << "unrecorded_cnt:" << unrecorded_cnt_[event];
    }
  }

  static bool IsEventRecorded(const std::shared_ptr<DeviceEvent> &event) {
    std::lock_guard<std::mutex> lock(unrecorded_cnt_mtx_);
    return unrecorded_cnt_.find(event) == unrecorded_cnt_.end();
  }

 private:
  // The num of event in pipline.
  static std::unordered_map<std::shared_ptr<DeviceEvent>, int64_t> unrecorded_cnt_;
  static std::mutex unrecorded_cnt_mtx_;
};

}  // namespace hal
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_HARDWARE_ABSTRACT_EVENT_EVENT_H
