/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pybind_api/hal/event_py.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/async/device_task.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/ms_context.h"
#include "include/common/pybind_api/api_register.h"
#include "pipeline/pynative/forward/forward_task.h"
#include "pipeline/pynative/pynative_utils.h"

namespace mindspore {
namespace hal {
std::unordered_map<std::shared_ptr<DeviceEvent>, int64_t> EventCnt::unrecorded_cnt_;
std::mutex EventCnt::unrecorded_cnt_mtx_;

void EventPy::CreateEvent(const StreamPyPtr &stream) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_LOG(DEBUG) << "enable_timing:" << enable_timing_ << ", blocking:" << blocking_;
  event_ = stream->device_ctx()->device_res_manager_->CreateEventWithFlag(enable_timing_, blocking_);
  is_created_ = true;
}

void EventPy::DispatchRecordEventTask(const StreamPyPtr &stream) {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(stream);

  // Record task is in pipline, record cnt firstly. Cnt will be decreased after event was recorded in device.
  EventCnt::IncreaseUnrecordedCnt(event_);

  // Record event async.
  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>([stream, event = event_]() {
    runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
      std::make_shared<pynative::PassthroughDeviceTask>([stream, event]() {
        auto stream_ptr = stream->stream();
        MS_LOG(DEBUG) << "RecordEvent stream_ptr:" << stream_ptr << ", event:" << event;
        event->set_record_stream(stream_ptr);
        event->RecordEvent();
        EventCnt::DecreaseUnrecordedCnt(event);
      }));
  }));
}

void EventPy::Record(const StreamPyPtr &stream) {
  MS_EXCEPTION_IF_NULL(stream);
  if (!is_created_) {
    CreateEvent(stream);
  }
  if (event_ != nullptr) {
    // event_ is nullptr in cpu
    DispatchRecordEventTask(stream);
  }
}

void EventPy::DispatchWaitEventTask(const StreamPyPtr &stream) {
  // Wait event async.
  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>([stream, event = event_]() {
    runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
      std::make_shared<pynative::PassthroughDeviceTask>([stream, event]() {
        auto stream_ptr = stream->stream();
        MS_LOG(DEBUG) << "WaitEvent stream_ptr:" << stream_ptr << ", event:" << event;
        event->set_wait_stream(stream_ptr);
        event->WaitEvent();
      }));
  }));
}

void EventPy::Wait(const StreamPyPtr &stream) {
  if (!is_created_) {
    MS_LOG(DEBUG) << "Event has not been created yet.";
    return;
  }

  if (event_ == nullptr) {
    // event_ is nullptr in cpu
    MS_LOG(DEBUG) << "Event is nullptr, no need to wait.";
    return;
  }

  DispatchWaitEventTask(stream);
}

void EventPy::Synchronize() {
  if (!is_created_) {
    MS_LOG(DEBUG) << "Event has not been created yet.";
    return;
  }

  if (event_ == nullptr) {
    // event_ is nullptr in cpu
    MS_LOG(DEBUG) << "Event is nullptr, no need to Sync.";
    return;
  }

  runtime::OpExecutor::GetInstance().WaitAll();
  event_->SyncEvent();
}

float EventPy::ElapsedTime(const EventPyPtr &other_event) {
  MS_EXCEPTION_IF_NULL(other_event);

  if (!is_created_ || !other_event->is_created()) {
    MS_LOG(EXCEPTION) << "event_ or other_event has not been recorded yet, event is created:" << is_created_
                      << "other_event is created:" << other_event->is_created();
  }

  if (event_ == nullptr || other_event->event() == nullptr) {
    MS_LOG(DEBUG) << "event_ or other_event is nullptr, event:" << event_ << " other_event:" << other_event->event();
    return 0;
  }

  runtime::OpExecutor::GetInstance().WaitAll();
  float elapsed_time = 0;
  event_->ElapsedTime(&elapsed_time, other_event->event().get());
  return elapsed_time;
}

bool EventPy::Query() {
  if (!is_created_) {
    MS_LOG(DEBUG) << "Event has not been created yet.";
    return true;
  }

  if (event_ == nullptr) {
    // event_ is nullptr when device is cpu
    MS_LOG(DEBUG) << "Event is nullptr, no need to Sync.";
    return true;
  }

  if (!EventCnt::IsEventRecorded(event_)) {
    // Event is dispatching, not recorded yet.
    MS_LOG(DEBUG) << "Event is dispatching by async queue";
    return false;
  }
  return event_->QueryEvent();
}

std::string EventPy::ToStringRepr() const {
  std::ostringstream buf;
  buf << "Event(enable_timing=" << enable_timing_ << ", blocking:" << blocking_ << ", is_created:" << is_created_
      << ", event:" << event_.get() << ")";
  return buf.str();
}

void RegEvent(py::module *m) {
  (void)py::class_<EventPy, std::shared_ptr<EventPy>>(*m, "Event")
    .def(py::init<bool, bool>())
    .def("query", &EventPy::Query)
    .def("synchronize", &EventPy::Synchronize, R"mydelimiter(
                             Wait for tasks captured by this event to complete.
                             )mydelimiter")
    .def("query", &EventPy::Query, R"mydelimiter(
                             Query completion status of all tasks captured by this event.
                             )mydelimiter")
    .def("wait", &EventPy::Wait)
    .def("record", &EventPy::Record)
    .def("elapsed_time", &EventPy::ElapsedTime, R"mydelimiter(
                             Return the elapsed time between two events.
                             )mydelimiter")
    .def("__repr__", &EventPy::ToStringRepr);
}
}  // namespace hal
}  // namespace mindspore
