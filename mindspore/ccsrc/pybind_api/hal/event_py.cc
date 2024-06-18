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
#include "runtime/pipeline/task/device_task.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/ms_context.h"
#include "include/common/pybind_api/api_register.h"
#include "pipeline/pynative/forward/forward_task.h"
#include "pipeline/pynative/pynative_utils.h"
#include "runtime/device/multi_stream_controller.h"

namespace mindspore {
namespace hal {
std::unordered_map<std::shared_ptr<DeviceEvent>, int64_t> EventCnt::unrecorded_cnt_;
std::mutex EventCnt::unrecorded_cnt_mtx_;

EventPy::~EventPy() {
  if (creator_stream_ != nullptr && event_ != nullptr) {
    const auto &device_ctx = creator_stream_->device_ctx();
    pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>([device_ctx, event = event_]() {
      auto destroy_fn = [device_ctx, event]() {
        MS_LOG(DEBUG) << "DestroyEvent, event:" << event;
        if (device_ctx != nullptr && device_ctx->initialized()) {
          device_ctx->device_res_manager_->DestroyEvent(event);
        }
      };
      if (!runtime::OpExecutor::NeedSync()) {
        runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
          std::make_shared<runtime::PassthroughDeviceTask>(destroy_fn));
      } else {
        destroy_fn();
      }
    }));
  }
  creator_stream_ = nullptr;
  event_ = nullptr;
}

void EventPy::CreateEvent(const StreamPyPtr &stream) {
  MS_EXCEPTION_IF_NULL(stream);
  creator_stream_ = stream;
  MS_LOG(DEBUG) << "enable_timing:" << enable_timing_ << ", blocking:" << blocking_;
  event_ = creator_stream_->device_ctx()->device_res_manager_->CreateEventWithFlag(enable_timing_, blocking_);
  is_created_ = true;
}

void EventPy::DispatchRecordEventTask(const StreamPyPtr &stream) {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(stream);

  // Record task is in pipline, record cnt firstly. Cnt will be decreased after event was recorded in device.
  EventCnt::IncreaseUnrecordedCnt(event_);

  // Record event async.
  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>(
    [stream, event = event_, record_stream_id = record_stream_id_, task_id_on_stream = task_id_on_stream_]() {
      auto record_fn = [stream, event, record_stream_id, task_id_on_stream]() {
        device::MultiStreamController::GetInstance()->Refresh(stream->device_ctx());
        auto task_id =
          device::MultiStreamController::GetInstance()->LaunchTaskIdOnStream(stream->device_ctx(), record_stream_id);
        *task_id_on_stream = task_id;
        auto stream_ptr = stream->stream();
        event->set_record_stream(stream_ptr);
        event->RecordEvent();
        MS_LOG(DEBUG) << "RecordEvent record_stream_id:" << record_stream_id << ", event:" << event << ", stream_ptr"
                      << stream_ptr << ", task_id_on_stream:" << *task_id_on_stream;
        EventCnt::DecreaseUnrecordedCnt(event);
      };
      if (!runtime::OpExecutor::NeedSync()) {
        runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
          std::make_shared<runtime::PassthroughDeviceTask>(record_fn));
      } else {
        record_fn();
      }
    }));
}

void EventPy::Record(const StreamPyPtr &stream) {
  MS_EXCEPTION_IF_NULL(stream);
  if (!is_created_) {
    CreateEvent(stream);
    device_ctx_ = stream->device_ctx();
  }
  if (event_ != nullptr) {
    record_stream_id_ = stream->stream_id();
    // event_ is nullptr in cpu
    DispatchRecordEventTask(stream);
  }
}

void EventPy::DispatchWaitEventTask(const StreamPyPtr &stream) {
  // Wait event async.
  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>(
    [stream, event = event_, record_stream_id = record_stream_id_, task_id_on_stream = task_id_on_stream_]() {
      auto wait_fn = [stream, event, record_stream_id, task_id_on_stream]() {
        auto stream_ptr = stream->stream();
        MS_LOG(DEBUG) << "WaitEvent wait stream id:" << stream->stream_id() << ", record_stream_id:" << record_stream_id
                      << ", event:" << event << ", task_id_on_stream:" << *task_id_on_stream;
        event->set_wait_stream(stream_ptr);
        event->WaitEventWithoutReset();

        // Release cross stream memory event, mark record_stream_id is use stream id, wait stream id is memory stream
        // id.
        (void)device::MultiStreamController::GetInstance()->WaitEvent(stream->device_ctx(), *task_id_on_stream,
                                                                      record_stream_id, stream->stream_id());
      };
      if (!runtime::OpExecutor::NeedSync()) {
        runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
          std::make_shared<runtime::PassthroughDeviceTask>(wait_fn));
      } else {
        wait_fn();
      }
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

  runtime::Pipeline::Get().WaitForward();
  event_->SyncEvent();
  MS_EXCEPTION_IF_NULL(device_ctx_);
  // Clear cross stream memory event which task id less than task_id_on_stream.
  (void)device::MultiStreamController::GetInstance()->WaitEvent(device_ctx_, *task_id_on_stream_, record_stream_id_);
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

  runtime::Pipeline::Get().WaitForward();
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
    .def("synchronize", &EventPy::Synchronize)
    .def("wait", &EventPy::Wait)
    .def("record", &EventPy::Record)
    .def("elapsed_time", &EventPy::ElapsedTime)
    .def("__repr__", &EventPy::ToStringRepr);
}
}  // namespace hal
}  // namespace mindspore
