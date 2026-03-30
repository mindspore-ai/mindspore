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

#include "pybind_api/runtime/event_py.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "include/pynative/utils/runtime/task/device_task.h"

#include "include/pynative/forward/pyboost/forward_task.h"
#include "include/pynative/utils/pynative_utils.h"
#include "pybind_api/runtime/runtime_api.h"

namespace mindspore {
namespace hal {

EventPy::~EventPy() = default;

void EventPy::DispatchRecordEventTask(const StreamPyPtr &stream) {
  MS_EXCEPTION_IF_NULL(event_);
  MS_EXCEPTION_IF_NULL(stream);

  // Record task is in pipeline, record cnt firstly. Cnt will be decreased after event was recorded in device.
  EventCnt::IncreaseUnrecordedCnt(event_);

  // Record event async, dispatch to frontend first then to backend.
  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>(
    [stream, event = event_, record_stream_id = record_stream_id_, task_id_on_stream = task_id_on_stream_]() {
      auto record_fn = [stream, event, record_stream_id, task_id_on_stream]() {
        Event::DoRecordEventTask(stream, event, record_stream_id, task_id_on_stream);
      };
      if (!runtime::OpExecutor::NeedSync()) {
        runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
          std::make_shared<runtime::PassthroughNoWaitDeviceTask>(record_fn));
      } else {
        record_fn();
      }
    }));
}

void EventPy::DispatchWaitEventTask(const StreamPyPtr &stream) {
  // Wait event async, dispatch to frontend first then to backend.
  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>(
    [stream, event = event_, record_stream_id = record_stream_id_, task_id_on_stream = task_id_on_stream_]() {
      auto wait_fn = [stream, event, record_stream_id, task_id_on_stream]() {
        Event::DoWaitEventTask(stream, event, record_stream_id, task_id_on_stream);
      };
      if (!runtime::OpExecutor::NeedSync()) {
        runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
          std::make_shared<runtime::PassthroughNoWaitDeviceTask>(wait_fn));
      } else {
        wait_fn();
      }
    }));
}

void RegEvent(py::module *m) {
  (void)py::class_<EventPy, std::shared_ptr<EventPy>>(*m, "Event")
    .def(py::init<bool, bool>())
    .def("query", &EventPy::Query)
    .def("synchronize", &EventPy::Synchronize)
    .def("wait", &Event::Wait)
    .def("record", &Event::Record)
    .def("elapsed_time", &Event::ElapsedTime)
    .def("__repr__", &EventPy::ToStringRepr);
}

}  // namespace hal
}  // namespace mindspore
