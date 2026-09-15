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

#ifndef MINDSPORE_CCSRC_PYBIND_API_HAL_EVENT_PY_H
#define MINDSPORE_CCSRC_PYBIND_API_HAL_EVENT_PY_H

#include <memory>
#include "include/pynative/utils/runtime/event.h"
#include "include/utils/visible.h"
#include "pybind_api/runtime/stream_py.h"

namespace mindspore {
namespace hal {
class EventPy;
using EventPyPtr = std::shared_ptr<EventPy>;

// EventPy inherits from Event for Python binding.
// Uses StreamPy (StreamPyPtr) for compatibility with existing Python API.
class PYBIND_EXPORT EventPy : public Event {
 public:
  EventPy() : Event() {}
  explicit EventPy(bool enable_timing, bool blocking) : Event(enable_timing, blocking) {}
  ~EventPy() override;

  // Return the time elapsed in milliseconds after this event was recorded and before the other_event was recorded.
  float ElapsedTime(const EventPyPtr &other_event) { return Event::ElapsedTime(other_event); }

 protected:
  void DispatchRecordEventTask(const StreamPyPtr &stream) override;
  void DispatchWaitEventTask(const StreamPyPtr &stream) override;
};

}  // namespace hal
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_HAL_EVENT_PY_H
