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

#include "pybind_api/runtime/stream_py.h"
#include "pybind_api/runtime/runtime_api.h"

namespace mindspore {
namespace hal {
void RegStream(py::module *m) {
  (void)py::class_<Stream, std::shared_ptr<Stream>>(*m, "Stream")
    .def(py::init<int>())
    .def(py::init<int, int>())
    .def(py::init<const Stream &>())
    .def("query", &Stream::Query)
    .def("synchronize", &Stream::Synchronize)
    .def("__repr__", &Stream::ToStringRepr)
    .def("__eq__", &Stream::StreamEqual)
    .def("device_stream", &Stream::stream)
    .def("stream_ptr", &Stream::stream_ptr)
    .def_property_readonly("id", &Stream::stream_id)
    .def_property_readonly("device_name", &Stream::device_name)
    .def_property_readonly("device_id", &Stream::device_id);

  (void)m->def("set_cur_stream", &mindspore::hal::SetCurStream, "Set current stream");
  (void)m->def("synchronize", &mindspore::hal::Synchronize, "Synchronize all stream");
  (void)m->def("current_stream", &mindspore::hal::CurrentStream, "Get current stream");
  (void)m->def("current_stream_id", &mindspore::hal::CurrentStreamId, "Get current stream id");
  (void)m->def("default_stream", &mindspore::hal::DefaultStream, "Get default stream");
  (void)m->def("communication_stream", &mindspore::hal::CommunicationStream, "Get communication stream");
}
}  // namespace hal
}  // namespace mindspore
