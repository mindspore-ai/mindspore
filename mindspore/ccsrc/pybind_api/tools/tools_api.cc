/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "pybind_api/tools/tools_api.h"
#include "pybind11/pybind11.h"
#include "tools/summary/event_writer.h"
#include "mindspore/ccsrc/pybind_api/tools/tools_api.h"

using EventWriter = mindspore::summary::EventWriter;

namespace mindspore {
namespace py = pybind11;

void RegToolsModule(py::module *m) {
  mindspore::profiler::RegProfilerManager(m);
  mindspore::profiler::RegProfiler(m);
  mindspore::profiler::RegHostProfile(m);
  mindspore::profiler::RegFrameworkProfiler(m);
  mindspore::profiler::RegFrameworkPythonProfileRecorder(m);
  RegStress(m);
  RegParamUtils(m);
  mindspore::datadump::RegDataDump(m);
  mindspore::silentdetect::RegSilentDetect(m);
  RegTFT(m);
  (void)py::class_<EventWriter, std::shared_ptr<EventWriter>>(*m, "EventWriter_")
    .def(py::init<const std::string &>())
    .def("GetFileName", &EventWriter::GetFileName, "Get the file name.")
    .def("Open", &EventWriter::Open, "Open the write file.")
    .def("Write", &EventWriter::Write, "Write the serialize event.")
    .def("EventCount", &EventWriter::GetWriteEventCount, "Write event count.")
    .def("Flush", &EventWriter::Flush, "Flush the event.")
    .def("Close", &EventWriter::Close, "Close the write.")
    .def("Shut", &EventWriter::Shut, "Final close the write.");
}
}  // namespace mindspore
