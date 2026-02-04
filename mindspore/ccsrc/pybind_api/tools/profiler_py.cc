
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
#include <utility>

#include "tools/profiler/profiling.h"
#include "tools/profiler/profiler.h"
#include "mindspore/ccsrc/pybind_api/tools/tools_api.h"

namespace mindspore {
namespace profiler {
void RegProfiler(const py::module *m) {
  (void)py::class_<Profiler, std::shared_ptr<Profiler>>(*m, "Profiler")
    .def_static("get_instance", &Profiler::GetInstance, py::arg("device_name"), "Profiler get_instance.")
    .def("init", &Profiler::Init, py::arg("profiling_path"), py::arg("device_id") = py::int_(0),
         py::arg("profiling_options") = py::str(""), "init")
    .def("start", &Profiler::Start, "start")
    .def("stop", &Profiler::Stop, "stop")
    .def("finalize", &Profiler::Finalize, "finalize")
    .def("sync_enable", &Profiler::SyncEnable, py::arg("enable_flag"))
    .def("data_process_enable", &Profiler::DataProcessEnable, py::arg("enable_flag"))
    .def("step_profiling_enable", &Profiler::StepProfilingEnable, py::arg("enable_flag"),
         "enable or disable step profiling")
    .def("enable_op_time", &Profiler::EnableOpTime, "Enable op_time.")
    .def("enable_profile_memory", &Profiler::EnableProfileMemory, "Enable profile_memory.")
    .def("mstx_mark", &Profiler::MstxMark, py::arg("message"), py::arg("stream") = py::none(),
         py::arg("domain") = py::str("default"), "Mark a profiling point")
    .def("mstx_range_start", &Profiler::MstxRangeStart, py::arg("message"), py::arg("stream") = py::none(),
         py::arg("domain") = py::str("default"), "Start a profiling range")
    .def("mstx_range_end", &Profiler::MstxRangeEnd, py::arg("range_id"), py::arg("domain") = py::str("default"),
         "End a profiling range");
}

void RegProfilerManager(const py::module *m) {
  (void)py::class_<ProfilerManager, std::shared_ptr<ProfilerManager>>(*m, "ProfilerManager")
    .def_static("get_instance", &ProfilerManager::GetInstance, "ProfilerManager get_instance.")
    .def("dynamic_status", &ProfilerManager::GetNetDynamicShapeStatus, "dynamic_status")
    .def("set_profile_framework", &ProfilerManager::SetProfileFramework, py::arg("profile_framework"));
}

// level: 0, for developer user, 1, for general user;
// Default parameter for host profile meaning: for developer user, collect both time and memory, record timestamp.
void RegHostProfile(py::module *m) {
  m->def("collect_host_info", &CollectHostInfo, py::arg("module"), py::arg("event"), py::arg("stage"),
         py::arg("start_time") = py::int_(0), py::arg("end_time") = py::int_(0), py::arg("level") = py::int_(0),
         py::arg("custom_info") = py::dict())
    .def("get_clock_time", &GetClockTime)
    .def("get_clock_syscnt", &GetClockSyscnt);
}

void RegFrameworkProfiler(py::module *m) {
  m->def(
     "_framework_profiler_step_start", []() { runtime::ProfilerAnalyzer::GetInstance().StartStep(); },
     "Profiler step start")
    .def(
      "_framework_profiler_step_end", []() { runtime::ProfilerAnalyzer::GetInstance().EndStep(); }, "Profiler step end")
    .def(
      "_framework_profiler_clear", []() { runtime::ProfilerAnalyzer::GetInstance().Clear(); },
      "Dump json and clear data")
    .def("_framework_profiler_enable_mi", []() { runtime::ProfilerAnalyzer::GetInstance().EnableMiProfile(); })
    .def("_framework_profiler_disable_mi", []() { runtime::ProfilerAnalyzer::GetInstance().DisableMiProfile(); });
}

void RegFrameworkPythonProfileRecorder(py::module *m) {
  (void)py::class_<runtime::PythonProfilerRecorder, std::shared_ptr<runtime::PythonProfilerRecorder>>(
    *m, "PythonProfilerRecorder")
    .def(py::init<const std::string &>())
    .def("record_start", &runtime::PythonProfilerRecorder::record_start, "record_start")
    .def("record_end", &runtime::PythonProfilerRecorder::record_end, "record_end");
}
}  // namespace profiler
}  // namespace mindspore
