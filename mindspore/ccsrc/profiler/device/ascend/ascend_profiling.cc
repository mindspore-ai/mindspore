/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "profiler/device/ascend/ascend_profiling.h"
#include <string>
#include "pybind_api/api_register.h"
#include "utils/log_adapter.h"
#include "utils/utils.h"

namespace mindspore {
namespace profiler {
namespace ascend {
std::shared_ptr<AscendProfiler> AscendProfiler::ascend_profiler_ = std::make_shared<AscendProfiler>();

std::shared_ptr<AscendProfiler> &AscendProfiler::GetInstance() { return ascend_profiler_; }

void AscendProfiler::StepProfilingEnable(const bool enable_flag) {
  MS_LOG(INFO) << "Start profiling";
  enable_flag_ = enable_flag;
}

void AscendProfiler::Start(const std::string &profiling_options) {
  profiling_options_ = profiling_options;
  StepProfilingEnable(true);
}

void AscendProfiler::Stop() {
  MS_LOG(INFO) << "Stop profiling";
  StepProfilingEnable(false);
}

REGISTER_PYBIND_DEFINE(AscendProfiler_, ([](const py::module *m) {
                         (void)py::class_<AscendProfiler, std::shared_ptr<AscendProfiler>>(*m, "AscendProfiler")
                           .def_static("get_instance", &AscendProfiler::GetInstance, "AscendProfiler get_instance.")
                           .def("start", &AscendProfiler::Start, py::arg("profiling_options"), "start")
                           .def("stop", &AscendProfiler::Stop, "stop");
                       }));
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
