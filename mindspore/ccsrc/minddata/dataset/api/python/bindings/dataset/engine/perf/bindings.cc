/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/engine/perf/monitor.h"
#include "minddata/dataset/engine/perf/profiling.h"

namespace mindspore {
namespace dataset {
PYBIND_REGISTER(ProfilingManager, 0, ([](const py::module *m) {
                  (void)py::class_<ProfilingManager, std::shared_ptr<ProfilingManager>>(*m, "ProfilingManager")
                    .def("init", [](ProfilingManager &prof_mgr) { THROW_IF_ERROR(prof_mgr.Init()); })
                    .def("start", [](ProfilingManager &prof_mgr) { THROW_IF_ERROR(prof_mgr.Start()); })
                    .def("stop", [](ProfilingManager &prof_mgr) { THROW_IF_ERROR(prof_mgr.Stop()); })
                    .def(
                      "save",
                      [](ProfilingManager &prof_mgr, const std::string &profile_data_path) {
                        THROW_IF_ERROR(prof_mgr.Save(profile_data_path));
                      },
                      py::arg("profile_data_path"));
                }));
}  // namespace dataset
}  // namespace mindspore
