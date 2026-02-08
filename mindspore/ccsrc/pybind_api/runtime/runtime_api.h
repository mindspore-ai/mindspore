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

#ifndef MINDSPORE_CCSRC_PYBIND_API_RUNTIME_RUNTIME_API_H_
#define MINDSPORE_CCSRC_PYBIND_API_RUNTIME_RUNTIME_API_H_

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "include/utils/visible.h"

namespace py = pybind11;
namespace mindspore {
namespace hal {
void RegStream(py::module *m);
void RegEvent(py::module *m);
PYNATIVE_EXPORT void RegCommHandle(py::module *m);
void RegMemory(py::module *m);
void RegUtils(py::module *m);
void RegResLimit(py::module *m);
}  // namespace hal

namespace runtime {
void RegRuntimeConf(py::module *m);
}  // namespace runtime

void RegDeviceManagerConf(const py::module *m);
void RegRuntimeModule(py::module *m);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_RUNTIME_RUNTIME_API_H_
