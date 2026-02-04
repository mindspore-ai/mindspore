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
#include "pybind11/pybind11.h"
#include "pybind_api/frontend/frontend_api.h"
#include "include/frontend/jit/ps/pipeline_interface.h"

namespace mindspore {
namespace py = pybind11;

void RegPreJit(py::module *m) { (void)m->def("PreJit", &pipeline::PreJit, "Init jit compile environment"); }
}  // namespace mindspore
