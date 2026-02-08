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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_EXTERNAL_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_EXTERNAL_H

#include "pybind11/pybind11.h"

namespace py = pybind11;
namespace mindspore {
py::bool_ pi_jit_enable();
py::bool_ pi_jit_disable();
bool pi_jit_should_compile(const py::handle &func_handle);
py::object get_code_extra(const py::object &);
void update_pijit_default_config(const py::kwargs &conf);
void PIJitSetContext(py::args, py::kwargs);
size_t FunctionId(const py::object &callable);
bool ClearJitCompileResults(const py::handle &func);
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_EXTERNAL_H
