/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/jit/pi/external.h"
#include "include/frontend/pybind_api.h"
namespace mindspore {
namespace pijit {
namespace py = pybind11;

// Interface with python
void RegPIJitInterface(py::module *m) {
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 12)
  // pijit cannot support python>=3.12 for now, but will be adapted very soon.
  (void)m->def("jit_mode_pi_enable", []() { return py::bool_(false); });
  (void)m->def("jit_mode_pi_disable", []() { return py::bool_(false); });
  (void)m->def("pi_jit_set_context", [](py::args, py::kwargs) { return py::none(); });
  (void)m->def(
    "update_pijit_default_config", [](py::args, py::kwargs) { return py::none(); }, "update pijit default config");
  (void)m->def(
    "get_code_extra", [](py::args, py::kwargs) { return py::none(); },
    "get copy of code extra which is the pijit compile result");
  (void)m->def(
    "function_id", [](py::args, py::kwargs) { return py::int_(0); },
    "Get cpp function pointer, or python function pointer, or object pointer");
  (void)m->def(
    "clear_jit_compile_results", [](const py::object &) { return py::bool_(false); }, "clear jit compile results");
#else
  // PIJit interface
  (void)m->def("jit_mode_pi_enable", &mindspore::pi_jit_enable, "enable jit from python byte code");
  (void)m->def("jit_mode_pi_disable", &mindspore::pi_jit_disable, "disable jit from python byte code");
  (void)m->def("pi_jit_set_context", &mindspore::PIJitSetContext, "set compile context attribute");
  (void)m->def("update_pijit_default_config", &mindspore::update_pijit_default_config, "update pijit default config");
  (void)m->def("get_code_extra", &mindspore::get_code_extra,
               "get copy of code extra which is the pijit compile result");

  (void)m->def("function_id", &mindspore::FunctionId,
               "Get cpp function pointer, or python function pointer, or object pointer");
  (void)m->def("clear_jit_compile_results", &mindspore::ClearJitCompileResults, "clear jit compile results");
#endif
}
}  // namespace pijit
}  // namespace mindspore
