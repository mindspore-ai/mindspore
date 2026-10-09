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

#include "pynative/forward/pyboost/customize/pyboost_custom_reg.h"

#include <memory>
#include "include/frontend/operator/primitive_py.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "tools/profiler/profiler.h"

namespace mindspore::pynative {
class CustomExtPrimAdapter : public PrimitiveFunctionAdapter {
 public:
  CustomExtPrimAdapter() : PrimitiveFunctionAdapter() {}
  ~CustomExtPrimAdapter() = default;
  std::string name() override { return "CustomExt"; }
  py::object Call(const py::list &args) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunOp, "CustomExt",
                                       false, true);
    return PyboostCustomExtBase(prim::kPrimCustomExt, args);
  }
};

void RegPyBoostCustomizeModule(py::module *m) {
  (void)py::class_<CustomExtPrimAdapter, PrimitiveFunctionAdapter, std::shared_ptr<CustomExtPrimAdapter>>(
    *m, "CustomExtPrim_")
    .def(py::init<>())
    .def("__call__", &CustomExtPrimAdapter::Call, "Call CustomExt op.");
  m->def("pyboost_custom_ext", &Pyboost_CustomExt, "Encrypt the data.");
  m->def("pyboost_cell_backward_hook", &Pyboost_CellBackwardHook, "CellBackwardHook Ops");
  m->def("pyboost_detach", &PyboostDetach, "Detach");
}
}  // namespace mindspore::pynative
