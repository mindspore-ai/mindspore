/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include <vector>
#include <string>

#include "include/utils/tensor_py.h"
#include "tools/tensor_dump/tensordump.h"
#include "tools/data_dump/dump_control.h"
#include "utils/ms_context.h"
#include "mindspore/ccsrc/pybind_api/tools/tools_api.h"

namespace py = pybind11;
namespace mindspore {
namespace datadump {
void RegDataDump(py::module *m) {
  m->def("_dump_set_dynamic", []() { DumpControl::GetInstance().SetDynamicDump(); })
    .def("_dump_start", []() { DumpControl::GetInstance().DynamicDumpStart(); })
    .def("_dump_stop", []() { DumpControl::GetInstance().DynamicDumpStop(); })
    .def(
      "_set_init_iter", [](std::uint32_t v) { DumpControl::GetInstance().SetInitialIteration(v); }, py::arg("v") = 0)
    .def("_tensordump_set_step", [](const std::vector<size_t> &v) { TensorDumpManager::GetInstance().SetDumpStep(v); })
    .def("_tensordump_exec",
         [](const std::string &filename, py::object &tensor_obj) {
           if (!tensor::IsTensorPy(tensor_obj)) {
             MS_EXCEPTION(TypeError) << "The second input of 'TensorDump': " << py::str(tensor_obj)
                                     << " should be a tensor but got "
                                     << py::cast<std::string>(tensor_obj.attr("__class__").attr("__name__")) << ".";
           }
           auto tensor = tensor::ConvertToTensor(tensor_obj);
           auto cpu_tensor = tensor->cpu();
           return TensorDumpManager::GetInstance().Exec(filename, cpu_tensor, kCallFromPython);
         })
    .def(
      "_dump_step", [](const std::uint32_t step) { DumpControl::GetInstance().UpdateUserDumpStep(step); },
      py::arg("step") = 1);
}

}  // namespace datadump

}  // namespace mindspore
