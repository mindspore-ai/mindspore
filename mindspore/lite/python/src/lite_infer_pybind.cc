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
#include "extendrt/cxx_api/model/model_impl.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/functional.h"

namespace mindspore::lite {
namespace py = pybind11;

std::vector<MSTensor> PyPredictModelImpl(ModelImpl *model, const std::vector<MSTensor> &inputs) {
  std::vector<MSTensor> outputs;
  if (!model->Predict(inputs, &outputs).IsOk()) {
    return {};
  }
  return outputs;
}

void LiteInferPyBind(const py::module &m) {
  (void)py::class_<ModelImpl, std::shared_ptr<ModelImpl>>(m, "LiteInferPyBind")
    .def(py::init<>())
    .def("build_from_func_graph",
         py::overload_cast<const FuncGraphPtr &, const std::shared_ptr<Context> &>(&ModelImpl::Build))
    .def("predict", &PyPredictModelImpl, py::call_guard<py::gil_scoped_release>())
    .def("resize", &ModelImpl::Resize)
    .def("get_inputs", &ModelImpl::GetInputs)
    .def("get_outputs", &ModelImpl::GetOutputs);
}
}  // namespace mindspore::lite
