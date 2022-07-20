/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "include/api/types.h"

namespace mindspore::lite {
namespace py = pybind11;

void ContextPyBind(const py::module &m);
void ConverterPyBind(const py::module &m);
void ModelPyBind(const py::module &m);
void ModelParallelRunnerPyBind(const py::module &m);
void TensorPyBind(const py::module &m);
MSTensor create_tensor();

PYBIND11_MODULE(_c_lite_wrapper, m) {
  m.doc() = "MindSpore Lite";
  ContextPyBind(m);
#ifdef ENABLE_CONVERTER
  ConverterPyBind(m);
#endif
  ModelPyBind(m);
  ModelParallelRunnerPyBind(m);
  TensorPyBind(m);
  m.def("create_tensor", &create_tensor);
}
}  // namespace mindspore::lite
