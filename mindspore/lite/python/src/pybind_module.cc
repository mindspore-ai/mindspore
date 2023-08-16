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
#include "include/api/types.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace mindspore::lite {
namespace py = pybind11;

void ContextPyBind(const py::module &m);
void ConverterPyBind(const py::module &m);
void ModelPyBind(const py::module &m);
#ifdef MSLITE_ENABLE_CLOUD_INFERENCE
void LiteInferPyBind(const py::module &m);
#endif
void ModelParallelRunnerPyBind(const py::module &m);
void ModelGroupPyBind(const py::module &m);
void TensorPyBind(const py::module &m);
void LLMEnginePyBind(const py::module &m);
std::shared_ptr<MSTensor> create_tensor(DataType data_type, const std::vector<int64_t> &shape,
                                        const std::string &device_type, int device_id);
std::shared_ptr<MSTensor> create_tensor_by_tensor(const MSTensor &tensor, const std::string &device_type,
                                                  int device_id);
std::shared_ptr<MSTensor> create_tensor_by_numpy(const py::array &input, const std::string &device_type,
                                                 int32_t device_id);
PYBIND11_MODULE(_c_lite_wrapper, m) {
  m.doc() = "MindSpore Lite";
  ContextPyBind(m);
#ifdef ENABLE_CONVERTER
  ConverterPyBind(m);
#endif
  ModelPyBind(m);
#ifdef MSLITE_ENABLE_CLOUD_INFERENCE
  LiteInferPyBind(m);
#endif
  ModelParallelRunnerPyBind(m);
  ModelGroupPyBind(m);
  TensorPyBind(m);
  LLMEnginePyBind(m);
  m.def("create_tensor", &create_tensor);
  m.def("create_tensor_by_tensor", &create_tensor_by_tensor);
  m.def("create_tensor_by_numpy", &create_tensor_by_numpy);
}
}  // namespace mindspore::lite
