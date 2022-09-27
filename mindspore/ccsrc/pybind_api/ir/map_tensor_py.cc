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

#include "pybind_api/ir/map_tensor_py.h"
#include <memory>
#include "pybind_api/ir/tensor_py.h"
#include "include/common/pybind_api/api_register.h"
#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"

namespace mindspore {
using tensor::TensorPy;

MapTensorPtr MapTensorPy::MakeMapTensor(const TypePtr &key_dtype, const TypePtr &value_dtype,
                                        const ShapeVector &value_shape) {
  TypeId key_dtype_id = ((key_dtype != nullptr) ? key_dtype->type_id() : TypeId::kNumberTypeInt32);
  TypeId value_dtype_id = ((value_dtype != nullptr) ? value_dtype->type_id() : TypeId::kNumberTypeFloat32);
  return std::make_shared<MapTensor>(key_dtype_id, value_dtype_id, value_shape);
}

void MapTensorPy::UpdateFromNumpy(const MapTensorPtr &map_tensor,
                                  const std::tuple<py::array, py::array, py::array> &numpy_data) {
  MS_EXCEPTION_IF_NULL(map_tensor);
  MapTensor::ExportData data;
  constexpr size_t key_index = 0;
  constexpr size_t value_index = 1;
  constexpr size_t status_index = 2;
  data.key_tensor = TensorPy::MakeTensorOfNumpy(std::get<key_index>(numpy_data));
  data.value_tensor = TensorPy::MakeTensorOfNumpy(std::get<value_index>(numpy_data));
  data.status_tensor = TensorPy::MakeTensorOfNumpy(std::get<status_index>(numpy_data));
  map_tensor->Update(data);
}

std::tuple<py::array, py::array, py::array> MapTensorPy::ExportAsNumpy(const MapTensorPtr &map_tensor, bool full) {
  MS_EXCEPTION_IF_NULL(map_tensor);
  auto data = map_tensor->Export(full);
  return std::make_tuple(TensorPy::AsNumpy(*data.key_tensor), TensorPy::AsNumpy(*data.value_tensor),
                         TensorPy::AsNumpy(*data.status_tensor));
}

namespace tensor {
void RegMapTensor(py::module *m) {
  // Define python MapTensor class.
  (void)py::class_<MapTensor, MapTensorPtr>(*m, "MapTensor_")
    .def(py::init(&MapTensorPy::MakeMapTensor), py::arg("key_dtype"), py::arg("value_dtype"), py::arg("value_shape"))
    .def_property_readonly("key_dtype", &MapTensor::KeyDtype)
    .def_property_readonly("value_dtype", &MapTensor::ValueDtype)
    .def_property_readonly("value_shape", &MapTensor::value_shape)
    .def("get", &MapTensor::Get)
    .def("put", &MapTensor::Put)
    .def("erase", &MapTensor::Erase)
    .def("export", &MapTensorPy::ExportAsNumpy)
    .def("update", &MapTensorPy::UpdateFromNumpy)
    .def("__str__", &MapTensor::ToString)
    .def("__repr__", &MapTensor::ToString);
}
}  // namespace tensor
}  // namespace mindspore
