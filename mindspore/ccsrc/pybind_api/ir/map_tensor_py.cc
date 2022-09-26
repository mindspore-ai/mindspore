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
#include <string>
#include "pybind11/pytypes.h"
#include "pybind_api/ir/tensor_py.h"
#include "include/common/pybind_api/api_register.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/parse/parse_base.h"
#include "utils/hash_set.h"
#include "utils/log_adapter.h"

namespace mindspore {
using tensor::TensorPy;

static ValuePtr ConvertMapTensorDefaultValue(const py::object &default_value_obj, const TypePtr &value_dtype) {
  static const mindspore::HashSet<std::string> support_init_names = {"zeros", "ones", "normal"};
  if (py::isinstance<py::str>(default_value_obj)) {
    std::string init_name = py::cast<std::string>(default_value_obj);
    if (support_init_names.find(init_name) == support_init_names.end()) {
      MS_EXCEPTION(ValueError) << "Unsupported init name for map parameter: " << init_name;
    }
    return std::make_shared<StringImm>(init_name);
  }
  ValuePtr default_value;
  bool convert_ok = parse::ConvertData(default_value_obj, &default_value, false, value_dtype, false);
  if (!convert_ok || default_value == nullptr) {
    MS_EXCEPTION(ValueError) << "Incorrect default value for map parameter: " << py::str(default_value_obj);
  }
  return default_value;
}

MapTensorPtr MapTensorPy::MakeMapTensor(const TypePtr &key_dtype, const TypePtr &value_dtype,
                                        const ShapeVector &value_shape, const py::object &default_value_obj) {
  TypeId key_dtype_id = ((key_dtype != nullptr) ? key_dtype->type_id() : TypeId::kNumberTypeInt32);
  TypeId value_dtype_id = ((value_dtype != nullptr) ? value_dtype->type_id() : TypeId::kNumberTypeFloat32);
  ValuePtr default_value = ConvertMapTensorDefaultValue(default_value_obj, value_dtype);
  return std::make_shared<MapTensor>(key_dtype_id, value_dtype_id, value_shape, default_value);
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

// Python wrapper for MapTensor::Get.
static tensor::TensorPtr PyMapTensorGet(const MapTensorPtr &map_tensor, const tensor::TensorPtr &key_tensor,
                                        const py::object &default_value_obj) {
  MS_EXCEPTION_IF_NULL(map_tensor);
  ValuePtr default_value =
    (default_value_obj.is_none() ? map_tensor->default_value()
                                 : ConvertMapTensorDefaultValue(default_value_obj, map_tensor->ValueDtype()));
  return map_tensor->Get(key_tensor, default_value);
}

namespace tensor {
void RegMapTensor(py::module *m) {
  // Define python MapTensor class.
  (void)py::class_<MapTensor, MapTensorPtr>(*m, "MapTensor_")
    .def(py::init(&MapTensorPy::MakeMapTensor), py::arg("key_dtype"), py::arg("value_dtype"), py::arg("value_shape"),
         py::arg("default_value"))
    .def_property_readonly("key_dtype", &MapTensor::KeyDtype)
    .def_property_readonly("value_dtype", &MapTensor::ValueDtype)
    .def_property_readonly("value_shape", &MapTensor::value_shape)
    .def("get", &PyMapTensorGet)
    .def("put", &MapTensor::Put)
    .def("erase", &MapTensor::Erase)
    .def("export", &MapTensorPy::ExportAsNumpy)
    .def("update", &MapTensorPy::UpdateFromNumpy)
    .def("__str__", &MapTensor::ToString)
    .def("__repr__", &MapTensor::ToString);
}
}  // namespace tensor
}  // namespace mindspore
