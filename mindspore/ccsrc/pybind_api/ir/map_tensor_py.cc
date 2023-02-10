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
#include <utility>
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

static ValuePtr ConvertMapTensorFilterValue(const py::object &filter_value_obj) {
  ValuePtr filter_value;
  bool convert_ok = parse::ConvertData(filter_value_obj, &filter_value);
  if (!convert_ok || filter_value == nullptr) {
    MS_EXCEPTION(ValueError) << "Incorrect filter value for map parameter: " << py::str(filter_value_obj);
  }
  return filter_value;
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

std::tuple<py::array, py::array, py::array> MapTensorPy::ExportAsNumpy(const MapTensorPtr &map_tensor,
                                                                       bool incremental) {
  MS_EXCEPTION_IF_NULL(map_tensor);
  auto data = map_tensor->Export(incremental);
  return std::make_tuple(TensorPy::AsNumpy(*data.key_tensor), TensorPy::AsNumpy(*data.value_tensor),
                         TensorPy::AsNumpy(*data.status_tensor));
}

static tensor::TensorPtr PyMapTensorGetKeys(const MapTensorPtr &map_tensor) {
  MS_EXCEPTION_IF_NULL(map_tensor);
  return map_tensor->key_tensor();
}

static tensor::TensorPtr PyMapTensorGetValues(const MapTensorPtr &map_tensor) {
  MS_EXCEPTION_IF_NULL(map_tensor);
  return map_tensor->value_tensor();
}

static std::pair<tensor::TensorPtr, tensor::TensorPtr> PyMapTensorGetData(const MapTensorPtr &map_tensor) {
  MS_EXCEPTION_IF_NULL(map_tensor);
  auto keys = map_tensor->key_tensor();
  auto values = map_tensor->value_tensor();
  return std::pair<tensor::TensorPtr, tensor::TensorPtr>(keys, values);
}

namespace tensor {
void RegMapTensor(const py::module *m) {
  // Define python MapTensor class.
  (void)py::class_<MapTensor, MapTensorPtr>(*m, "MapTensor_")
    .def(py::init([](const TypePtr &key_dtype, const TypePtr &value_dtype, const ShapeVector &value_shape,
                     const py::object &default_value_obj, const py::object &permit_filter_obj,
                     const py::object &evict_filter_obj) {
           TypeId key_dtype_id = ((key_dtype != nullptr) ? key_dtype->type_id() : TypeId::kNumberTypeInt32);
           TypeId value_dtype_id = ((value_dtype != nullptr) ? value_dtype->type_id() : TypeId::kNumberTypeFloat32);
           ValuePtr default_value = ConvertMapTensorDefaultValue(default_value_obj, value_dtype);
           ValuePtr permit_filter_value = ConvertMapTensorFilterValue(permit_filter_obj);
           ValuePtr evict_filter_value = ConvertMapTensorFilterValue(evict_filter_obj);
           return std::make_shared<MapTensor>(key_dtype_id, value_dtype_id, value_shape, default_value,
                                              permit_filter_value, evict_filter_value);
         }),
         py::arg("key_dtype"), py::arg("value_dtype"), py::arg("value_shape"), py::arg("default_value"),
         py::arg("permit_filter_value"), py::arg("evict_filter_value"))
    .def(py::init([](const Tensor &key_tensor, const Tensor &value_tensor, const py::object &default_value_obj,
                     const py::object &permit_filter_obj, const py::object &evict_filter_obj) {
           auto key_tensor_ptr = std::make_shared<tensor::Tensor>(key_tensor);
           auto value_tensor_ptr = std::make_shared<tensor::Tensor>(value_tensor);
           auto status_tensor_ptr = std::make_shared<Tensor>(kNumberTypeInt, key_tensor.shape());
           auto value_dtype = key_tensor_ptr->Dtype();
           ValuePtr default_value = ConvertMapTensorDefaultValue(default_value_obj, value_dtype);
           ValuePtr permit_filter_value = ConvertMapTensorFilterValue(permit_filter_obj);
           ValuePtr evict_filter_value = ConvertMapTensorFilterValue(evict_filter_obj);
           return std::make_shared<MapTensor>(key_tensor_ptr, value_tensor_ptr, status_tensor_ptr, default_value,
                                              permit_filter_value, evict_filter_value);
         }),
         py::arg("key_tensor"), py::arg("value_tensor"), py::arg("default_value"), py::arg("permit_filter_value"),
         py::arg("evict_filter_value"))
    .def_property_readonly("key_dtype", &MapTensor::KeyDtype)
    .def_property_readonly("value_dtype", &MapTensor::ValueDtype)
    .def_property_readonly("value_shape", &MapTensor::value_shape)
    .def_property_readonly("size", &MapTensor::size)
    .def("export_data", &MapTensorPy::ExportAsNumpy)
    .def("import_data", &MapTensorPy::UpdateFromNumpy)
    .def("__str__", &MapTensor::ToString)
    .def("__repr__", &MapTensor::ToString)
    .def("get_keys", &PyMapTensorGetKeys)
    .def("get_values", &PyMapTensorGetValues)
    .def("get_data", &PyMapTensorGetData);
}
}  // namespace tensor
}  // namespace mindspore
