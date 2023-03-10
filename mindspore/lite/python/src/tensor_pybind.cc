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
#include "include/api/data_type.h"
#include "include/api/format.h"
#include "src/common/log_adapter.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "pybind11/stl.h"

namespace mindspore::lite {
namespace py = pybind11;

py::buffer_info GetPyBufferInfo(const MSTensor &tensor);

void TensorPyBind(const py::module &m) {
  (void)py::enum_<DataType>(m, "DataType")
    .value("kTypeUnknown", DataType::kTypeUnknown)
    .value("kObjectTypeString", DataType::kObjectTypeString)
    .value("kObjectTypeList", DataType::kObjectTypeList)
    .value("kObjectTypeTuple", DataType::kObjectTypeTuple)
    .value("kObjectTypeTensorType", DataType::kObjectTypeTensorType)
    .value("kNumberTypeBool", DataType::kNumberTypeBool)
    .value("kNumberTypeInt8", DataType::kNumberTypeInt8)
    .value("kNumberTypeInt16", DataType::kNumberTypeInt16)
    .value("kNumberTypeInt32", DataType::kNumberTypeInt32)
    .value("kNumberTypeInt64", DataType::kNumberTypeInt64)
    .value("kNumberTypeUInt8", DataType::kNumberTypeUInt8)
    .value("kNumberTypeUInt16", DataType::kNumberTypeUInt16)
    .value("kNumberTypeUInt32", DataType::kNumberTypeUInt32)
    .value("kNumberTypeUInt64", DataType::kNumberTypeUInt64)
    .value("kNumberTypeFloat16", DataType::kNumberTypeFloat16)
    .value("kNumberTypeFloat32", DataType::kNumberTypeFloat32)
    .value("kNumberTypeFloat64", DataType::kNumberTypeFloat64)
    .value("kInvalidType", DataType::kInvalidType);

  (void)py::enum_<Format>(m, "Format")
    .value("DEFAULT_FORMAT", Format::DEFAULT_FORMAT)
    .value("NCHW", Format::NCHW)
    .value("NHWC", Format::NHWC)
    .value("NHWC4", Format::NHWC4)
    .value("HWKC", Format::HWKC)
    .value("HWCK", Format::HWCK)
    .value("KCHW", Format::KCHW)
    .value("CKHW", Format::CKHW)
    .value("KHWC", Format::KHWC)
    .value("CHWK", Format::CHWK)
    .value("HW", Format::HW)
    .value("HW4", Format::HW4)
    .value("NC", Format::NC)
    .value("NC4", Format::NC4)
    .value("NC4HW4", Format::NC4HW4)
    .value("NCDHW", Format::NCDHW)
    .value("NWC", Format::NWC)
    .value("NCW", Format::NCW)
    .value("NDHWC", Format::NDHWC)
    .value("NC8HW8", Format::NC8HW8);

  (void)py::class_<MSTensor, std::shared_ptr<MSTensor>>(m, "TensorBind")
    .def(py::init<>())
    .def("set_tensor_name", [](MSTensor &tensor, const std::string &name) { tensor.SetTensorName(name); })
    .def("get_tensor_name", &MSTensor::Name)
    .def("set_data_type", &MSTensor::SetDataType)
    .def("get_data_type", &MSTensor::DataType)
    .def("set_shape", &MSTensor::SetShape)
    .def("get_shape", &MSTensor::Shape)
    .def("set_format", &MSTensor::SetFormat)
    .def("get_format", &MSTensor::format)
    .def("get_element_num", &MSTensor::ElementNum)
    .def("get_data_size", &MSTensor::DataSize)
    .def("set_data", &MSTensor::SetData)
    .def("get_data", &MSTensor::MutableData)
    .def("is_null", [](const MSTensor &tensor) { return tensor == nullptr; })
    .def("set_data_from_numpy",
         [](MSTensor &tensor, const py::array &input) {
           PyArrayObject *darray = PyArray_GETCONTIGUOUS(reinterpret_cast<PyArrayObject *>(input.ptr()));
           void *data = PyArray_DATA(darray);
           auto tensor_data = tensor.MutableData();
           memcpy(tensor_data, data, tensor.DataSize());
           Py_DECREF(darray);
         })
    .def("get_data_to_numpy", [](MSTensor &tensor) -> py::array {
      auto info = GetPyBufferInfo(tensor);
      py::object self = py::cast(&tensor);
      return py::array(py::dtype(info), info.shape, info.strides, info.ptr, self);
    });
}

MSTensor create_tensor() {
  auto tensor = mindspore::MSTensor::CreateTensor("", DataType::kNumberTypeFloat32, {}, nullptr, 0);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "create tensor failed.";
    return {};
  }
  auto copy_tensor = *tensor;
  delete tensor;
  return copy_tensor;
}

std::string GetPyTypeFormat(DataType data_type) {
  switch (data_type) {
    case DataType::kNumberTypeFloat32:
      return py::format_descriptor<float>::format();
    case DataType::kNumberTypeFloat64:
      return py::format_descriptor<double>::format();
    case DataType::kNumberTypeUInt8:
      return py::format_descriptor<uint8_t>::format();
    case DataType::kNumberTypeUInt16:
      return py::format_descriptor<uint16_t>::format();
    case DataType::kNumberTypeUInt32:
      return py::format_descriptor<uint32_t>::format();
    case DataType::kNumberTypeUInt64:
      return py::format_descriptor<uint64_t>::format();
    case DataType::kNumberTypeInt8:
      return py::format_descriptor<int8_t>::format();
    case DataType::kNumberTypeInt16:
      return py::format_descriptor<int16_t>::format();
    case DataType::kNumberTypeInt32:
      return py::format_descriptor<int32_t>::format();
    case DataType::kNumberTypeInt64:
      return py::format_descriptor<int64_t>::format();
    case DataType::kNumberTypeBool:
      return py::format_descriptor<bool>::format();
    case DataType::kObjectTypeString:
      return py::format_descriptor<uint8_t>::format();
    default:
      MS_LOG(ERROR) << "Unsupported DataType " << static_cast<int>(data_type) << ".";
      return "";
  }
}

py::buffer_info GetPyBufferInfo(const MSTensor &tensor) {
  ssize_t item_size = tensor.DataSize() / tensor.ElementNum();
  std::string format = GetPyTypeFormat(tensor.DataType());
  ssize_t ndim = tensor.Shape().size();
  std::vector<ssize_t> shape(tensor.Shape().begin(), tensor.Shape().end());
  std::vector<ssize_t> strides(ndim);
  ssize_t element_num = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strides[i] = element_num * item_size;
    element_num *= shape[i];
  }
  return py::buffer_info{const_cast<MSTensor &>(tensor).MutableData(), item_size, format, ndim, shape, strides};
}
}  // namespace mindspore::lite
