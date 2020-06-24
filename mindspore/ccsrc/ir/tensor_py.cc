/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ir/tensor_py.h"

#include <functional>
#include <numeric>
#include <vector>
#include <sstream>
#include <string>

#include "device/device_address.h"
#include "pybind_api/api_register.h"
#include "pybind_api/export_flags.h"
#include "pipeline/static_analysis/abstract_value.h"

namespace mindspore {
namespace tensor {

static TypeId GetDataType(const py::buffer_info &buf) {
  if (buf.format.size() == 1) {
    switch (buf.format.front()) {
      case 'e':
      case 'f':
      case 'd':
        switch (buf.itemsize) {
          case 2:
            return TypeId::kNumberTypeFloat16;
          case 4:
            return TypeId::kNumberTypeFloat32;
          case 8:
            return TypeId::kNumberTypeFloat64;
        }
        break;
      case 'b':
      case 'h':
      case 'i':
      case 'l':
      case 'q':
        switch (buf.itemsize) {
          case 1:
            return TypeId::kNumberTypeInt8;
          case 2:
            return TypeId::kNumberTypeInt16;
          case 4:
            return TypeId::kNumberTypeInt32;
          case 8:
            return TypeId::kNumberTypeInt64;
        }
        break;
      case 'B':
      case 'H':
      case 'I':
      case 'L':
      case 'Q':
        switch (buf.itemsize) {
          case 1:
            return TypeId::kNumberTypeUInt8;
          case 2:
            return TypeId::kNumberTypeUInt16;
          case 4:
            return TypeId::kNumberTypeUInt32;
          case 8:
            return TypeId::kNumberTypeUInt64;
        }
        break;
      case '?':
        return TypeId::kNumberTypeBool;
    }
  }
  MS_LOG(WARNING) << "Unsupported DataType format " << buf.format << " item size " << buf.itemsize;
  return TypeId::kTypeUnknown;
}

static std::string GetPyTypeFormat(TypeId data_type) {
  switch (data_type) {
    case TypeId::kNumberTypeFloat16:
      return "e";
    case TypeId::kNumberTypeFloat32:
      return py::format_descriptor<float>::format();
    case TypeId::kNumberTypeFloat64:
      return py::format_descriptor<double>::format();
    case TypeId::kNumberTypeUInt8:
      return py::format_descriptor<uint8_t>::format();
    case TypeId::kNumberTypeUInt16:
      return py::format_descriptor<uint16_t>::format();
    case TypeId::kNumberTypeUInt32:
      return py::format_descriptor<uint32_t>::format();
    case TypeId::kNumberTypeUInt64:
      return py::format_descriptor<uint64_t>::format();
    case TypeId::kNumberTypeInt8:
      return py::format_descriptor<int8_t>::format();
    case TypeId::kNumberTypeInt16:
      return py::format_descriptor<int16_t>::format();
    case TypeId::kNumberTypeInt32:
      return py::format_descriptor<int32_t>::format();
    case TypeId::kNumberTypeInt64:
      return py::format_descriptor<int64_t>::format();
    case TypeId::kNumberTypeBool:
      return py::format_descriptor<bool>::format();
    default:
      MS_LOG(WARNING) << "Unsupported DataType " << data_type << ".";
      return "";
  }
}

static bool IsCContiguous(const py::array &input) {
  auto flags = static_cast<unsigned int>(input.flags());
  return (flags & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) != 0;
}

TensorPtr TensorPy::MakeTensor(const py::array &input, const TypePtr &type_ptr) {
  // Get input buffer info.
  py::buffer_info buf = input.request();
  // Check data types.
  auto data_type = type_ptr ? type_ptr->type_id() : TypeId::kTypeUnknown;
  auto buf_type = GetDataType(buf);
  if (buf_type == TypeId::kTypeUnknown && data_type == TypeId::kTypeUnknown) {
    MS_LOG(EXCEPTION) << "Unsupported tensor type!";
  }
  // Use buf type as data type if type_ptr not set.
  if (data_type == TypeId::kTypeUnknown) {
    data_type = buf_type;
  }
  // Convert input array to C contiguous if need.
  std::unique_ptr<char[]> tmp_buf;
  if (!IsCContiguous(input)) {
    Py_buffer pybuf;
    if (PyObject_GetBuffer(input.ptr(), &pybuf, PyBUF_ANY_CONTIGUOUS)) {
      MS_LOG(EXCEPTION) << "Failed to get buffer from the input!";
    }
    tmp_buf = std::make_unique<char[]>(pybuf.len);
    if (PyBuffer_ToContiguous(tmp_buf.get(), &pybuf, pybuf.len, 'C')) {
      MS_LOG(EXCEPTION) << "Can't copy numpy.ndarray to a contiguous buffer.";
    }
    PyBuffer_Release(&pybuf);
    buf.ptr = tmp_buf.get();
  }
  // Get tensor shape.
  std::vector<int> shape(buf.shape.begin(), buf.shape.end());
  if (data_type == buf_type) {
    // Use memory copy if input data type is same as the required type.
    return std::make_shared<Tensor>(data_type, shape, buf.ptr, buf.size * buf.itemsize);
  }
  // Create tensor with data type converted.
  return std::make_shared<Tensor>(data_type, shape, buf.ptr, buf_type);
}

static std::vector<ssize_t> GetStrides(const std::vector<ssize_t> &shape, ssize_t item_size) {
  std::vector<ssize_t> strides;
  strides.reserve(shape.size());
  const auto ndim = shape.size();
  for (size_t i = 0; i < ndim; ++i) {
    auto stride = item_size;
    for (size_t j = i + 1; j < ndim; ++j) {
      stride *= shape[j];
    }
    strides.push_back(stride);
  }
  return strides;
}

static py::buffer_info GetPyBufferInfo(const Tensor &tensor) {
  std::vector<ssize_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<ssize_t> strides = GetStrides(shape, tensor.data().itemsize());
  return py::buffer_info{
    tensor.data_c(), tensor.data().itemsize(), GetPyTypeFormat(tensor.data_type()), tensor.DataDim(), shape, strides};
}

py::tuple TensorPy::GetPyTupleShape(const Tensor &tensor) {
  auto &shape = tensor.shape();
  py::tuple dims(shape.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = py::int_(shape[i]);
  }
  return dims;
}

py::array TensorPy::SyncAsNumpy(const Tensor &tensor) {
  tensor.data_sync();
  auto info = GetPyBufferInfo(tensor);
  py::object self = py::cast(&tensor);
  return py::array(py::dtype(info), info.shape, info.strides, info.ptr, self);
}

py::array TensorPy::AsNumpy(const Tensor &tensor) {
  auto info = GetPyBufferInfo(tensor);
  py::object self = py::cast(&tensor);
  return py::array(py::dtype(info), info.shape, info.strides, info.ptr, self);
}

static std::vector<int> GetShapeFromTuple(const py::tuple &tuple) {
  std::vector<int> shape;
  const size_t size = tuple.size();
  shape.reserve(tuple.size());
  for (size_t i = 0; i < size; ++i) {
    shape.push_back(py::int_(tuple[i]));
  }
  return shape;
}

REGISTER_PYBIND_DEFINE(Tensor, ([](const py::module *m) {
                         // Define python Tensor class.
                         // dtype should define before Tensor, because Tensor init depend dtype
                         (void)py::class_<Tensor, std::shared_ptr<Tensor>>(*m, "Tensor")
                           .def(py::init([](const Tensor &tensor) { return std::make_shared<Tensor>(tensor); }),
                                py::arg("input"))
                           .def(py::init([](const Tensor &tensor, const TypePtr &type_ptr) {
                                  TypeId data_type = type_ptr ? type_ptr->type_id() : kTypeUnknown;
                                  if (data_type == kTypeUnknown || tensor.data_type() == data_type) {
                                    return std::make_shared<Tensor>(tensor);
                                  }
                                  return std::make_shared<Tensor>(tensor, data_type);
                                }),
                                py::arg("input"), py::arg("dtype"))
                           .def(py::init([](const TypePtr &type_ptr, const py::tuple &shape) {
                                  auto data_type = type_ptr ? type_ptr->type_id() : TypeId::kNumberTypeFloat64;
                                  return std::make_shared<Tensor>(data_type, GetShapeFromTuple(shape));
                                }),
                                py::arg("dtype"), py::arg("shape"))
                           .def(py::init([](const py::array &input, const TypePtr &type_ptr) {
                                  return TensorPy::MakeTensor(input, type_ptr);
                                }),
                                py::arg("input"), py::arg("dtype") = nullptr)
                           .def(py::init([](py::float_ input, const TypePtr &type_ptr) {
                                  return TensorPy::MakeTensor(py::array(input), type_ptr);
                                }),
                                py::arg("input"), py::arg("dtype") = nullptr)
                           .def(py::init([](py::int_ input, const TypePtr &type_ptr) {
                                  return TensorPy::MakeTensor(py::array(input), type_ptr);
                                }),
                                py::arg("input"), py::arg("dtype") = nullptr)
                           .def(py::init([](py::list input, const TypePtr &type_ptr) {
                                  return TensorPy::MakeTensor(py::array(input), type_ptr);
                                }),
                                py::arg("input"), py::arg("dtype") = nullptr)
                           .def(py::init([](py::tuple input, const TypePtr &type_ptr) {
                                  return TensorPy::MakeTensor(py::array(input), type_ptr);
                                }),
                                py::arg("input"), py::arg("dtype") = nullptr)
                           .def_readonly(PYTHON_TENSOR_FLAG, &Tensor::parse_info_)
                           .def_property_readonly("dtype", &Tensor::Dtype, R"mydelimiter(
                             Get the tensor's data type.

                             Returns:
                                 type, the data type of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.dtype
                                 Int32
                             )mydelimiter")
                           .def_property_readonly("shape", TensorPy::GetPyTupleShape, R"mydelimiter(
                             Get the tensor's shape.

                             Returns:
                                 tuple[int], the shape of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((3, 3)))
                                 >>> data.shape()
                                 (3, 3)
                             )mydelimiter")
                           .def("asnumpy", TensorPy::SyncAsNumpy, R"mydelimiter(
                             Convert tensor to numpy.ndarray.

                             Returns:
                                 numpy.ndarray.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> array = data.asnumpy()
                                 >>> array
                                 array([[1., 1., 1.],
                                        [1., 1., 1.]])
                             )mydelimiter")
                           .def("size", &Tensor::DataSize, R"mydelimiter(
                             Get tensor's data size.

                             Returns:
                                 int, the size of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.size()
                                 6
                             )mydelimiter")
                           .def("is_init", &Tensor::is_init, R"mydelimiter(
                             Get tensor init_flag.

                             Returns:
                                 bool, whether the tensor init.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.is_init()
                                 False
                             )mydelimiter")
                           .def("set_init_flag", &Tensor::set_init_flag, R"mydelimiter(
                             Set tensor init_flag.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.set_init_flag(True)
                             )mydelimiter")
                           .def("dim", &Tensor::DataDim, R"mydelimiter(
                             Get tensor's data dimension.

                             Returns:
                                 int, the dimension of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.dim()
                                 2
                             )mydelimiter")
                           .def("set_dtype", &Tensor::SetDtype, R"mydelimiter(
                              Set the tensor's data type.

                              Arg:
                                  dtype (:class:`mindspore.dtype`): The type of output tensor.

                              Examples:
                                  >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                  >>> data.set_dtype(mindspore.int32)
                                  mindspore.int32
                              )mydelimiter")
                           .def("__str__", &Tensor::ToString)
                           .def("__repr__", &Tensor::ToStringRepr)
                           .def(py::pickle(
                             [](const Tensor &t) {  // __getstate__
                               /* Return a tuple that fully encodes the state of the object */
                               return py::make_tuple(TensorPy::AsNumpy(t));
                             },
                             [](const py::tuple &t) {  // __setstate__
                               if (t.size() != 1) {
                                 throw std::runtime_error("Invalid state!");
                               }
                               /* Create a new C++ instance */
                               return TensorPy::MakeTensor(t[0].cast<py::array>());
                             }));
                         // Define python MetaTensor class.
                         (void)py::class_<MetaTensor, std::shared_ptr<MetaTensor>>(*m, "MetaTensor")
                           .def(py::init<TypePtr, const std::vector<int>>(), py::arg("dtype"), py::arg("shape"))
                           .def_readonly(PYTHON_META_TENSOR_FLAG, &MetaTensor::parse_info_)
                           .def_property_readonly("dtype", &MetaTensor::Dtype, "Get the MetaTensor's dtype.")
                           .def_property_readonly("shape", &MetaTensor::shape, "Get the MetaTensor's shape.")
                           .def(py::pickle(
                             [](const MetaTensor &t) {  // __getstate__
                               /* Return a tuple that fully encodes the state of the object */
                               return py::make_tuple(static_cast<int>(t.data_type()), t.shape());
                             },
                             [](const py::tuple &t) {  // __setstate__
                               if (t.size() != 2) {
                                 throw std::runtime_error("Invalid state!");
                               }
                               /* Create a new C++ instance */
                               MetaTensor tensor(TypeId(t[0].cast<int>()), t[1].cast<std::vector<int>>());
                               return tensor;
                             }));
                       }));

}  // namespace tensor
}  // namespace mindspore
