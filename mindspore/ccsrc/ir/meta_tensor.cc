/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "ir/meta_tensor.h"

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

void DataBuf2Contiguous(const py::array &src, py::array *const dest) {
  if (dest == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to copy data to a contiguous buffer as dest is nullptr!";
  }

  Py_buffer pybuf_src;
  if (PyObject_GetBuffer(src.ptr(), &pybuf_src, PyBUF_ANY_CONTIGUOUS)) {
    MS_LOG(EXCEPTION) << "Failed to get buffer info from the src!";
  }

  if (!PyBuffer_IsContiguous(&pybuf_src, 'C')) {
    if (PyBuffer_ToContiguous(dest->request(true).ptr, &pybuf_src, pybuf_src.len, 'C')) {
      MS_LOG(EXCEPTION) << "Can't copy numpy.ndarray to a contiguous buffer.";
    }
  } else {
    *dest = src;
  }

  PyBuffer_Release(&pybuf_src);
}

// MetaTensor has default type_id_ which is TypeId::kTypeUnknown.
MetaTensor::MetaTensor() : data_type_(TypeId::kTypeUnknown) {}

MetaTensor::MetaTensor(const TypeId data_type, const std::vector<int> &shape) : data_type_(data_type), shape_(shape) {}

MetaTensor::MetaTensor(const TypePtr &type_ptr, const py::tuple &shape) {
  TypeId data_type = TypeId::kTypeUnknown;
  if (type_ptr != nullptr) {
    data_type = type_ptr->type_id();
  }
  data_type_ = data_type;
  shape_.resize(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_[i] = py::int_(shape[i]);
  }
}

MetaTensor::MetaTensor(const MetaTensor &meta_tensor)
    : Value(meta_tensor), data_type_(meta_tensor.data_type()), shape_(meta_tensor.shape()) {}

MetaTensor &MetaTensor::operator=(const MetaTensor &meta_tensor) {
  if (&meta_tensor == this) {
    return *this;
  }

  data_type_ = meta_tensor.data_type();
  shape_ = meta_tensor.shape();
  device_info_ = meta_tensor.device_info();

  return *this;
}

bool MetaTensor::operator==(const MetaTensor &meta_tensor) const {
  return data_type_ == meta_tensor.data_type() && shape_ == meta_tensor.shape();
}

// Get the size of a given dimension by its index number.
// The given index number should be in [0, shape_.size()).
// param index Dimension index number.
// return The size of the dimension if succeed, or -1 if failed.
int MetaTensor::DimensionSize(const size_t index) const {
  int dim_size = -1;
  if (index < shape_.size()) {
    dim_size = shape_[index];
  } else {
    MS_LOG(ERROR) << "Dimension index is wrong: " << index;
  }
  return dim_size;
}

int MetaTensor::ElementsNum() const {
  return std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int>());
}

TypePtr MetaTensor::Dtype() const { return TypeIdToType(data_type_); }

TypePtr MetaTensor::SetDtype(const TypePtr type_ptr) {
  if (type_ptr == nullptr) {
    MS_LOG(ERROR) << "Dtype to be set is nullptr.";
    return nullptr;
  }
  (void)set_data_type(type_ptr->type_id());
  return type_ptr;
}

void MetaTensor::SetDeviceInfo(const std::string &format, const TypePtr &data_type) {
  DeviceInfo info(format, data_type);
  set_device_info(info);
}

std::string MetaTensor::ToString() const {
  std::ostringstream buf;
  buf << "MetaTensor shape:[" << shape() << "]";
  return buf.str();
}

std::string MetaTensor::DumpText() const {
  std::ostringstream oss;
  oss << type_name() << "(" << SizeToInt(data_type_) << ")[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    oss << (i > 0 ? ", " : "") << shape_[i];
  }
  oss << "]";
  return oss.str();
}

Tensor::Tensor(const TypePtr &type_ptr, const py::tuple &shape) {
  TypeId data_type = TypeId::kTypeUnknown;
  if (type_ptr != nullptr) {
    data_type = type_ptr->type_id();
  }
  data_type_ = data_type;
  shape_.resize(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    shape_[i] = py::int_(shape[i]);
  }
  init(data_type_, shape_, &data_);
}

Tensor::Tensor(TypeId data_type, const std::vector<int> &shape) { init(data_type, shape, &data_); }

Tensor::Tensor(const py::array &input, const TypePtr &data_type) { init(input, data_type); }

Tensor::Tensor(const py::list &input, const TypePtr &data_type) { init(py::array(input), data_type); }

Tensor::Tensor(const py::tuple &input, const TypePtr &data_type) { init(py::array(input), data_type); }

Tensor::Tensor(const py::float_ &input, const TypePtr &data_type) { init(py::array(input), data_type); }

Tensor::Tensor(const py::int_ &input, const TypePtr &data_type) { init(py::array(input), data_type); }

Tensor::Tensor(const Tensor &tensor, const TypePtr &data_type)
    : MetaTensor(tensor), device_address_(tensor.device_address_) {
  init(tensor.data_, data_type);
  dirty_ = tensor.is_dirty();
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  if (this != &tensor) {
    MetaTensor::operator=(tensor);
    dirty_ = tensor.is_dirty();
    device_address_ = tensor.device_address();
    data_ = tensor.data_;
  }
  return *this;
}

bool Tensor::operator==(const Tensor &tensor) const {
  return (MetaTensor::operator==(tensor) && data_ == tensor.data_);
}

bool Tensor::ValueEqual(const Tensor &other) const {
  auto equal = [&other, this]() -> bool {
    auto np = py::module::import("numpy");
    auto equal = np.attr("equal")(data_, other.data_);
    auto all_equal = np.attr("all")(equal);
    return all_equal.cast<bool>();
  };
  return (MetaTensor::operator==(other) && (data_.is(other.data_) || equal()));
}

int Tensor::DataDim() const { return static_cast<int>(data_.ndim()); }

int Tensor::DataSize() const { return static_cast<int>(data_.size()); }

py::tuple Tensor::GetPyTupleShape() const {
  py::tuple dims(shape_.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = py::int_(shape_[i]);
  }
  return dims;
}

py::array Tensor::data() const { return data_; }

int Tensor::data_type_c() const { return static_cast<int>(data_type_); }

std::vector<int> Tensor::shape_c(void) const { return shape(); }

void *Tensor::data_c(bool writable) {
  // operand of bit operation should be unsigned int.
  unsigned int flags = ((unsigned int)data_.flags()) & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_;
  bool is_c_contiguous = (flags != 0) ? true : false;
  if (!is_c_contiguous) {
    py::array data_c;
    init(data_type_, shape_, &data_c);
    DataBuf2Contiguous(data_, &data_c);
    data_ = data_c;
  }
  return data_.request(writable).ptr;
}

TypeId Tensor::GetDataType(const py::buffer_info &buf) const {
  TypeId data_type = TypeId::kTypeUnknown;
  if (buf.format.compare("e") == 0) {
    data_type = TypeId::kNumberTypeFloat16;
  } else if (buf.format.compare("f") == 0) {
    data_type = TypeId::kNumberTypeFloat32;
  } else if (buf.format.compare("d") == 0) {
    data_type = TypeId::kNumberTypeFloat64;
  } else if (buf.format.compare("B") == 0) {
    data_type = TypeId::kNumberTypeUInt8;
  } else if (buf.format.compare("H") == 0) {
    data_type = TypeId::kNumberTypeUInt16;
  } else if (buf.format.compare("I") == 0) {
    data_type = TypeId::kNumberTypeUInt32;
  } else if (buf.format.compare("L") == 0 || buf.format.compare("Q") == 0) {
    data_type = TypeId::kNumberTypeUInt64;
  } else if (buf.format.compare("b") == 0) {
    data_type = TypeId::kNumberTypeInt8;
  } else if (buf.format.compare("h") == 0) {
    data_type = TypeId::kNumberTypeInt16;
  } else if (buf.format.compare("i") == 0) {
    data_type = TypeId::kNumberTypeInt32;
  } else if (buf.format.compare("l") == 0 || buf.format.compare("q") == 0) {
    data_type = TypeId::kNumberTypeInt64;
  } else if (buf.format.compare("?") == 0) {
    data_type = TypeId::kNumberTypeBool;
  } else {
    MS_LOG(WARNING) << "Get unsupported DataType " << buf.format << ".";
  }
  return data_type;
}

void Tensor::init(const py::array &input, const TypePtr &type_ptr) {
  TypeId data_type = TypeId::kTypeUnknown;
  if (type_ptr != nullptr) {
    data_type = type_ptr->type_id();
  }
  init(input, data_type);
}

void Tensor::init(const py::array &input, const TypeId &data_type) {
  py::buffer_info buf = input.request();

  data_type_ = GetDataType(buf);
  if (TypeId::kTypeUnknown == data_type && TypeId::kTypeUnknown == data_type_) {
    MS_LOG(EXCEPTION) << "Unsupported tensor type!";
  }

  std::vector<ssize_t> tm = buf.shape;
  size_t len = tm.size();
  std::vector<int> dims(len);
  for (size_t i = 0; i < len; ++i) {
    dims[i] = static_cast<int>(tm[i]);
  }
  (void)set_shape(dims);

  if (TypeId::kTypeUnknown != data_type && TypeId::kTypeUnknown != data_type_ && data_type_ != data_type) {
    // If user defined data type is not same as GetDataType from the data
    bool success = convert_data(input, data_type_, &data_, data_type);
    if (success) {
      data_type_ = data_type;
    } else {
      data_type_ = TypeId::kTypeUnknown;
      MS_LOG(EXCEPTION) << "Convert data from " << data_type_ << " to " << data_type << " failed!";
    }
  } else {
    data_ = input;
  }
  dirty_ = true;
}

void Tensor::init(TypeId data_type, const std::vector<int> &shape, py::array *const data) {
  data_type_ = data_type;
  shape_ = shape;
  switch (data_type) {
    case kNumberTypeBool:
      *data = py::array_t<bool, py::array::c_style>(shape);
      break;
    case kNumberTypeInt8:
      *data = py::array_t<int8_t, py::array::c_style>(shape);
      break;
    case kNumberTypeInt16:
      *data = py::array_t<int16_t, py::array::c_style>(shape);
      break;
    case kNumberTypeInt32:
      *data = py::array_t<int32_t, py::array::c_style>(shape);
      break;
    case kNumberTypeInt64:
      *data = py::array_t<int64_t, py::array::c_style>(shape);
      break;
    case kNumberTypeUInt8:
      *data = py::array_t<uint8_t, py::array::c_style>(shape);
      break;
    case kNumberTypeUInt16:
      *data = py::array_t<uint16_t, py::array::c_style>(shape);
      break;
    case kNumberTypeUInt32:
      *data = py::array_t<uint32_t, py::array::c_style>(shape);
      break;
    case kNumberTypeUInt64:
      *data = py::array_t<uint64_t, py::array::c_style>(shape);
      break;
    case kNumberTypeFloat16:
      *data = py::array_t<float16, py::array::c_style>(shape);
      break;
    case kNumberTypeFloat32:
      *data = py::array_t<float, py::array::c_style>(shape);
      break;
    case kNumberTypeFloat64:
      *data = py::array_t<double, py::array::c_style>(shape);
      break;
    default:
      MS_LOG(EXCEPTION) << "Cannot construct Tensor because of unsupported data type: " << data_type << ".";
      break;
  }
}

TypePtr Tensor::SetDtype(const TypePtr type_ptr) {
  MS_EXCEPTION_IF_NULL(type_ptr);
  (void)set_data_type(type_ptr->type_id());
  return type_ptr;
}

TypeId Tensor::set_data_type(const TypeId data_type) {
  if (data_.size() > 0 && data_type_ != data_type) {
    bool success = convert_data(data_, data_type_, &data_, data_type);
    if (success) {
      data_type_ = data_type;
    } else {
      MS_LOG(EXCEPTION) << "Convert data from " << data_type_ << " to " << data_type << " failed!";
    }
  } else if (data_.size() == 0) {
    data_type_ = data_type;
  }

  return data_type_;
}

bool Tensor::convert_data(const py::array &in, const TypeId in_data_type, py::array *const out,
                          const TypeId out_data_type) {
  if (out == nullptr) {
    return false;
  }

  bool result = true;
  if (TypeId::kTypeUnknown == in_data_type || TypeId::kTypeUnknown == out_data_type) {
    result = false;
  } else if (in_data_type == out_data_type) {
    *out = in;
  } else if (TypeId::kNumberTypeFloat64 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("float64").cast<py::array>();
  } else if (TypeId::kNumberTypeFloat32 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("float32").cast<py::array>();
  } else if (TypeId::kNumberTypeFloat16 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("float16").cast<py::array>();
  } else if (TypeId::kNumberTypeInt64 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("int64").cast<py::array>();
  } else if (TypeId::kNumberTypeInt32 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("int32").cast<py::array>();
  } else if (TypeId::kNumberTypeInt16 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("int16").cast<py::array>();
  } else if (TypeId::kNumberTypeInt8 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("int8").cast<py::array>();
  } else if (TypeId::kNumberTypeUInt8 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("uint8").cast<py::array>();
  } else if (TypeId::kNumberTypeUInt16 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("uint16").cast<py::array>();
  } else if (TypeId::kNumberTypeUInt32 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("uint32").cast<py::array>();
  } else if (TypeId::kNumberTypeUInt64 == out_data_type) {
    *out = in.attr("astype").cast<py::function>()("uint64").cast<py::array>();
  } else {
    data_type_ = TypeId::kTypeUnknown;
    MS_LOG(EXCEPTION) << "Cannot convert from " << TypeIdLabel(in_data_type) << " to " << TypeIdLabel(out_data_type)
                      << ".";
  }

  return result;
}

abstract::AbstractBasePtr Tensor::ToAbstract() {
  auto tens = shared_from_base<Tensor>();
  auto dtype = tens->Dtype();
  if (!IsSubType(dtype, kNumber)) {
    MS_LOG(EXCEPTION) << "Expect tensor type kNumber but got: " << dtype->ToString() << ".";
  }
  auto tensor_shape = tens->shape();
  auto abs_tensor = std::make_shared<abstract::AbstractTensor>(dtype, tensor_shape);
  abs_tensor->set_value(shared_from_base<Tensor>());
  return abs_tensor;
}

std::string Tensor::GetShapeAndDataTypeInfo() const {
  std::ostringstream buf;
  buf << "Tensor \nshape:[" << shape() << "]" << this->Dtype()->ToString();
  return buf.str();
}

std::string Tensor::ToString() const {
  const int small_tensor_size = 30;
  std::ostringstream buf;
  buf << "Tensor \nshape:[" << shape() << "]" << this->Dtype()->ToString();
  // only print small tensor
  if (DataSize() < small_tensor_size) {
    buf << "val:" << std::string(py::str(data()));
  }
  return buf.str();
}

std::string Tensor::ToStringRepr() const {
  std::ostringstream buf;
  auto type_ptr = this->Dtype();
  MS_EXCEPTION_IF_NULL(type_ptr);
  buf << "Tensor shape:[" << shape() << "]" << type_ptr->ToString();
  buf << "\nval:" << std::string(py::str(data()));
  return buf.str();
}

py::array Tensor::data_sync() {
  if (device_address_ != nullptr) {
    if (!device_address_->SyncDeviceToHost(this->shape(), static_cast<size_t>(this->data().nbytes()), this->data_type(),
                                           this->data_c(true))) {
      MS_LOG(EXCEPTION) << "SyncDeviceToHost when asnumpy.";
    }
  }
  return data_;
}

REGISTER_PYBIND_DEFINE(Tensor, ([](const py::module *m) {
                         // dtype should define before Tensor, because Tensor init depend dtype
                         (void)py::class_<Tensor, std::shared_ptr<Tensor>>(*m, "Tensor")
                           .def(py::init<TypePtr, py::tuple>(), py::arg("dtype"), py::arg("shape"))
                           .def(py::init<py::array, TypePtr>(), py::arg("input"), py::arg("dtype") = nullptr)
                           .def(py::init<py::float_, TypePtr>(), py::arg("input"), py::arg("dtype") = nullptr)
                           .def(py::init<py::int_, TypePtr>(), py::arg("input"), py::arg("dtype") = nullptr)
                           .def(py::init<py::list, TypePtr>(), py::arg("input"), py::arg("dtype") = nullptr)
                           .def(py::init<py::tuple, TypePtr>(), py::arg("input"), py::arg("dtype") = nullptr)
                           .def(py::init<Tensor, TypePtr>(), py::arg("input"), py::arg("dtype") = nullptr)
                           .def_readonly(PYTHON_TENSOR_FLAG, &Tensor::parse_info_)
                           .def("asnumpy", &Tensor::data_sync, R"mydelimiter(
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
                           .def("dim", &Tensor::DataDim, R"mydelimiter(
                             Get tensor's data dimension.

                             Returns:
                                 int, the dimension of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.dim()
                                 2
                             )mydelimiter")
                           .def("dtype", &Tensor::Dtype, R"mydelimiter(
                             Get the tensor's data type.

                             Returns:
                                 type, the data type of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.dtype()
                                 Int32
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
                           .def("shape", &Tensor::GetPyTupleShape, R"mydelimiter(
                             Get the tensor's shape.

                             Returns:
                                 tuple[int], the shape of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((3, 3)))
                                 >>> data.shape()
                                 (3, 3)
                             )mydelimiter")
                           .def("__str__", &Tensor::ToString)
                           .def("__repr__", &Tensor::ToStringRepr)
                           .def(py::pickle(
                             [](const Tensor &t) {  // __getstate__
                               /* Return a tuple that fully encodes the state of the object */
                               return py::make_tuple(t.data());
                             },
                             [](const py::tuple &t) {  // __setstate__
                               if (t.size() != 1) {
                                 throw std::runtime_error("Invalid state!");
                               }
                               /* Create a new C++ instance */
                               Tensor tensor(t[0].cast<py::array>());
                               return tensor;
                             }));
                         (void)py::class_<MetaTensor, std::shared_ptr<MetaTensor>>(*m, "MetaTensor")
                           .def(py::init<TypePtr, py::tuple>(), py::arg("dtype"), py::arg("shape"));
                       }));

}  // namespace tensor
}  // namespace mindspore
