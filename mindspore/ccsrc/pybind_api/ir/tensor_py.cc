/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "pybind_api/ir/tensor_py.h"

#include <utility>

#include "include/common/pybind_api/api_register.h"
#include "abstract/abstract_value.h"
#include "utils/shape_utils.h"
#include "utils/cache_embedding_hashmap_struct.h"
#include "include/common/utils/python_adapter.h"

namespace mindspore {
namespace tensor {
namespace {
struct TensorToNumpyRegister {
  TensorToNumpyRegister() { python_adapter::PyAdapterCallback::SetTensorToNumpyHandler(tensor::TensorPy::AsNumpy); }
} callback_register;
}  // namespace
constexpr ssize_t kPyBufItemSize1 = 1;
constexpr ssize_t kPyBufItemSize2 = 2;
constexpr ssize_t kPyBufItemSize4 = 4;
constexpr ssize_t kPyBufItemSize8 = 8;

static TypeId GetDataType(const py::buffer_info &buf) {
  if (buf.format.size() == 1) {
    switch (buf.format.front()) {
      case 'e':
      case 'f':
      case 'd':
        switch (buf.itemsize) {
          case kPyBufItemSize2:
            return TypeId::kNumberTypeFloat16;
          case kPyBufItemSize4:
            return TypeId::kNumberTypeFloat32;
          case kPyBufItemSize8:
            return TypeId::kNumberTypeFloat64;
        }
        break;
      case 'b':
      case 'h':
      case 'i':
      case 'l':
      case 'q':
        switch (buf.itemsize) {
          case kPyBufItemSize1:
            return TypeId::kNumberTypeInt8;
          case kPyBufItemSize2:
            return TypeId::kNumberTypeInt16;
          case kPyBufItemSize4:
            return TypeId::kNumberTypeInt32;
          case kPyBufItemSize8:
            return TypeId::kNumberTypeInt64;
          default:
            break;
        }
        break;
      case 'B':
      case 'H':
      case 'I':
      case 'L':
      case 'Q':
        switch (buf.itemsize) {
          case kPyBufItemSize1:
            return TypeId::kNumberTypeUInt8;
          case kPyBufItemSize2:
            return TypeId::kNumberTypeUInt16;
          case kPyBufItemSize4:
            return TypeId::kNumberTypeUInt32;
          case kPyBufItemSize8:
            return TypeId::kNumberTypeUInt64;
          default:
            break;
        }
        break;
      case '?':
        return TypeId::kNumberTypeBool;
      default:
        break;
    }
  } else if (buf.format.size() >= 2) {
    // Support np.str_ dtype, format: {x}w. {x} is a number that means the maximum length of the string items.
    if (buf.format.back() == 'w' || buf.format.back() == 's') {
      return TypeId::kObjectTypeString;
    } else if (buf.format == "Zf") {
      return TypeId::kNumberTypeComplex64;
    } else if (buf.format == "Zd") {
      return TypeId::kNumberTypeComplex128;
    }
  }
  MS_LOG(WARNING) << "Unsupported DataType format " << buf.format << ", item size " << buf.itemsize;
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
    case TypeId::kObjectTypeString:
      return py::format_descriptor<uint8_t>::format();
    case TypeId::kNumberTypeComplex64:
      return py::format_descriptor<std::complex<float>>::format();
    case TypeId::kNumberTypeComplex128:
      return py::format_descriptor<std::complex<double>>::format();
    case TypeId::kMetaTypeType:
    case TypeId::kMetaTypeEllipsis:
    default:
      MS_LOG(WARNING) << "Unsupported DataType " << data_type << ".";
      return "";
  }
}

static bool IsCContiguous(const py::array &input) {
  auto flags = static_cast<unsigned int>(input.flags());
  return (flags & static_cast<unsigned int>(pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) != 0;
}

// TensorDataNumpy implements TensorData using numpy array.
class TensorDataNumpy : public TensorData {
 public:
  explicit TensorDataNumpy(py::buffer_info &&buffer) : buffer_(std::move(buffer)) {}

  ~TensorDataNumpy() override = default;

  /// Total number of elements.
  ssize_t size() const override { return buffer_.size; }

  /// Byte size of a single element.
  ssize_t itemsize() const override { return buffer_.itemsize; }

  /// Total number of bytes.
  ssize_t nbytes() const override { return buffer_.itemsize * buffer_.size; }

  /// Number of dimensions.
  ssize_t ndim() const override { return buffer_.ndim; }

  /// Data pointer.
  void *data() override { return buffer_data(); }

  const void *const_data() const override { return buffer_.ptr; }

  bool is_sub_data() const override { return false; }

  bool has_sub_data() const override { return false; }

  bool is_from_numpy() const override { return true; }

  /// To string.
  std::string ToString(const TypeId, const ShapeVector &, bool use_comma) const override {
    if (use_comma) {
      // Call python np.array2string(data_, separator=', ') to convert string with comma.
      py::dict kwargs;
      kwargs["separator"] = ", ";
      auto np = py::module::import("numpy");
      auto array2string = np.attr("array2string");
      return py::str(array2string(py_array(), **kwargs));
    }
    // without comma.
    return py::str(py_array());
  }

  /// py::array object. by default, use py::str() as the dummy owner to prevent data copy.
  py::array py_array(const py::handle &owner = py::str()) const {
    return py::array(py::dtype(buffer_), buffer_.shape, buffer_.strides, buffer_.ptr, owner);
  }

 private:
  void *buffer_data() const { return buffer_.ptr; }

  // The internal buffer.
  py::buffer_info buffer_;
};

TensorPtr TensorPy::MakeTensor(const py::array &input, const TypePtr &type_ptr) {
  // Get input buffer info.
  py::buffer_info buf = input.request();
  // Check data types.
  auto data_type = type_ptr ? type_ptr->type_id() : TypeId::kTypeUnknown;
  auto buf_type = GetDataType(buf);
  if (buf_type == TypeId::kTypeUnknown && data_type == TypeId::kTypeUnknown) {
    MS_LOG(EXCEPTION) << "Unsupported tensor type!";
  }
  MS_LOG(DEBUG) << "data_type: " << data_type << ", buf_type: " << buf_type;
  if (data_type == TypeId::kObjectTypeString || buf_type == TypeId::kObjectTypeString) {
    return TensorPy::MakeTensorOfNumpy(input);
  }
  // Use buf type as data type if type_ptr not set.
  if (data_type == TypeId::kTypeUnknown) {
    data_type = buf_type;
  }
  // Convert input array to C contiguous if need.
  std::unique_ptr<char[]> tmp_buf;
  if (!IsCContiguous(input)) {
    Py_buffer pybuf;
    if (PyObject_GetBuffer(input.ptr(), &pybuf, PyBUF_ANY_CONTIGUOUS) != 0) {
      MS_LOG(EXCEPTION) << "Failed to get buffer from the input!";
    }
    tmp_buf = std::make_unique<char[]>(pybuf.len);
    if (PyBuffer_ToContiguous(tmp_buf.get(), &pybuf, pybuf.len, 'C') != 0) {
      MS_LOG(EXCEPTION) << "Can't copy numpy.ndarray to a contiguous buffer.";
    }
    PyBuffer_Release(&pybuf);
    buf.ptr = tmp_buf.get();
  }
  // Get tensor shape.
  ShapeVector shape(buf.shape.begin(), buf.shape.end());
  if (data_type == buf_type) {
    // Use memory copy if input data type is the same as the required type.
    return std::make_shared<Tensor>(data_type, shape, buf.ptr, buf.size * buf.itemsize);
  }
  // Create tensor with data type converted.
  return std::make_shared<Tensor>(data_type, shape, buf.ptr, buf_type);
}

/// Creates a Tensor from a numpy array without copy
TensorPtr TensorPy::MakeTensorOfNumpy(const py::array &input) {
  // Check format.
  if (!IsCContiguous(input)) {
    MS_LOG(EXCEPTION) << "Array should be C contiguous.";
  }
  // Get input buffer info.
  py::buffer_info buf = input.request();
  // Get tensor dtype and check it.
  auto dtype = GetDataType(buf);
  if (dtype == TypeId::kTypeUnknown) {
    MS_LOG(EXCEPTION) << "Unsupported data type!";
  }
  // Get tensor shape.
  ShapeVector shape(buf.shape.begin(), buf.shape.end());
  // Make a tensor with shared data with numpy array.
  auto tensor_data = std::make_shared<TensorDataNumpy>(std::move(buf));
  return std::make_shared<Tensor>(dtype, shape, tensor_data);
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

py::tuple TensorPy::GetPyTupleStrides(const Tensor &tensor) {
  std::vector<ssize_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<ssize_t> strides = GetStrides(shape, tensor.data().itemsize());
  py::tuple py_strides(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    py_strides[i] = py::int_(strides[i]);
  }
  return py_strides;
}

py::int_ TensorPy::GetPyItemSize(const Tensor &tensor) { return tensor.data().itemsize(); }

py::int_ TensorPy::GetPyNBytes(const Tensor &tensor) { return tensor.data().nbytes(); }

template <typename T>
void MemCopyFromCacheToHost(void *hashmap_addr, void *host_addr, void *cache_addr, size_t host_max, size_t cache_max,
                            size_t hashmap_size, size_t col_size) {
  auto host_data = static_cast<char *>(host_addr);
  auto cache_data = static_cast<char *>(cache_addr);
  auto hashmap_data = static_cast<HashmapEntry<T> *>(hashmap_addr);
  // default param type float
  const size_t param_type_size = 4;
  size_t single_col_bytes = param_type_size * col_size;
  for (size_t i = 0; i < hashmap_size; ++i) {
    if (!hashmap_data[i].IsEmpty()) {
      size_t host_offset = single_col_bytes * LongToSize(hashmap_data[i].key_);
      size_t cache_offset = single_col_bytes * LongToSize(hashmap_data[i].value_);
      if (cache_offset + single_col_bytes <= cache_max) {
        auto ret =
          memcpy_s(host_data + host_offset, host_max - host_offset, cache_data + cache_offset, single_col_bytes);
        if (ret != 0) {
          MS_LOG(EXCEPTION) << "Memcpy failed.";
        }
      }
    }
  }
  MS_LOG(INFO) << "Memcpy from cache to host success!";
}

void TensorPy::FlushFromCache(const Tensor &tensor) {
  py::gil_scoped_release gil_release;
  if (tensor.NeedWait()) {
    tensor.Wait();
  }
  tensor.data_sync();

  if (tensor.cache_enable()) {
    MS_LOG(INFO) << tensor.ToString() << " is cache enable.";
    auto hashmap_tensor_ptr = tensor.hashmap_tensor_ptr();
    auto cache_tensor_ptr = tensor.cache_tensor_ptr();
    if (hashmap_tensor_ptr != nullptr && cache_tensor_ptr != nullptr) {
      hashmap_tensor_ptr->data_sync();
      cache_tensor_ptr->data_sync();
      auto hashmap_size = hashmap_tensor_ptr->shape_c()[0];
      auto host_shape = tensor.shape_c();
      auto cache_shape = cache_tensor_ptr->shape_c();
      if (host_shape.size() != 2 && host_shape.size() != 2 && host_shape[1] != cache_shape[1]) {
        MS_LOG(EXCEPTION) << "Got host shape and cache shape invalid."
                          << "host shape:" << host_shape << ", cache shape:" << cache_shape;
      }
      auto host_data_max_size = static_cast<size_t>(tensor.Size());
      auto cache_data_max_size = static_cast<size_t>(cache_tensor_ptr->Size());
      auto hashmap_data_type = hashmap_tensor_ptr->data_type();
      if (hashmap_data_type == TypeId::kNumberTypeInt32) {
        MemCopyFromCacheToHost<int32_t>(hashmap_tensor_ptr->data_c(), tensor.data_c(), cache_tensor_ptr->data_c(),
                                        host_data_max_size, cache_data_max_size, hashmap_size, host_shape[1]);
      } else if (hashmap_data_type == TypeId::kNumberTypeInt64) {
        MemCopyFromCacheToHost<int32_t>(hashmap_tensor_ptr->data_c(), tensor.data_c(), cache_tensor_ptr->data_c(),
                                        host_data_max_size, cache_data_max_size, hashmap_size, host_shape[1]);
      } else {
        MS_LOG(ERROR) << "Hashmap dtype only suppotr int32, in64.";
      }
    }
  }
}

py::array TensorPy::SyncAsNumpy(const Tensor &tensor) {
  {
    py::gil_scoped_release gil_release;
    if (tensor.NeedWait()) {
      tensor.Wait();
    }
    tensor.data_sync();

    // Release device address of graph output tensor.
    if (tensor.need_release_device_mem()) {
      const_cast<Tensor &>(tensor).set_device_address(nullptr);
    }
  }
  return AsNumpy(tensor);
}

py::array TensorPy::AsNumpy(const Tensor &tensor) {
  // Use TensorData as the owner to prevent use-after-free problem.
  // We can NOT use Tensor as the owner since its TensorData may change
  // by other operations such as AssignValue().
  py::object owner = py::cast(tensor.data_ptr());
  auto data_numpy = dynamic_cast<const TensorDataNumpy *>(&tensor.data());
  if (data_numpy != nullptr) {
    // Return internal numpy array if tensor data is implemented base on it.
    return data_numpy->py_array(owner);
  }
  // Otherwise, create numpy array by buffer protocol.
  auto info = GetPyBufferInfo(tensor);
  return py::array(py::dtype(info), info.shape, info.strides, info.ptr, owner);
}

static ShapeVector GetShapeFromTuple(const py::tuple &tuple) {
  ShapeVector shape;
  const size_t size = tuple.size();
  shape.reserve(tuple.size());
  for (size_t i = 0; i < size; ++i) {
    shape.push_back(py::int_(tuple[i]));
  }
  return shape;
}
void RegMetaTensor(const py::module *m) {
  // Define python MetaTensor class.
  (void)py::class_<MetaTensor, std::shared_ptr<MetaTensor>>(*m, "MetaTensor")
    .def(py::init<TypePtr, const ShapeVector>(), py::arg("dtype"), py::arg("shape"))
    .def_property_readonly("dtype", &MetaTensor::Dtype, "Get the MetaTensor's dtype.")
    .def_property_readonly("shape", &MetaTensor::shape, "Get the MetaTensor's shape.")
    .def_property("param_info", &MetaTensor::param_info, &MetaTensor::set_param_info)
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
        MetaTensor tensor(TypeId(t[0].cast<int>()), t[1].cast<ShapeVector>());
        return tensor;
      }));
  // Define TensorData as a python class so that ownership of tensor data can be managed.
  (void)py::class_<TensorData, TensorDataPtr>(*m, "_TensorData");
  // Define python Tensor class.
  // dtype should define before Tensor, because Tensor init depend dtype
  (void)py::class_<Tensor, MetaTensor, std::shared_ptr<Tensor>>(*m, "Tensor")
    .def(py::init([](const Tensor &tensor) { return std::make_shared<Tensor>(tensor); }), py::arg("input"))
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
    .def(py::init([](const TypePtr &type_ptr, const py::list &shape) {
           auto data_type = type_ptr ? type_ptr->type_id() : TypeId::kNumberTypeFloat64;
           return std::make_shared<Tensor>(data_type, GetShapeFromTuple(shape));
         }),
         py::arg("dtype"), py::arg("shape"))
    .def(
      py::init([](const py::array &input, const TypePtr &type_ptr) { return TensorPy::MakeTensor(input, type_ptr); }),
      py::arg("input"), py::arg("dtype") = nullptr)
    .def(py::init([](const py::float_ input, const TypePtr &type_ptr) {
           return TensorPy::MakeTensor(py::array(input), type_ptr);
         }),
         py::arg("input"), py::arg("dtype") = nullptr)
    .def(py::init([](const py::int_ input, const TypePtr &type_ptr) {
           return TensorPy::MakeTensor(py::array(input), type_ptr);
         }),
         py::arg("input"), py::arg("dtype") = nullptr)
    .def(py::init([](const py::list &input, const TypePtr &type_ptr) {
           return TensorPy::MakeTensor(py::array(input), type_ptr);
         }),
         py::arg("input"), py::arg("dtype") = nullptr)
    .def(py::init([](const py::tuple &input, const TypePtr &type_ptr) {
           return TensorPy::MakeTensor(py::array(input), type_ptr);
         }),
         py::arg("input"), py::arg("dtype") = nullptr)
    // We only suppot array/bool_/int_/float_/list/tuple/complex pybind objects as tensor input,
    // and array/bool_/int_/float_/list/tuple init will be matched above, other pybind objects
    // input will raise error except complex data type.
    .def(py::init([](const py::object &input, const TypePtr &type_ptr) {
           if (!PyComplex_CheckExact(input.ptr())) {
             MS_LOG(EXCEPTION) << "Unsupported tensor type: " << input.get_type();
           }
           return TensorPy::MakeTensor(py::array(input), type_ptr);
         }),
         py::arg("input"), py::arg("dtype") = nullptr)
    .def_property("init_flag", &Tensor::is_init, &Tensor::set_init_flag)
    .def_property_readonly("_dtype", &Tensor::Dtype, R"mydelimiter(
                             Get the tensor's data type.

                             Returns:
                                 type, the data type of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.dtype
                                 Int32
                             )mydelimiter")
    .def_property_readonly("_shape", TensorPy::GetPyTupleShape, R"mydelimiter(
                             Get the tensor's shape.

                             Returns:
                                 tuple[int], the shape of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((3, 3)))
                                 >>> data.shape()
                                 (3, 3)
                             )mydelimiter")
    .def_property_readonly("_size", &Tensor::DataSize, R"mydelimiter(
                             Get tensor's data size.

                             Returns:
                                 size_t, the size of tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data.size
                                 6
                             )mydelimiter")
    .def_property_readonly("_itemsize", TensorPy::GetPyItemSize, R"mydelimiter(
                             Get the tensor's length of one element in bytes.

                             Returns:
                                 itemsize, length of one element in bytes.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.itemsize
                                 4
                             )mydelimiter")
    .def_property_readonly("_nbytes", TensorPy::GetPyNBytes, R"mydelimiter(
                             Get the tensor's total number of bytes.

                             Returns:
                                 nbytes, total number of bytes taken by the tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.nbytes
                                 4
                             )mydelimiter")
    .def_property_readonly("_strides", TensorPy::GetPyTupleStrides, R"mydelimiter(
                             Get the tensor's tuple of bytes to step in each dimension
                             when traversing an array.

                             Returns:
                                 tuple[int], the strides of the tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 1), np.int32))
                                 >>> data.strides
                                 (4, 4)
                             )mydelimiter")
    .def("_flatten_tensors", Tensor::FlattenTensors, py::arg("fusion_size") = 0)
    .def("_is_flattened", Tensor::IsFlattened)
    .def("_get_flattened_tensors", Tensor::GetFlattenedTensors)
    .def("_get_fusion_size", Tensor::GetFusionSize)
    .def("from_numpy", TensorPy::MakeTensorOfNumpy, R"mydelimiter(
                             Creates a Tensor from a numpy.ndarray without copy.

                             Arg:
                                 array (numpy.ndarray): The input ndarray.

                             Returns:
                                 Tensor, tensor with shared data to input ndarray.

                             Examples:
                                 >>> a = np.ones((2, 3))
                                 >>> t = mindspore.Tensor.from_numpy(a)
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
    .def("_flush_from_cache", TensorPy::FlushFromCache, R"mydelimiter(
                             Flush Cache data to Host if tensor is cache enable.

                             Returns:
                                 None.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((2, 3)))
                                 >>> data._flush_from_cache()
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
    .def("assign_value_cpp", &Tensor::AssignValue, R"mydelimiter(
                             Assign another tensor value to this.

                             Arg:
                                 value (:class:`mindspore.tensor`): The value tensor.

                             Examples:
                                 >>> data = mindspore.Tensor(np.ones((1, 2), np.float32))
                                 >>> data2 = mindspore.Tensor(np.ones((2, 2), np.float32))
                                 >>> data.assign_value(data2)
                                 >>> data.shape
                                 (2, 2)
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
    .def("set_cast_dtype", &Tensor::set_cast_dtype, py::arg("dtype") = nullptr)
    .def("data_sync", &Tensor::data_sync)
    .def("__str__", &Tensor::ToString)
    .def("__repr__", &Tensor::ToStringRepr)
    .def(py::pickle(
      [](const Tensor &t) {  // __getstate__
        /* Return a tuple that fully encodes the state of the object */
        return py::make_tuple(TensorPy::SyncAsNumpy(t));
      },
      [](const py::tuple &t) {  // __setstate__
        if (t.size() != 1) {
          throw std::runtime_error("Invalid state!");
        }
        /* Create a new C++ instance */
        return TensorPy::MakeTensor(t[0].cast<py::array>());
      }));
}

template <typename T>
py::tuple GetSparseTensorShape(const T &sparse_tensor) {
  auto &shape = sparse_tensor.shape();
  py::tuple dims(shape.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = py::int_(shape[i]);
  }
  return dims;
}

py::tuple CSRTensorPy::GetPyTupleShape(const CSRTensor &csr_tensor) { return GetSparseTensorShape(csr_tensor); }

void RegCSRTensor(const py::module *m) {
  // Define python CSRTensor class.
  (void)py::class_<CSRTensor, std::shared_ptr<CSRTensor>>(*m, "CSRTensor")
    .def(py::init([](const Tensor &indptr, const Tensor &indices, const Tensor &values, const py::tuple &shape) {
           return std::make_shared<CSRTensor>(std::make_shared<Tensor>(indptr), std::make_shared<Tensor>(indices),
                                              std::make_shared<Tensor>(values), GetShapeFromTuple(shape));
         }),
         py::arg("indptr"), py::arg("indices"), py::arg("values"), py::arg("shape"))
    .def(py::init([](const CSRTensor &csr_tensor) { return std::make_shared<CSRTensor>(csr_tensor); }),
         py::arg("input"))
    .def_property_readonly("_shape", CSRTensorPy::GetPyTupleShape)
    .def_property_readonly("_dtype", &CSRTensor::Dtype)
    .def_property_readonly("_indptr", &CSRTensor::GetIndptr)
    .def_property_readonly("_indices", &CSRTensor::GetIndices)
    .def_property_readonly("_values", &CSRTensor::GetValues)
    .def("__str__", &CSRTensor::ToString)
    .def("__repr__", &CSRTensor::ToString);
}

py::tuple COOTensorPy::GetPyTupleShape(const COOTensor &coo_tensor) { return GetSparseTensorShape(coo_tensor); }

void RegCOOTensor(const py::module *m) {
  // Define python COOTensor class.
  (void)py::class_<COOTensor, std::shared_ptr<COOTensor>>(*m, "COOTensor")
    .def(py::init([](const Tensor &indices, const Tensor &values, const py::tuple &shape) {
           return std::make_shared<COOTensor>(std::make_shared<Tensor>(indices), std::make_shared<Tensor>(values),
                                              GetShapeFromTuple(shape));
         }),
         py::arg("indices"), py::arg("values"), py::arg("shape"))
    .def(py::init([](const COOTensor &coo_tensor) { return std::make_shared<COOTensor>(coo_tensor); }),
         py::arg("input"))
    .def_property_readonly("_shape", COOTensorPy::GetPyTupleShape)
    .def_property_readonly("_dtype", &COOTensor::Dtype)
    .def_property_readonly("_indices", &COOTensor::GetIndices)
    .def_property_readonly("_values", &COOTensor::GetValues)
    .def("__str__", &COOTensor::ToString)
    .def("__repr__", &COOTensor::ToString);
}

py::tuple RowTensorPy::GetPyTupleShape(const RowTensor &row_tensor) { return GetSparseTensorShape(row_tensor); }

void RegRowTensor(const py::module *m) {
  // Define python RowTensor class.
  (void)py::class_<RowTensor, std::shared_ptr<RowTensor>>(*m, "RowTensor")
    .def(py::init([](const Tensor &indices, const Tensor &values, const py::tuple &shape) {
           return std::make_shared<RowTensor>(std::make_shared<Tensor>(indices), std::make_shared<Tensor>(values),
                                              GetShapeFromTuple(shape));
         }),
         py::arg("indices"), py::arg("values"), py::arg("shape"))
    .def(py::init([](const RowTensor &row_tensor) { return std::make_shared<RowTensor>(row_tensor); }),
         py::arg("input"))
    .def_property_readonly("_shape", RowTensorPy::GetPyTupleShape)
    .def_property_readonly("_dtype", &RowTensor::Dtype)
    .def_property_readonly("_indices", &RowTensor::GetIndices)
    .def_property_readonly("_values", &RowTensor::GetValues)
    .def("__str__", &RowTensor::ToString)
    .def("__repr__", &RowTensor::ToString);
}
}  // namespace tensor
}  // namespace mindspore
