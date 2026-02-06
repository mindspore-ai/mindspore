/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "include/utils/tensor_py.h"

#include <string>
#include <utility>
#include <memory>
#include <vector>
#include "ir/value.h"
#include "ir/tensor_new.h"
#include "utils/log_adapter.h"
#include "include/utils/convert_utils_py.h"
#include "tools/profiler/profiler.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/utils/pyobj_manager.h"
#include "include/runtime/pipeline/pipeline.h"
#include "ops_utils/callback.h"

namespace mindspore {
namespace tensor {
PyTypeObject *TensorPy_Type;

TensorPy::TensorPy(const TensorPtr &input) { tensor_ = input; }

TensorPy::TensorPy(const stub::StubNodePtr &stub_node) { stub_ = stub_node; }

TensorPy::TensorPy(int64_t input, const TypePtr &data_type) { tensor_ = tensor::from_scalar(input, data_type); }

TensorPy::TensorPy(int32_t input, const TypePtr &data_type) { tensor_ = tensor::from_scalar(input, data_type); }

TensorPy::TensorPy(int16_t input, const TypePtr &data_type) { tensor_ = tensor::from_scalar(input, data_type); }

TensorPy::TensorPy(int8_t input, const TypePtr &data_type) { tensor_ = tensor::from_scalar(input, data_type); }

TensorPy::TensorPy(const std::vector<int64_t> &input, const TypePtr &data_type) {
  tensor_ = tensor::from_vector(input, data_type);
}

TensorPy::TensorPy(const std::vector<int32_t> &input, const TypePtr &data_type) {
  tensor_ = tensor::from_vector(input, data_type);
}

TensorPy::TensorPy(const std::vector<double> &input, const TypePtr &data_type) {
  tensor_ = tensor::from_vector(input, data_type);
}

TensorPy::TensorPy(const std::vector<float> &input, const TypePtr &data_type) {
  tensor_ = tensor::from_vector(input, data_type);
}

TensorPy::TensorPy(const TensorPy &input)
    : init_finished_flag_(input.init_finished_flag_),
      const_arg_flag_(input.const_arg_flag_),
      virtual_flag_(input.virtual_flag_),
      ms_parameter_output_(input.ms_parameter_output_),
      initializer_(input.initializer_),
      parent_tensor_(input.parent_tensor_),
      index_of_parent_(input.index_of_parent_),
      symbolic_shape_(input.symbolic_shape_),
      device_(input.device_) {
  tensor_ = input.GetTensor();
}

TensorPy::TensorPy(TypeId data_type, const ShapeVector &shape) {
  // todo: check.
  tensor_ = tensor::from_spec(data_type, shape, device::DeviceType::kCPU);
}

bool TensorPy::IsInitFinished() { return init_finished_flag_; }

void TensorPy::SetInitFinished(bool flag) { init_finished_flag_ = flag; }

bool TensorPy::IsConstArg() { return const_arg_flag_; }

void TensorPy::SetConstArg(bool flag) { const_arg_flag_ = flag; }

bool TensorPy::IsVirtual() { return virtual_flag_; }

void TensorPy::SetVirtualFlag(bool flag) { virtual_flag_ = flag; }

const py::object TensorPy::GetInitializer() const {
  if (!initializer_.check() || initializer_.is_none()) {
    return py::none();
  }
  return initializer_;
}

void TensorPy::SetInitializer(const py::object &init) { initializer_ = init; }

const std::string TensorPy::GetDevice() const { return device_; }

void TensorPy::SetDevice(const std::string &dev) { device_ = dev; }

TensorPtr TensorPy::GetTensor() const {
  if (tensor_ == nullptr) {
    MS_EXCEPTION_IF_NULL(stub_);
    return std::static_pointer_cast<Tensor>(stub_->WaitValue());
  }
  return tensor_;
}

void TensorPy::SetTensor(const TensorPtr &tensor) { tensor_ = tensor; }

void TensorPy::UpdateStub(const TensorPtr &tensor) { stub_->SetValue(tensor); }

const py::object TensorPy::GetStorage() const { return storage_; }

void TensorPy::SetStorage(py::object storage) { storage_ = storage; }

const py::object TensorPy::GetParentTensor() {
  if (!parent_tensor_.check() || parent_tensor_.is_none()) {
    return py::none();
  }
  return parent_tensor_;
}

void TensorPy::SetParentTensor(const py::object &parent) { parent_tensor_ = parent; }

const py::object TensorPy::GetIndexOfParent() {
  if (!index_of_parent_.check() || index_of_parent_.is_none()) {
    return py::none();
  }
  return index_of_parent_;
}

void TensorPy::SetIndexOfParent(const py::object &index) { index_of_parent_ = index; }

const py::object TensorPy::GetSymbolicShape() const {
  if (!symbolic_shape_.check() || symbolic_shape_.is_none()) {
    return py::none();
  }
  return symbolic_shape_;
}

void TensorPy::SetSymbolicShape(const py::object &symbolic) { symbolic_shape_ = symbolic; }

py::tuple TensorPy::GetPyTupleShape() {
  auto &shape = GetTensor()->shape();
  py::tuple dims(shape.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    dims[i] = py::int_(shape[i]);
  }
  return dims;
}

py::int_ TensorPy::GetPyItemSize() { return GetTensor()->DataItemSize(); }

py::int_ TensorPy::GetPyNBytes() { return GetTensor()->DataNBytes(); }

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

py::tuple TensorPy::GetPyTupleStrides() {
  auto tensor = GetTensor();
  std::vector<ssize_t> shape(tensor->shape().begin(), tensor->shape().end());
  std::vector<ssize_t> strides = GetStrides(shape, tensor->DataItemSize());
  py::tuple py_strides(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    py_strides[i] = py::int_(strides[i]);
  }
  return py_strides;
}

TypePtr TensorPy::GetDtype() const { return GetTensor()->Dtype(); }

TypePtr TensorPy::SetDtype(const TypePtr type) { return GetTensor()->SetDtype(type); }

TypeId TensorPy::GetDataType() const { return GetTensor()->data_type(); }

const ShapeVector &TensorPy::GetShape() const { return GetTensor()->shape(); }

bool TensorPy::IsInit() const { return GetTensor()->is_init(); }

void TensorPy::SetInitFlag(bool flag) { GetTensor()->set_init_flag(flag); }

void TensorPy::SetShape(const ShapeVector &shape) { GetTensor()->set_shape(shape); }

int TensorPy::DataDim() const { return GetTensor()->DataDim(); }

TensorPy &TensorPy::AssignValue(const TensorPy &tensorpy) {
  // todo: assign value for base tensor.
  auto tensor = tensorpy.GetTensor();
  GetTensor()->AssignValue(*tensor);
  return *this;
}

bool TensorPy::Offload(const std::string &file_path) { return GetTensor()->Offload(file_path); }

const std::string TensorPy::GetOffloadFilePath() const { return GetTensor()->GetOffloadFilePath(); }

void TensorPy::SetCastDtype(const TypePtr &dtype) { GetTensor()->set_cast_dtype(dtype); }

void TensorPy::DataSync(bool need_wait) const { MS_LOG(EXCEPTION) << "Tensor not support data_sync."; }

void TensorPy::ExecuteLazyTask() const { GetTensor()->ExecuteLazyTask(); }

bool TensorPy::IsContiguous() const { return GetTensor()->is_contiguous(); }

std::vector<int64_t> TensorPy::GetStride() const { return GetTensor()->stride(); }

const int64_t TensorPy::GetStorageOffset() const { return SizeToLong(GetTensor()->storage_offset()); }

std::string TensorPy::ToString() const { return GetTensor()->ToString(); }

std::string TensorPy::ToStringRepr() const { return GetTensor()->ToStringRepr(); }

bool TensorPy::CheckStub() { return Tensor::CheckStub(); }

ParamInfoPtr TensorPy::GetParamInfo() const { return GetTensor()->param_info(); }

void TensorPy::SetParamInfo(const ParamInfoPtr &param_info) {
  auto base_tensor = GetTensor();
  MS_EXCEPTION_IF_NULL(base_tensor);
  base_tensor->set_param_info(param_info);
}

bool TensorPy::IsComplex() const {
  auto base_tensor = GetTensor();
  TypeId type_id = base_tensor->data_type();
  switch (type_id) {
    case TypeId::kNumberTypeComplex:
    case TypeId::kNumberTypeComplex64:
    case TypeId::kNumberTypeComplex128:
      return true;
    default:
      break;
  }
  return false;
}

bool TensorPy::IsSigned() const {
  auto base_tensor = GetTensor();
  TypeId type_id = base_tensor->data_type();
  switch (type_id) {
    case TypeId::kNumberTypeInt:
    case TypeId::kNumberTypeInt8:
    case TypeId::kNumberTypeInt16:
    case TypeId::kNumberTypeInt32:
    case TypeId::kNumberTypeInt64:
    case TypeId::kNumberTypeFloat:
    case TypeId::kNumberTypeFloat32:
    case TypeId::kNumberTypeFloat64:
    case TypeId::kNumberTypeBFloat16:
    case TypeId::kNumberTypeComplex:
    case TypeId::kNumberTypeComplex64:
    case TypeId::kNumberTypeComplex128:
      return true;
    default:
      break;
  }
  return false;
}

const size_t TensorPy::GetDataSize() const { return GetTensor()->DataSize(); }

void *TensorPy::GetTensorDataObject() const { return GetTensor()->data_c(); }

const DeviceAddressPtr TensorPy::GetDeviceAddress() const { return GetTensor()->device_address(); }

bool TensorPy::IsMSParameterOutput() const { return ms_parameter_output_; }

void TensorPy::SetMSParameterOutput(bool flag) { ms_parameter_output_ = flag; }

bool TensorPy::HasAutoGrad() const { return GetTensor()->HasAutoGrad(); }

bool TensorPy::NeedContiguous() const { return GetTensor()->NeedContiguous(); }

/* =========================================== Common Function ================================================= */
constexpr ssize_t kPyBufItemSize1 = 1;
constexpr ssize_t kPyBufItemSize2 = 2;
constexpr ssize_t kPyBufItemSize4 = 4;
constexpr ssize_t kPyBufItemSize8 = 8;

static TypeId GetBaseDataType(const char data_type, ssize_t item_size) {
  switch (data_type) {
    case 'e':
    case 'f':
    case 'd':
      switch (item_size) {
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
      switch (item_size) {
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
      switch (item_size) {
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
    case 'E':
      return TypeId::kNumberTypeBFloat16;
    default:
      MS_LOG(WARNING) << "Unsupported DataType format " << data_type << ", item size " << item_size;
      return TypeId::kTypeUnknown;
  }
  return TypeId::kTypeUnknown;
}

static TypeId GetDataType(const py::buffer_info &buf) {
  if (buf.format.size() == 1) {
    char data_type = buf.format.front();
    return GetBaseDataType(data_type, buf.itemsize);
  } else if (buf.format.size() >= 2) {
    // Support np.str_ dtype, format: {x}w. {x} is a number that means the maximum length of the string items.
    if (buf.format.back() == 'w' || buf.format.back() == 's') {
      return TypeId::kObjectTypeString;
    } else if (buf.format == "Zf") {
      return TypeId::kNumberTypeComplex64;
    } else if (buf.format == "Zd") {
      return TypeId::kNumberTypeComplex128;
    }

    if (buf.format.size() == 2) {
      char byte_order = buf.format[0];
      char data_type = buf.format[1];
      if (byte_order == '<' || byte_order == '>' || byte_order == '=' || byte_order == '|' || byte_order == '!') {
        return GetBaseDataType(data_type, buf.itemsize);
      }
    }
  }
  MS_LOG(WARNING) << "Unsupported DataType format " << buf.format << ", item size " << buf.itemsize;
  return TypeId::kTypeUnknown;
}

static std::string GetPyTypeFormat(TypeId data_type) {
  switch (data_type) {
    case TypeId::kNumberTypeFloat16:
      return "e";
    case TypeId::kNumberTypeBFloat16:
      return "E";
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
    case TypeId::kNumberTypeInt4:
    case TypeId::kNumberTypeInt8:
      return py::format_descriptor<int8_t>::format();
    case TypeId::kNumberTypeInt16:
      return py::format_descriptor<int16_t>::format();
    case TypeId::kNumberTypeInt:
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
  explicit TensorDataNumpy(py::buffer_info &&buffer) : buffer_(std::make_unique<py::buffer_info>(std::move(buffer))) {}

  ~TensorDataNumpy() override {
    py::gil_scoped_acquire acquire;
    buffer_.reset();
  }

  /// Total number of elements.
  ssize_t size() const override { return buffer()->size; }

  /// Byte size of a single element.
  ssize_t itemsize() const override { return buffer()->itemsize; }

  /// Total number of bytes.
  ssize_t nbytes() const override { return buffer()->itemsize * buffer()->size; }

  /// Number of dimensions.
  ssize_t ndim() const override { return buffer()->ndim; }

  /// Data pointer.
  void *data() override { return buffer_data(); }

  void *const_data() const override { return buffer()->ptr; }

  bool is_from_numpy() const override { return true; }

  const std::vector<ssize_t> &shape() const { return buffer()->shape; }

  /// To string.
  std::string ToString(const TypeId, const ShapeVector &, bool use_comma) const override {
    py::gil_scoped_acquire gil_acquire;
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
    py::gil_scoped_acquire acquire;
    py::dtype np_dtype =
      (buffer()->format == "E") ? py::detail::npy_format_descriptor<bfloat16>::dtype() : py::dtype(*buffer());
    return py::array(np_dtype, buffer()->shape, buffer()->strides, buffer()->ptr, owner);
  }

 private:
  void *buffer_data() const { return buffer_->ptr; }
  std::unique_ptr<py::buffer_info> const &buffer() const {
    MS_EXCEPTION_IF_NULL(buffer_);
    return buffer_;
  }

  // The internal buffer.
  std::unique_ptr<py::buffer_info> buffer_;
};

py::buffer_info GetPyBufferFromPyArray(const py::array &input) {
  py::buffer_info buf;
  auto descr = py::detail::array_descriptor_proxy(py::detail::array_proxy(input.ptr())->descr);
  // For bfloat16, modify descr->type_num to support acquiring buffer_info from numpy.
  if (descr->type == 'E') {
    // convert descr->type_num from E(NPY_BFLOAT16) to H(NPY_USHORT)
    const int NPY_USHORT = 4;
    int orig_type_num = descr->type_num;
    descr->type_num = NPY_USHORT;
    // acquire buffer_info with type of NPY_USHORT
    buf = input.request();
    // convert buffer_info.format from H(NPY_USHORT) to E(NPY_BFLOAT16)
    buf.format = "E";
    // change back descr->type_num
    descr->type_num = orig_type_num;
  } else {
    buf = input.request();
  }
  return buf;
}

TensorPtr MakeTensor(const py::array &input, const TypePtr &type_ptr) {
  py::gil_scoped_acquire acquire;
  // Get input buffer info.
  py::buffer_info buf = tensor::GetPyBufferFromPyArray(input);
  // Check data types.
  auto data_type = type_ptr ? type_ptr->type_id() : TypeId::kTypeUnknown;
  auto buf_type = GetDataType(buf);
  if (buf_type == TypeId::kTypeUnknown && data_type == TypeId::kTypeUnknown) {
    MS_LOG(EXCEPTION) << "Unsupported tensor type!";
  }
  MS_LOG(DEBUG) << "data_type: " << data_type << ", buf_type: " << buf_type;
  if (data_type == TypeId::kObjectTypeString || buf_type == TypeId::kObjectTypeString) {
    return tensor::MakeTensorOfNumpy(input);
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
    return tensor::from_buffer(data_type, shape, buf.ptr, buf.size * buf.itemsize);
  }
  // Create tensor with data type converted.
  return tensor::from_buffer(data_type, shape, buf.ptr, buf_type);
}

/// Creates a Tensor from a numpy array without copy
TensorPtr MakeTensorOfNumpy(const py::array &input) {
  py::gil_scoped_acquire acquire;
  // Check format.
  if (!IsCContiguous(input)) {
    MS_LOG(EXCEPTION) << "Array should be C contiguous.";
  }
  // Get input buffer info.
  py::buffer_info buf = tensor::GetPyBufferFromPyArray(input);
  // Get tensor dtype and check it.
  auto dtype = GetDataType(buf);
  if (dtype == TypeId::kTypeUnknown) {
    MS_LOG(EXCEPTION) << "Unsupported data type!";
  }
  // Get tensor shape.
  ShapeVector shape(buf.shape.begin(), buf.shape.end());

  // Make a tensor with shared data with numpy array.
  auto tensor_data = std::make_shared<TensorDataNumpy>(std::move(buf));

  auto device_address = DeviceAddressMaker(tensor_data->data(), dtype, shape)
                          .set_deleter([tensor_data](void *, bool) {})
                          .set_maker(GetDeviceAddressMaker(device::DeviceType::kCPU))
                          .make_device_address();
  device_address->set_data(std::move(tensor_data));

  return std::make_shared<Tensor>(dtype, shape, device_address);
}

static py::buffer_info GetPyBufferInfo(const TensorPtr &tensor) {
  std::vector<ssize_t> shape(tensor->shape().begin(), tensor->shape().end());
  std::vector<ssize_t> strides = GetStrides(shape, tensor->DataItemSize());
  return py::buffer_info{
    tensor->data_c(), tensor->DataItemSize(), GetPyTypeFormat(tensor->data_type()), tensor->DataDim(), shape, strides};
}

static py::buffer_info GetPyBufferInfo(const Tensor &tensor) {
  std::vector<ssize_t> shape(tensor.shape().begin(), tensor.shape().end());
  std::vector<ssize_t> strides = GetStrides(shape, tensor.DataItemSize());
  return py::buffer_info{
    tensor.data_c(), tensor.DataItemSize(), GetPyTypeFormat(tensor.data_type()), tensor.DataDim(), shape, strides};
}

py::array NumpyNonBlocking(const Tensor &tensor) {
  const auto &device_address = tensor.device_address();
  if (device_address == nullptr) {
    MS_LOG(EXCEPTION) << "Tensor " << tensor.ToString() << " is uninitialized. "
                      << "Maybe you need to call Tensor.init_data first.";
  }
  if (device_address->GetDeviceType() != device::DeviceType::kCPU) {
    MS_LOG(EXCEPTION) << "Only support convert CPU Tensor to Numpy array, but got Tensor on "
                      << device::GetDeviceNameByType(device_address->GetDeviceType());
  }
  py::object owner = py::cast(device_address);
  if (device_address->has_data()) {
    const auto &data = device_address->data();
    auto raw_data = dynamic_cast<TensorDataNumpy *>(data.get());
    if (raw_data != nullptr) {
      return raw_data->py_array(owner);
    }
  }
  // Create numpy array by buffer protocol.
  auto info = GetPyBufferInfo(tensor);
  py::dtype np_dtype = (tensor.data_type() == kNumberTypeBFloat16)
                         ? py::detail::npy_format_descriptor<bfloat16>::dtype()
                         : py::dtype(info);
  return py::array(np_dtype, info.shape, info.strides, info.ptr, owner);
}

py::array AsNumpy(const Tensor &tensor) {
  // Use TensorData as the owner to prevent use-after-free problem.
  // We can NOT use Tensor as the owner since its TensorData may change
  // by other operations such as AssignValue().
  py::gil_scoped_acquire acquire;
  auto tensor_cpu = tensor.cpu();
  py::object owner = py::cast(tensor_cpu->device_address());
  if (tensor_cpu->device_address() != nullptr && tensor_cpu->device_address()->has_data()) {
    const auto &data = tensor_cpu->device_address()->data();
    auto raw_data = dynamic_cast<TensorDataNumpy *>(data.get());
    if (raw_data != nullptr) {
      return raw_data->py_array(owner);
    }
  }
  // Create numpy array by buffer protocol.
  auto info = GetPyBufferInfo(tensor_cpu);
  py::dtype np_dtype = (tensor_cpu->data_type() == kNumberTypeBFloat16)
                         ? py::detail::npy_format_descriptor<bfloat16>::dtype()
                         : py::dtype(info);
  return py::array(np_dtype, info.shape, info.strides, info.ptr, owner);
}

bool IsTensorPy(const py::handle &obj) {
  if (TensorPy_Type == nullptr || !obj.check()) {
    return false;
  }
  PyObject *raw_ptr = obj.ptr();
  return PyObject_TypeCheck(raw_ptr, TensorPy_Type);
}
REGISTER_OPS_CALLBACK(IsTensorPy);

bool IsPyObjectTensorPy(PyObject *obj) {
  if (TensorPy_Type == nullptr || obj == nullptr) {
    return false;
  }
  return PyObject_TypeCheck(obj, TensorPy_Type);
}

bool IsPyObjectPythonTensorStrict(PyObject *obj) {
  if (obj == nullptr) {
    return false;
  }
  return Py_TYPE(obj) == PyObjManager::Get().GetTensorPythonType();
}

py::object GetPythonTensor() {
  auto tensor_module = PyObjManager::Get().GetTensorModule();
  return py::reinterpret_borrow<py::object>(tensor_module);
}

const ValuePtr ConvertToValue(const py::handle &obj) {
  PyObject *raw_ptr = obj.ptr();
  if (PyObject_TypeCheck(raw_ptr, TensorPy_Type)) {
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)raw_ptr;
    auto &value = tensor->value;
    if (value.has_stub()) {
      return value.stub();
    }
    return value.GetTensor();
  }
  MS_LOG(EXCEPTION) << "Not TensorPy object";
}

const ValuePtr ConvertPyObjectToValue(PyObject *obj) {
  if (PyObject_TypeCheck(obj, TensorPy_Type)) {
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)obj;
    auto &value = tensor->value;
    if (value.has_stub()) {
      return value.stub();
    }
    return value.GetTensor();
  }
  MS_LOG(EXCEPTION) << "Not TensorPy object";
}

TensorPtr ConvertToTensor(const py::handle &obj) {
  PyObject *raw_ptr = obj.ptr();
  if (PyObject_TypeCheck(raw_ptr, TensorPy_Type)) {
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)raw_ptr;
    auto tensor_ptr = tensor->value.GetTensor();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    return tensor_ptr;
  }
  return nullptr;
}

void SetTensorValue(const py::handle &obj, const TensorPtr &tensor_value) {
  PyObject *raw_ptr = obj.ptr();
  if (PyObject_TypeCheck(raw_ptr, TensorPy_Type)) {
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)raw_ptr;
    tensor->value.SetTensor(tensor_value);
  } else {
    MS_LOG(EXCEPTION) << "Not TensorPy object";
  }
}

TensorPtr ConvertPyObjectToTensor(PyObject *obj) {
  if (PyObject_TypeCheck(obj, TensorPy_Type)) {
    PyType<TensorPy> *tensor = (PyType<TensorPy> *)obj;
    auto tensor_ptr = tensor->value.GetTensor();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    return tensor_ptr;
  }
  return nullptr;
}

PyType<TensorPy> *ConvertPyObject2TensorPyType(const py::object obj) {
  PyType<TensorPy> *tensor_type = reinterpret_cast<PyType<TensorPy> *>(obj.ptr());

  return tensor_type;
}

const py::handle ConvertToTensorPy(const py::handle &obj) {
  PyObject *raw_ptr = obj.ptr();
  if (PyObject_TypeCheck(raw_ptr, TensorPy_Type)) {
    return obj;
  }

  return nullptr;
}

PyObject *TensorPythonInit(const TensorPtr &tensor) {
  PyObject *python_tensor_class = PyObjManager::Get().GetTensorPythonClass();
  PyObject *obj = (reinterpret_cast<PyTypeObject *>(python_tensor_class))
                    ->tp_alloc(reinterpret_cast<PyTypeObject *>(python_tensor_class), 0);
  if (obj == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  auto result = (PyType<TensorPy> *)obj;
  if (tensor == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }

  new (&result->value) TensorPy(tensor);
  result->value.SetInitFinished(true);

  return reinterpret_cast<PyObject *>(result);
}

PyObject *TensorPythonInitFromTensor(TensorPtr tensor) {
  PyType<TensorPy> *result = (PyType<TensorPy> *)TensorPy_Type->tp_alloc(TensorPy_Type, 0);
  if (result == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  if (tensor == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }

  new (&result->value) TensorPy(tensor);
  // set to adapt python __repr__
  result->value.SetInitFinished(true);

  return reinterpret_cast<PyObject *>(result);
}

PyTypeObject *GetTensorPyType() { return TensorPy_Type; }

void SetTensorPyType(PyTypeObject *TensorPyType) { TensorPy_Type = TensorPyType; }

py::object PackTensorToPyObject(const TensorPtr &tensor) {
  PyObject *tensor_py = TensorPythonInit(tensor);
  return py::reinterpret_steal<py::object>(tensor_py);
}

PyObject *PackTensor(const TensorPtr &tensor, bool has_side_effect) {
  PyObject *python_tensor_class = PyObjManager::Get().GetTensorPythonClass();
  auto tensor_py_type = reinterpret_cast<PyTypeObject *>(python_tensor_class);
  PyObject *obj = tensor_py_type->tp_alloc(tensor_py_type, 0);
  if (obj == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  auto result = (PyType<TensorPy> *)obj;
  new (&result->value) TensorPy(tensor);
  result->value.SetInitFinished(true);
  result->value.set_has_side_effect(has_side_effect);
  return reinterpret_cast<PyObject *>(result);
}

PyObject *PackStubTensor(const stub::StubNodePtr &stub_node) {
  PyObject *python_tensor_class = PyObjManager::Get().GetTensorPythonClass();
  auto tensor_py_type = reinterpret_cast<PyTypeObject *>(python_tensor_class);
  PyObject *obj = tensor_py_type->tp_alloc(tensor_py_type, 0);
  if (obj == nullptr) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create TensorPy object");
    return nullptr;
  }
  auto result = (PyType<TensorPy> *)obj;
  new (&result->value) TensorPy(stub_node);
  result->value.SetInitFinished(true);
  return reinterpret_cast<PyObject *>(result);
}

PyObject *Wrap(const TensorPtr &tensor) { return PackTensor(tensor); }

PyObject *Wrap(const std::vector<TensorPtr> &tensors) {
  PyObject *output = PyTuple_New(static_cast<Py_ssize_t>(tensors.size()));
  for (size_t i = 0; i < tensors.size(); ++i) {
    PyTuple_SET_ITEM(output, i, Wrap(tensors[i]));
  }
  return output;
}

PyObject *Wrap(const ValuePtrList &values) {
  PyObject *output = PyTuple_New(static_cast<Py_ssize_t>(values.size()));
  for (size_t i = 0; i < values.size(); ++i) {
    PyTuple_SET_ITEM(output, i, Wrap(values[i]));
  }
  return output;
}

PyObject *Wrap(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    return Wrap(value->cast<tensor::TensorPtr>());
  } else if (value->isa<ValueSequeue>()) {
    auto sequeue = value->cast<ValueSequencePtr>();
    const auto &values = sequeue->value();
    size_t size = values.size();
    bool is_tuple = value->isa<ValueTuple>();
    PyObject *output =
      is_tuple ? PyTuple_New(static_cast<Py_ssize_t>(size)) : PyList_New(static_cast<Py_ssize_t>(size));
    for (size_t i = 0; i < size; ++i) {
      if (is_tuple) {
        PyTuple_SET_ITEM(output, i, Wrap(values[i]));
      } else {
        PyList_SET_ITEM(output, i, Wrap(values[i]));
      }
    }
    return output;
  } else {
    return ValueToPyData(value).release().ptr();
  }
}

PyTypeObject *getTensorPyType() { return TensorPy_Type; }

void setTensorPyType(PyTypeObject *TensorPyType) { TensorPy_Type = TensorPyType; }
}  // namespace tensor
}  // namespace mindspore
