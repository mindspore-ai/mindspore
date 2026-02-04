/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "include/utils/pynative/py_parse.h"

#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <set>
#include <memory>

#include "numpy/arrayobject.h"
#include "mindspore/core/include/ir/tensor_new.h"
#include "mindspore/core/include/utils/value_utils.h"
#include "include/utils/tensor_py.h"
#include "include/utils/stub_tensor.h"
#include "include/utils/tensor_utils.h"
#include "include/utils/operator/primitive_utils.h"

#if NPY_API_VERSION < 0x0000000d
#error Current Numpy version is too low, the required version is not less than 1.19.3.
#endif

namespace mindspore {
namespace py_parse {
static std::string ReportPyBindType(const py::object &obj) {
  py::gil_scoped_acquire gil;
  return obj.get_type().attr("__name__").cast<std::string>();
}

using Tensor = mindspore::tensor::Tensor;
using TensorPtr = mindspore::tensor::TensorPtr;

TensorPtr ConvertPyObjectTensorValue(PyObject *obj) {
  if (tensor::IsPyObjectTensorPy(obj)) {
    return tensor::ConvertPyObjectToTensor(obj);
  }

  return nullptr;
}

ValuePtr ConvertPyObjectTensor(PyObject *obj) {
  if (tensor::IsPyObjectTensorPy(obj)) {
    return tensor::ConvertPyObjectToValue(obj);
  }

  return nullptr;
}

ValuePtr ConvertTensor(const py::object &obj) {
  if (tensor::IsTensorPy(obj)) {
    return tensor::ConvertToValue(obj);
  }

  return nullptr;
}

TensorPtr ConvertTensorValue(const py::object &obj) {
  if (tensor::IsTensorPy(obj)) {
    return tensor::ConvertToTensor(obj);
  }

  return nullptr;
}
namespace {
bool IsNumpyAvailable() {
  static bool available = []() {
    if (_import_array() >= 0) {
      return true;
    } else {
      MS_LOG(WARNING) << "Numpy init failed.";
      return false;
    }
  }();
  return available;
}

inline static bool IsPythonInt(PyObject *obj) { return !PyBool_Check(obj) && PyLong_Check(obj); }

inline static bool IsNumpyInt(PyObject *obj) { return IsNumpyAvailable() && PyArray_IsScalar(obj, Integer); }

inline static bool IsNumpyBool(PyObject *obj) { return IsNumpyAvailable() && PyArray_IsScalar(obj, Bool); }

inline static bool IsNumpyFloat(PyObject *obj) { return IsNumpyAvailable() && PyArray_IsScalar(obj, Floating); }

inline static bool IsNumpyScalar(PyObject *obj) {
  return IsNumpyAvailable() &&
         (PyArray_IsScalar(obj, Integer) || PyArray_IsScalar(obj, Bool) || PyArray_IsScalar(obj, Floating));
}

inline static bool IsNumpyScalarIntArray(PyObject *obj) {
  if (!IsNumpyAvailable()) {
    return false;
  }
  // numpy array with empty shape and one int element
  if (!PyArray_Check(obj)) {
    return false;
  }
  PyArrayObject *array_obj = reinterpret_cast<PyArrayObject *>(obj);
  if (!PyArray_IsZeroDim(array_obj)) {
    return false;
  }
  static std::set<int> valid_int_types{NPY_SHORT, NPY_USHORT, NPY_INT,      NPY_UINT,
                                       NPY_LONG,  NPY_ULONG,  NPY_LONGLONG, NPY_ULONGLONG};
  int array_data_type = PyArray_TYPE(array_obj);
  return valid_int_types.find(array_data_type) != valid_int_types.end();
}

inline int64_t ConvertNumpyScalarIntArray(PyObject *obj) {
  PyArrayObject *array_obj = reinterpret_cast<PyArrayObject *>(obj);
  void *data = PyArray_DATA(array_obj);
  int type_num = PyArray_TYPE(array_obj);
  int64_t value;
  switch (type_num) {
    case NPY_SHORT:
      value = *reinterpret_cast<short *>(data);
      break;
    case NPY_USHORT:
      value = *reinterpret_cast<unsigned short *>(data);
      break;
    case NPY_INT:
      value = *reinterpret_cast<int *>(data);
      break;
    case NPY_UINT:
      value = *reinterpret_cast<unsigned int *>(data);
      break;
    case NPY_LONG:
      value = *reinterpret_cast<long *>(data);
      break;
    case NPY_ULONG:
      value = *reinterpret_cast<unsigned long *>(data);
      break;
    case NPY_LONGLONG:
      value = *reinterpret_cast<long long *>(data);
      break;
    case NPY_ULONGLONG:
      value = *reinterpret_cast<unsigned long long *>(data);
      break;
    default:
      MS_EXCEPTION(ValueError) << "Unsupported data type " << type_num;
  }
  return value;
}

template <typename T, bool allow_size_one>
static bool IsScalarTensor(PyObject *obj) {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, double> || std::is_same_v<T, bool>,
                "T must be int64_t, double, or bool");

  auto tensor = ConvertPyObjectTensorValue(obj);
  if (!tensor) {
    return false;
  }

  // allow_size_one == true: allow [1], [1,1], ... (size-1 tensors)
  // allow_size_one == false: require [] (true 0-D scalar)
  if constexpr (allow_size_one) {
    auto tensor_size = tensor->DataSize();
    if (tensor_size != 1) {
      return false;
    }
  } else {
    if (!tensor->shape().empty()) {
      return false;
    }
  }

  auto tensor_type = tensor->data_type();
  if constexpr (std::is_same_v<T, int64_t>) {
    constexpr std::array<TypeId, 6> valid_types{kNumberTypeUInt8, kNumberTypeInt8,  kNumberTypeInt16,
                                                kNumberTypeInt,   kNumberTypeInt32, kNumberTypeInt64};
    return std::find(valid_types.begin(), valid_types.end(), tensor_type) != valid_types.end();
  } else if constexpr (std::is_same_v<T, double>) {
    constexpr std::array<TypeId, 3> valid_types{kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
    return std::find(valid_types.begin(), valid_types.end(), tensor_type) != valid_types.end();
  } else {
    return tensor_type == kNumberTypeBool;
  }
}

std::optional<int64_t> FetchScalarTensorIntValue(const tensor::TensorPtr &tensor) {
  const auto &tensor_type_id = tensor->data_type();
  static const std::unordered_map<TypeId, std::function<int64_t(const tensor::TensorPtr &tensor)>> valid_integral{
    {kNumberTypeUInt8,
     [](const tensor::TensorPtr &tensor) -> int64_t { return static_cast<int64_t>(TensorItem<uint8_t>(tensor)); }},
    {kNumberTypeInt8,
     [](const tensor::TensorPtr &tensor) -> int64_t { return static_cast<int64_t>(TensorItem<int8_t>(tensor)); }},
    {kNumberTypeInt16,
     [](const tensor::TensorPtr &tensor) -> int64_t { return static_cast<int64_t>(TensorItem<int16_t>(tensor)); }},
    {kNumberTypeInt,
     [](const tensor::TensorPtr &tensor) -> int64_t { return static_cast<int64_t>(TensorItem<int>(tensor)); }},
    {kNumberTypeInt32,
     [](const tensor::TensorPtr &tensor) -> int64_t { return static_cast<int64_t>(TensorItem<int32_t>(tensor)); }},
    {kNumberTypeInt64,
     [](const tensor::TensorPtr &tensor) -> int64_t { return static_cast<int64_t>(TensorItem<int64_t>(tensor)); }},
  };
  auto it = valid_integral.find(tensor_type_id);
  if (MS_UNLIKELY(tensor->DataSize() != 1 || it == valid_integral.end())) {
    return std::nullopt;
  }
  return it->second(tensor);
}

std::optional<double> FetchScalarTensorFloatValue(const tensor::TensorPtr &tensor) {
  const auto &tensor_type_id = tensor->data_type();
  static const std::unordered_map<TypeId, std::function<double(const tensor::TensorPtr &tensor)>> valid_integral{
    {kNumberTypeFloat16,
     [](const tensor::TensorPtr &tensor) -> double { return static_cast<double>(TensorItem<float16>(tensor)); }},
    {kNumberTypeFloat32,
     [](const tensor::TensorPtr &tensor) -> double { return static_cast<double>(TensorItem<float>(tensor)); }},
    {kNumberTypeFloat64, [](const tensor::TensorPtr &tensor) -> double { return TensorItem<double>(tensor); }}};
  auto it = valid_integral.find(tensor_type_id);
  if (MS_UNLIKELY(tensor->DataSize() != 1 || it == valid_integral.end())) {
    return std::nullopt;
  }
  return it->second(tensor);
}

std::optional<bool> FetchScalarTensorBoolValue(const tensor::TensorPtr &tensor) {
  const auto &tensor_type_id = tensor->data_type();
  if (tensor->DataSize() == 1 && tensor_type_id == kNumberTypeBool) {
    return TensorItem<bool>(tensor);
  }
  return std::nullopt;
}

template <typename T>
static std::optional<T> ConvertScalarTensor(PyObject *obj) {
  static_assert(std::is_same_v<T, int64_t> || std::is_same_v<T, double> || std::is_same_v<T, bool>,
                "T must be int64_t, double, or bool");
  // convert tensor with one int element to int
  auto tensor = ConvertPyObjectTensorValue(obj);
  if (MS_UNLIKELY(tensor == nullptr)) {
    return std::nullopt;
  }
  if constexpr (std::is_same_v<T, int64_t>) {
    return FetchScalarTensorIntValue(tensor);
  } else if constexpr (std::is_same_v<T, double>) {
    return FetchScalarTensorFloatValue(tensor);
  } else {
    return FetchScalarTensorBoolValue(tensor);
  }
}
}  // namespace

namespace {
template <typename T, typename U>
ValuePtr PyCast(const py::object &obj) {
  return std::make_shared<T>(py::cast<U>(obj));
}

ValuePtr ConvertIntegerWithType(const py::object &obj) {
  // The mutable _Bool class inherits from int, because base class 'bool' is a marked final.
  if (py::hasattr(obj, "__ms_mutable_bool__")) {
    auto obj_int64 = py::cast<int64_t>(obj);
    bool obj_bool = obj_int64 != 0;
    return std::make_shared<BoolImm>(obj_bool);
  }
  return std::make_shared<Int64Imm>(*ParseUtilsConvertInt(obj.ptr()));
}

inline static ValuePtr MakeFP32Imm(double obj_double) {
  auto obj_float32 = static_cast<pyfloat>(obj_double);
  auto ret = std::make_shared<FP32Imm>(obj_float32);
  ret->set_prim_value(obj_double);
  return ret;
}

ValuePtr ConvertFloatWithType(const py::object &obj) {
  auto value_opt = ParseUtilsConvertFloat(obj.ptr());
  if (value_opt) {
    return MakeFP32Imm(*value_opt);
  }
  return nullptr;
}

ValuePtr ConvertBool(const py::object &obj) {
  if (!py::isinstance<py::bool_>(obj)) {
    // The mutable _Bool class inherits from int, because base class 'bool' is a marked final.
    if (py::isinstance<py::int_>(obj) && py::hasattr(obj, "__ms_mutable_bool__")) {
      auto obj_int64 = py::cast<int64_t>(obj);
      bool obj_bool = obj_int64 != 0;
      return std::make_shared<BoolImm>(obj_bool);
    }
    return nullptr;
  }
  return PyCast<BoolImm, bool>(obj);
}

ValuePtr ConvertInt(const py::object &obj) {
  // bool is also an instance of py::int_
  if (!ParseUtilsCheckInt(obj.ptr())) {
    return nullptr;
  }
  return ConvertIntegerWithType(obj);
}

template <bool allow_scalar_tensor>
ValuePtr ConvertFloat(const py::object &obj) {
  if (!ParseUtilsCheckFloat(obj.ptr(), allow_scalar_tensor)) {
    return nullptr;
  }
  return ConvertFloatWithType(obj);
}

ValuePtr ConvertNumber(const py::object &obj) {
  if (ParseUtilsCheckBool(obj.ptr())) {
    return std::make_shared<BoolImm>(*ParseUtilsConvertBool(obj.ptr()));
  }
  if (ParseUtilsCheckInt(obj.ptr())) {
    return ConvertIntegerWithType(obj);
  }
  if (ParseUtilsCheckFloat(obj.ptr())) {
    return ConvertFloatWithType(obj);
  }
  return nullptr;
}

ValuePtr ConvertStr(const py::object &obj) {
  if (!py::isinstance<py::str>(obj)) {
    return nullptr;
  }
  return PyCast<StringImm, string>(obj);
}

ValuePtr ConvertDtype(const py::object &obj) {
  if (!py::isinstance<mindspore::Type>(obj)) {
    MS_LOG(EXCEPTION) << "Get arg is not mindspore type " << ReportPyBindType(obj);
  }
  return obj.cast<TypePtr>();
}

template <typename T1, typename T2>
ValuePtr ConvertSingleElementToTensor(const py::object &obj) {
  if (!py::isinstance<T1>(obj)) {
    return nullptr;
  }

  auto v = py::cast<T2>(obj);
  return tensor::from_scalar(v);
}

ValuePtr ConvertNumberToTensor(const py::object &obj) {
  if (ParseUtilsCheckBool(obj.ptr(), false)) {
    auto v = py::cast<bool>(obj);
    return tensor::from_scalar(v);
  }

  if (ParseUtilsCheckInt(obj.ptr(), false)) {
    auto v = py::cast<int64_t>(obj);
    return tensor::from_scalar(v);
  }

  if (ParseUtilsCheckFloat(obj.ptr(), false)) {
    auto v = py::cast<pyfloat>(obj);
    return tensor::from_scalar(v);
  }

  return nullptr;
}

inline ValuePtr ConvertPythonFloatToScalarValue(double value) {
  auto ret = std::make_shared<FP32Imm>(static_cast<float>(value));
  ret->set_prim_value(value);
  return ret;
}

template <typename TS, typename TSE, typename TDE>
ValuePtr ConvertSequenceToTensor(const py::object &obj) {
  if (!py::isinstance<TS>(obj)) {
    return nullptr;
  }

  auto seq = obj.cast<TS>();
  if (seq.size() == 0) {
    return nullptr;
  }

  std::vector<TDE> value_list;
  for (size_t it = 0; it < seq.size(); ++it) {
    if (!py::isinstance<TSE>(seq[it])) {
      return nullptr;
    }

    auto value = py::cast<TDE>(seq[it]);
    value_list.emplace_back(value);
  }

  return tensor::from_vector(value_list);
}

template <typename TS>
ValuePtr ConvertSequenceBoolToTensor(const py::object &obj) {
  if (!py::isinstance<TS>(obj)) {
    return nullptr;
  }

  auto seq = obj.cast<TS>();
  if (seq.size() == 0) {
    return nullptr;
  }

  auto tensor =
    tensor::from_spec(kNumberTypeBool, ShapeVector({static_cast<int64_t>(seq.size())}), device::DeviceType::kCPU);
  auto data = static_cast<bool *>(tensor->data_c());
  for (size_t it = 0; it < seq.size(); ++it) {
    if (!py::isinstance<py::bool_>(seq[it])) {
      return nullptr;
    }

    auto value = py::cast<bool>(seq[it]);
    data[it] = value;
  }

  return tensor;
}

template <typename TD, typename TDE, typename IMMTYPE, TypeId tid>
ValuePtr ConvertTensorToSequence(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    MS_LOG(INFO) << "Can not convert python object with type [" << ReportPyBindType(obj) << "] to Tensor.";
    return nullptr;
  }

  auto data_type = tensor->data_type();
  // Since the dst object type is only, once the src object is validated as Tensor, the other converting errors should
  // be thrown. There is no other paths for this case to run successfully.
  if (data_type != tid) {
    MS_LOG(ERROR) << "Can not convert Tensor with type " << TypeIdToString(data_type) << "to Sequence with type "
                  << TypeIdToString(tid) << ".";
    return nullptr;
  }

  auto shape = tensor->shape();
  if (shape.size() > 1) {
    MS_LOG(ERROR) << "Only support converting 1-D Tensor or scalar Tensor to sequence. But got the shape of Tensor: "
                  << shape;
    return nullptr;
  }

  // The data_c is a raw pointer and will become invalid when cpu_tensor is destructed.
  auto cpu_tensor = tensor->cpu();
  auto data = static_cast<TDE *>(cpu_tensor->data_c());
  auto size = tensor->DataSize();
  std::vector<ValuePtr> value_list;
  for (size_t i = 0; i < size; ++i) {
    (void)value_list.emplace_back(std::make_shared<IMMTYPE>(data[i]));
  }
  return std::make_shared<TD>(value_list);
}

template <typename TD>
ValuePtr ConvertTensorToSequenceInt(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    MS_LOG(INFO) << "Can not convert python object with type [" << ReportPyBindType(obj) << "] to Tensor.";
    return nullptr;
  }

  auto shape = tensor->shape();
  if (shape.size() > 1) {
    MS_LOG(ERROR) << "Only support converting 1-D Tensor or scalar Tensor to sequence. But got the shape of Tensor: "
                  << shape;
    return nullptr;
  }

  auto data_type = tensor->data_type();
  if (data_type != kNumberTypeInt64 && data_type != kNumberTypeInt32) {
    MS_LOG(ERROR) << "Can not convert Tensor with type " << TypeIdToString(data_type) << "to Int Sequence.";
    return nullptr;
  }
  auto size = tensor->DataSize();
  std::vector<ValuePtr> value_list;
  // The data_c is a raw pointer and will become invalid when cpu_tensor is destructed.
  auto cpu_tensor = tensor->cpu();
  if (data_type == kNumberTypeInt64) {
    auto data = static_cast<int64_t *>(cpu_tensor->data_c());
    std::transform(data, data + size, std::back_inserter(value_list),
                   [](int64_t num) { return std::make_shared<Int64Imm>(num); });
  } else {
    auto data = static_cast<int32_t *>(cpu_tensor->data_c());
    std::transform(data, data + size, std::back_inserter(value_list),
                   [](int32_t num) { return std::make_shared<Int64Imm>(num); });
  }
  return std::make_shared<TD>(value_list);
}

template <typename TD>
ValuePtr ConvertTensorToSequenceFloat(const py::object &obj) {
  auto float_tensor = ConvertTensorValue(obj);
  if (float_tensor == nullptr) {
    MS_LOG(INFO) << "Can not convert python object with type [" << ReportPyBindType(obj) << "] to Tensor.";
    return nullptr;
  }

  auto data_type = float_tensor->data_type();
  if (data_type != kNumberTypeFloat64) {
    MS_LOG(ERROR) << "Can not convert Tensor with type " << TypeIdToString(data_type) << "to Float64 Sequence.";
    return nullptr;
  }

  auto shape = float_tensor->shape();
  if (shape.size() > 1) {
    MS_LOG(ERROR) << "Only support converting 1-D Tensor or scalar Tensor to sequence. But got the shape of Tensor: "
                  << shape;
    return nullptr;
  }

  // The data_c is a raw pointer and will become invalid when cpu_tensor is destructed.
  auto cpu_tensor = float_tensor->cpu();
  auto data = static_cast<double *>(cpu_tensor->data_c());
  auto size = float_tensor->DataSize();
  std::vector<ValuePtr> value_list(size);
  for (size_t i = 0; i < size; ++i) {
    (void)value_list.emplace_back(ConvertPythonFloatToScalarValue(data[i]));
  }

  return std::make_shared<TD>(value_list);
}

template <typename TD>
ValuePtr ConvertTensorToSequenceAny(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    MS_LOG(INFO) << "Can not convert python object with type [" << ReportPyBindType(obj) << "] to Tensor.";
    return nullptr;
  }

  auto shape = tensor->shape();
  if (shape.size() > 1) {
    MS_LOG(ERROR) << "Only support converting 1-D Tensor or scalar Tensor to sequence. But got the shape of Tensor: "
                  << shape;
    return nullptr;
  }

  auto data_type = tensor->data_type();
  auto size = tensor->DataSize();
  std::vector<ValuePtr> value_list(size);
  // The data_c is a raw pointer and will become invalid when cpu_tensor is destructed.
  auto cpu_tensor = tensor->cpu();
  if (data_type == kNumberTypeInt64) {
    auto data = static_cast<int64_t *>(cpu_tensor->data_c());
    for (size_t i = 0; i < size; ++i) {
      (void)value_list.emplace_back(std::make_shared<Int64Imm>(data[i]));
    }
  } else if (data_type == kNumberTypeFloat64) {
    auto data = static_cast<double *>(cpu_tensor->data_c());
    for (size_t i = 0; i < size; ++i) {
      (void)value_list.emplace_back(ConvertPythonFloatToScalarValue(data[i]));
    }
  } else if (data_type == kNumberTypeBool) {
    auto data = static_cast<bool *>(cpu_tensor->data_c());
    for (size_t i = 0; i < size; ++i) {
      (void)value_list.emplace_back(std::make_shared<BoolImm>(data[i]));
    }
  } else {
    MS_LOG(ERROR) << "Can not convert Tensor with type " << TypeIdToString(data_type) << " to sequence.";
    return nullptr;
  }

  return std::make_shared<TD>(value_list);
}

ValuePtr ConvertTensorToInt(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    return nullptr;
  }
  if (tensor->DataSize() != 1) {
    MS_LOG(ERROR) << "Can only convert tensor with one element to int, but got " << tensor->ToString();
    return nullptr;
  }
  switch (tensor->data_type()) {
    case kNumberTypeInt64:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<int64_t>(tensor));
    case kNumberTypeInt32:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<int32_t>(tensor));
    case kNumberTypeInt16:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<int16_t>(tensor));
    case kNumberTypeInt8:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<int8_t>(tensor));
    case kNumberTypeUInt64:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<uint64_t>(tensor));
    case kNumberTypeUInt32:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<uint32_t>(tensor));
    case kNumberTypeUInt16:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<uint16_t>(tensor));
    case kNumberTypeUInt8:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<uint8_t>(tensor));
    default:
      MS_EXCEPTION(TypeError) << "Can not convert " << tensor->ToString() << " to Int.";
  }
}

ValuePtr ConvertTensorAndInt(const py::object &obj) {
  auto value_opt = ConvertGeneralizedIntToBasicInt(obj.ptr());
  if (value_opt) {
    return std::make_shared<Int64Imm>(*value_opt);
  }
  return nullptr;
}

ValuePtr ConvertTensorToFloat(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    return nullptr;
  }
  if (tensor->DataSize() != 1) {
    MS_LOG(ERROR) << "Can only convert tensor with one element to float, but got " << tensor->ToString();
    return nullptr;
  }
  if (tensor->data_type() != kNumberTypeFloat64) {
    MS_LOG(ERROR) << "Can not convert " << tensor->ToString() << " to float";
    return nullptr;
  }
  return ConvertPythonFloatToScalarValue(tensor::GetTensorData<double>(tensor));
}

ValuePtr ConvertTensorToBool(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    return nullptr;
  }
  if (tensor->data_type() != kNumberTypeBool) {
    MS_LOG(ERROR) << "Can not convert " << tensor->ToString() << " to bool";
    return nullptr;
  }
  return std::make_shared<BoolImm>(tensor::GetTensorData<bool>(tensor));
}

ValuePtr ConvertTensorToNumber(const py::object &obj) {
  auto tensor = ConvertTensorValue(obj);
  if (tensor == nullptr) {
    return nullptr;
  }
  if (tensor->DataSize() != 1) {
    MS_EXCEPTION(ValueError) << "Can only convert tensor with one element to number, but got " << tensor->ToString();
  }

  switch (tensor->data_type()) {
    case kNumberTypeBool:
      return std::make_shared<BoolImm>(tensor::GetTensorData<bool>(tensor));
    case kNumberTypeInt64:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<int64_t>(tensor));
    case kNumberTypeInt32:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<int32_t>(tensor));
    case kNumberTypeInt16:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<int16_t>(tensor));
    case kNumberTypeInt8:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<int8_t>(tensor));
    case kNumberTypeUInt64:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<uint64_t>(tensor));
    case kNumberTypeUInt32:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<uint32_t>(tensor));
    case kNumberTypeUInt16:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<uint16_t>(tensor));
    case kNumberTypeUInt8:
      return std::make_shared<Int64Imm>(tensor::GetTensorData<uint8_t>(tensor));
    case kNumberTypeFloat64:
      return ConvertPythonFloatToScalarValue(tensor::GetTensorData<double>(tensor));
    case kNumberTypeFloat32:
      return ConvertPythonFloatToScalarValue(tensor::GetTensorData<float>(tensor));
    case kNumberTypeFloat16:
      return ConvertPythonFloatToScalarValue(static_cast<float>(tensor::GetTensorData<float16>(tensor)));
    default:
      MS_EXCEPTION(TypeError) << "Can not convert " << tensor->ToString() << " to number";
  }
}

ValuePtr ConvertBoolOrIntToFloat(const py::object &obj) {
  if (PyLong_Check(obj.ptr()) || IsNumpyScalar(obj.ptr())) {
    return MakeFP32Imm(PyFloat_AsDouble(obj.ptr()));
  }
  if (IsScalarTensor<int64_t, false>(obj.ptr())) {
    auto value_opt = ConvertScalarTensor<double>(obj.ptr());
    return value_opt ? MakeFP32Imm(static_cast<double>(*value_opt)) : nullptr;
  }
  if (IsScalarTensor<bool, false>(obj.ptr())) {
    auto value_opt = ConvertScalarTensor<bool>(obj.ptr());
    return value_opt ? MakeFP32Imm(static_cast<double>(*value_opt)) : nullptr;
  }
  return nullptr;
}

// convert functions without type_cast
ConverterMap InitBasicConverters() {
  ConverterMap kConverters = {
    {static_cast<int32_t>(mindspore::ops::DT_BOOL), ConvertBool},
    {static_cast<int32_t>(mindspore::ops::DT_INT), ConvertInt},
    {static_cast<int32_t>(mindspore::ops::DT_FLOAT), ConvertFloat<true>},
    {static_cast<int32_t>(mindspore::ops::DT_NUMBER), ConvertNumber},
    {static_cast<int32_t>(mindspore::ops::DT_TENSOR), ConvertTensor},
    {static_cast<int32_t>(mindspore::ops::DT_STR), ConvertStr},
    {static_cast<int32_t>(mindspore::ops::DT_TYPE), ConvertDtype},
    {static_cast<int32_t>(mindspore::ops::DT_TUPLE_BOOL), ConvertSequence<py::tuple, ValueTuple, ConvertBool>},
    {static_cast<int32_t>(mindspore::ops::DT_TUPLE_INT), ConvertSequence<py::tuple, ValueTuple, ConvertTensorAndInt>},
    {static_cast<int32_t>(mindspore::ops::DT_TUPLE_FLOAT), ConvertSequence<py::tuple, ValueTuple, ConvertFloat<false>>},
    {static_cast<int32_t>(mindspore::ops::DT_TUPLE_NUMBER), ConvertSequence<py::tuple, ValueTuple, ConvertNumber>},
    {static_cast<int32_t>(mindspore::ops::DT_TUPLE_TENSOR), ConvertSequence<py::tuple, ValueTuple, ConvertTensor>},
    {static_cast<int32_t>(mindspore::ops::DT_TUPLE_STR), ConvertSequence<py::tuple, ValueTuple, ConvertStr>},
    {static_cast<int32_t>(mindspore::ops::DT_LIST_BOOL), ConvertSequence<py::list, ValueTuple, ConvertBool>},
    {static_cast<int32_t>(mindspore::ops::DT_LIST_INT), ConvertSequence<py::list, ValueTuple, ConvertTensorAndInt>},
    {static_cast<int32_t>(mindspore::ops::DT_LIST_FLOAT), ConvertSequence<py::list, ValueTuple, ConvertFloat<false>>},
    {static_cast<int32_t>(mindspore::ops::DT_LIST_NUMBER), ConvertSequence<py::list, ValueTuple, ConvertNumber>},
    {static_cast<int32_t>(mindspore::ops::DT_LIST_TENSOR), ConvertSequence<py::list, ValueTuple, ConvertTensor>},
    {static_cast<int32_t>(mindspore::ops::DT_LIST_STR), ConvertSequence<py::list, ValueTuple, ConvertStr>}};
  return kConverters;
}

// TypeCast1: convert single element to sequence
ConverterMap InitSingleToSequenceConverters() {
  ConverterMap kConverters = {{CombineTypesForTypeCast(mindspore::ops::DT_NUMBER, mindspore::ops::DT_TUPLE_INT),
                               ConvertSingleElementToSequence<ValueTuple, ConvertNumber>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_NUMBER, mindspore::ops::DT_LIST_INT),
                               ConvertSingleElementToSequence<ValueTuple, ConvertNumber>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_TUPLE_INT),
                               ConvertSingleElementToSequence<ValueTuple, ConvertInt>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_LIST_INT),
                               ConvertSingleElementToSequence<ValueTuple, ConvertInt>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_FLOAT, mindspore::ops::DT_TUPLE_INT),
                               ConvertSingleElementToSequence<ValueTuple, ConvertFloat<false>>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_FLOAT, mindspore::ops::DT_LIST_INT),
                               ConvertSingleElementToSequence<ValueTuple, ConvertFloat<false>>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_BOOL, mindspore::ops::DT_TUPLE_INT),
                               ConvertSingleElementToSequence<ValueTuple, ConvertBool>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_BOOL, mindspore::ops::DT_LIST_INT),
                               ConvertSingleElementToSequence<ValueTuple, ConvertBool>}};
  return kConverters;
}

// TypeCast2: convert sequence to sequence, such as py::tuple to ValueList
ConverterMap InitSequenceToSequenceConverters() {
  ConverterMap kConverters = {{CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_INT, mindspore::ops::DT_LIST_INT),
                               ConvertSequence<py::tuple, ValueTuple, ConvertTensorAndInt>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_FLOAT, mindspore::ops::DT_LIST_FLOAT),
                               ConvertSequence<py::tuple, ValueTuple, ConvertFloat<false>>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_BOOL, mindspore::ops::DT_LIST_BOOL),
                               ConvertSequence<py::tuple, ValueTuple, ConvertBool>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_TENSOR, mindspore::ops::DT_LIST_TENSOR),
                               ConvertSequence<py::tuple, ValueTuple, ConvertTensor>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_LIST_INT, mindspore::ops::DT_TUPLE_INT),
                               ConvertSequence<py::list, ValueTuple, ConvertTensorAndInt>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_LIST_FLOAT, mindspore::ops::DT_TUPLE_FLOAT),
                               ConvertSequence<py::list, ValueTuple, ConvertFloat<false>>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_LIST_BOOL, mindspore::ops::DT_TUPLE_BOOL),
                               ConvertSequence<py::list, ValueTuple, ConvertBool>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_LIST_TENSOR, mindspore::ops::DT_TUPLE_TENSOR),
                               ConvertSequence<py::list, ValueTuple, ConvertTensor>}};
  return kConverters;
}

// TypeCast3: convert single element to Tensor
ConverterMap InitSingleToTensorConverters() {
  ConverterMap kConverters = {
    {CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_TENSOR),
     ConvertSingleElementToTensor<py::int_, pyint>},
    {CombineTypesForTypeCast(mindspore::ops::DT_FLOAT, mindspore::ops::DT_TENSOR),
     ConvertSingleElementToTensor<py::float_, pyfloat>},
    {CombineTypesForTypeCast(mindspore::ops::DT_BOOL, mindspore::ops::DT_TENSOR),
     ConvertSingleElementToTensor<py::bool_, bool>},
    {CombineTypesForTypeCast(mindspore::ops::DT_NUMBER, mindspore::ops::DT_TENSOR), ConvertNumberToTensor}};
  return kConverters;
}

// TypeCast4: convert between sequence and tensor
ConverterMap InitSequenceAndTensorConverters() {
  ConverterMap kConverters = {{CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_INT, mindspore::ops::DT_TENSOR),
                               ConvertSequenceToTensor<py::tuple, py::int_, pyint>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_FLOAT, mindspore::ops::DT_TENSOR),
                               ConvertSequenceToTensor<py::tuple, py::float_, pyfloat>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_BOOL, mindspore::ops::DT_TENSOR),
                               ConvertSequenceBoolToTensor<py::tuple>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_LIST_INT, mindspore::ops::DT_TENSOR),
                               ConvertSequenceToTensor<py::list, py::int_, pyint>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_LIST_FLOAT, mindspore::ops::DT_TENSOR),
                               ConvertSequenceToTensor<py::list, py::float_, pyfloat>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_LIST_BOOL, mindspore::ops::DT_TENSOR),
                               ConvertSequenceBoolToTensor<py::list>},

                              {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TUPLE_INT),
                               ConvertTensorToSequenceInt<ValueTuple>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TUPLE_FLOAT),
                               ConvertTensorToSequenceFloat<ValueTuple>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TUPLE_BOOL),
                               ConvertTensorToSequence<ValueTuple, bool, BoolImm, kNumberTypeBool>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TUPLE_BOOL),
                               ConvertTensorToSequenceAny<ValueTuple>},

                              {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_LIST_INT),
                               ConvertTensorToSequenceInt<ValueTuple>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_LIST_FLOAT),
                               ConvertTensorToSequenceFloat<ValueTuple>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_LIST_BOOL),
                               ConvertTensorToSequence<ValueTuple, bool, BoolImm, kNumberTypeBool>},
                              {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_LIST_BOOL),
                               ConvertTensorToSequenceAny<ValueTuple>}};
  return kConverters;
}

// TypeCast5: convert tensor to single element
ConverterMap InitTensorToSingleConverters() {
  ConverterMap kConverters = {
    {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_INT), ConvertTensorToInt},
    {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_FLOAT), ConvertTensorToFloat},
    {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_BOOL), ConvertTensorToBool},
    {CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_NUMBER), ConvertTensorToNumber}};
  return kConverters;
}

// TypeCas6: convert int/bool to float
ConverterMap InitNumberToFloatConverters() {
  ConverterMap kConverters = {
    {CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_FLOAT), ConvertBoolOrIntToFloat},
    {CombineTypesForTypeCast(mindspore::ops::DT_BOOL, mindspore::ops::DT_FLOAT), ConvertBoolOrIntToFloat}};
  return kConverters;
}
}  // namespace

const ConverterMap &GetConverters() {
  static const ConverterMap kStaticConverters = []() {
    ConverterMap kConverters;
    kConverters.merge(InitBasicConverters());
    kConverters.merge(InitSingleToSequenceConverters());
    kConverters.merge(InitSequenceToSequenceConverters());
    kConverters.merge(InitSingleToTensorConverters());
    kConverters.merge(InitSequenceAndTensorConverters());
    kConverters.merge(InitTensorToSingleConverters());
    kConverters.merge(InitNumberToFloatConverters());
    return kConverters;
  }();
  return kStaticConverters;
}

void ReportGetConverterError(int32_t dtype) {
  if ((dtype >> kTypeShiftBits) == 0) {
    MS_LOG(EXCEPTION) << "Can not find converter for dtype[" << prim::OpDTypeToString(static_cast<ops::OP_DTYPE>(dtype))
                      << "].";
  } else {
    MS_LOG(EXCEPTION) << "Can not find converter for src_type["
                      << prim::OpDTypeToString(static_cast<ops::OP_DTYPE>(dtype >> kTypeShiftBits)) << "] and dst_type["
                      << prim::OpDTypeToString(static_cast<ops::OP_DTYPE>(dtype & kDstMask)) << "].";
  }
}

OpDefConvertFunc GetConverterByType(int32_t dtype) {
  const auto &kConverters = GetConverters();
  auto it = kConverters.find(dtype);
  if (it == kConverters.end()) {
    ReportGetConverterError(dtype);
  }
  return it->second;
}

bool ParseUtilsCheckInt(PyObject *obj, bool allow_scalar_tensor) {
  if (IsPythonInt(obj) || IsNumpyInt(obj)) {
    return true;
  }
  return allow_scalar_tensor ? IsScalarTensor<int64_t, false>(obj) : false;
}

std::optional<int64_t> ParseUtilsConvertInt(PyObject *obj) {
  if (IsPythonInt(obj) || IsNumpyInt(obj)) {
    return static_cast<int64_t>(PyLong_AsLongLong(obj));
  }
  return ConvertScalarTensor<int64_t>(obj);
}

bool ParseUtilsCheckFloat(PyObject *obj, bool allow_scalar_tensor) {
  if (PyFloat_Check(obj) || IsNumpyFloat(obj)) {
    return true;
  }
  return allow_scalar_tensor ? IsScalarTensor<double, false>(obj) : false;
}

std::optional<double> ParseUtilsConvertFloat(PyObject *obj) {
  if (PyFloat_Check(obj) || IsNumpyFloat(obj)) {
    return PyFloat_AsDouble(obj);
  }
  return ConvertScalarTensor<double>(obj);
}

bool ParseUtilsCheckBool(PyObject *obj, bool allow_scalar_tensor) {
  if (PyBool_Check(obj) || IsNumpyBool(obj)) {
    return true;
  }
  return allow_scalar_tensor ? IsScalarTensor<bool, false>(obj) : false;
}

std::optional<bool> ParseUtilsConvertBool(PyObject *obj) {
  if (PyBool_Check(obj) || IsNumpyBool(obj)) {
    return (PyObject_IsTrue(obj) == 1);
  }
  return ConvertScalarTensor<bool>(obj);
}

bool ParseUtilsCheckScalar(PyObject *obj, bool allow_scalar_tensor) {
  if (PyFloat_Check(obj) || PyLong_Check(obj) || IsNumpyScalar(obj)) {
    return true;
  }
  return allow_scalar_tensor ? (IsScalarTensor<int64_t, false>(obj) || IsScalarTensor<double, false>(obj) ||
                                IsScalarTensor<bool, false>(obj))
                             : false;
}

bool IsGeneralizedInt(PyObject *obj) {
  return IsPythonInt(obj) || IsNumpyInt(obj) || IsNumpyScalarIntArray(obj) || IsScalarTensor<int64_t, true>(obj);
}

// convert int, numpy int, numpy scalar int array or scalar int tensor to int
std::optional<int64_t> ConvertGeneralizedIntToBasicInt(PyObject *obj) {
  if (MS_UNLIKELY(IsNumpyScalarIntArray(obj))) {
    return ConvertNumpyScalarIntArray(obj);
  }
  return ParseUtilsConvertInt(obj);
}
}  // namespace py_parse
}  // namespace mindspore
