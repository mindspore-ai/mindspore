/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#include "include/pynative/forward/pyboost/converter.h"
#include <vector>
#include <utility>
#include <string>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "include/utils/convert_utils_py.h"
#include "include/pynative/forward/pyboost/fallback.h"
#include "include/pynative/utils/pynative_utils.h"
#include "include/utils/tensor_py.h"
#include "include/utils/tensor_utils.h"
#include "include/utils/python_attr.h"
#include "include/utils/pynative/py_parse.h"
#include "include/utils/operator/primitive_utils.h"
#include "mindspore/core/include/utils/value_utils.h"
#include "include/utils/pynative/storage_py.h"

namespace mindspore {
namespace pynative {

using mindspore::pynative::PyNativeAlgo::PyParser;

bool IsPyObjNone(PyObject *obj) { return obj == NULL || obj == Py_None; }

Py_ssize_t GetListOrTupleSize(PyObject *args) { return PyList_Check(args) ? PyList_Size(args) : PyTuple_Size(args); }

#define RAISE_PARSE_ERROR(out_error_msg, raise_error, msg, func_name) \
  if (out_error_msg || raise_error) {                                 \
    std::string error_msg = msg;                                      \
    if (raise_error) {                                                \
      MS_EXCEPTION(TypeError) << func_name << "()" << error_msg;      \
    } else if (out_error_msg) {                                       \
      out_error_msg->append(error_msg);                               \
    }                                                                 \
  }
using OpDefConvertFunc = std::function<ValuePtr(PyObject *obj)>;
using OpIntVectorConvertFunc = std::function<std::optional<std::vector<int64_t>>(PyObject *obj)>;
using OpIntConvertFunc = std::function<std::optional<int64_t>(PyObject *obj)>;
namespace {
using OP_DTYPE = mindspore::ops::OP_DTYPE;

template <typename T, typename U>
std::shared_ptr<U> PyCast(PyObject *obj);

template <>
std::shared_ptr<BoolImm> PyCast<bool, BoolImm>(PyObject *obj) {
  bool value = (PyObject_IsTrue(obj) == 1);
  return std::make_shared<BoolImm>(value);
}

template <>
std::shared_ptr<StringImm> PyCast<string, StringImm>(PyObject *obj) {
  const char *obj_str = PyUnicode_AsUTF8(obj);
  return std::make_shared<StringImm>(std::string(obj_str));
}

inline static std::shared_ptr<FP32Imm> MakeFP32Imm(double value) {
  auto value_float32 = static_cast<float>(value);
  auto ret = std::make_shared<FP32Imm>(value_float32);
  ret->set_prim_value(value);
  return ret;
}

BoolImmPtr ConvertBool(PyObject *obj) {
  if (!PyBool_Check(obj)) {
    // The mutable _Bool class inherits from int, because base class 'bool' is a marked final.
    if (PyLong_Check(obj) && PyObject_HasAttrString(obj, "__ms_mutable_bool__")) {
      auto obj_int64 = PyLong_AsLong(obj);
      bool obj_bool = obj_int64 != 0;
      return std::make_shared<BoolImm>(obj_bool);
    }
    return nullptr;
  }
  return PyCast<bool, BoolImm>(obj);
}

Int64ImmPtr ConvertInt(PyObject *obj) {
  if (py_parse::ParseUtilsCheckInt(obj)) {
    return std::make_shared<Int64Imm>(*py_parse::ParseUtilsConvertInt(obj));
  }
  return nullptr;
}

FP32ImmPtr ConvertFloat(PyObject *obj) {
  if (py_parse::ParseUtilsCheckFloat(obj)) {
    return MakeFP32Imm(*py_parse::ParseUtilsConvertFloat(obj));
  }
  return nullptr;
}

ScalarPtr ConvertNumber(PyObject *obj) {
  if (py_parse::ParseUtilsCheckFloat(obj)) {
    return MakeFP32Imm(*py_parse::ParseUtilsConvertFloat(obj));
  }
  if (py_parse::ParseUtilsCheckBool(obj)) {
    return std::make_shared<BoolImm>(*py_parse::ParseUtilsConvertBool(obj));
  }
  if (py_parse::ParseUtilsCheckInt(obj)) {
    return std::make_shared<Int64Imm>(*py_parse::ParseUtilsConvertInt(obj));
  }
  return nullptr;
}

bool IsPyStr(PyObject *obj) { return PyUnicode_Check(obj); }

StringImmPtr ConvertStr(PyObject *obj) {
  if (!IsPyStr(obj)) {
    return nullptr;
  }
  return PyCast<string, StringImm>(obj);
}

template <typename T>
ValueTuplePtr ConvertIntSequence(PyObject *obj) {
  if (!T::TypeCheck(obj)) {
    return nullptr;
  }
  Py_ssize_t size = T::GetSize(obj);
  std::vector<ValuePtr> convert(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    // borrow reference
    PyObject *item = T::GetItem(obj, i);
    auto value = py_parse::ConvertGeneralizedIntToBasicInt(item);
    if (value) {
      convert[i] = std::make_shared<Int64Imm>(*value);
      continue;
    }
    return nullptr;
  }
  return std::make_shared<ValueTuple>(std::move(convert));
}

template <typename T>
ValueTuplePtr ConvertBoolSequence(PyObject *obj) {
  if (!T::TypeCheck(obj)) {
    return nullptr;
  }
  Py_ssize_t size = T::GetSize(obj);
  std::vector<ValuePtr> convert(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    // borrow reference
    PyObject *item = T::GetItem(obj, i);
    if (!PyBool_Check(item)) {
      return nullptr;
    }
    auto out = PyCast<bool, BoolImm>(item);
    if (out == nullptr) {
      return nullptr;
    }
    convert[i] = out;
  }
  return std::make_shared<ValueTuple>(std::move(convert));
}

template <typename T>
ValueTuplePtr ConvertFloatSequence(PyObject *obj) {
  if (!T::TypeCheck(obj)) {
    return nullptr;
  }
  Py_ssize_t size = T::GetSize(obj);
  std::vector<ValuePtr> convert(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    // borrow reference
    PyObject *item = T::GetItem(obj, i);
    if (py_parse::ParseUtilsCheckFloat(item, false)) {
      convert[i] = MakeFP32Imm(*py_parse::ParseUtilsConvertFloat(item));
      continue;
    }
    return nullptr;
  }
  return std::make_shared<ValueTuple>(std::move(convert));
}

void EnablePipelineForTupleTensor(const ValueTuplePtr &tuple) {
  const auto &values = tuple->value();
  for (auto &value : values) {
    if (value->isa<tensor::Tensor>()) {
      auto t = value->cast<TensorPtr>();
      t->set_need_pipeline_sync(true);
    }
  }
}

std::optional<std::vector<int64_t>> ConvertIntToIntVector(PyObject *obj) {
  if (py_parse::ParseUtilsCheckInt(obj)) {
    return std::vector<int64_t>({*py_parse::ParseUtilsConvertInt(obj)});
  }
  return std::nullopt;
}

template <typename T>
std::optional<std::vector<int64_t>> ConvertIntVector(PyObject *obj) {
  if (!T::TypeCheck(obj)) {
    return std::nullopt;
  }
  Py_ssize_t size = T::GetSize(obj);
  std::vector<int64_t> convert(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    // borrow reference
    PyObject *item = T::GetItem(obj, i);
    auto value_opt = py_parse::ConvertGeneralizedIntToBasicInt(item);
    if (value_opt) {
      convert[i] = *value_opt;
      continue;
    }
    return std::nullopt;
  }
  return convert;
}

std::string GenerateSignatureStr(const std::string &function_name, const FunctionSignaturePtr &signature,
                                 bool is_method) {
  return is_method ? "Tensor." + function_name + signature->ToString() : function_name + signature->ToString();
}

std::vector<int> CollectValidSignatureIndices(const std::vector<FunctionSignaturePtr> &signatures,
                                              Py_ssize_t total_arg_size) {
  std::vector<int> valid_indices;
  for (const auto &signature : signatures) {
    if (static_cast<size_t>(total_arg_size) >= signature->min_args_ &&
        static_cast<size_t>(total_arg_size) <= signature->max_args_) {
      valid_indices.emplace_back(signature->index_);
    }
  }
  return valid_indices;
}

std::vector<std::string> GenerateAllSignatureStrings(const std::vector<FunctionSignaturePtr> &signatures,
                                                     const std::string &function_name, bool is_method) {
  std::vector<std::string> sig_strings;
  std::unordered_set<std::string> sig_set;
  for (const auto &signature : signatures) {
    auto sig_str = GenerateSignatureStr(function_name, signature, is_method);
    bool inserted = sig_set.insert(sig_str).second;
    if (inserted) {
      sig_strings.emplace_back("\"" + sig_str + "\"");
    }
  }
  return sig_strings;
}

bool IsFunctionalRegInMap(const std::string &function_name, bool is_method) {
  const auto &signature_map = prim::GetFunctionalSignatureMap(is_method);
  return signature_map.find(function_name) != signature_map.end();
}

bool PyObjectHasFallbackAttr(PyObject *obj) {
  // Only Tensor subclass need to check fallback attr.
  if (tensor::IsPyObjectPythonTensorStrict(obj)) {
    return false;
  }

  if (!tensor::IsPyObjectTensorPy(obj)) {
    return false;
  }
  auto attr = FastGetPyObjectAttr(obj, GetFallbackStr().c_str());
  return attr.ptr() != nullptr;
}
}  // namespace
namespace py = pybind11;

Converter::Converter(ops::OpDef *op_def)
    : op_def_(op_def), source_type_(std::vector<ops::OP_DTYPE>(op_def->args_.size())) {}

int64_t Converter::ToBasicInt(PyObject *python_args, size_t i) {
  // python_args should be list
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  if (py_parse::ParseUtilsCheckInt(obj)) {
    return PyLong_AsLongLong(obj);
  }
  const auto &op_arg = op_def_->args_[i];
  return ConvertIntByCastDtype(python_args, op_arg, i);
}

std::optional<int64_t> Converter::ToBasicIntOptional(PyObject *python_args, size_t i) {
  // python_args should be list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToBasicInt(python_args, i));
}

template <typename T>
std::vector<int64_t> Converter::ToBasicIntVector(PyObject *python_args, size_t i) {
  // python_args should be list
  PyObject *obj = PyList_GetItem(python_args, i);
  auto convert = ConvertIntVector<T>(obj);
  if (convert.has_value()) {
    return convert.value();
  }
  const auto &op_arg = op_def_->args_[i];
  return ConvertIntVectorByCastDtype(python_args, op_arg, i);
}

template <typename T>
std::optional<std::vector<int64_t>> Converter::ToBasicIntVectorOptional(PyObject *python_args, size_t i) {
  // python_args should be list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToBasicIntVector<T>(python_args, i));
}

void Converter::Parse(PyObject *python_args) {
  // Converter is a static object
  // reset the flag every time.
  has_fallback_ = false;
  Py_ssize_t args_size = (python_args && python_args != Py_None) ? GetListOrTupleSize(python_args) : 0;
  if (op_def_->args_.size() != static_cast<size_t>(args_size)) {
    MS_LOG(EXCEPTION) << "For operator " << op_def_->name_ << ", it requires " << op_def_->args_.size()
                      << "parameters, bug got " << static_cast<size_t>(args_size) << "parameters!";
  }
}

ValuePtr Converter::ToTensor(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  const auto &op_arg = op_def_->args_[i];
  PyObject *obj = PyList_GetItem(python_args, i);

  if (!has_fallback_) {
    if (fallback_enabled() && PyObjectHasFallbackAttr(obj)) {
      has_fallback_ = true;
    }
  }

  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto tensor = py_parse::ConvertPyObjectTensor(obj);
  if (tensor != nullptr) {
    if (tensor->isa<tensor::Tensor>()) {
      tensor->cast<tensor::TensorPtr>()->set_need_pipeline_sync(true);
    }
    return tensor;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg, i);
    if (convert != nullptr && convert->isa<tensor::Tensor>()) {
      auto converted_tensor = convert->cast<tensor::TensorPtr>();
      converted_tensor->set_source_type(source_type_[i]);
      return converted_tensor;
    }
  }

  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, i);
  return nullptr;
}

std::optional<ValuePtr> Converter::ToTensorOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToTensor(python_args, i));
}

template <typename T>
ValueTuplePtr Converter::ToTensorList(PyObject *python_args, size_t i) {
  // convert to py::object and then used in data_converter, because data_converter haven't been refactored to PyObject
  // AllFinite the python_args is a tuple
  py::object obj = py::reinterpret_borrow<py::object>(PyList_Check(python_args) ? PyList_GetItem(python_args, i)
                                                                                : PyTuple_GetItem(python_args, i));
  const auto &op_arg = op_def_->args_[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto val_seq = py_parse::ConvertSequence<typename T::pybind_type, ValueTuple, py_parse::ConvertTensor>(obj);
  if (val_seq != nullptr && val_seq->template isa<ValueTuple>()) {
    auto value_tuple = val_seq->template cast<ValueTuplePtr>();
    if (!has_fallback_ && fallback_enabled()) {
      MS_EXCEPTION_IF_NULL(value_tuple);
      const auto &values = value_tuple->value();
      has_fallback_ = std::any_of(values.begin(), values.end(), [](const ValuePtr &value) {
        return value != nullptr && value->isa<Tensor>() && value->cast<TensorPtr>()->has_fallback();
      });
    }
    EnablePipelineForTupleTensor(value_tuple);
    return value_tuple;
  }
  return ConvertValueTupleByCastDtype(python_args, op_arg, i);
}

template <typename T>
std::optional<ValueTuplePtr> Converter::ToTensorListOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToTensorList<T>(python_args, i));
}

Int64ImmPtr Converter::ToInt(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  const auto &op_arg = op_def_->args_[i];
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertInt(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<Int64Imm>()) {
      return convert_value->cast<Int64ImmPtr>();
    }
  }
  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, i);
  return nullptr;
}

std::optional<Int64ImmPtr> Converter::ToIntOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToInt(python_args, i));
}

template <typename T>
ValueTuplePtr Converter::ToIntList(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  const auto &op_arg = op_def_->args_[i];
  PyObject *obj = PyList_GetItem(python_args, i);
  ValueTuplePtr convert = ConvertIntSequence<T>(obj);
  if (convert != nullptr) {
    return convert;
  }
  return ConvertValueTupleByCastDtype(python_args, op_arg, i);
}

template <typename T>
std::optional<ValueTuplePtr> Converter::ToIntListOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToIntList<T>(python_args, i));
}

BoolImmPtr Converter::ToBool(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  const auto &op_arg = op_def_->args_[i];
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertBool(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<BoolImm>()) {
      return convert_value->cast<BoolImmPtr>();
    }
  }
  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, i);
  return nullptr;
}

std::optional<BoolImmPtr> Converter::ToBoolOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToBool(python_args, i));
}

template <typename T>
ValueTuplePtr Converter::ToBoolList(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  const auto &op_arg = op_def_->args_[i];
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  ValueTuplePtr convert = ConvertBoolSequence<T>(obj);
  if (convert != nullptr) {
    return convert;
  }
  return ConvertValueTupleByCastDtype(python_args, op_arg, i);
}

template <typename T>
std::optional<ValueTuplePtr> Converter::ToBoolListOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToBoolList<T>(python_args, i));
}

FP32ImmPtr Converter::ToFloat(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  const auto &op_arg = op_def_->args_[i];
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertFloat(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<FP32Imm>()) {
      return convert_value->cast<FP32ImmPtr>();
    }
  }
  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, i);
  return nullptr;
}

std::optional<FP32ImmPtr> Converter::ToFloatOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToFloat(python_args, i));
}

template <typename T>
ValueTuplePtr Converter::ToFloatList(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  const auto &op_arg = op_def_->args_[i];
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  ValueTuplePtr convert = ConvertFloatSequence<T>(obj);
  if (convert != nullptr) {
    return convert;
  }
  return ConvertValueTupleByCastDtype(python_args, op_arg, i);
}

template <typename T>
std::optional<ValueTuplePtr> Converter::ToFloatListOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToFloatList<T>(python_args, i));
}

ScalarPtr Converter::ToScalar(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  const auto &op_arg = op_def_->args_[i];
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertNumber(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<Scalar>()) {
      return convert_value->cast<ScalarPtr>();
    }
  }
  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, i);
  return nullptr;
}

std::optional<ScalarPtr> Converter::ToScalarOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToScalar(python_args, i));
}

StringImmPtr Converter::ToString(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  const auto &op_arg = op_def_->args_[i];
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertStr(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<StringImm>()) {
      return convert_value->cast<StringImmPtr>();
    }
  }
  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, i);
  return nullptr;
}

std::optional<StringImmPtr> Converter::ToStringOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToString(python_args, i));
}

Int64ImmPtr Converter::ToDtype(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertInt(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    TypePtr type = py::cast<mindspore::TypePtr>(obj);
    return std::make_shared<Int64Imm>(static_cast<int>(type->type_id()));
  }
  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, i);
  return nullptr;
}

std::optional<Int64ImmPtr> Converter::ToDtypeOptional(PyObject *python_args, size_t i) {
  // type of python_args is py::list
  PyObject *obj = PyList_GetItem(python_args, i);
  if (obj == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToDtype(python_args, i));
}

ValuePtr Converter::ConvertByCastDtype(PyObject *input, const ops::OpInputArg &op_arg, size_t index) {
  // convert to py::object and then used in data_converter, because data_converter haven't been refactored to PyObject
  py::object py_input = py::reinterpret_borrow<py::object>(input);
  for (auto &cast_dtype : op_arg.cast_dtype_) {
    auto convert_func = py_parse::GetConverterByType(py_parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
    if (convert_func == nullptr) {
      MS_LOG(EXCEPTION) << "Can't find convert function for src_dtype[" << cast_dtype << "] and dst_type"
                        << op_arg.arg_dtype_ << "].";
    }
    auto value = convert_func(py_input);
    if (value != nullptr) {
      source_type_[index] = cast_dtype;
      return value;
    }
  }
  return nullptr;
}

std::optional<std::vector<int64_t>> ConvertTensorToIntVector(PyObject *obj) {
  auto tensor = tensor::ConvertPyObjectToTensor(obj);
  if (tensor == nullptr) {
    PyObject *obj_type = PyObject_Str(PyObject_Type(obj));
    const char *type_str = PyUnicode_AsUTF8(obj_type);
    MS_LOG(INFO) << "Can not convert python object with type [" << type_str << "] to Tensor.";
    Py_DECREF(obj_type);
    return std::nullopt;
  }

  auto shape = tensor->shape();
  if (shape.size() > 1) {
    MS_LOG(ERROR) << "Only support converting 1-D Tensor or scalar Tensor to sequence. But got the shape of Tensor: "
                  << shape;
    return std::nullopt;
  }

  auto data_type = tensor->data_type();
  if (data_type != kNumberTypeInt64 && data_type != kNumberTypeInt32) {
    MS_LOG(ERROR) << "Can not convert Tensor with type " << TypeIdToString(data_type) << "to Int Sequence.";
    return std::nullopt;
  }
  auto size = tensor->DataSize();
  MS_EXCEPTION_IF_NULL(tensor->device_address());
  tensor = tensor->device_address()->GetDeviceType() == device::DeviceType::kCPU ? tensor : tensor->cpu();
  if (data_type == kNumberTypeInt64) {
    auto data = static_cast<int64_t *>(tensor->data_c());
    return std::vector<int64_t>(data, data + size);
  } else {
    auto data = static_cast<int32_t *>(tensor->data_c());
    return std::vector<int64_t>(data, data + size);
  }
}

static const std::unordered_map<int32_t, OpIntVectorConvertFunc> kIntVectorConverters = {
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_TUPLE_INT), ConvertIntToIntVector},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_INT, mindspore::ops::DT_LIST_INT), ConvertIntToIntVector},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_TUPLE_INT),
   ConvertIntVector<CPythonTuple>},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_LIST_INT),
   ConvertIntVector<CPythonList>},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_LIST_INT, mindspore::ops::DT_TUPLE_INT),
   ConvertIntVector<CPythonList>},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_TUPLE_INT, mindspore::ops::DT_LIST_INT),
   ConvertIntVector<CPythonTuple>},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TUPLE_INT),
   ConvertTensorToIntVector},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_LIST_INT),
   ConvertTensorToIntVector}};

OpIntVectorConvertFunc GetIntVectorConverterByBaseType(int32_t dtype) {
  auto it = kIntVectorConverters.find(dtype);
  if (it == kIntVectorConverters.end()) {
    return nullptr;
  }
  return it->second;
}

std::vector<int64_t> Converter::ConvertIntVectorByCastDtype(PyObject *python_args, const ops::OpInputArg &op_arg,
                                                            size_t index) {
  // python_args should be list
  PyObject *input = PyList_GetItem(python_args, index);
  if (!op_arg.cast_dtype_.empty()) {
    for (auto &cast_dtype : op_arg.cast_dtype_) {
      OpIntVectorConvertFunc convert_func =
        GetIntVectorConverterByBaseType(py_parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
      if (convert_func != nullptr) {
        auto value = convert_func(input);
        if (value.has_value()) {
          source_type_[index] = cast_dtype;
          return value.value();
        }
      } else {
        MS_LOG(EXCEPTION) << "Can't find convert function for src_dtype[" << cast_dtype << "] and dst_type"
                          << op_arg.arg_dtype_ << "].";
      }
    }
  }
  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, index);
  return {};
}

std::optional<int64_t> ConvertTensorToInt64(PyObject *obj) {
  auto tensor = py_parse::ConvertPyObjectTensorValue(obj);
  if (tensor == nullptr) {
    return std::nullopt;
  }
  if (tensor->DataSize() != 1) {
    MS_LOG(ERROR) << "Can only convert tensor with one element to int, but got " << tensor->ToString();
    return std::nullopt;
  }
  if (tensor->data_type() == kNumberTypeInt64) {
    return tensor::GetTensorData<int64_t>(tensor);
  } else if (tensor->data_type() == kNumberTypeInt32) {
    return static_cast<int64_t>(tensor::GetTensorData<int32_t>(tensor));
  } else if (tensor->data_type() == kNumberTypeInt16) {
    return static_cast<int64_t>(tensor::GetTensorData<int16_t>(tensor));
  } else if (tensor->data_type() == kNumberTypeInt8) {
    return static_cast<int64_t>(tensor::GetTensorData<int8_t>(tensor));
  } else if (tensor->data_type() == kNumberTypeUInt8) {
    return static_cast<int64_t>(tensor::GetTensorData<uint8_t>(tensor));
  } else {
    MS_LOG(ERROR) << "Can not convert " << tensor->ToString() << " to int.";
    return std::nullopt;
  }
}

std::optional<int64_t> ConvertToInt64(PyObject *obj) {
  if (py_parse::ParseUtilsCheckInt(obj)) {
    return py_parse::ParseUtilsConvertInt(obj);
  }
  return std::nullopt;
}

static const std::unordered_map<int32_t, OpIntConvertFunc> kIntConverters = {
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_INT), ConvertToInt64},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_INT), ConvertTensorToInt64}};

OpIntConvertFunc GetInConverterByBaseType(int32_t dtype) {
  auto it = kIntConverters.find(dtype);
  if (it == kIntConverters.end()) {
    return nullptr;
  }
  return it->second;
}

int64_t Converter::ConvertIntByCastDtype(PyObject *python_args, const ops::OpInputArg &op_arg, size_t index) {
  // python_args should be list
  PyObject *input = PyList_GetItem(python_args, index);
  if (!op_arg.cast_dtype_.empty()) {
    for (auto &cast_dtype : op_arg.cast_dtype_) {
      OpIntConvertFunc convert_func =
        GetInConverterByBaseType(py_parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
      if (convert_func != nullptr) {
        auto value = convert_func(input);
        if (value.has_value()) {
          source_type_[index] = cast_dtype;
          return value.value();
        }
      } else {
        MS_LOG(EXCEPTION) << "Can't find convert function for src_dtype[" << cast_dtype << "] and dst_type"
                          << op_arg.arg_dtype_ << "].";
      }
    }
  }
  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, index);
  return 0;
}

ValueTuplePtr Converter::ConvertValueTupleByCastDtype(PyObject *python_args, const ops::OpInputArg &op_arg,
                                                      size_t index) {
  // python_args should be list
  PyObject *input =
    PyList_Check(python_args) ? PyList_GetItem(python_args, index) : PyTuple_GetItem(python_args, index);
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(input, op_arg, index);
    if (convert_value != nullptr && convert_value->isa<ValueTuple>()) {
      EnablePipelineForTupleTensor(convert_value->cast<ValueTuplePtr>());
      return convert_value->cast<ValueTuplePtr>();
    }
  }
  PyParser::PrintTypeCastErrorForPyObject(op_def_, python_args, index);
  return nullptr;
}

PythonArgParser::PythonArgParser(std::vector<std::string> fmts, const std::string &function_name)
    : function_name_(function_name), max_args_(0) {
  int index = 0;
  for (auto &stmt : fmts) {
    signatures_.emplace_back(std::make_shared<FunctionSignature>(stmt, index, function_name_));
    index++;
  }
  for (auto &signature : signatures_) {
    if (signature->max_args_ > max_args_) {
      max_args_ = signature->max_args_;
    }
  }
}

const std::vector<std::string> PythonArgParser::GetParseTypeListString(PyObject *args, PyObject *kwargs) {
  std::vector<std::string> type_list;
  Py_ssize_t args_size = (args && args != Py_None) ? GetListOrTupleSize(args) : 0;

  for (Py_ssize_t i = 0; i < args_size; ++i) {
    PyObject *py_arg = PyTuple_GetItem(args, i);
    (void)type_list.emplace_back(PyParser::BuildPyObjectInputTypeString(py_arg));
  }
  if (!kwargs || kwargs == Py_None) {
    return type_list;
  }
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(kwargs, &pos, &key, &value)) {
    const char *name_str = PyUnicode_AsUTF8(key);
    std::string kwarg_info(name_str);
    kwarg_info += "=";
    kwarg_info += PyParser::BuildPyObjectInputTypeString(value);
    (void)type_list.emplace_back(kwarg_info);
  }
  return type_list;
}

using CheckFunc = bool (*)(PyObject *);
template <typename T>
bool CheckPySequenceType(PyObject *obj, int &idx, CheckFunc check_func, bool fullcheck = false) {
  if (!T::TypeCheck(obj)) {
    return false;
  }
  Py_ssize_t size = T::GetSize(obj);
  if (size == 0) {
    return true;
  }
  size = fullcheck ? size : 1;
  for (Py_ssize_t i = 0; i < size; ++i) {
    // borrow reference
    PyObject *item = T::GetItem(obj, i);
    if (!check_func(item)) {
      idx = i;
      return false;
    }
  }
  return true;
}

bool IsTensor(PyObject *obj) {
  if (mindspore::tensor::IsPyObjectTensorPy(obj)) {
    return true;
  }
  return false;
}
bool IsStorage(PyObject *obj) {
  if (IsPyObjectStoragePy(obj)) {
    return true;
  }
  return false;
}

bool CheckArgsAsIntlist(PyObject *obj, bool as_intlist) {
  int idx;
  return as_intlist &&
         (py_parse::IsGeneralizedInt(obj) || CheckPySequenceType<CPythonTuple>(obj, idx, py_parse::IsGeneralizedInt) ||
          CheckPySequenceType<CPythonList>(obj, idx, py_parse::IsGeneralizedInt));
}

std::string GetTypeErrorMsg(bool is_kwd, int error_idx, PyObject *obj, const FunctionParameter &param, size_t arg_pos) {
  std::string error_msg;
  PyObject *type_obj = PyObject_Type(obj);
  PyObject *name_attr = PyObject_GetAttrString(type_obj, "__name__");
  Py_DECREF(type_obj);
  const char *name_str = PyUnicode_AsUTF8(name_attr);
  if (is_kwd) {
    error_msg = ": argument '" + param.name_ + "' must be " + prim::OpDTypeToString(param.type_) + " but got " +
                std::string(name_str) + ".";
  } else {
    error_msg = ": argument '" + param.name_ + "' (position " + std::to_string(arg_pos) + ")" + " must be " +
                prim::OpDTypeToString(param.type_);
    error_msg += (error_idx >= 0)
                   ? " but found type of " + std::string(name_str) + " at pos " + std::to_string(error_idx) + "."
                   : ", not " + std::string(name_str) + ".";
  }
  Py_DECREF(name_attr);
  return error_msg;
}

bool FunctionSignature::CheckParamValid(PyObject *obj, const FunctionParameter &param, bool raise_error,
                                        std::string *out_error_msg, ConvertPair &convert_type, int &error_idx) {
  if (param.is_any_) {
    // only when py_method dispatch to skip type check
    return true;
  }
  if (obj == Py_None) {
    if (!param.allow_none_) {
      RAISE_PARSE_ERROR(out_error_msg, raise_error, ": missing 1 required positional argument: " + param.name_ + ".",
                        name_);
      return false;
    }
    return true;
  } else if (param.Check(obj, convert_type, error_idx)) {
    return true;
  }
  return false;
}

bool FunctionSignature::Parse(PyObject *args, PyObject *kwargs, ParserArgs &parser_args, bool raise_error,
                              std::string *out_error_msg) {
  size_t nargs = 0;
  if (!IsPyObjNone(args)) {
    nargs = static_cast<size_t>(GetListOrTupleSize(args));
  }
  size_t nkwargs = 0;
  if (!IsPyObjNone(kwargs)) {
    nkwargs = static_cast<size_t>(PyDict_Size(kwargs));
  }
  size_t arg_pos = 0;
  size_t out_arglist_index = 0;

  if (nargs > max_pos_args_ && !allow_int_as_list_) {
    RAISE_PARSE_ERROR(out_error_msg, raise_error,
                      " takes " + std::to_string(max_pos_args_) + " positional arguments but " + std::to_string(nargs) +
                        (nargs > 1 ? " were" : " was") + " given.",
                      name_);
    return false;
  }
  for (auto &param : params_) {
    bool is_kwd = false;
    param.is_any_ = param.type_ == OP_DTYPE::DT_ANY;
    PyObject *obj = NULL;
    if (arg_pos < nargs) {
      obj = PyTuple_GetItem(args, arg_pos++);
      if (param.kw_only_) {
        RAISE_PARSE_ERROR(out_error_msg, raise_error, " got extra positional args.", name_);
        return false;
      }
    } else if (!IsPyObjNone(kwargs)) {
      is_kwd = true;
      PyObject *key_object = PyUnicode_FromString(param.name_.c_str());
      if (PyDict_Contains(kwargs, key_object)) {
        obj = PyDict_GetItem(kwargs, key_object);
        nkwargs--;
      }
      Py_DECREF(key_object);
    }
    bool check_arg_as_intlist = !is_kwd && (arg_pos == kIndex1) && param.allow_vararg_;
    int error_idx = check_arg_as_intlist ? kIndex0 : -1;
    ConvertPair convert_type({OP_DTYPE::DT_BEGIN, param.type_});
    if (!obj) {
      if (!param.optional_) {
        RAISE_PARSE_ERROR(out_error_msg, raise_error, " missing 1 required positional argument: " + param.name_ + ".",
                          name_);
        return false;
      }
      parser_args.SetArg(param.GetDefaultValue(), convert_type, out_arglist_index++);
    } else if (CheckArgsAsIntlist(args, check_arg_as_intlist)) {
      // tensor.reshape(1, 2, 3) as tensor.reshape((1, 2, 3))
      if (PyTuple_Check(args)) {
        parser_args.SetArg(args, {OP_DTYPE::DT_BEGIN, param.type_}, out_arglist_index++);
      } else {
        parser_args.SetArg(args, {OP_DTYPE::DT_LIST_INT, param.type_}, out_arglist_index++);
      }
      arg_pos = nargs;
    } else if (CheckParamValid(obj, param, raise_error, out_error_msg, convert_type, error_idx)) {
      parser_args.SetArg(obj, convert_type, out_arglist_index++);
    } else {
      RAISE_PARSE_ERROR(out_error_msg, raise_error, GetTypeErrorMsg(is_kwd, error_idx, obj, param, arg_pos), name_);
      return false;
    }
  }
  return RaiseParseKeywordArgsError(nkwargs, raise_error, out_error_msg, nargs, kwargs);
}

bool FunctionSignature::RaiseParseKeywordArgsError(size_t nkwargs, bool raise_error, std::string *out_error_msg,
                                                   size_t nargs, PyObject *kwargs) {
  std::string error_msg;
  if (nkwargs == 0) {
    return true;
  }
  if (raise_error || out_error_msg) {
    if (kwargs && kwargs != Py_None) {
      PyObject *key;
      PyObject *value;
      Py_ssize_t kwarg_pos = 0;
      while (PyDict_Next(kwargs, &kwarg_pos, &key, &value)) {
        const char *name_str = PyUnicode_AsUTF8(key);
        std::string arg_name(name_str);
        int64_t pos = -1;
        for (size_t i = 0; i < params_.size(); ++i) {
          if (arg_name == params_[i].name_) {
            pos = i;
          }
        }
        if (pos < 0) {
          error_msg = " got an unexpected keyword argument '" + arg_name + "'.";
        } else if (pos < static_cast<int64_t>(nargs)) {
          error_msg = " got multiple values for argument '" + arg_name + "'.";
        }
      }
    }
    if (error_msg.empty()) {
      error_msg = "(): invalid keyword arguments.";
    }
    if (out_error_msg) {
      out_error_msg->append(error_msg);
    }
    if (raise_error) {
      MS_EXCEPTION(TypeError) << name_ << "()" << error_msg;
    }
  }
  return false;
}

FunctionSignature::FunctionSignature(const std::string &fmt, int index, const std::string &name)
    : name_(name), max_pos_args_(0), max_args_(0), min_args_(0), allow_int_as_list_(false), index_(index) {
  auto open_paren = fmt.find('(');
  if (open_paren == std::string::npos) {
    MS_LOG(EXCEPTION) << "parse failed";
  }

  auto last_offset = open_paren + 1;
  bool done = false;
  bool is_kwonlyargs = false;
  while (!done) {
    auto offset = fmt.find(", ", last_offset);
    auto next_offset = offset + 2;
    if (offset == std::string::npos) {
      offset = fmt.find(')', last_offset);
      done = true;
      next_offset = offset + 1;
      if (offset == last_offset) {
        last_offset = next_offset;
        break;
      }
    }

    if (offset == std::string::npos || offset == last_offset) {
      MS_LOG(EXCEPTION) << "parse failed";
    }

    auto param_str = fmt.substr(last_offset, offset - last_offset);
    if (param_str.compare("*") != 0) {
      if (!is_kwonlyargs) {
        max_pos_args_++;
      }
      params_.emplace_back(param_str, is_kwonlyargs);
      allow_int_as_list_ |= params_.back().allow_vararg_;
      if (!params_.back().optional_) {
        min_args_++;
      }
      max_args_++;
    } else {
      is_kwonlyargs = true;
    }
    last_offset = next_offset;
  }
}

std::string FunctionSignature::ToString() {
  std::stringstream param_ss;
  bool kw_only_flag = false;
  for (auto &param : params_) {
    if (param.kw_only_ && !kw_only_flag) {
      kw_only_flag = true;
      param_ss << "*, ";
    }
    std::vector<std::string> type_list = {prim::OpDTypeToString(param.type_)};
    std::transform(param.cast_types_.begin(), param.cast_types_.end(), std::back_inserter(type_list),
                   [](const auto &type) { return prim::OpDTypeToString(type); });
    if (param.allow_none_) {
      type_list.emplace_back("None");
    }
    std::sort(type_list.begin(), type_list.end());
    std::string type_list_str = std::accumulate(
      type_list.begin(), type_list.end(), std::string(),
      [](const std::string &a, const std::string &b) -> std::string { return a.empty() ? b : a + ", " + b; });
    param_ss << param.name_ << "=<" << type_list_str << ">, ";
  }
  auto type_str = param_ss.str().substr(0, param_ss.str().length() - 2);
  return "(" + type_str + ")";
}

ops::OP_DTYPE GetOpDtype(const std::string &type_str) {
  auto it = type_str_map.find(type_str);
  if (it == type_str_map.end()) {
    it = type_not_in_yaml_str_map.find(type_str);
    if (it == type_not_in_yaml_str_map.end()) {
      MS_LOG(EXCEPTION) << "Parse function parameter failed! invalid type string:" << type_str;
    }
  }
  return it->second;
}

FunctionParameter::FunctionParameter(const std::string &fmt, bool is_kw_only) {
  kw_only_ = is_kw_only;
  auto space = fmt.find(' ');
  if (space == std::string::npos) {
    MS_LOG(EXCEPTION) << "Parse function parameter failed! missing type:" << fmt;
  }

  auto types_str = fmt.substr(0, space);
  std::istringstream iss(types_str);
  std::string first_valid_type_str;
  std::string substring;

  bool first_token = true;
  bool first_type = true;
  while (std::getline(iss, substring, '|')) {
    // handle disable scalar tensor argument
    if (substring == "!tensor") {
      if (first_token) {
        MS_LOG(EXCEPTION) << "Invalid parameter type format: '!tensor' cannot be the first token. "
                          << "fmt: " << fmt;
      }
      allow_scalar_tensor_ = false;
      first_token = false;
      continue;
    }

    if (first_type) {
      type_ = GetOpDtype(substring);
      first_valid_type_str = substring;
      first_type = false;
    } else {
      cast_types_.emplace_back(GetOpDtype(substring));
    }
    first_token = false;
  }

  auto name_str = fmt.substr(space + 1);
  auto eq = name_str.find('=');
  if (eq != std::string::npos) {
    name_ = name_str.substr(0, eq);
    optional_ = true;
    auto value_str = name_str.substr(eq + 1);
    if (value_str == "None") {
      allow_none_ = true;
    }
    if (first_valid_type_str.empty()) {
      MS_LOG(EXCEPTION) << "Default value specified but no valid type found: " << fmt;
    }
    default_str_.assign(first_valid_type_str).append(",").append(value_str);
    ParserDefaultObjects::GetInstance().Set(type_, value_str, default_str_);
  } else {
    optional_ = false;
    name_ = name_str;
  }

  auto varargs = name_str.find('*');
  if (varargs != std::string::npos) {
    allow_vararg_ = true;
    name_ = name_.substr(1);
  }
}

bool IsPyBool(PyObject *obj) {
  return PyBool_Check(obj) || (PyLong_Check(obj) && PyObject_HasAttrString(obj, "__ms_mutable_bool__"));
}

static inline std::vector<int64_t> ParseListInt(const std::string &s) {
  if (s.empty()) return std::vector<int64_t>();
  if (s[0] != '[' && s[0] != '(') {
    return std::vector<int64_t>{std::stol(s)};
  }
  auto args = std::vector<int64_t>();
  std::istringstream ss(s.substr(1, s.length() - kIndex2));
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    args.emplace_back(std::stol(tok));
  }
  return args;
}

namespace {
static bool ListElementCheckFloat(PyObject *obj) { return py_parse::ParseUtilsCheckFloat(obj, false); }
static bool ListElementCheckScalar(PyObject *obj) { return py_parse::ParseUtilsCheckScalar(obj, false); }
}  // namespace

bool ListTypeCheck(PyObject *obj, const ops::OP_DTYPE &type, int &idx, bool fullcheck = false) {
  switch (type) {
    case OP_DTYPE::DT_ANY:
      return true;
    case OP_DTYPE::DT_LIST_TENSOR:
      return CheckPySequenceType<CPythonList>(obj, idx, IsTensor, fullcheck);
    case OP_DTYPE::DT_LIST_ANY:
      return PyList_Check(obj);
    case OP_DTYPE::DT_LIST_INT:
      return CheckPySequenceType<CPythonList>(obj, idx, py_parse::IsGeneralizedInt, fullcheck);
    case OP_DTYPE::DT_LIST_FLOAT:
      return CheckPySequenceType<CPythonList>(obj, idx, ListElementCheckFloat, fullcheck);
    case OP_DTYPE::DT_LIST_BOOL:
      return CheckPySequenceType<CPythonList>(obj, idx, IsPyBool, fullcheck);
    case OP_DTYPE::DT_LIST_STR:
      return CheckPySequenceType<CPythonList>(obj, idx, IsPyStr, fullcheck);
    case OP_DTYPE::DT_LIST_NUMBER:
      return CheckPySequenceType<CPythonList>(obj, idx, ListElementCheckScalar, fullcheck);
    case OP_DTYPE::DT_TUPLE_ANY:
      return PyTuple_Check(obj);
    case OP_DTYPE::DT_TUPLE_INT:
      return CheckPySequenceType<CPythonTuple>(obj, idx, py_parse::IsGeneralizedInt, fullcheck);
    case OP_DTYPE::DT_TUPLE_FLOAT:
      return CheckPySequenceType<CPythonTuple>(obj, idx, ListElementCheckFloat, fullcheck);
    case OP_DTYPE::DT_TUPLE_BOOL:
      return CheckPySequenceType<CPythonTuple>(obj, idx, IsPyBool, fullcheck);
    case OP_DTYPE::DT_TUPLE_TENSOR:
      return CheckPySequenceType<CPythonTuple>(obj, idx, IsTensor, fullcheck);
    case OP_DTYPE::DT_TUPLE_NUMBER:
      return CheckPySequenceType<CPythonTuple>(obj, idx, ListElementCheckScalar, fullcheck);
    default:
      MS_LOG(EXCEPTION) << "Performing a list type check and encountered an unexpected type, which is "
                        << prim::OpDTypeToString(type);
  }
  return false;
}

static bool IsConverttablePythonType(PyObject *obj) {
  if (obj == reinterpret_cast<PyObject *>(&PyBool_Type)) {
    return true;
  }
  return false;
}

bool FunctionParameter::TypeCheck(PyObject *obj, const ops::OP_DTYPE &type, int &idx, ConvertPair &convert_type) const {
  switch (type) {
    case OP_DTYPE::DT_TENSOR:
      return IsTensor(obj);
    case OP_DTYPE::DT_STORAGE:
      return IsStorage(obj);
    case OP_DTYPE::DT_NUMBER:
      return py_parse::ParseUtilsCheckScalar(obj, allow_scalar_tensor_);
    case OP_DTYPE::DT_FLOAT:
      if (py_parse::ParseUtilsCheckFloat(obj, allow_scalar_tensor_)) {
        return true;
      } else if (py_parse::ParseUtilsCheckInt(obj, allow_scalar_tensor_)) {
        convert_type.first = OP_DTYPE::DT_INT;
        return true;
      }
      return false;
    case OP_DTYPE::DT_INT:
      return py_parse::ParseUtilsCheckInt(obj, allow_scalar_tensor_);
    case OP_DTYPE::DT_BOOL:
      if (PyBool_Check(obj)) {
        return true;
      }
      if (PyLong_Check(obj) && PyObject_HasAttrString(obj, "__ms_mutable_bool__")) {
        convert_type.first = OP_DTYPE::DT_BOOL;
        return true;
      }
      return false;
    case OP_DTYPE::DT_TYPE:
      return IsConverttablePythonType(obj) || py::isinstance<mindspore::Type>(obj);
    case OP_DTYPE::DT_STR:
      return IsPyStr(obj);
    default:
      return ListTypeCheck(obj, type, idx);
  }
  return false;
}

bool FunctionParameter::Check(PyObject *obj, ConvertPair &convert_type, int &error_idx) const {
  if (!TypeCheck(obj, type_, error_idx, convert_type)) {
    auto it = std::find_if(cast_types_.begin(), cast_types_.end(), [&](const ops::OP_DTYPE &cast_type) {
      return TypeCheck(obj, cast_type, error_idx, convert_type);
    });
    if (it != cast_types_.end() && convert_type.first == OP_DTYPE::DT_BEGIN) {
      convert_type.first = *it;
      return true;
    }
    return false;
  }
  return true;
}

PyObject *GetPyListInt(const std::vector<int64_t> &vec) {
  PyObject *list_py = PyList_New(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    PyList_SetItem(list_py, i, PyLong_FromLong(vec[i]));
  }
  return list_py;
}

PyObject *GetPyTupleInt(const std::vector<int64_t> &vec) {
  PyObject *list_py = PyTuple_New(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    PyTuple_SetItem(list_py, i, PyLong_FromLong(vec[i]));
  }
  return list_py;
}

std::optional<PyObject *> ParseIntStr(const std::string &str) {
  char *str_end;
  auto defalut_int = strtol(str.c_str(), &str_end, 0);
  return (*str_end == 0) ? std::optional<PyObject *>(PyLong_FromLong(defalut_int)) : std::nullopt;
}

std::optional<PyObject *> ParseBoolStr(const std::string &str) {
  if (str == "True" || str == "true" || str == "False" || str == "false") {
    return std::optional<PyObject *>(PyBool_FromLong((str == "True" || str == "true")));
  }
  return std::nullopt;
}

PyObject *ParseNumber(const std::string &str) {
  auto cast_bool = ParseBoolStr(str);
  if (cast_bool.has_value()) {
    return cast_bool.value();
  }
  auto cast_int = ParseIntStr(str);
  if (cast_int.has_value()) {
    return cast_int.value();
  }
  return PyFloat_FromDouble(stof(str));
}

std::string RemoveQuotes(const std::string &str) {
  if (str.size() >= kIndex2 && str.front() == '\'' && str.back() == '\'') {
    return str.substr(1, str.size() - kIndex2);
  }
  return str;
}

PyObject *ParserDefaultObjects::StrToPyObj(const ops::OP_DTYPE &type, const std::string &str) {
  if (str == "None") {
    Py_RETURN_NONE;
  }
  switch (type) {
    case ops::OP_DTYPE::DT_INT:
      return PyLong_FromLong(stol(str));
    case ops::OP_DTYPE::DT_FLOAT:
      return PyFloat_FromDouble(stof(str));
    case ops::OP_DTYPE::DT_BOOL:
      return PyBool_FromLong((str == "True" || str == "true"));
    case ops::OP_DTYPE::DT_NUMBER:
      return ParseNumber(str);
    case ops::OP_DTYPE::DT_TUPLE_INT:
      return GetPyTupleInt(ParseListInt(str));
    case ops::OP_DTYPE::DT_TUPLE_TENSOR:
      // now only support default=None
      if (str != "None") {
        MS_LOG(EXCEPTION) << "default value for Tensor must be none, got: " << str;
      }
      Py_RETURN_NONE;
    case ops::OP_DTYPE::DT_STR:
      return PyUnicode_FromString(RemoveQuotes(str).c_str());
    case ops::OP_DTYPE::DT_TENSOR:
      if (str != "None") {
        MS_LOG(EXCEPTION) << "default value for Tensor must be None, but got: " << str;
      }
      Py_RETURN_NONE;
    case ops::OP_DTYPE::DT_LIST_INT:
      return GetPyListInt(ParseListInt(str));
    case ops::OP_DTYPE::DT_LIST_FLOAT:
      if (str != "None") {
        MS_LOG(EXCEPTION) << "Defaults not supported for float[]";
      }
      Py_RETURN_NONE;
    default:
      MS_LOG(EXCEPTION) << "The" << type << " is an unknown type "
                        << ", or the default value cannot be set.";
      break;
  }
}

ValuePtr ConvertSimpleBool(PyObject *obj) {
  bool value = (PyObject_IsTrue(obj) == 1);
  return std::make_shared<BoolImm>(value);
}

ValuePtr ConvertMutableBool(PyObject *obj) {
  auto obj_int64 = PyLong_AsLongLong(obj);
  bool obj_bool = obj_int64 != 0;
  return std::make_shared<BoolImm>(obj_bool);
}

ValuePtr ConvertSimpleTensor(PyObject *obj) {
  auto tensor = tensor::ConvertPyObjectToValue(obj);
  if (tensor != nullptr) {
    if (tensor->isa<tensor::Tensor>()) {
      tensor->cast<tensor::TensorPtr>()->set_need_pipeline_sync(true);
    }
    return tensor;
  }
  return tensor;
}

template <typename T>
ValuePtr ConvertTensorList(PyObject *obj) {
  // convert to py::object and then used in data_converter, because data_converter haven't been refactored to PyObject
  py::object py_obj = py::reinterpret_borrow<py::object>(obj);
  auto val_seq = py_parse::ConvertSequence<typename T::pybind_type, ValueTuple, py_parse::ConvertTensor>(py_obj);
  if (val_seq != nullptr && val_seq->template isa<ValueTuple>()) {
    EnablePipelineForTupleTensor(val_seq->template cast<ValueTuplePtr>());
    return val_seq;
  }
  return val_seq;
}

static const std::unordered_map<int32_t, OpDefConvertFunc> kParseConverters = {
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_TENSOR), ConvertSimpleTensor},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_TUPLE_TENSOR),
   ConvertTensorList<CPythonTuple>},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_LIST_TENSOR),
   ConvertTensorList<CPythonList>},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_BOOL, mindspore::ops::DT_BOOL), ConvertMutableBool},
  {py_parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_BOOL), ConvertSimpleBool}};

OpDefConvertFunc GetSimpleConverterByType(int32_t dtype) {
  auto it = kParseConverters.find(dtype);
  if (it == kParseConverters.end()) {
    return nullptr;
  }
  return it->second;
}

ValuePtr ParserArgs::ConvertByParseDtype(size_t index) {
  auto src = src_types_[index];
  auto dst = dst_types_[index];
  OpDefConvertFunc convert_func = GetSimpleConverterByType(py_parse::CombineTypesForTypeCast(src, dst));
  if (convert_func == nullptr) {
    py_parse::OpDefConvertFunc data_convert_func =
      py_parse::GetConverterByType(src == OP_DTYPE::DT_BEGIN ? dst : py_parse::CombineTypesForTypeCast(src, dst));
    if (data_convert_func == nullptr) {
      MS_EXCEPTION(NotImplementedError) << "Can't find convert function for src_dtype[" << src << "] and dst_type"
                                        << dst << "].";
    }
    // borrow reference
    PyObject *item = arg_list_[index];
    // convert to py::object and then used in data_converter, because data_converter haven't been refactored to PyObject
    py::object py_item = py::reinterpret_borrow<py::object>(item);
    auto value = data_convert_func(py_item);
    if (value && (dst == mindspore::ops::DT_TENSOR)) {
      auto t = value->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(t);
      t->set_source_type(src);
    }
    return (value != nullptr) ? value : nullptr;
  } else {
    // borrow reference
    PyObject *item = arg_list_[index];
    auto value = convert_func(item);
    if (value != nullptr) {
      src_types_[index] = mindspore::ops::DT_BEGIN;
      return value;
    }
    return nullptr;
  }
}

std::vector<int64_t> ParserArgs::ToBasicIntVector(size_t index) {
  auto src = src_types_[index];
  auto dst = dst_types_[index];

  auto convert_func = GetIntVectorConverterByBaseType(py_parse::CombineTypesForTypeCast(src, dst));
  if (convert_func == nullptr) {
    MS_EXCEPTION(NotImplementedError) << "Can't find convert function for src_dtype[" << src << "] and dst_type[" << dst
                                      << "].";
  }
  auto value = convert_func(arg_list_[index]);
  if (value.has_value()) {
    src_types_[index] = mindspore::ops::DT_BEGIN;
    return value.value();
  }
  PrintConvertError(index);
  return {};
}

std::optional<std::vector<int64_t>> ParserArgs::ToBasicIntVectorOptional(size_t index) {
  if (arg_list_[index] == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToBasicIntVector(index));
}

int64_t ParserArgs::ToBasicInt(size_t index) {
  auto src = src_types_[index];
  auto dst = dst_types_[index];
  auto convert_func = GetInConverterByBaseType(py_parse::CombineTypesForTypeCast(src, dst));
  if (convert_func == nullptr) {
    MS_EXCEPTION(NotImplementedError) << "Can't find convert function for src_dtype[" << src << "] and dst_type" << dst
                                      << "].";
  }
  auto value = convert_func(arg_list_[index]);
  if (value.has_value()) {
    src_types_[index] = mindspore::ops::DT_BEGIN;
    return value.value();
  }
  PrintConvertError(index);
  return 0;
}

std::optional<int64_t> ParserArgs::ToBasicIntOptional(size_t index) {
  if (arg_list_[index] == Py_None) {
    return std::nullopt;
  }
  return std::make_optional(ToBasicInt(index));
}

void ParserArgs::InsertInputTensor(size_t index, PyObject *input) {
  arg_list_.insert(arg_list_.begin() + index, input);
  src_types_.insert(src_types_.begin() + index, ops::OP_DTYPE::DT_BEGIN);
  dst_types_.insert(dst_types_.begin() + index, ops::OP_DTYPE::DT_TENSOR);
}

ValuePtr UnpackTensor(PyObject *input, const std::string &func_name) {
  if (tensor::IsPyObjectTensorPy(input)) {
    return ConvertSimpleTensor(input);
  } else {
    MS_EXCEPTION(TypeError) << "Tensor." << func_name << "() doesn't apply to '"
                            << PyParser::BuildPyObjectInputTypeString((input)) << "' object.";
  }
  return nullptr;
}

void ParserArgs::SetArg(PyObject *arg, const ConvertPair &convert_type, size_t index) {
  if (index > arg_list_.size()) {
    MS_LOG(EXCEPTION) << "Invalid argument index.";
  }
  arg_list_[index] = arg;
  src_types_[index] = convert_type.first;
  dst_types_[index] = convert_type.second;
}

void ParserArgs::ClearArgs() {
  arg_list_.clear();
  src_types_.clear();
  dst_types_.clear();
}

void ParserArgs::PrintConvertError(size_t index) {
  const auto &obj = arg_list_[index];
  std::stringstream ss;
  size_t param_idx = index;
  // In tensor api, 'input' is not included in the signature's parameter list
  if (arg_list_.size() > signature_->params_.size()) {
    if (param_idx == 0) {
      MS_LOG(EXCEPTION) << "Invalid param idx, please check.";
    }
    param_idx -= 1;
  }
  ss << signature_->name_ << "():";
  ss << " argument \'" << signature_->params_[param_idx].name_ << "\'(position " << param_idx << ") should be "
     << prim::OpDTypeToString(dst_types_[index]);
  if (!PyTuple_Check(obj) && !PyList_Check(obj)) {
    ss << ", but got " << PyParser::BuildPyObjectInputTypeString(obj) << ".";
  } else {
    int error_pos = 0;
    PyObject *element;
    const auto expect_type = src_types_[index] == ops::OP_DTYPE::DT_BEGIN ? dst_types_[index] : src_types_[index];
    (void)ListTypeCheck(obj, expect_type, error_pos, true);
    if (PyTuple_Check(obj)) {
      element = PyTuple_GetItem(obj, error_pos);
    } else {
      element = PyList_GetItem(obj, error_pos);
    }
    ss << ", but unpack failed at pos " << error_pos << " (got " << PyParser::BuildPyObjectInputTypeString(element)
       << ").";
  }
  MS_EXCEPTION(TypeError) << ss.str();
}

void ParserArgs::CheckHasFallback() {
  if (!fallback_enabled()) {
    return;
  }
  for (const auto &arg : arg_list_) {
    MS_LOG(DEBUG) << "Has fallback " << PyObjectHasFallbackAttr(arg) << " for " << PyUnicode_AsUTF8(PyObject_Str(arg));
    if (PyObjectHasFallbackAttr(arg)) {
      has_fallback_ = true;
      return;
    }
  }
}

void ParserArgs::CheckHasFallback(PyObject *obj) {
  if (!fallback_enabled()) {
    return;
  }
  MS_LOG(DEBUG) << "Has fallback " << PyObjectHasFallbackAttr(obj) << " for " << PyUnicode_AsUTF8(PyObject_Str(obj));
  if (PyObjectHasFallbackAttr(obj)) {
    has_fallback_ = true;
    return;
  }
}

std::vector<std::string> GetInvalidKwargsName(PyObject *kwargs, const std::vector<FunctionParameter> &params) {
  std::vector<std::string> invalid_names;
  Py_ssize_t kwargs_size = PyDict_Size(kwargs);
  const size_t kw_start_idx = params.size() - static_cast<size_t>(kwargs_size);
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(kwargs, &pos, &key, &value)) {
    bool is_vailed = true;
    const char *name_str = PyUnicode_AsUTF8(key);
    std::string arg_name(name_str);
    for (auto idx = kw_start_idx; idx < params.size(); ++idx) {
      if (arg_name == params[idx].name_) {
        is_vailed = false;
        break;
      }
    }
    if (is_vailed) {
      invalid_names.emplace_back(arg_name);
    }
  }
  return invalid_names;
}

std::vector<std::string> ParamMatchInfo(PyObject *args, PyObject *kwargs,
                                        const std::vector<FunctionParameter> &params) {
  size_t argpos = 0;
  std::string type_info = "    match failed because invalid types: (";
  std::string guide_line(type_info.size(), ' ');
  Py_ssize_t args_size = (args && args != Py_None) ? GetListOrTupleSize(args) : 0;
  while (argpos < params.size()) {
    bool is_kwd = argpos >= static_cast<size_t>(args_size);
    PyObject *obj = NULL;
    if (is_kwd) {
      PyObject *key_object = PyUnicode_FromString(params[argpos].name_.c_str());
      if (!IsPyObjNone(kwargs) && PyDict_Contains(kwargs, key_object)) {
        obj = PyDict_GetItem(kwargs, key_object);
      }
      Py_DECREF(key_object);
    } else {
      // borrow reference
      obj = PyTuple_GetItem(args, argpos);
    }
    if (!obj || obj == Py_None) {
      if (!params[argpos].optional_) {
        return {"    missing required argument: " + params[argpos].name_};
      }
    } else {
      auto input_type_str = PyParser::BuildPyObjectInputTypeString(obj) + ", ";
      if (is_kwd) {
        input_type_str = params[argpos].name_ + "=" + input_type_str;
      }
      type_info += input_type_str;
      ConvertPair convert_type({params[argpos].type_, params[argpos].type_});
      int error_idx = 0;
      if ((!is_kwd && params[argpos].kw_only_) || (obj == Py_None && !params[argpos].allow_none_) ||
          !params[argpos].Check(obj, convert_type, error_idx)) {
        guide_line += std::string(input_type_str.size(), '~');
      } else {
        guide_line += std::string(input_type_str.size(), ' ');
      }
    }
    ++argpos;
  }
  type_info = type_info.substr(0, type_info.size() - kIndex2);
  return {type_info + ")", guide_line};
}

std::string PythonArgParser::PrintParseError(PyObject *args, PyObject *kwargs, const bool &is_method) {
  Py_ssize_t args_size = (!IsPyObjNone(args)) ? GetListOrTupleSize(args) : 0;
  Py_ssize_t kwargs_size = (!IsPyObjNone(kwargs)) ? PyDict_Size(kwargs) : 0;
  const Py_ssize_t total_arg_size = args_size + kwargs_size;
  const std::vector<int> valid_signature_idx = CollectValidSignatureIndices(signatures_, total_arg_size);
  if (valid_signature_idx.size() == 1) {
    ParserArgs parser_args(signatures_[valid_signature_idx[0]]);
    signatures_[valid_signature_idx[0]]->Parse(args, kwargs, parser_args, true);
  }
  std::vector<std::string> error_msg;
  std::unordered_set<std::string> signatures_str;
  for (auto idx : valid_signature_idx) {
    const auto &signature = signatures_[idx];
    auto sig_str = GenerateSignatureStr(function_name_, signature, is_method);
    if (signatures_str.count(sig_str)) continue;
    signatures_str.insert(sig_str);
    error_msg.emplace_back("\"" + sig_str + "\"");
    std::vector<std::string> invalid_names;
    if (kwargs && kwargs != Py_None) {
      invalid_names = GetInvalidKwargsName(kwargs, signature->params_);
      if (!invalid_names.empty()) {
        auto invalid_kw = std::accumulate(
          invalid_names.begin(), invalid_names.end(), std::string(),
          [](const std::string &a, const std::string &b) -> std::string { return a.empty() ? b : a + ", " + b; });
        error_msg.emplace_back("    match failed because incorrect keyword name: " + invalid_kw);
      }
    }
    if (invalid_names.empty()) {
      auto match_infos = ParamMatchInfo(args, kwargs, signature->params_);
      if (!match_infos.empty()) {
        error_msg.insert(error_msg.end(), match_infos.begin(), match_infos.end());
      } else {
        ParserArgs parser_args(signature);
        std::string error_info;
        signature->Parse(args, kwargs, parser_args, false, &error_info);
        error_msg.emplace_back("    " + error_info);
      }
    }
  }
  auto type_list = GetParseTypeListString(args, kwargs);
  if (error_msg.empty()) {
    bool in_map = IsFunctionalRegInMap(function_name_, is_method);
    if (in_map) {
      return prim::BuildFunctionalErrorMsg(function_name_, type_list, is_method);
    } else {
      auto function_signature = GenerateAllSignatureStrings(signatures_, function_name_, is_method);
      auto ss = prim::BuildApiInputInfo(function_name_, type_list);
      for (auto &function_sign : function_signature) {
        ss << function_sign << "\n";
      }
      return ss.str();
    }
  } else {
    auto ss = prim::BuildApiInputInfo(function_name_, type_list);
    for (auto &error_info : error_msg) {
      ss << error_info << "\n";
    }
    return ss.str();
  }
}

ParserDefaultObjects &ParserDefaultObjects::GetInstance() {
  static ParserDefaultObjects default_objs_instance;
  return default_objs_instance;
}

// Declare template to compile corresponding method.
template std::vector<int64_t> Converter::ToBasicIntVector<CPythonTuple>(PyObject *python_args, size_t i);
template std::vector<int64_t> Converter::ToBasicIntVector<CPythonList>(PyObject *python_args, size_t i);
template std::optional<std::vector<int64_t>> Converter::ToBasicIntVectorOptional<CPythonTuple>(PyObject *python_args,
                                                                                               size_t i);
template std::optional<std::vector<int64_t>> Converter::ToBasicIntVectorOptional<CPythonList>(PyObject *python_args,
                                                                                              size_t i);
template ValueTuplePtr Converter::ToIntList<CPythonTuple>(PyObject *python_args, size_t i);
template ValueTuplePtr Converter::ToIntList<CPythonList>(PyObject *python_args, size_t i);
template ValueTuplePtr Converter::ToTensorList<CPythonTuple>(PyObject *python_args, size_t i);
template ValueTuplePtr Converter::ToTensorList<CPythonList>(PyObject *python_args, size_t i);
template ValueTuplePtr Converter::ToBoolList<CPythonTuple>(PyObject *python_args, size_t i);
template ValueTuplePtr Converter::ToBoolList<CPythonList>(PyObject *python_args, size_t i);
template ValueTuplePtr Converter::ToFloatList<CPythonTuple>(PyObject *python_args, size_t i);
template ValueTuplePtr Converter::ToFloatList<CPythonList>(PyObject *python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToTensorListOptional<CPythonTuple>(PyObject *python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToTensorListOptional<CPythonList>(PyObject *python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToIntListOptional<CPythonTuple>(PyObject *python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToIntListOptional<CPythonList>(PyObject *python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToBoolListOptional<CPythonTuple>(PyObject *python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToBoolListOptional<CPythonList>(PyObject *python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToFloatListOptional<CPythonTuple>(PyObject *python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToFloatListOptional<CPythonList>(PyObject *python_args, size_t i);

}  // namespace pynative
}  // namespace mindspore
