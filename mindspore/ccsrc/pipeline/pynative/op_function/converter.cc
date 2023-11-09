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
#include "pipeline/pynative/op_function/converter.h"
#include <unordered_map>
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/ps/parse/data_converter.h"

namespace mindspore {
namespace pynative {
namespace {
using OP_DTYPE = mindspore::ops::OP_DTYPE;
std::string CTypeToPythonType(const OP_DTYPE &type) {
  static std::unordered_map<OP_DTYPE, std::string> convert_map = {{OP_DTYPE::DT_BOOL, "bool"},
                                                                  {OP_DTYPE::DT_INT, "int"},
                                                                  {OP_DTYPE::DT_FLOAT, "float"},
                                                                  {OP_DTYPE::DT_NUMBER, "number"},
                                                                  {OP_DTYPE::DT_TENSOR, "tensor"},
                                                                  {OP_DTYPE::DT_STR, "str"},
                                                                  {OP_DTYPE::DT_TUPLE_BOOL, "tuple[bool]"},
                                                                  {OP_DTYPE::DT_TUPLE_INT, "tuple[int]"},
                                                                  {OP_DTYPE::DT_TUPLE_FLOAT, "tuple[float]"},
                                                                  {OP_DTYPE::DT_TUPLE_NUMBER, "tuple[number]"},
                                                                  {OP_DTYPE::DT_TUPLE_TENSOR, "tuple[tensor]"},
                                                                  {OP_DTYPE::DT_TUPLE_STR, "tuple[str]"},
                                                                  {OP_DTYPE::DT_LIST_BOOL, "list[bool]"},
                                                                  {OP_DTYPE::DT_LIST_INT, "list[int]"},
                                                                  {OP_DTYPE::DT_LIST_FLOAT, "list[float]"},
                                                                  {OP_DTYPE::DT_LIST_NUMBER, "list[number]"},
                                                                  {OP_DTYPE::DT_LIST_TENSOR, "list[tensor]"},
                                                                  {OP_DTYPE::DT_LIST_STR, "list[str]"}};
  if (convert_map.find(type) == convert_map.end()) {
    MS_LOG(EXCEPTION) << "Can not found type in convert map" << type;
  }
  return convert_map[type];
}

ValuePtr ConvertByCastDtype(const py::object &input, const ops::OpInputArg &op_arg) {
  for (auto &cast_dtype : op_arg.cast_dtype_) {
    auto convert_func = parse::GetConverterByType(parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
    if (convert_func == nullptr) {
      MS_LOG(EXCEPTION) << "Can't find convert function for src_dtype[" << cast_dtype << "] and dst_type"
                        << op_arg.arg_dtype_ << "].";
    }
    auto value = convert_func(input);
    if (value != nullptr) {
      return value;
    }
  }
  return nullptr;
}

template <typename T, typename U>
std::shared_ptr<U> PyCast(const py::object &obj) {
  return std::make_shared<U>(py::cast<T>(obj));
}

BoolImmPtr ConvertBool(const py::object &obj) {
  if (!py::isinstance<py::bool_>(obj)) {
    return nullptr;
  }
  return PyCast<bool, BoolImm>(obj);
}

Int64ImmPtr ConvertInt(const py::object &obj) {
  if (!py::isinstance<py::int_>(obj)) {
    return nullptr;
  }
  return PyCast<int64_t, Int64Imm>(obj);
}

FP32ImmPtr ConvertFloat(const py::object &obj) {
  if (!py::isinstance<py::float_>(obj)) {
    return nullptr;
  }
  return PyCast<double, FP32Imm>(obj);
}

ScalarPtr ConvertNumber(const py::object &obj) {
  if (py::isinstance<py::int_>(obj)) {
    return std::make_shared<Int64Imm>(py::cast<int64_t>(obj));
  } else if (py::isinstance<py::float_>(obj)) {
    return std::make_shared<FP32Imm>(py::cast<double>(obj));
  } else if (py::isinstance<py::bool_>(obj)) {
    return std::make_shared<BoolImm>(py::cast<bool>(obj));
  }
  return nullptr;
}

template <typename T, typename U, typename N>
ValueTuplePtr ConvertList(const py::object &obj) {
  if (!py::isinstance<T>(obj)) {
    return nullptr;
  }
  auto seq = obj.cast<T>();
  std::vector<ValuePtr> convert;
  for (size_t i = 0; i < seq.size(); ++i) {
    if (!py::isinstance<U>(seq[i])) {
      return {};
    }
    auto out = PyCast<U, N>(seq[i]);
    if (out == nullptr) {
      return nullptr;
    }
    (void)convert.emplace_back(out);
  }
  return std::make_shared<ValueTuple>(convert);
}
}  // namespace

Converter::Converter(const ops::OpDef &op_def) { op_def_ = op_def; }

void Converter::Parse(py::list python_args) {
  python_args_ = &python_args;
  if (op_def_.args_.size() != python_args.size()) {
    MS_LOG(EXCEPTION) << "For operator " << op_def_.name_ << ", it requires " << op_def_.args_.size()
                      << "parameters, bug got " << python_args.size() << "parameters!";
  }
}

ValuePtr Converter::ToTensor(size_t i) {
  const auto &op_arg = op_def_.args_[i];
  const py::object &obj = (*python_args_)[i];
  auto tensor = parse::ConvertTensor(obj);
  if (tensor != nullptr) {
    return tensor;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg)->cast<TensorPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  ThrowException(i);
  return nullptr;
}

std::optional<ValuePtr> Converter::ToTensorOptional(size_t i) {
  const py::object &obj = (*python_args_)[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(std::move(ToTensor(i)));
}

template <typename T>
ValueTuplePtr Converter::ToTensorList(size_t i) {
  const auto &op_arg = op_def_.args_[i];
  const py::object &obj = (*python_args_)[i];
  auto val_seq = parse::ConvertSequence<py::tuple, ValueTuple, parse::ConvertTensor>(obj);

  if (val_seq != nullptr && val_seq->isa<ValueTuple>()) {
    return val_seq->cast<ValueTuplePtr>();
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg);
    if (convert_value != nullptr && convert_value->isa<ValueTuple>()) {
      return convert_value->cast<ValueTuplePtr>();
    }
  }
  ThrowException(i);
  return nullptr;
}

Int64ImmPtr Converter::ToInt(size_t i) {
  const auto &op_arg = op_def_.args_[i];
  const py::object &obj = (*python_args_)[i];
  auto convert = ConvertInt(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    convert = ConvertByCastDtype(obj, op_arg)->cast<Int64ImmPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  ThrowException(i);
  return nullptr;
}

std::optional<Int64ImmPtr> Converter::ToIntOptional(size_t i) {
  const py::object &obj = (*python_args_)[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(std::move(ToInt(i)));
}

template <typename T>
ValueTuplePtr Converter::ToIntList(size_t i) {
  const auto &op_arg = op_def_.args_[i];
  const py::object &obj = (*python_args_)[i];
  ValueTuplePtr convert = ConvertList<T, py::int_, Int64Imm>(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg);
    if (convert_value != nullptr && convert_value->isa<ValueTuple>()) {
      return convert_value->cast<ValueTuplePtr>();
    }
  }
  ThrowException(i);
  return nullptr;
}

BoolImmPtr Converter::ToBool(size_t i) {
  const auto &op_arg = op_def_.args_[i];
  const py::object &obj = (*python_args_)[i];
  auto convert = ConvertBool(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    convert = ConvertByCastDtype(obj, op_arg)->cast<BoolImmPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  ThrowException(i);
  return nullptr;
}

template <typename T>
ValueTuplePtr Converter::ToBoolList(size_t i) {
  const auto &op_arg = op_def_.args_[i];
  const py::object &obj = (*python_args_)[i];
  ValueTuplePtr convert = ConvertList<T, py::bool_, BoolImm>(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg);
    if (convert_value != nullptr && convert_value->isa<ValueTuple>()) {
      return convert_value->cast<ValueTuplePtr>();
    }
  }
  ThrowException(i);
  return nullptr;
}

FP32ImmPtr Converter::ToFloat(size_t i) {
  const auto &op_arg = op_def_.args_[i];
  const py::object &obj = (*python_args_)[i];
  auto convert = ConvertFloat(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    convert = ConvertByCastDtype(obj, op_arg)->cast<FP32ImmPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  ThrowException(i);
  return nullptr;
}

template <typename T>
ValueTuplePtr Converter::ToFloatList(size_t i) {
  const auto &op_arg = op_def_.args_[i];
  const py::object &obj = (*python_args_)[i];
  ValueTuplePtr convert = ConvertList<T, py::float_, FP32Imm>(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg);
    if (convert_value != nullptr && convert_value->isa<ValueTuple>()) {
      return convert_value->cast<ValueTuplePtr>();
    }
  }
  ThrowException(i);
  return nullptr;
}

ScalarPtr Converter::ToScalar(size_t i) {
  const auto &op_arg = op_def_.args_[i];
  const py::object &obj = (*python_args_)[i];
  auto convert = ConvertNumber(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    convert = ConvertByCastDtype(obj, op_arg)->cast<ScalarPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  ThrowException(i);
  return nullptr;
}

TypePtr Converter::ToDtype(size_t i) {
  const py::object &obj = (*python_args_)[i];
  if (!py::isinstance<mindspore::Type>(obj)) {
    MS_LOG(EXCEPTION) << "Get arg is not mindspore type " << py::str(obj);
  }
  return obj.cast<TypePtr>();
}

py::object Converter::Wrap(const TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->NeedWait()) {
    py::gil_scoped_release release;
    tensor->Wait();
  }
  py::tuple v(1);
  v[0] = tensor;
  return v[0];
}

void Converter::ThrowException(size_t i) {
  MS_LOG(EXCEPTION) << "For op " << op_def_.name_ << ", the " << i << "th arg dtype is not right!"
                    << "expect dtype: " << CTypeToPythonType(op_def_.args_[i].arg_dtype_)
                    << "but got dtype: " << py::type((*python_args_)[i]);
}

// Declare template to compile corresponding method.
template ValueTuplePtr Converter::ToTensorList<py::tuple>(size_t i);
template ValueTuplePtr Converter::ToTensorList<py::list>(size_t i);
template ValueTuplePtr Converter::ToIntList<py::tuple>(size_t i);
template ValueTuplePtr Converter::ToIntList<py::list>(size_t i);
template ValueTuplePtr Converter::ToBoolList<py::tuple>(size_t i);
template ValueTuplePtr Converter::ToBoolList<py::list>(size_t i);
template ValueTuplePtr Converter::ToFloatList<py::tuple>(size_t i);
template ValueTuplePtr Converter::ToFloatList<py::list>(size_t i);

}  // namespace pynative
}  // namespace mindspore
