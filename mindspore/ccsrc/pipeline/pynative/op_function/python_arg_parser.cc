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
#include "pipeline/pynative/op_function/python_arg_parser.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/ps/parse/data_converter.h"
namespace mindspore {
namespace pynative {
namespace {
ValuePtr ConvertByCastDtype(const py::object &input, const ops::OpArg &op_arg) {
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

FP64ImmPtr ConvertFloat(const py::object &obj) {
  if (!py::isinstance<py::int_>(obj)) {
    return nullptr;
  }
  return PyCast<double, FP64Imm>(obj);
}

ScalarPtr ConvertNumber(const py::object &obj) {
  if (py::isinstance<py::int_>(obj)) {
    return std::make_shared<Int64Imm>(py::cast<int64_t>(obj));
  } else if (py::isinstance<py::float_>(obj)) {
    return std::make_shared<FP64Imm>(py::cast<double>(obj));
  } else if (py::isinstance<py::bool_>(obj)) {
    return std::make_shared<BoolImm>(py::cast<bool>(obj));
  }
  return nullptr;
}

TensorPtr ConvertTensor(const py::object &obj) {
  if (!py::isinstance<tensor::Tensor>(obj)) {
    return nullptr;
  }
  return obj.cast<tensor::TensorPtr>();
}

// StringImmPtr ConvertStr(const py::object &obj) {
//   if (!py::isinstance<py::str>(obj)) {
//     return nullptr;
//   }
//   return PyCast<std::string, StringImm>(obj);
// }

template <typename T, typename U, typename N>
std::vector<std::shared_ptr<N>> ConvertList(const py::object &obj) {
  if (!py::isinstance<T>(obj)) {
    return {};
  }
  auto seq = obj.cast<T>();
  std::vector<std::shared_ptr<N>> convert;
  for (size_t i = 0; i < seq.size(); ++i) {
    if (!py::isinstance<U>(seq[i])) {
      return {};
    }
    auto out = PyCast<U, N>(seq[i]);
    if (out == nullptr) {
      return {};
    }
    (void)convert.emplace_back(out);
  }
  return convert;
}
}  // namespace

void Parser::Parse(py::list python_args) {
  python_args_ = &python_args;
  if (op_def_->args_.size() != python_args.size()) {
    MS_LOG(EXCEPTION) << "For operator " << op_def_->name_ << ", it requires " << op_def_->args_.size()
                      << "paramters, bug got " << python_args.size() << "parameters!";
  }
}

TensorPtr Parser::ToTensor(size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = (*python_args_)[i];
  TensorPtr tensor = ConvertTensor(obj);
  if (tensor != nullptr) {
    return tensor;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg)->cast<TensorPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  PrintError(i);
  return nullptr;
}

ValuePtrList Parser::ToTensorList(size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = (*python_args_)[i];
  auto val_seq = parse::ConvertSequence<py::list, ValueList, parse::ConvertTensor>(obj)->cast<ValueSequencePtr>();
  if (val_seq != nullptr) {
    return val_seq->value();
  }
  val_seq = parse::ConvertSequence<py::tuple, ValueTuple, parse::ConvertTensor>(obj)->cast<ValueSequencePtr>();
  if (val_seq != nullptr) {
    return val_seq->value();
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg)->cast<ValueSequencePtr>();
    if (convert != nullptr) {
      return convert->value();
    }
  }
  PrintError(i);
  return {};
}

Int64ImmPtr Parser::ToInt(size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = (*python_args_)[i];
  Int64ImmPtr convert = ConvertInt(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg)->cast<Int64ImmPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  PrintError(i);
  return nullptr;
}

std::vector<Int64ImmPtr> Parser::ToIntList(size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = (*python_args_)[i];
  std::vector<Int64ImmPtr> convert = ConvertList<py::list, py::int_, Int64Imm>(obj);
  if (!convert.empty()) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg)->cast<ValueSequencePtr>();
    if (convert != nullptr) {
      std::vector<Int64ImmPtr> result;
      std::transform(convert->value().begin(), convert->value().end(), std::back_inserter(result),
                     [](const ValuePtr &value) { return value->cast<Int64ImmPtr>(); });
      return result;
    }
  }
  PrintError(i);
  return {};
}

BoolImmPtr Parser::ToBool(size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = (*python_args_)[i];
  BoolImmPtr convert = ConvertBool(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg)->cast<BoolImmPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  PrintError(i);
  return nullptr;
}

std::vector<BoolImmPtr> Parser::ToBoolList(size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = (*python_args_)[i];
  std::vector<BoolImmPtr> convert = ConvertList<py::list, py::bool_, BoolImm>(obj);
  if (!convert.empty()) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg)->cast<ValueSequencePtr>();
    if (convert != nullptr) {
      std::vector<BoolImmPtr> result;
      std::transform(convert->value().begin(), convert->value().end(), std::back_inserter(result),
                     [](const ValuePtr &value) { return value->cast<BoolImmPtr>(); });
      return result;
    }
  }
  PrintError(i);
  return {};
}

FloatImmPtr Parser::ToFloat(size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = (*python_args_)[i];
  FloatImmPtr convert = ConvertFloat(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg)->cast<FloatImmPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  PrintError(i);
  return nullptr;
}

ScalarPtr Parser::ToScalar(size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = (*python_args_)[i];
  auto convert = ConvertNumber(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg)->cast<ScalarPtr>();
    if (convert != nullptr) {
      return convert;
    }
  }
  PrintError(i);
  return nullptr;
}

std::vector<ScalarPtr> Parser::ToScalarList(size_t i) { return {}; }

py::object Parser::Wrap(const TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  if (tensor->NeedWait()) {
    py::gil_scoped_release release;
    tensor->Wait();
  }
  py::tuple v(1);
  v[0] = tensor;
  return v[0];
}

void Parser::PrintError(size_t i) { return; }
}  // namespace pynative
}  // namespace mindspore
