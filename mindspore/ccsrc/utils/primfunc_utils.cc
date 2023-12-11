/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "include/common/utils/primfunc_utils.h"
#include "include/common/utils/convert_utils_py.h"

namespace mindspore::ops {
std::string EnumToString(OP_DTYPE dtype) {
  static const std::unordered_map<OP_DTYPE, std::string> kEnumToStringMap = {
    {OP_DTYPE::DT_BOOL, "bool"},
    {OP_DTYPE::DT_INT, "int"},
    {OP_DTYPE::DT_FLOAT, "float"},
    {OP_DTYPE::DT_NUMBER, "Number"},
    {OP_DTYPE::DT_TENSOR, "Tensor"},
    {OP_DTYPE::DT_STR, "string"},
    {OP_DTYPE::DT_ANY, "Any"},
    {OP_DTYPE::DT_TUPLE_BOOL, "tuple of bool"},
    {OP_DTYPE::DT_TUPLE_INT, "tuple of int"},
    {OP_DTYPE::DT_TUPLE_FLOAT, "tuple of float"},
    {OP_DTYPE::DT_TUPLE_NUMBER, "tuple of Number"},
    {OP_DTYPE::DT_TUPLE_TENSOR, "tuple of Tensor"},
    {OP_DTYPE::DT_TUPLE_STR, "tuple of string"},
    {OP_DTYPE::DT_TUPLE_ANY, "tuple of Any"},
    {OP_DTYPE::DT_LIST_BOOL, "list of bool"},
    {OP_DTYPE::DT_LIST_INT, "list of int"},
    {OP_DTYPE::DT_LIST_FLOAT, "list of float"},
    {OP_DTYPE::DT_LIST_NUMBER, "list of number"},
    {OP_DTYPE::DT_LIST_TENSOR, "list of tensor"},
    {OP_DTYPE::DT_LIST_STR, "list of string"},
    {OP_DTYPE::DT_LIST_ANY, "list of Any"},
  };

  auto it = kEnumToStringMap.find(dtype);
  if (it == kEnumToStringMap.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Failed to map Enum[" << dtype << "] to String.";
  }
  return it->second;
}

namespace {
template <typename T>
bool ValidateSequenceType(const AbstractBasePtr &abs_seq, OP_DTYPE type_elem) {
  if (!abs_seq->isa<T>()) {
    return false;
  }
  if (type_elem == OP_DTYPE::DT_ANY) {
    return true;
  }
  auto abs = abs_seq->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->dynamic_len()) {
    return true;
  }
  for (const auto &abs_elem : abs->elements()) {
    if (!ValidateArgsType(abs_elem, type_elem)) {
      return false;
    }
  }
  return true;
}

bool ValidateArgsSequenceType(const AbstractBasePtr &abs_arg, OP_DTYPE type_arg) {
  switch (static_cast<int>(type_arg)) {
    case OP_DTYPE::DT_TUPLE_BOOL: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_BOOL);
    }
    case OP_DTYPE::DT_TUPLE_INT: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_INT);
    }
    case OP_DTYPE::DT_TUPLE_FLOAT: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_FLOAT);
    }
    case OP_DTYPE::DT_TUPLE_NUMBER: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_NUMBER);
    }
    case OP_DTYPE::DT_TUPLE_TENSOR: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_TENSOR);
    }
    case OP_DTYPE::DT_TUPLE_STR: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_STR);
    }
    case OP_DTYPE::DT_TUPLE_ANY: {
      return ValidateSequenceType<abstract::AbstractTuple>(abs_arg, OP_DTYPE::DT_ANY);
    }
    case OP_DTYPE::DT_LIST_BOOL: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_BOOL);
    }
    case OP_DTYPE::DT_LIST_INT: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_INT);
    }
    case OP_DTYPE::DT_LIST_FLOAT: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_FLOAT);
    }
    case OP_DTYPE::DT_LIST_NUMBER: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_NUMBER);
    }
    case OP_DTYPE::DT_LIST_TENSOR: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_TENSOR);
    }
    case OP_DTYPE::DT_LIST_STR: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_STR);
    }
    case OP_DTYPE::DT_LIST_ANY: {
      return ValidateSequenceType<abstract::AbstractList>(abs_arg, OP_DTYPE::DT_ANY);
    }
    default: {
      MS_EXCEPTION(ValueError) << "Unknown op dtype " << EnumToString(type_arg);
    }
  }
}
}  // namespace

bool ValidateArgsType(const AbstractBasePtr &abs_arg, OP_DTYPE type_arg) {
  auto abs_type = abs_arg->BuildType();
  MS_EXCEPTION_IF_NULL(abs_type);
  switch (static_cast<int>(type_arg)) {
    case OP_DTYPE::DT_ANY: {
      return true;
    }
    case OP_DTYPE::DT_BOOL: {
      return abs_arg->isa<abstract::AbstractScalar>() && abs_type->isa<Bool>();
    }
    case OP_DTYPE::DT_INT: {
      return abs_arg->isa<abstract::AbstractScalar>() && (abs_type->isa<Int>() || abs_type->isa<UInt>());
    }
    case OP_DTYPE::DT_FLOAT: {
      return abs_arg->isa<abstract::AbstractScalar>() && (abs_type->isa<Float>() || abs_type->isa<BFloat>());
    }
    case OP_DTYPE::DT_NUMBER: {
      return abs_arg->isa<abstract::AbstractScalar>() && abs_type->isa<Number>();
    }
    case OP_DTYPE::DT_STR: {
      return abs_arg->isa<abstract::AbstractScalar>() && abs_type->isa<String>();
    }
    case OP_DTYPE::DT_TENSOR: {
      return abs_arg->isa<abstract::AbstractTensor>();
    }
    case OP_DTYPE::DT_TYPE: {
      return abs_arg->isa<abstract::AbstractType>() && abs_type->isa<Type>();
    }
    default: {
      return ValidateArgsSequenceType(abs_arg, type_arg);
    }
  }
  return false;
}

std::string BuildOpErrorMsg(const OpDefPtr &op_def, const std::vector<std::string> &op_type_list) {
  std::stringstream init_arg_ss;
  std::stringstream input_arg_ss;
  for (const auto &op_arg : op_def->args_) {
    if (op_arg.as_init_arg_) {
      init_arg_ss << op_arg.arg_name_ << "=<";
      for (const auto &dtype : op_arg.cast_dtype_) {
        init_arg_ss << EnumToString(dtype) << ", ";
      }
      init_arg_ss << EnumToString(op_arg.arg_dtype_) << ">, ";
    } else {
      input_arg_ss << op_arg.arg_name_ << "=<";
      for (const auto &dtype : op_arg.cast_dtype_) {
        input_arg_ss << EnumToString(dtype) << ", ";
      }
      input_arg_ss << EnumToString(op_arg.arg_dtype_) << ">, ";
    }
  }

  auto init_arg_str = init_arg_ss.str();
  auto input_arg_str = input_arg_ss.str();
  constexpr size_t truncate_offset = 2;
  init_arg_str =
    init_arg_str.empty() ? "" : init_arg_str.replace(init_arg_str.end() - truncate_offset, init_arg_str.end(), "");
  input_arg_str =
    input_arg_str.empty() ? "" : input_arg_str.replace(input_arg_str.end() - truncate_offset, input_arg_str.end(), "");

  std::stringstream real_init_arg_ss;
  std::stringstream real_input_arg_ss;
  for (size_t i = 0; i < op_type_list.size(); i++) {
    const auto &op_arg = op_def->args_[i];
    if (op_arg.as_init_arg_) {
      real_init_arg_ss << op_arg.arg_name_ << "=" << op_type_list[i] << ", ";
    } else {
      real_input_arg_ss << op_arg.arg_name_ << "=" << op_type_list[i] << ", ";
    }
  }
  auto real_init_arg_str = real_init_arg_ss.str();
  auto real_input_arg_str = real_input_arg_ss.str();
  real_init_arg_str = real_init_arg_str.empty() ? ""
                                                : real_init_arg_str.replace(real_init_arg_str.end() - truncate_offset,
                                                                            real_init_arg_str.end(), "");
  real_input_arg_str =
    real_input_arg_str.empty()
      ? ""
      : real_input_arg_str.replace(real_input_arg_str.end() - truncate_offset, real_input_arg_str.end(), "");

  std::stringstream ss;
  ss << "Failed calling " << op_def->name_ << " with \"" << op_def->name_ << "(" << real_init_arg_str << ")("
     << real_input_arg_str << ")\"." << std::endl;
  ss << "The valid calling should be: " << std::endl;
  ss << "\"" << op_def->name_ << "(" << init_arg_str << ")(" << input_arg_str << ")\".";
  return ss.str();
}
}  // namespace mindspore::ops
