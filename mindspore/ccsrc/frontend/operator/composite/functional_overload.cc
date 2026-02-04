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

#include "frontend/operator/composite/functional_overload.h"

#include <set>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include "include/utils/operator/primitive_utils.h"
#include "primitive/structure_ops.h"
#include "ir/core_ops_primitive.h"
#include "ir/dtype/tensor_type.h"
#include "frontend/operator/ops.h"
#include "abstract/abstract_value.h"
#include "frontend/jit/ps/resource.h"
#include "frontend/jit/ps/fallback.h"
#include "frontend/jit/ps/parse/parse_base.h"
#include "frontend/jit/ps/static_analysis/prim.h"
#include "frontend/jit/ps/static_analysis/prim_utils.h"
#include "frontend/operator/composite/auto_generate/functional_map.h"
#include "primitive/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace prim {
namespace {
using OP_DTYPE = mindspore::ops::OP_DTYPE;

std::vector<OP_DTYPE> GetSourceDtypeByArgHandler(const std::string &arg_handler_func) {
  static std::map<std::string, std::vector<OP_DTYPE>> arg_handler_map = {
    {"to_pair", {OP_DTYPE::DT_INT, OP_DTYPE::DT_FLOAT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_kernel_size", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_strides", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_rates", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_dilations", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_output_padding", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"to_2d_paddings", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY}},
    {"dtype_to_type_id", {OP_DTYPE::DT_TYPE}},
    {"str_to_enum", {OP_DTYPE::DT_STR}},
    {"_scalar_tensor_to_scalar", {OP_DTYPE::DT_NUMBER, OP_DTYPE::DT_TENSOR}},
    {"_scalar_tensor_to_int", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TENSOR}},
    {"_scalar_tensor_to_float", {OP_DTYPE::DT_FLOAT, OP_DTYPE::DT_TENSOR}},
    {"_normalize_int_sequence", {OP_DTYPE::DT_INT, OP_DTYPE::DT_TUPLE_ANY, OP_DTYPE::DT_LIST_ANY, OP_DTYPE::DT_TENSOR}},
  };
  auto iter = arg_handler_map.find(arg_handler_func);
  if (iter == arg_handler_map.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Miss definition of arg_handler '" << arg_handler_func << "' here.";
  }
  return iter->second;
}

size_t GetHashIdForFunctionalCache(const std::string &functional_name, const AbstractBasePtrList &args_abs_list,
                                   bool is_method) {
  return hash_combine(
    {std::hash<std::string>()(functional_name), AbstractBasePtrListHash(args_abs_list), std::hash<bool>{}(is_method)});
}

bool MatchExpectedDtype(const ops::OP_DTYPE &input_dtype, const ops::OP_DTYPE &expected_dtype) {
  MS_LOG(DEBUG) << "Input dtype is: '" << prim::OpDTypeToString(input_dtype) << "' and expected dtype is '"
                << prim::OpDTypeToString(expected_dtype) << "'.";
  // Check if the types match.
  if (input_dtype == expected_dtype || expected_dtype == ops::OP_DTYPE::DT_ANY) {
    return true;
  }
  static std::set<ops::OP_DTYPE> number_dtype_list = {ops::OP_DTYPE::DT_BOOL, ops::OP_DTYPE::DT_INT,
                                                      ops::OP_DTYPE::DT_FLOAT};
  static std::set<ops::OP_DTYPE> tuple_dtype_list = {ops::OP_DTYPE::DT_TUPLE_BOOL,   ops::OP_DTYPE::DT_TUPLE_INT,
                                                     ops::OP_DTYPE::DT_TUPLE_FLOAT,  ops::OP_DTYPE::DT_TUPLE_NUMBER,
                                                     ops::OP_DTYPE::DT_TUPLE_TENSOR, ops::OP_DTYPE::DT_TUPLE_STR,
                                                     ops::OP_DTYPE::DT_TUPLE_ANY};
  static std::set<ops::OP_DTYPE> list_dtype_list = {ops::OP_DTYPE::DT_LIST_BOOL,   ops::OP_DTYPE::DT_LIST_INT,
                                                    ops::OP_DTYPE::DT_LIST_FLOAT,  ops::OP_DTYPE::DT_LIST_NUMBER,
                                                    ops::OP_DTYPE::DT_LIST_TENSOR, ops::OP_DTYPE::DT_LIST_STR,
                                                    ops::OP_DTYPE::DT_LIST_ANY};
  // Check number.
  if (expected_dtype == ops::OP_DTYPE::DT_NUMBER && number_dtype_list.find(input_dtype) != number_dtype_list.end()) {
    return true;
  }
  // Check Tuple without checking its elements.
  if (input_dtype == ops::OP_DTYPE::DT_TUPLE_ANY && tuple_dtype_list.find(expected_dtype) != tuple_dtype_list.end()) {
    return true;
  }
  // Check List without checking its elements.
  if (input_dtype == ops::OP_DTYPE::DT_LIST_ANY && list_dtype_list.find(expected_dtype) != list_dtype_list.end()) {
    return true;
  }
  return false;
}

bool MatchPrimitiveArgDtype(const std::string &prim_name, const ops::OpInputArg &op_arg,
                            const ops::OP_DTYPE &input_dtype) {
  MS_LOG(DEBUG) << "Matching arg '" << op_arg.arg_name_ << "' for Primitive[" << prim_name << "] with dtype "
                << prim::OpDTypeToString(input_dtype) << ".";
  if (MatchExpectedDtype(input_dtype, op_arg.arg_dtype_) ||
      (op_arg.is_optional_ && input_dtype == ops::OP_DTYPE::DT_NONE)) {
    return true;
  }
  if (!op_arg.cast_dtype_.empty()) {
    return std::any_of(op_arg.cast_dtype_.cbegin(), op_arg.cast_dtype_.cend(),
                       [&input_dtype](const ops::OP_DTYPE &dtype) { return MatchExpectedDtype(input_dtype, dtype); });
  }
  if (!op_arg.arg_handler_.empty()) {
    auto src_dtypes = GetSourceDtypeByArgHandler(op_arg.arg_handler_);
    return std::any_of(src_dtypes.cbegin(), src_dtypes.cend(),
                       [&input_dtype](const ops::OP_DTYPE &dtype) { return MatchExpectedDtype(input_dtype, dtype); });
  }
  return false;
}

ops::OP_DTYPE GetOpDtypeFromAbstract(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  const auto &abs_type = abs->BuildType();
  MS_EXCEPTION_IF_NULL(abs_type);
  if (abs->isa<abstract::AbstractTensor>()) {
    return ops::OP_DTYPE::DT_TENSOR;
  }
  if (abs->isa<abstract::AbstractTuple>()) {
    return ops::OP_DTYPE::DT_TUPLE_ANY;
  }
  if (abs->isa<abstract::AbstractList>()) {
    return ops::OP_DTYPE::DT_LIST_ANY;
  }
  if (abs->isa<abstract::AbstractNone>()) {
    return ops::OP_DTYPE::DT_NONE;
  }
  if (abs->isa<abstract::AbstractType>() && abs_type->isa<Type>()) {
    return ops::OP_DTYPE::DT_TYPE;
  }
  if (abs->isa<abstract::AbstractScalar>()) {
    if (abs_type->isa<Bool>()) {
      return ops::OP_DTYPE::DT_BOOL;
    }
    if (abs_type->isa<Int>() || abs_type->isa<UInt>()) {
      return ops::OP_DTYPE::DT_INT;
    }
    if (abs_type->isa<Float>() || abs_type->isa<BFloat>()) {
      return ops::OP_DTYPE::DT_FLOAT;
    }
    if (abs_type->isa<Number>()) {
      return ops::OP_DTYPE::DT_NUMBER;
    }
    if (abs_type->isa<String>()) {
      return ops::OP_DTYPE::DT_STR;
    }
  }
  return ops::OP_DTYPE::DT_ANY;
}

std::set<std::string> *GetMethodKwonlyArgs(const std::string &prim_name) {
  const auto &iter = ops::tensor_method_kwonlyargs_map.find(prim_name);
  if (iter != ops::tensor_method_kwonlyargs_map.end()) {
    return &iter->second;
  }
  return nullptr;
}

std::set<std::string> *GetFunctionKwonlyArgs(const std::string &prim_name) {
  const auto &iter = ops::function_kwonlyargs_map.find(prim_name);
  if (iter != ops::function_kwonlyargs_map.end()) {
    return &iter->second;
  }
  return nullptr;
}

std::pair<size_t, bool> GetVarargsIndex(const std::string &prim_name, bool is_method) {
  if (is_method) {
    const auto &iter = ops::tensor_method_varargs_map.find(prim_name);
    if (iter != ops::tensor_method_varargs_map.end()) {
      return std::pair<size_t, bool>(iter->second, true);
    }
    return std::pair<size_t, bool>(0, false);
  } else {
    const auto &iter = ops::function_varargs_map.find(prim_name);
    if (iter != ops::function_varargs_map.end()) {
      return std::pair<size_t, bool>(iter->second, true);
    }
    return std::pair<size_t, bool>(0, false);
  }
}

std::string GetPrimName(const ValuePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  if (prim->isa<Primitive>()) {
    return prim->cast<PrimitivePtr>()->name();
  }
  if (prim->isa<DeprecatedTensorMethod>()) {
    return prim->cast<DeprecatedTensorMethodPtr>()->name();
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Expect Primitive or MetaFuncGraph, but got " << prim->ToString();
}

template <typename ContainerType>
std::string HandleContainer(const ContainerType *container, const std::string &type_name) {
  MS_EXCEPTION_IF_NULL(container);
  if (container->dynamic_len()) {
    return type_name;
  }

  std::stringstream ss;
  ss << type_name << "<";
  for (size_t i = 0; i < container->size(); ++i) {
    if (i != 0) {
      ss << ", ";
    }
    ss << BuildArgsTypeString(container->elements()[i]);
  }
  ss << ">";
  return ss.str();
}

std::string BuildOtherTypeString(const TypePtr &arg_type) {
  if (arg_type->isa<Keyword>()) {
    std::stringstream ss;
    auto kw_type = arg_type->cast_ptr<Keyword>();
    ss << kw_type->GetKey() << "=" << BuildArgsTypeString(kw_type->GetValue());
    return ss.str();
  }

  if (arg_type->isa<Tuple>()) {
    auto tuple_type = arg_type->cast_ptr<Tuple>();
    return HandleContainer(tuple_type, "tuple");
  }

  if (arg_type->isa<List>()) {
    auto list_type = arg_type->cast_ptr<List>();
    return HandleContainer(list_type, "list");
  }

  return arg_type->ToString();
}
}  // namespace

void PrimitiveConverter::GetOpDtypeList() {
  for (const auto &abs : input_args_abs_list_) {
    // Ignore monad.
    if (abs->isa<abstract::AbstractMonad>()) {
      MS_LOG(DEBUG) << "Arg is a: AbstractMonad";
      continue;
    }
    if (abs->isa<abstract::AbstractKeywordArg>()) {
      auto kw_abs = abs->cast<abstract::AbstractKeywordArgPtr>();
      const std::string &key = kw_abs->get_key();
      auto op_dtype = GetOpDtypeFromAbstract(kw_abs->get_arg());
      input_keyword_args_dtype_.insert(std::make_pair(key, op_dtype));
      MS_LOG(DEBUG) << "Keyword arg is: " << key;
    } else {
      auto op_dtype = GetOpDtypeFromAbstract(abs);
      (void)input_position_args_dtype_.emplace_back(op_dtype);
      MS_LOG(DEBUG) << "Arg is a: " << op_dtype;
    }
  }
}

bool PrimitiveConverter::CheckKwargs(PrimitiveAttr *cur_prim) {
  for (const auto &[key, value] : input_keyword_args_dtype_) {
    const auto &op_indexes = cur_prim->op_def->indexes_;
    const auto &expect_op_args = cur_prim->op_def->args_;
    const auto &iter = op_indexes.find(key);
    if (iter == op_indexes.end()) {
      error_msgs_.push_back("match failed because incorrect keyword name: " + key);
      is_keyword_ = true;
      return false;
    }
    // Check key index.
    auto index_key = iter->second;
    if (index_key < input_position_args_dtype_.size() && !cur_prim->has_varargs) {
      error_msgs_.push_back("Mismatch: Primitive[" + cur_prim->prim_name + "] got multiple values for argument '" +
                            key + "'.");
      return false;
    }
    // Check value dtype.
    auto op_arg = expect_op_args[index_key];
    if (!MatchPrimitiveArgDtype(cur_prim->prim_name, op_arg, value)) {
      error_msgs_.push_back(functional_name_ + "(): argument '" + key + "' must be " +
                            prim::OpDTypeToString(op_arg.arg_dtype_) + " but got " + prim::OpDTypeToString(value));
      return false;
    }
  }
  return true;
}

size_t PrimitiveConverter::GetPrimDefaultSize(const std::vector<ops::OpInputArg> &expect_op_args,
                                              const std::string &prim_name, size_t varargs_index,
                                              bool has_varargs) const {
  auto default_dict = parse::GetPrimDefaultDict(prim_name);
  bool has_default = !py::isinstance<py::none>(default_dict);
  // The default value of vararg is ().
  bool vararg_non_default =
    has_varargs && ((has_default && !default_dict.contains(expect_op_args[varargs_index].arg_name_)) || !has_default);
  size_t varargs_count = vararg_non_default ? 1 : 0;
  if (!has_default) {
    return varargs_count;
  }
  return varargs_count + default_dict.cast<py::dict>().size();
}

bool PrimitiveConverter::CheckExplicitSequence(PrimitiveAttr *cur_prim,
                                               const std::vector<ops::OpInputArg> &expect_op_args) const {
  MS_LOG(DEBUG) << "Check explicit sequence";
  return input_position_args_dtype_.size() == (cur_prim->varargs_index + 1) &&
         MatchPrimitiveArgDtype(cur_prim->prim_name, expect_op_args[cur_prim->varargs_index],
                                input_position_args_dtype_[cur_prim->varargs_index]);
}

bool PrimitiveConverter::CheckImplicitTuple(PrimitiveAttr *cur_prim,
                                            const std::vector<ops::OpInputArg> &expect_op_args) {
  MS_LOG(DEBUG) << "Check implicit tuple";
  for (size_t i = cur_prim->varargs_index; i < input_position_args_dtype_.size(); ++i) {
    const auto &arg_name = expect_op_args[cur_prim->varargs_index].arg_name_;
    // Variable parameters support only the integer type.
    if (!MatchExpectedDtype(ops::OP_DTYPE::DT_INT, input_position_args_dtype_[i])) {
      error_msgs_.push_back(functional_name_ + "(): argument '" + arg_name + "' (position " + std::to_string(i) +
                            ") must be " + prim::OpDTypeToString(expect_op_args[cur_prim->varargs_index].arg_dtype_) +
                            ", not " + prim::OpDTypeToString(input_position_args_dtype_[i]) + ".");
      return false;
    }
  }
  *need_pack_ = true;
  return true;
}

bool PrimitiveConverter::CheckPositionArgs(PrimitiveAttr *cur_prim) {
  size_t check_position = input_position_args_dtype_.size();
  const auto kwonly_list =
    is_method_ ? GetMethodKwonlyArgs(cur_prim->prim_name) : GetFunctionKwonlyArgs(cur_prim->prim_name);
  const auto &op_def = ops::GetOpDef(cur_prim->prim_name);
  MS_EXCEPTION_IF_NULL(op_def);
  const auto &expect_op_args = op_def->args_;

  if (cur_prim->has_varargs && input_position_args_dtype_.size() > cur_prim->varargs_index) {
    check_position = cur_prim->varargs_index;
    if (!CheckExplicitSequence(cur_prim, expect_op_args) && !CheckImplicitTuple(cur_prim, expect_op_args)) {
      return false;
    }
    // MetaFuncGraph does not support *args scenario
    if (cur_prim->prim_name.find("DeprecatedTranspose") != std::string::npos) {
      *need_pack_ = true;
    }
  }

  MS_LOG(DEBUG) << "Check position args";
  for (size_t i = 0; i < check_position; ++i) {
    // position argument should not be keyword-only.
    const auto &arg_name = expect_op_args[i].arg_name_;
    if (kwonly_list != nullptr && kwonly_list->find(arg_name) != kwonly_list->end()) {
      error_msgs_.push_back("Find arg_name " + arg_name + " in kwonly_list");
      first_failed_position_ = i;
      return false;
    }
    if (!MatchPrimitiveArgDtype(cur_prim->prim_name, expect_op_args[i], input_position_args_dtype_[i])) {
      error_msgs_.push_back(functional_name_ + "(): argument '" + arg_name + "' (position " + std::to_string(i) +
                            ") must be " + prim::OpDTypeToString(expect_op_args[i].arg_dtype_) + ", not " +
                            prim::OpDTypeToString(input_position_args_dtype_[i]) + ".");
      first_failed_position_ = i;
      return false;
    }
  }
  return true;
}

bool PrimitiveConverter::CheckArgsSize(PrimitiveAttr *cur_prim) {
  // If no varargs , check args size
  auto has_varargs_index_pair = GetVarargsIndex(cur_prim->prim_name, is_method_);
  cur_prim->varargs_index = has_varargs_index_pair.first;
  cur_prim->has_varargs = has_varargs_index_pair.second;
  const auto &expect_op_args = cur_prim->op_def->args_;
  size_t kwonly_size = 0;
  auto kwonly_list = is_method_ ? GetMethodKwonlyArgs(cur_prim->prim_name) : GetFunctionKwonlyArgs(cur_prim->prim_name);
  if (kwonly_list != nullptr) {
    kwonly_size = kwonly_list->size();
  }
  auto inputs_size = input_position_args_dtype_.size() + input_keyword_args_dtype_.size();
  size_t expect_positional_size = expect_op_args.size() - kwonly_size;
  if (cur_prim->has_varargs || (inputs_size <= expect_op_args.size() && inputs_size >= expect_positional_size)) {
    cur_prim->is_match_args_size = true;
  }
  MS_LOG(DEBUG) << "Expect args number: " << expect_op_args.size() << " kwonly_size: " << kwonly_size
                << ". The number of input position args: " << input_position_args_dtype_.size()
                << " and that of input keyword args: " << input_keyword_args_dtype_.size()
                << " has_varargs: " << cur_prim->has_varargs;

  if (!cur_prim->has_varargs && input_position_args_dtype_.size() > expect_positional_size) {
    error_msgs_.push_back(
      functional_name_ + "() takes " +
      (is_method_ ? std::to_string(expect_positional_size - 1) : std::to_string(expect_positional_size)) +
      " positional arguments but " + (is_method_ ? std::to_string(inputs_size - 1) : std::to_string(inputs_size)) +
      (inputs_size > 2 ? " were" : " was") + " given.");
    return false;
  }

  // Check the number of arguments.
  auto least_size = expect_op_args.size() - GetPrimDefaultSize(expect_op_args, cur_prim->prim_name,
                                                               cur_prim->varargs_index, cur_prim->has_varargs);
  if (inputs_size < least_size) {
    error_msgs_.push_back(functional_name_ + "() missing " + std::to_string(least_size - inputs_size) +
                          " required argument");
    return false;
  }
  return true;
}

bool PrimitiveConverter::MatchPrimitiveArgs(PrimitiveAttr *cur_prim) {
  MS_LOG(DEBUG) << "Matching Primitive" << cur_prim->prim_name;
  if (!CheckArgsSize(cur_prim) || !CheckKwargs(cur_prim) || !CheckPositionArgs(cur_prim)) {
    return false;
  }
  return true;
}

std::string BuildArgsTypeString(const TypePtr &arg_type) {
  MS_EXCEPTION_IF_NULL(arg_type);
  if (arg_type->isa<Bool>()) {
    return "bool";
  }
  if (arg_type->isa<Int>() || arg_type->isa<UInt>()) {
    return "int";
  }
  if (arg_type->isa<Float>() || arg_type->isa<BFloat>()) {
    return "float";
  }
  if (arg_type->isa<String>()) {
    return "string";
  }
  if (arg_type->isa<TypeNone>()) {
    return "None";
  }
  if (arg_type->isa<TensorType>()) {
    return "Tensor";
  }
  return BuildOtherTypeString(arg_type);
}

std::string PrimitiveConverter::BuildMatchInfo(const std::vector<std::string> &arg_info_list) const {
  MS_LOG(DEBUG) << "first_failed_position_ = " << first_failed_position_;
  std::stringstream ss;
  std::string result;
  std::string info = "    match failed because invalid types: (";
  std::string guide_line(info.size(), ' ');
  result = std::accumulate(
    arg_info_list.begin(), arg_info_list.end(), std::string(),
    [](const std::string &a, const std::string &b) -> std::string { return a.empty() ? b : a + ", " + b; });
  for (size_t i = 0; i < arg_info_list.size(); ++i) {
    MS_LOG(DEBUG) << "size[" << i << "] = " << arg_info_list[i].size();
    if (i + 1 < first_failed_position_) {
      guide_line += std::string(arg_info_list[i].size(), ' ') + "  ";
    } else {
      guide_line += std::string(arg_info_list[i].size(), '~') + "~~";
    }
  }
  ss << info << result << ")\n" << guide_line << "\n";
  return ss.str();
}

std::string PrimitiveConverter::BuildDetailedErrorMsg(const std::vector<std::string> &arg_info_list) const {
  const auto &signature_map = GetFunctionalSignatureMap(is_method_);
  auto it = signature_map.find(functional_name_);
  std::stringstream ss;
  if (it != signature_map.end()) {
    if (prim_list_size_ == 1) {
      if (!error_msgs_.empty()) {
        ss << error_msgs_[0] << "\n";
      }
    } else if (match_index_.size() == 1) {
      if (match_index_[0] < error_msgs_.size()) {
        ss << error_msgs_[match_index_[0]] << "\n";
      }
    } else {
      const std::vector<std::string> &valid_arg_options = it->second;
      ss << BuildApiInputInfo(functional_name_, arg_info_list).str();
      for (size_t i = 0; i < valid_arg_options.size(); ++i) {
        const std::string &arg_option = valid_arg_options[i];
        ss << "\"" << arg_option << "\"\n";
        if (std::find(match_index_.begin(), match_index_.end(), i) == match_index_.end()) {
          continue;
        }
        if (!is_keyword_) {
          ss << BuildMatchInfo(arg_info_list);
        } else {
          ss << "\t" << error_msgs_[i] << "\n";
        }
      }
    }
    ss << std::endl;
  } else {
    MS_LOG(EXCEPTION) << "Valid arg options are not correctly generated." << std::endl;
  }
  return ss.str();
}

FuncGraphPtr DeprecatedTensorMethod::GenerateFuncGraph(const abstract::AbstractBasePtrList &) {
  static const std::string module_path = "mindspore.graph._parse.deprecated.deprecated_tensor_method";
  static const std::string method_map = "deprecated_tensor_method_map";
  py::dict map_obj = python_adapter::GetPyFn(module_path, method_map);
  const auto &method_name = method();
  if (!map_obj.contains(py::str(method_name))) {
    MS_LOG(INTERNAL_EXCEPTION) << "As a deprecated Tensor method, '" << method_name
                               << "' should be registered in graph/_parse/deprecated/deprecated_tensor_method.py::"
                               << method_map;
  }
  const std::string &function_name = map_obj[py::str(method_name)].cast<std::string>();
  auto value = prim::GetPythonOps(function_name, parse::PYTHON_MOD_OPS_TENSOR_METHOD_MODULE);
  if (!value->isa<FuncGraph>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Expect FuncGraph, but got " << value->ToString();
  }
  return value->cast<FuncGraphPtr>();
}

std::map<size_t, std::pair<ValuePtr, bool>> &GetFunctionalConvertCache() {
  static std::map<size_t, std::pair<ValuePtr, bool>> functional_convert_cache;
  return functional_convert_cache;
}

bool IsFunctionalMethod(const TypeId &type_id, const std::string &method_name) {
  return NormalizeTypeId(type_id) == kObjectTypeTensorType &&
         ops::tensor_method_overload_map.find(method_name) != ops::tensor_method_overload_map.end();
}

void PrimitiveConverter::PrintErrorMessages() {
  // The order of tensor_method_overload_map is different with that of tensor_method_overload_signature_map.
  if (has_deprecated_) {
    if (!error_msgs_.empty()) {
      std::string first_error = error_msgs_[0];
      error_msgs_.erase(error_msgs_.begin());
      error_msgs_.push_back(first_error);
    }
    for (auto &index : this->match_index_) {
      if (index == 0) {
        index = prim_list_size_ - 1;
      } else {
        index--;
      }
    }
  }

  for (auto &i : match_index_) {
    MS_LOG(DEBUG) << "index=" << i;
  }
  for (auto &e : error_msgs_) {
    MS_LOG(DEBUG) << "error_msgs_=" << e;
  }

  std::vector<std::string> arg_info_list;
  // The first arg of method map is input tenspr
  auto start = is_method_ ? input_args_abs_list_.begin() + 1 : input_args_abs_list_.begin();
  (void)std::transform(start, input_args_abs_list_.cend(), std::back_inserter(arg_info_list),
                       [](const AbstractBasePtr &op_abs) { return BuildArgsTypeString(op_abs->BuildType()); });
  if (prim_list_size_ == 1 || match_index_.size() != 0) {
    MS_EXCEPTION(TypeError) << BuildDetailedErrorMsg(arg_info_list);
  } else {
    MS_EXCEPTION(TypeError) << BuildFunctionalErrorMsg(functional_name_, arg_info_list, is_method_);
  }
}

ValuePtr PrimitiveConverter::Convert() {
  // Convert Function to Primitive.
  const auto &overload_map = is_method_ ? ops::tensor_method_overload_map : ops::function_overload_map;
  const auto &iter = overload_map.find(functional_name_);
  if (iter == overload_map.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Functional[" << functional_name_ << "] does not support overloading.";
  }
  GetOpDtypeList();
  const auto &prim_list = iter->second;
  prim_list_size_ = prim_list.size();
  ValuePtr match_prim = nullptr;
  for (size_t i = 0; i < prim_list.size(); ++i) {
    const auto &prim = prim_list[i];
    const auto &prim_name = GetPrimName(prim);
    const auto &op_def = ops::GetOpDef(prim_name);
    if (op_def == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Cannot find OpDef of Primitive[" << prim_name << "].";
    }
    PrimitiveAttr cur_prim = {
      prim_name,
      op_def,
    };

    if (MatchPrimitiveArgs(&cur_prim)) {
      match_prim = prim;
      break;
    }

    if (cur_prim.is_match_args_size) {
      match_index_.push_back(i);
    }
    if (prim_name.find("Deprecated") != std::string::npos) {
      has_deprecated_ = true;
    }
  }

  if (match_prim != nullptr) {
    return match_prim;
  }

  PrintErrorMessages();
  return nullptr;
}

ValuePtr TransformFunctionalToPrimitive(const std::string &functional_name,
                                        const abstract::AbstractBasePtrList &input_args_abs_list, bool is_method,
                                        bool *need_pack) {
  // Check cache.
  auto hash_id = GetHashIdForFunctionalCache(functional_name, input_args_abs_list, is_method);
  const auto &cache_iter = GetFunctionalConvertCache().find(hash_id);
  if (cache_iter != GetFunctionalConvertCache().end()) {
    const auto &prim = cache_iter->second.first;
    *need_pack = cache_iter->second.second;
    MS_LOG(DEBUG) << "Get functional cache: " << functional_name << ", primitive name: " << prim->ToString();
    return prim;
  }

  MS_LOG(DEBUG) << "Start looking for matched primitive for Functional[" << functional_name << "].";
  PrimitiveConverter converter(functional_name, input_args_abs_list, is_method, need_pack);
  ValuePtr match_prim = converter.Convert();

  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "] to Primitive: " << match_prim->ToString();
  GetFunctionalConvertCache()[hash_id] = std::make_pair(match_prim, *need_pack);
  return match_prim;
}

AnfNodePtrList GeneratePrimitivePackPositionArgs(const FuncGraphPtr &func_graph,
                                                 const std::vector<AnfNodePtr> &args_list, size_t position_args_size,
                                                 size_t var_args_index) {
  AnfNodePtrList nodes;
  if (position_args_size <= var_args_index) {
    for (size_t i = 0; i < position_args_size; ++i) {
      (void)nodes.emplace_back(args_list[i]);
    }
  } else {
    for (size_t i = 0; i < var_args_index; ++i) {
      (void)nodes.emplace_back(args_list[i]);
    }
    AnfNodePtrList pack_args{NewValueNode(prim::kPrimMakeTuple)};
    (void)std::copy(args_list.begin() + var_args_index, args_list.begin() + position_args_size,
                    std::back_inserter(pack_args));
    auto pack_args_node = func_graph->NewCNodeInOrder(pack_args);
    nodes.emplace_back(pack_args_node);
  }
  return nodes;
}

void GeneratePrimitivePackKeywordArgs(const std::string &prim_name, const std::vector<ops::OpInputArg> &op_args,
                                      std::map<std::string, AnfNodePtr> *key_map, size_t var_args_index,
                                      AnfNodePtrList *nodes) {
  size_t nodes_size = nodes->size();
  for (size_t i = nodes_size; i < op_args.size(); ++i) {
    const auto &arg_name = op_args[i].arg_name_;
    const auto &iter = key_map->find(arg_name);
    if (iter != key_map->end()) {
      MS_LOG(DEBUG) << "Get args for Primitive[" << prim_name << "]: " << iter->second->DebugString();
      (void)nodes->emplace_back(iter->second);
      (void)key_map->erase(arg_name);
    } else {
      if (i == var_args_index) {
        auto empty_tuple_value = std::make_shared<ValueTuple>(ValuePtrList());
        (void)nodes->emplace_back(NewValueNode(empty_tuple_value));
        continue;
      }
      auto default_arg = parse::GetArgDefaultValue(prim_name, arg_name);
      if (default_arg == nullptr) {
        break;
      }
      MS_LOG(DEBUG) << "Get the default value of '" << arg_name << "' attribute of Primitive[" << prim_name
                    << "], which is " << default_arg->ToString() << ".";
      (void)nodes->emplace_back(NewValueNode(default_arg));
    }
  }
}

AnfNodePtrList GeneratePrimitivePackArgs(const std::pair<std::string, bool> &params,
                                         const std::vector<AnfNodePtr> &args_list,
                                         const std::vector<ops::OpInputArg> &op_args,
                                         const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                         const FuncGraphPtr &graph) {
  const std::string &prim_name = params.first;
  bool is_method = params.second;
  size_t var_args_index = GetVarargsIndex(prim_name, is_method).first;
  size_t args_size = args_list.size();
  std::map<std::string, AnfNodePtr> key_map;
  for (size_t idx = 0; idx < args_list.size(); ++idx) {
    auto input = args_list[idx];
    if (mindspore::IsMonad(input)) {
      --args_size;
      continue;
    }
    auto input_abs = eval_func(input);
    if (input_abs->isa<abstract::AbstractKeywordArg>()) {
      abstract::GetKeywordArgsMap(input_abs, op_args, input, graph, &key_map);
    }
  }
  args_size -= key_map.size();
  AnfNodePtrList nodes = GeneratePrimitivePackPositionArgs(graph, args_list, args_size, var_args_index);
  GeneratePrimitivePackKeywordArgs(prim_name, op_args, &key_map, var_args_index, &nodes);

  if (nodes.size() != op_args.size()) {
    std::string args_type_str = (op_args.size() != 0 && op_args[0].as_init_arg_) ? "init arguments" : "inputs";
    MS_EXCEPTION(TypeError) << "For Operator[" << prim_name << "], the number of " << args_type_str
                            << " (including default arguments) should be " << op_args.size()
                            << ", but the actual number of inputs is not satisfied, which is " << args_size << ".";
  }
  return nodes;
}

AnfNodePtr ConvertFunctionalToPrimitive(const std::string &functional_name, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                        const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                        bool is_method) {
  bool need_pack = false;
  auto prim = TransformFunctionalToPrimitive(functional_name, args_abs_list, is_method, &need_pack);
  auto prim_name = GetPrimName(prim);
  const auto &op_def = ops::GetOpDef(prim_name);
  MS_EXCEPTION_IF_NULL(op_def);
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtrList prim_inputs_list;
  if (prim->isa<Primitive>()) {
    auto do_trans = std::make_shared<prim::DoTransPrimitiveFunction>(prim->cast<PrimitivePtr>());
    (void)prim_inputs_list.emplace_back(NewValueNode(do_trans));
  } else {
    (void)prim_inputs_list.emplace_back(NewValueNode(prim));
  }
  AnfNodePtrList args_node_list;
  if (need_pack) {
    args_node_list = GeneratePrimitivePackArgs(std::make_pair(prim_name, is_method), inputs_list, op_def->args_,
                                               eval_func, func_graph);
  } else {
    args_node_list =
      abstract::GeneratePrimitiveDefaultArgs(prim_name, inputs_list, op_def->args_, eval_func, func_graph);
  }
  (void)std::copy(args_node_list.begin(), args_node_list.end(), std::back_inserter(prim_inputs_list));
  auto prim_node = func_graph->NewCNodeInOrder(prim_inputs_list);
  prim_node->set_debug_info(cnode->debug_info());
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "]: " << cnode->DebugString()
                << " to Primitive: " << prim_node->DebugString();
  return prim_node;
}

AnfNodePtr ConvertFunctionalToPyExecute(const FunctionalPtr &functional, const AnfNodePtrList &inputs_list,
                                        const AbstractBasePtrList &args_abs_list, const CNodePtr &cnode,
                                        bool is_method) {
  auto functional_name = functional->name();
  MS_LOG(DEBUG) << "Functional[" << functional_name << "] receives arguments that contain Any.";
  std::stringstream script_buffer;
  constexpr auto index_fn = 0;
  std::string data_str = fallback::ConvertRealStrToUnicodeStr(functional_name, index_fn);
  auto fn_node = inputs_list[index_fn];
  auto fn_key_node = NewValueNode(std::make_shared<StringImm>(data_str));
  std::vector<AnfNodePtr> keys_inputs{NewValueNode(prim::kPrimMakeTuple), fn_key_node};
  std::vector<AnfNodePtr> values_inputs{NewValueNode(prim::kPrimMakeTuple), fn_node};
  if (is_method) {
    // Convert Functional node to PyExectue("x.method_name(arg1, arg2)", local_keys, local_values)
    script_buffer << data_str << "." << functional_name + "(";
  } else {
    // Convert Functional node to PyExectue("mint_function_str(x, arg1, arg2)", local_keys, local_values)
    std::string mint_function_str = "__mint_" + functional_name + "__";
    script_buffer << mint_function_str << "(" << data_str << ", ";
    (void)keys_inputs.emplace_back(NewValueNode(std::make_shared<StringImm>(mint_function_str)));
    (void)values_inputs.emplace_back(NewValueNode(std::make_shared<parse::InterpretedObject>(py::cast(functional))));
  }
  for (size_t index = 1; index < inputs_list.size(); ++index) {
    auto internal_arg = fallback::ConvertRealStrToUnicodeStr(functional_name, index);
    if (args_abs_list[index]->isa<abstract::AbstractKeywordArg>()) {
      auto key = args_abs_list[index]->cast<abstract::AbstractKeywordArgPtr>()->get_key();
      script_buffer << key << "=" << internal_arg << ", ";
    } else {
      script_buffer << internal_arg << ", ";
    }
    (void)keys_inputs.emplace_back(NewValueNode(std::make_shared<StringImm>(internal_arg)));
    (void)values_inputs.emplace_back(inputs_list[index]);
  }
  script_buffer << ")";
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "] to script: " << script_buffer.str();
  auto script_node = NewValueNode(std::make_shared<StringImm>(script_buffer.str()));
  auto fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  auto keys_tuple_node = fg->NewCNodeInOrder(keys_inputs);
  auto values_tuple_node = fg->NewCNodeInOrder(values_inputs);
  auto pyexecute_node =
    fallback::CreatePyExecuteCNodeInOrder(fg, script_node, keys_tuple_node, values_tuple_node, cnode->debug_info());
  pyexecute_node->set_debug_info(cnode->debug_info());
  MS_LOG(DEBUG) << "Convert Functional[" << functional_name << "]: " << cnode->DebugString()
                << " to PyExecute: " << pyexecute_node->DebugString();
  return pyexecute_node;
}
}  // namespace prim
}  // namespace mindspore
