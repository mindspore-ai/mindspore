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

#include "pipeline/jit/fallback.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"
#include "ops/core_ops.h"
#include "pipeline/jit/debug/trace.h"
#include "pipeline/jit/parse/resolve.h"
#include "abstract/abstract_value.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace fallback {
AnfNodePtr ConvertPyObjectToPyExecute(const FuncGraphPtr &fg, const std::string &key, const py::object value,
                                      const AnfNodePtr &node) {
  auto value_node_key = key;
  (void)value_node_key.erase(
    std::remove_if(value_node_key.begin(), value_node_key.end(), [](char c) { return !std::isalnum(c); }),
    value_node_key.end());

  // Set the value node into dict firstly.
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  constexpr auto set_local_variable = "set_local_variable";
  MS_LOG(DEBUG) << set_local_variable << "([" << key << "]/" << value_node_key << ", " << value << ")";
  (void)python_adapter::CallPyModFn(mod, set_local_variable, value_node_key, value);

  // Get the value node from the dict in IR.
  std::stringstream script_buffer;
  script_buffer << "__import__('mindspore')._extends.parse.get_local_variable(" << value_node_key << ")";
  const std::string &script = script_buffer.str();
  const auto script_str = std::make_shared<StringImm>(script);

  // Build new CNode for value node.
  ValuePtrList keys({std::make_shared<StringImm>(value_node_key)});
  ValuePtrList values({std::make_shared<StringImm>(value_node_key)});
  const auto interpreted_cnode = fg->NewCNode({NewValueNode(prim::kPrimPyExecute), NewValueNode(script_str),
                                               NewValueNode(std::make_shared<ValueTuple>(keys)),
                                               NewValueNode(std::make_shared<ValueTuple>(values))});
  constexpr auto debug_recursive_level = 2;
  MS_LOG(DEBUG) << "original node: " << node->DebugString(debug_recursive_level)
                << ", interpreted_cnode: " << interpreted_cnode->DebugString(debug_recursive_level);
  interpreted_cnode->set_debug_info(node->debug_info());
  fg->ReplaceInOrder(node, interpreted_cnode);
  return interpreted_cnode;
}

AnfNodePtr ConvertInterpretedObjectToPyExecute(const FuncGraphPtr &fg, const ValuePtr &value, const AnfNodePtr &node) {
  const auto &interpreted_value = dyn_cast<parse::InterpretedObject>(value);
  if (interpreted_value == nullptr) {
    return nullptr;
  }
  return ConvertPyObjectToPyExecute(fg, interpreted_value->name(), interpreted_value->obj(), node);
}

namespace {
// Get the type from python type string, defined in Python module 'mindspore.common.dtype'.
TypePtr GetTypeFromString(const std::string &dtype) {
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  constexpr auto get_dtype_python_function = "get_dtype";
  auto type = python_adapter::CallPyModFn(mod, get_dtype_python_function, py::str(dtype));
  MS_LOG(DEBUG) << "type: " << type;
  if (py::isinstance<py::none>(type)) {
    return nullptr;
  }
  auto type_ptr = py::cast<TypePtr>(type);
  if (type_ptr == nullptr) {
    return nullptr;
  }
  return type_ptr->Clone();
}

std::string GetErrorFormatMessage(const AnfNodePtr &node, const std::string &comment) {
  std::stringstream err_buf;
  err_buf << "Wrong comment format for JIT type annotation: '" << comment
          << "'.\ne.g. '# @jit.typing: () -> tensor[int32]' or:"
          << "\n---\n\tdtype_var = ms.int32\n\t# @jit.typing: () -> tensor[{dtype_var}]\n\t...\n---\n\n"
          << trace::GetDebugInfo(node->debug_info());
  return err_buf.str();
}
}  // namespace

TypePtr GetJitAnnotationTypeFromComment(const AnfNodePtr &node, const FormatedVariableTypeFunc &format_type_func) {
  const auto &debug_info = trace::GetSourceCodeDebugInfo(node->debug_info());
  const auto &location = debug_info->location();
  if (location == nullptr) {
    MS_LOG(INFO) << "Location info is null, node: " << node->DebugString();
    return nullptr;
  }
  const auto &comments = location->comments();
  if (comments.empty()) {
    return nullptr;
  }
  // Only use the last comment.
  const auto &comment = comments.back();
  std::regex regex("^#\\s*@jit.typing\\s*:\\s*\\(\\)\\s*->\\s*([a-zA-Z0-9{}]+)?\\[?([a-zA-Z0-9{}]+)?\\]?$");
  std::smatch matched_results;
  if (std::regex_match(comment, matched_results, regex)) {
    constexpr auto container_match_count = 3;
    // Not match.
    if (matched_results.size() != container_match_count) {
      return nullptr;
    }
    const auto &container_type_str = matched_results[1].str();
    const auto &dtype_str = matched_results[container_match_count - 1].str();
    MS_LOG(DEBUG) << "matched_results: " << matched_results[0] << ", " << container_type_str << ", " << dtype_str;
    // Match nothing.
    if (container_type_str.empty()) {
      MS_LOG(EXCEPTION) << GetErrorFormatMessage(node, comment);
    }
    // Handle base type only.
    if (dtype_str.empty()) {
      TypePtr base_type = nullptr;
      // Handle dtype.
      if (container_type_str.front() == '{' && container_type_str.back() == '}') {  // Handle format variable type.
        if (!format_type_func) {
          MS_LOG(EXCEPTION) << GetErrorFormatMessage(node, comment);
        }
        constexpr auto excluded_size = 2;
        const auto &variable_base_type = container_type_str.substr(1, container_type_str.size() - excluded_size);
        // Find variable type.
        if (!variable_base_type.empty()) {
          base_type = format_type_func(variable_base_type);
          if (base_type == nullptr) {  // Not throw exception if not match any variable.
            return nullptr;
          }
        }
      } else {  // Handle string type.
        const auto &base_type_str = container_type_str;
        base_type = GetTypeFromString(base_type_str);
      }
      if (base_type == nullptr) {
        MS_LOG(EXCEPTION) << GetErrorFormatMessage(node, comment);
      }
      return base_type;
    }
    // Handle container type: tensor, list_ and tuple_.
    const auto &container_type = GetTypeFromString(container_type_str);
    if (container_type == nullptr) {
      MS_LOG(EXCEPTION) << GetErrorFormatMessage(node, comment);
    }
    if (!container_type->isa<Tuple>() && !container_type->isa<List>() && !container_type->isa<TensorType>()) {
      MS_LOG(EXCEPTION) << "JIT type annotation only support tensor/list_/tuple_, but got '" << container_type_str;
    }
    TypePtr dtype = nullptr;
    // Handle dtype.
    if (dtype_str.front() == '{' && dtype_str.back() == '}') {  // Handle format variable dtype.
      if (!format_type_func) {
        MS_LOG(EXCEPTION) << GetErrorFormatMessage(node, comment);
      }
      constexpr auto excluded_size = 2;
      const auto &variable_dtype = dtype_str.substr(1, dtype_str.size() - excluded_size);
      // Find variable dtype.
      if (!variable_dtype.empty()) {
        dtype = format_type_func(variable_dtype);
        if (dtype == nullptr) {  // Not throw exception if not match any variable.
          return nullptr;
        }
      }
    } else {  // Handle string dtype.
      dtype = GetTypeFromString(dtype_str);
    }
    if (dtype == nullptr) {
      MS_LOG(EXCEPTION) << GetErrorFormatMessage(node, comment);
    }
    if (container_type->isa<TensorType>()) {  // Handle tensor type.
      if (!dtype->isa<Number>()) {
        MS_LOG(EXCEPTION) << "Cannot get dtype for by input string: '" << dtype_str << "', for '" << container_type_str
                          << "'\n"
                          << trace::GetDebugInfo(node->debug_info());
      }
      container_type->cast<TensorTypePtr>()->set_element(dtype);
    } else if (container_type->isa<Tuple>() || container_type->isa<List>()) {  // Handle list_/tuple_ type.
      // To handle nested sequence later.
      if (!dtype->isa<Number>() && !dtype->isa<TensorType>()) {
        MS_LOG(EXCEPTION) << "Cannot get element type for by input string: '" << dtype_str << "', for '"
                          << container_type_str << "'\n"
                          << trace::GetDebugInfo(node->debug_info());
      }
      if (container_type->isa<Tuple>()) {
        container_type->cast<TuplePtr>()->set_elements(TypePtrList({dtype}));
      } else if (container_type->isa<List>()) {
        container_type->cast<ListPtr>()->set_elements(TypePtrList({dtype}));
      }
      return nullptr;  // Supports tuple_[...] / list_[...] later.
    }
    return container_type;
  }
  return nullptr;
}

namespace {
std::string ConvertRealStrToUnicodeStr(const std::string &target, size_t index) {
  std::stringstream script_buffer;
  script_buffer << kPyExecPrefix << std::to_string(index);
  std::vector<size_t> convert_pos;
  for (size_t i = 0; i < target.size(); ++i) {
    auto c = target[i];
    if (!std::isalnum(c)) {
      convert_pos.push_back(i);
    }
  }
  size_t start = 0;
  for (auto end : convert_pos) {
    std::string sub_non_convert = target.substr(start, end - start);
    if (sub_non_convert.size() != 0) {
      script_buffer << kUnderLine << sub_non_convert;
    }
    char sub_convert = target[end];
    std::stringstream hex_s;
    hex_s << kUnderLine << kHexPrefix << std::hex << static_cast<int>(sub_convert);
    script_buffer << hex_s.str();
    start = end + 1;
  }
  if (target.substr(start).size() != 0) {
    script_buffer << kUnderLine << target.substr(start);
  }
  script_buffer << kPyExecSuffix;
  auto unicode_str = script_buffer.str();
  MS_LOG(DEBUG) << "Get Unicode str: " << unicode_str;
  return script_buffer.str();
}

std::string ConvertUnicodeStrToRealStr(const std::string &target) {
  constexpr size_t non_script_size = 3;
  size_t sub_target_len = target.size() - std::strlen(kPyExecPrefix) - non_script_size;
  auto sub_target = target.substr(std::strlen(kPyExecPrefix) + 2, sub_target_len);
  std::stringstream script_buffer;
  sub_target = sub_target + "_";
  auto pos = sub_target.find("_");
  constexpr size_t base_16 = 16;
  while (pos != sub_target.npos) {
    auto cur_str = sub_target.substr(0, pos);
    if (cur_str.size() == 0) {
      break;
    }
    if (cur_str.substr(0, std::strlen(kHexPrefix)) == kHexPrefix) {
      script_buffer << char(std::stoi(cur_str, nullptr, base_16));
    } else {
      script_buffer << cur_str;
    }
    sub_target = sub_target.substr(pos + 1);
    pos = sub_target.find("_");
  }
  auto real_str = script_buffer.str();
  return real_str;
}

std::string ConvertToRealStr(const std::string &target) {
  if (target.find(kPyExecPrefix) == string::npos) {
    return target;
  }
  std::string real_str = "";
  size_t pos = 0;
  size_t start_pos, end_pos;
  while ((start_pos = target.find(kPyExecPrefix, pos)) != std::string::npos &&
         (end_pos = target.find(kPyExecSuffix, start_pos + std::strlen(kPyExecPrefix))) != std::string::npos) {
    if (start_pos > pos) {
      real_str += target.substr(pos, start_pos - pos);
    }
    auto substr = target.substr(start_pos, end_pos - start_pos + std::strlen(kPyExecSuffix));
    pos = end_pos + std::strlen(kPyExecSuffix);
    real_str += ConvertUnicodeStrToRealStr(substr);
  }
  if (pos < target.size()) {
    real_str += target.substr(pos);
  }
  return real_str;
}
}  // namespace

std::vector<std::string> GetPyExecuteInputFromUnicodeStr(const std::string &script) {
  // Get substr from script, substr start with kPyExecPrefix and end with kPyExecSuffix.
  std::vector<std::string> res;
  size_t pos = 0;
  size_t start_pos, end_pos;
  while ((start_pos = script.find(kPyExecPrefix, pos)) != string::npos &&
         (end_pos = script.find(kPyExecSuffix, start_pos + std::strlen(kPyExecPrefix))) != string::npos) {
    auto substr = script.substr(start_pos, end_pos - start_pos + std::strlen(kPyExecSuffix));
    pos = end_pos + std::strlen(kPyExecSuffix);
    res.push_back(substr);
    MS_LOG(DEBUG) << "Found input: " << substr;
  }
  return res;
}

AnfNodePtr GeneratePyExecuteNodeWithScriptSrc(const FuncGraphPtr &func_graph, const TypePtrList &types,
                                              const AnfNodePtrList &node_inputs, std::string script_str) {
  // Pack local parameters keys.
  auto input_str_list = GetPyExecuteInputFromUnicodeStr(script_str);
  if (input_str_list.empty()) {
    MS_LOG(EXCEPTION) << "Not found PyExecute input. script: " << script_str;
  }
  if (input_str_list.size() != node_inputs.size()) {
    if (script_str.find(kPyExecuteSlice) == string::npos) {
      MS_LOG(EXCEPTION) << "Input string size is " << input_str_list.size()
                        << " and input node size is: " << node_inputs.size() << ". Size not match.";
    }
    if (input_str_list.size() == 1 && types.size() > 1 && types[1]->isa<AnyType>()) {
      // The script is subscript, and slice input is PyExecute node.
      auto new_slice_str = ConvertRealStrToUnicodeStr("__slice__", 1);
      script_str = input_str_list[0] + "[" + new_slice_str + "]";
      input_str_list = {input_str_list[0], new_slice_str};
    }
  }

  std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < node_inputs.size(); ++i) {
    if (types[i]->isa<Slice>()) {
      (void)key_value_names_list.emplace_back(NewValueNode("__start__"));
      (void)key_value_names_list.emplace_back(NewValueNode("__stop__"));
      (void)key_value_names_list.emplace_back(NewValueNode("__step__"));
    } else {
      auto input_str = input_str_list[i];
      (void)key_value_names_list.emplace_back(NewValueNode(input_str));
    }
  }
  const auto key_value_name_tuple = func_graph->NewCNode(key_value_names_list);

  // Pack the local parameters values.
  std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < node_inputs.size(); ++i) {
    auto input = node_inputs[i];
    if (types[i]->isa<Slice>()) {
      auto start_node = func_graph->NewCNode({NewValueNode(prim::kPrimSliceGetItem), input, NewValueNode("start")});
      auto end_node = func_graph->NewCNode({NewValueNode(prim::kPrimSliceGetItem), input, NewValueNode("stop")});
      auto step_node = func_graph->NewCNode({NewValueNode(prim::kPrimSliceGetItem), input, NewValueNode("step")});
      (void)key_value_list.emplace_back(start_node);
      (void)key_value_list.emplace_back(end_node);
      (void)key_value_list.emplace_back(step_node);
    } else {
      (void)key_value_list.emplace_back(input);
    }
  }
  const auto key_value_tuple = func_graph->NewCNode(key_value_list);

  // Build the PyExecute node.
  auto ret_node = func_graph->NewCNodeInOrder(
    {NewValueNode(prim::kPrimPyExecute), NewValueNode(script_str), key_value_name_tuple, key_value_tuple});
  MS_LOG(DEBUG) << "Generate PyExecute node: " << ret_node;
  return ret_node;
}

void SetNodeExprSrc(const AnfNodePtr &node, const std::string &expr_src) {
  auto node_debug_info = node->debug_info();
  MS_EXCEPTION_IF_NULL(node_debug_info);
  auto node_location = node_debug_info->location();
  MS_EXCEPTION_IF_NULL(node_location);
  node_location->set_expr_src(expr_src);
  MS_LOG(DEBUG) << "Set new expr src '" << expr_src << "' for node: " << node->DebugString();
}

std::string GetNodeExprSrc(const AnfNodePtr &node) {
  auto node_debug_info = node->debug_info();
  MS_EXCEPTION_IF_NULL(node_debug_info);
  auto node_location = node_debug_info->location();
  MS_EXCEPTION_IF_NULL(node_location);
  return node_location->expr_src();
}

std::string GeneratePyExecuteScriptForBinOrComp(const std::string &left, const std::string &right,
                                                const std::string &op) {
  auto real_left = ConvertToRealStr(left);
  auto real_right = ConvertToRealStr(right);
  auto unicode_left = ConvertRealStrToUnicodeStr(real_left, 0);
  auto unicode_right = ConvertRealStrToUnicodeStr(real_right, 1);
  auto res = unicode_left + op + unicode_right;
  MS_LOG(DEBUG) << "Generate new script for BinOp/Compare: " << res;
  return res;
}

std::string GeneratePyExecuteScriptForUnary(const std::string &operand, const std::string &op) {
  auto real_operand = ConvertToRealStr(operand);
  auto unicode_operand = ConvertRealStrToUnicodeStr(real_operand, 0);
  auto res = op + " " + unicode_operand;
  MS_LOG(DEBUG) << "Generate new script for UnaryOp: " << res;
  return res;
}

std::string GeneratePyExecuteScriptForSubscript(const std::string &value, const std::string &slice, bool is_slice) {
  auto real_value = ConvertToRealStr(value);
  auto unicode_value = ConvertRealStrToUnicodeStr(real_value, 0);
  std::string res;
  if (is_slice) {
    res = unicode_value + kPyExecuteSlice;
  } else {
    auto unicode_slice = ConvertRealStrToUnicodeStr(slice, 1);
    res = unicode_value + "[" + unicode_slice + "]";
  }
  MS_LOG(DEBUG) << "Generate new script for SubScript: " << res;
  return res;
}
}  // namespace fallback

namespace raiseutils {
std::string MakeRaiseKey(const int index) { return "__internal_error_value" + std::to_string(index) + "__"; }

bool CheckIsStr(const AbstractBasePtr &abs) {
  auto scalar = abs->cast_ptr<abstract::AbstractScalar>();
  auto scalar_type = scalar->BuildType();
  MS_EXCEPTION_IF_NULL(scalar_type);
  if (scalar_type->IsSameTypeId(String::kTypeId)) {
    return true;
  }
  return false;
}

bool CheckNeedSymbol(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  bool need_symbol = false;
  if (abs->isa<abstract::AbstractScalar>()) {
    need_symbol = CheckIsStr(abs);
  } else if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_list = abs->cast_ptr<abstract::AbstractSequence>();
    const auto &elements = abs_list->elements();
    for (auto &element : elements) {
      MS_EXCEPTION_IF_NULL(element);
      if (element->isa<abstract::AbstractScalar>()) {
        need_symbol = CheckIsStr(element);
        if (need_symbol) {
          return need_symbol;
        }
      }
    }
  }
  return need_symbol;
}

std::string GetExceptionString(const AbstractBasePtr &arg, const AnfNodePtr &input,
                               const std::shared_ptr<KeyValueInfo> &key_value, bool need_symbol, bool need_comma) {
  std::string exception_str;
  MS_EXCEPTION_IF_NULL(arg);
  if (arg->isa<abstract::AbstractSequence>() && !IsPrimitiveCNode(input, prim::kPrimGetAttr)) {
    return GetTupleOrListString(arg, input, key_value, need_symbol, need_comma);
  } else if (arg->BuildValue() == kValueAny || arg->isa<abstract::AbstractTensor>() ||
             IsPrimitiveCNode(input, prim::kPrimGetAttr)) {
    exception_str = GetVariable(input, need_symbol, key_value, exception_str);
  } else if (arg->isa<abstract::AbstractDictionary>()) {
    MS_LOG(EXCEPTION) << "Dictionary type is currently not supporting";
  } else if (arg->isa<abstract::AbstractScalar>()) {
    // Process raise ValueError
    exception_str += GetScalarStringValue(arg);
  } else {
    MS_LOG(EXCEPTION) << "Unexpected abstract: " << arg->ToString();
  }
  return exception_str;
}

std::string GetVariable(const AnfNodePtr &input, const bool need_symbol, const std::shared_ptr<KeyValueInfo> &key_value,
                        std::string exception_str) {
  std::string key = MakeRaiseKey(key_value->num_str);
  key_value->num_str += 1;
  if (need_symbol) {
    exception_str = exception_str + "'+f'{" + key + "}'+'";
  } else {
    exception_str = exception_str + key;
  }
  (void)key_value->keys.emplace_back(NewValueNode(std::make_shared<StringImm>(key)));
  (void)key_value->values.emplace_back(input);
  return exception_str;
}

bool CheckHasVariable(const AbstractBasePtr &arg) {
  if (arg->isa<abstract::AbstractSequence>()) {
    auto arg_tuple = arg->cast_ptr<abstract::AbstractSequence>();
    MS_EXCEPTION_IF_NULL(arg_tuple);
    const auto &arg_tuple_elements = arg_tuple->elements();
    if (arg_tuple_elements.size() == 0) {
      MS_LOG(EXCEPTION) << "The arg_tuple_elements can't be empty.";
    }
    for (size_t index = 0; index < arg_tuple_elements.size(); ++index) {
      auto &element = arg_tuple_elements[index];
      if (CheckHasVariable(element)) {
        return true;
      }
    }
  } else if (arg->BuildValue() == kValueAny || arg->isa<abstract::AbstractTensor>()) {
    return true;
  }
  return false;
}

std::string GetTupleOrListString(const AbstractBasePtr &arg, const AnfNodePtr &input,
                                 const std::shared_ptr<KeyValueInfo> &key_value, bool need_symbol, bool need_comma) {
  MS_EXCEPTION_IF_NULL(arg);
  bool has_variable = CheckHasVariable(arg);
  std::string exception_str;
  bool is_tuple = arg->isa<abstract::AbstractTuple>();
  // Process raise ValueError("str")
  auto arg_tuple = arg->cast_ptr<abstract::AbstractSequence>();
  MS_EXCEPTION_IF_NULL(arg_tuple);
  const auto &arg_tuple_elements = arg_tuple->elements();
  if (!input->isa<CNode>() && has_variable) {
    return GetVariable(input, need_symbol, key_value, exception_str);
  }
  if (arg_tuple_elements.size() > 1 && !IsPrimitiveCNode(input, prim::kPrimJoinedStr)) {
    if (is_tuple) {
      exception_str += "(";
    } else {
      exception_str += "[";
    }
  }
  auto cnode = input->cast_ptr<CNode>();
  bool not_variable = !has_variable;
  if (has_variable) {
    not_variable = (arg->BuildValue() != kValueAny) || IsValueNode<prim::DoSignaturePrimitive>(cnode->input(0));
  }
  for (size_t index = 0; index < arg_tuple_elements.size(); ++index) {
    auto &element = arg_tuple_elements[index];
    if (has_variable) {
      const auto &inputs = cnode->inputs();
      if (arg_tuple_elements.size() >= cnode->inputs().size()) {
        MS_LOG(EXCEPTION) << "Size of cnode should be greater than arg_tuple_elements, "
                          << "but got cnode size: " << cnode->inputs().size()
                          << " arg_tuple_elements size: " << arg_tuple_elements.size();
      }
      auto inputs_in_tuple = inputs[index + 1];
      exception_str += GetExceptionString(element, inputs_in_tuple, key_value, need_symbol, need_comma);
    } else {
      exception_str += GetExceptionString(element, input, key_value, need_symbol, need_comma);
    }
    if (index != arg_tuple_elements.size() - 1 && need_comma && not_variable) {
      exception_str += ", ";
    }
  }
  if (arg_tuple_elements.size() > 1 && !IsPrimitiveCNode(input, prim::kPrimJoinedStr)) {
    if (is_tuple) {
      exception_str += ")";
    } else {
      exception_str += "]";
    }
  }
  return exception_str;
}

std::string GetExceptionType(const AbstractBasePtr &abs, const AnfNodePtr &node,
                             const std::shared_ptr<KeyValueInfo> &key_value, bool has_variable) {
  MS_EXCEPTION_IF_NULL(node);
  auto clt = GetValueNode<ClassTypePtr>(node);
  if (clt != nullptr) {
    const auto &class_name = clt->name();
    auto begin = class_name.find("'") + 1;
    auto end = class_name.substr(begin).find("'");
    auto class_type = class_name.substr(begin, end);
    return class_type;
  }
  std::string str;
  if (abs->isa<abstract::AbstractScalar>()) {
    auto scalar = abs->cast_ptr<abstract::AbstractScalar>();
    MS_EXCEPTION_IF_NULL(scalar);
    auto scalar_value = scalar->BuildValue();
    MS_EXCEPTION_IF_NULL(scalar_value);
    if (scalar_value->isa<StringImm>()) {
      str = GetValue<std::string>(scalar_value);
      if (GetValueNode<StringImmPtr>(node) == nullptr && has_variable) {
        (void)key_value->keys.emplace_back(NewValueNode(std::make_shared<StringImm>(str)));
        (void)key_value->values.emplace_back(node);
      }
      return str;
    }
  }
  MS_LOG(EXCEPTION) << "The abstract of exception type is not scalar: " << abs->ToString();
}

std::string GetScalarStringValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto scalar = abs->cast<abstract::AbstractScalarPtr>();
  MS_EXCEPTION_IF_NULL(scalar);
  auto scalar_value = scalar->BuildValue();
  return scalar_value->ToString();
}

bool HasVariableCondition(FuncGraphPtr cur_graph, FuncGraphPtr prev_graph) {
  if (cur_graph == nullptr) {
    return false;
  }
  if (cur_graph->is_tensor_condition_branch()) {
    return true;
  }
  auto cur_fg_map = cur_graph->func_graph_cnodes_index();
  for (auto &cur_fg_use : cur_fg_map) {
    auto temp_node = cur_fg_use.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(temp_node);
    if (prev_graph != nullptr && temp_node->func_graph() == prev_graph) {
      return false;
    }
    if (HasVariableCondition(temp_node->func_graph(), cur_graph)) {
      return true;
    }
  }
  if (HasVariableCondition(cur_graph->parent())) {
    return true;
  }
  return false;
}
}  // namespace raiseutils
}  // namespace mindspore
