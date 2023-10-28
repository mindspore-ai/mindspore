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

#include "pipeline/jit/ps/fallback.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>
#include <utility>

#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/fallback.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/convert_utils_py.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/interpret_node_recorder.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "abstract/abstract_value.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace fallback {
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
          << "'.\ne.g. '# @jit.typing: () -> tensor_type[int32]' or:"
          << "\n---\n\tdtype_var = ms.int32\n\t# @jit.typing: () -> tensor_type[{dtype_var}]\n\t...\n---\n\n"
          << trace::GetDebugInfoStr(node->debug_info());
  return err_buf.str();
}

std::string ConvertUnicodeStrToRealStr(const std::string &target) {
  constexpr size_t non_script_size = 3;
  size_t sub_target_len = target.size() - std::strlen(kPyExecPrefix) - non_script_size;
  auto sub_target = target.substr(std::strlen(kPyExecPrefix) + 2, sub_target_len);
  std::stringstream script_buffer;
  sub_target = sub_target + "_";
  auto pos = sub_target.find("_");
  constexpr size_t base_16 = 16;
  while (pos != std::string::npos) {
    auto cur_str = sub_target.substr(0, pos);
    if (cur_str.size() == 0) {
      break;
    }
    if (cur_str.substr(0, std::strlen(kHexPrefix)) == kHexPrefix) {
      try {
        script_buffer << char(std::stoll(cur_str, nullptr, base_16));
      } catch (std::invalid_argument &) {
        MS_LOG(EXCEPTION) << "Invalid argument";
      } catch (std::out_of_range const &) {
        MS_LOG(EXCEPTION) << "Out of range";
      } catch (...) {
        MS_LOG(EXCEPTION) << "Other unexpected error";
      }
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
  size_t start_pos = target.find(kPyExecPrefix, pos);
  if (start_pos != std::string::npos) {
    size_t end_pos = target.find(kPyExecSuffix, start_pos + std::strlen(kPyExecPrefix));

    while (end_pos != std::string::npos) {
      if (start_pos > pos) {
        real_str += target.substr(pos, start_pos - pos);
      }
      auto substr = target.substr(start_pos, end_pos - start_pos + std::strlen(kPyExecSuffix));
      pos = end_pos + std::strlen(kPyExecSuffix);
      real_str += ConvertUnicodeStrToRealStr(substr);
      start_pos = target.find(kPyExecPrefix, pos);
      if (start_pos == std::string::npos) {
        break;
      }
      end_pos = target.find(kPyExecSuffix, start_pos + std::strlen(kPyExecPrefix));
    }
  }
  if (pos < target.size()) {
    real_str += target.substr(pos);
  }
  return real_str;
}

TypePtr HandleBaseTypeForAnnotation(const std::string &dtype_str, const std::string &container_type_str,
                                    const FormatedVariableTypeFunc &format_type_func, const AnfNodePtr &node,
                                    const std::string &comment) {
  if (!dtype_str.empty()) {
    return nullptr;
  }
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

std::pair<bool, TypePtr> GetDTypeFromDTypeStr(const std::string &dtype_str,
                                              const FormatedVariableTypeFunc &format_type_func, const AnfNodePtr &node,
                                              const std::string &comment) {
  TypePtr dtype = nullptr;
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
        return std::make_pair(false, nullptr);
      }
    }
  } else {  // Handle string dtype.
    dtype = GetTypeFromString(dtype_str);
  }
  return std::make_pair(true, dtype);
}

TypePtr HandleContainerTypeForAnnotation(const std::string &dtype_str, const std::string &container_type_str,
                                         const FormatedVariableTypeFunc &format_type_func, const AnfNodePtr &node,
                                         const std::string &comment) {
  const auto &container_type = GetTypeFromString(container_type_str);
  if (container_type == nullptr) {
    MS_LOG(EXCEPTION) << GetErrorFormatMessage(node, comment);
  }
  if (!container_type->isa<Tuple>() && !container_type->isa<List>() && !container_type->isa<TensorType>()) {
    MS_LOG(EXCEPTION) << "JIT type annotation only support tensor/list_/tuple_, but got '" << container_type_str;
  }

  auto [is_match, dtype] = GetDTypeFromDTypeStr(dtype_str, format_type_func, node, comment);
  if (!is_match) {
    return nullptr;
  }
  if (dtype == nullptr) {
    MS_LOG(EXCEPTION) << GetErrorFormatMessage(node, comment);
  }
  if (container_type->isa<TensorType>()) {  // Handle tensor type.
    if (!dtype->isa<Number>()) {
      MS_LOG(EXCEPTION) << "Cannot get dtype for by input string: '" << dtype_str << "', for '" << container_type_str
                        << "'\n"
                        << trace::GetDebugInfoStr(node->debug_info());
    }
    container_type->cast<TensorTypePtr>()->set_element(dtype);
  } else if (container_type->isa<Tuple>() || container_type->isa<List>()) {  // Handle list_/tuple_ type.
    // To handle nested sequence later.
    if (!dtype->isa<Number>() && !dtype->isa<TensorType>()) {
      MS_LOG(EXCEPTION) << "Cannot get element type for by input string: '" << dtype_str << "', for '"
                        << container_type_str << "'\n"
                        << trace::GetDebugInfoStr(node->debug_info());
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
}  // namespace

CNodePtr CreatePyExecuteCNode(const FuncGraphPtr &fg, const AnfNodePtr &script, const AnfNodePtr &keys,
                              const AnfNodePtr &values, const NodeDebugInfoPtr &debug_info) {
  const auto interpreted_cnode = fg->NewCNode({NewValueNode(prim::kPrimPyExecute), script, keys, values});
  if (debug_info != nullptr) {
    interpreted_cnode->set_debug_info(debug_info);
  }
  // Record the PyExecute node.
  InterpretNodeRecorder::GetInstance().PushPyExecuteNode(interpreted_cnode);
  return interpreted_cnode;
}

CNodePtr CreatePyExecuteCNode(const AnfNodePtr &orig_node, const AnfNodePtr &script, const AnfNodePtr &keys,
                              const AnfNodePtr &values) {
  const FuncGraphPtr &fg = orig_node->func_graph();
  if (fg == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The func graph is null. orig_node: " << orig_node->DebugString();
  }
  const auto interpreted_cnode = CreatePyExecuteCNode(fg, script, keys, values, orig_node->debug_info());
  return interpreted_cnode;
}

CNodePtr CreatePyExecuteCNodeInOrder(const FuncGraphPtr &fg, const AnfNodePtr &script, const AnfNodePtr &keys,
                                     const AnfNodePtr &values, const NodeDebugInfoPtr &debug_info) {
  const auto interpreted_cnode = fg->NewCNodeInOrder({NewValueNode(prim::kPrimPyExecute), script, keys, values});
  interpreted_cnode->set_debug_info(debug_info);
  // Record the PyExecute node.
  InterpretNodeRecorder::GetInstance().PushPyExecuteNode(interpreted_cnode);
  return interpreted_cnode;
}

CNodePtr CreatePyExecuteCNodeInOrder(const AnfNodePtr &orig_node, const AnfNodePtr &script, const AnfNodePtr &keys,
                                     const AnfNodePtr &values) {
  const FuncGraphPtr &fg = orig_node->func_graph();
  if (fg == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The func graph is null. orig_node: " << orig_node->DebugString();
  }
  const auto interpreted_cnode = CreatePyExecuteCNodeInOrder(fg, script, keys, values, orig_node->debug_info());
  return interpreted_cnode;
}

CNodePtr CreatePyInterpretCNodeInOrder(const FuncGraphPtr &fg, const std::string &script_text,
                                       const py::object &global_dict_obj, const AnfNodePtr &local_dict_node,
                                       const NodeDebugInfoPtr &debug_info) {
  auto script = std::make_shared<parse::Script>(script_text);
  auto script_node = NewValueNode(script);
  parse::PyObjectWrapperPtr global_dict_wrapper = std::make_shared<parse::InterpretedObject>(global_dict_obj);
  auto global_dict_node = NewValueNode(global_dict_wrapper);
  auto node =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimPyInterpret), script_node, global_dict_node, local_dict_node});
  node->set_debug_info(debug_info);
  return node;
}

void SetPyObjectToLocalVariable(const std::string &key, const py::object &value) {
  py::module mod = python_adapter::GetPyModule("mindspore.common._jit_fallback_utils");
  constexpr auto set_local_variable = "set_local_variable";
  MS_LOG(DEBUG) << set_local_variable << "([" << key << "]/" << key << ", " << value << ")";
  (void)python_adapter::CallPyModFn(mod, set_local_variable, key, value);
}

AnfNodePtr ConvertPyObjectToPyExecute(const FuncGraphPtr &fg, const std::string &key, const py::object value,
                                      const AnfNodePtr &node, bool replace) {
  auto value_node_key = ConvertRealStrToUnicodeStr(key, 0);
  // Set the value node into dict firstly.
  SetPyObjectToLocalVariable(value_node_key, value);

  // Get the value node from the dict in IR.
  std::stringstream script_buffer;
  script_buffer << "__import__('mindspore').common._jit_fallback_utils.get_local_variable(" << value_node_key << ")";
  const std::string &script = script_buffer.str();
  const auto script_str = std::make_shared<StringImm>(script);

  // Build new CNode for value node.
  ValuePtrList keys({std::make_shared<StringImm>(value_node_key)});
  ValuePtrList values({std::make_shared<StringImm>(value_node_key)});
  const auto interpreted_cnode =
    CreatePyExecuteCNode(fg, NewValueNode(script_str), NewValueNode(std::make_shared<ValueTuple>(keys)),
                         NewValueNode(std::make_shared<ValueTuple>(values)), node->debug_info());
  constexpr auto debug_recursive_level = 2;
  MS_LOG(DEBUG) << "original node: " << node->DebugString(debug_recursive_level)
                << ", interpreted_cnode: " << interpreted_cnode->DebugString(debug_recursive_level);
  if (replace) {
    fg->ReplaceInOrder(node, interpreted_cnode);
  }
  return interpreted_cnode;
}

AnfNodePtr ConvertPyObjectToPyInterpret(const FuncGraphPtr &fg, const std::string &key, const py::object value,
                                        const AnfNodePtr &node, bool replace) {
  auto value_node_key = ConvertRealStrToUnicodeStr(key, 0);
  // Set the value node into dict firstly.
  SetPyObjectToLocalVariable(value_node_key, value);

  // Build the script
  std::stringstream script_buffer;
  script_buffer << "__import__('mindspore').common._jit_fallback_utils.get_local_variable(" << value_node_key << ")";
  const std::string &script = script_buffer.str();
  auto script_str = std::make_shared<parse::Script>(script);
  auto script_node = NewValueNode(script_str);

  // Build the global dict.
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  constexpr auto python_get_dict = "get_global_params";
  const auto &global_dict = python_adapter::CallPyModFn(mod, python_get_dict);
  parse::PyObjectWrapperPtr interpreted_global_dict = std::make_shared<parse::InterpretedObject>(global_dict);
  auto global_dict_node = NewValueNode(interpreted_global_dict);

  // Build the local dict.
  ValuePtrList local_keys({std::make_shared<StringImm>(value_node_key)});
  ValuePtrList local_values({std::make_shared<StringImm>(value_node_key)});
  auto local_key_tuple = NewValueNode(std::make_shared<ValueTuple>(local_keys));
  auto local_value_tuple = NewValueNode(std::make_shared<ValueTuple>(local_values));
  auto local_dict_node = fg->NewCNode({NewValueNode(prim::kPrimMakeDict), local_key_tuple, local_value_tuple});
  auto prim = NewValueNode(prim::kPrimPyInterpret);
  auto interpret_node = fg->NewCNode({prim, script_node, global_dict_node, local_dict_node});
  if (replace) {
    fg->ReplaceInOrder(node, interpret_node);
  }
  return interpret_node;
}

AnfNodePtr ConvertMsClassObjectToPyExecute(const FuncGraphPtr &fg, const ValuePtr &value, const AnfNodePtr &node) {
  const auto &ms_class_value = dyn_cast<parse::MsClassObject>(value);
  if (ms_class_value == nullptr) {
    return nullptr;
  }
  return ConvertPyObjectToPyExecute(fg, ms_class_value->name(), ms_class_value->obj(), node, true);
}

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
  std::regex regex("^#\\s*@jit.typing\\s*:\\s*\\(\\)\\s*->\\s*([a-zA-Z0-9{}_]+)?\\[?([a-zA-Z0-9{}_]+)?\\]?$");
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
    auto base_type = HandleBaseTypeForAnnotation(dtype_str, container_type_str, format_type_func, node, comment);
    if (base_type != nullptr) {
      return base_type;
    }
    // Handle container type: tensor, list_ and tuple_.
    return HandleContainerTypeForAnnotation(dtype_str, container_type_str, format_type_func, node, comment);
  }
  return nullptr;
}

bool GetJitAnnotationSideEffectFromComment(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &debug_info = trace::GetSourceCodeDebugInfo(node->debug_info());
  const auto &location = debug_info->location();
  if (location == nullptr) {
    MS_LOG(DEBUG) << "Location info is null, node: " << node->DebugString();
    return false;
  }
  const auto &comments = location->comments();
  if (comments.empty()) {
    return false;
  }
  // Only use the last comment.
  const auto &comment = comments.back();
  std::regex regex("^#\\s*@jit.typing:\\s*side_effect");
  if (std::regex_match(comment, regex)) {
    return true;
  }
  return false;
}

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

std::vector<std::string> GetPyExecuteInputFromUnicodeStr(const std::string &script) {
  // Get substr from script, substr start with kPyExecPrefix and end with kPyExecSuffix.
  std::vector<std::string> res;
  size_t pos = 0;
  size_t start_pos = script.find(kPyExecPrefix, pos);
  if (start_pos == string::npos) {
    return res;
  }
  size_t end_pos = script.find(kPyExecSuffix, start_pos + std::strlen(kPyExecPrefix));
  while (end_pos != string::npos) {
    auto substr = script.substr(start_pos, end_pos - start_pos + std::strlen(kPyExecSuffix));
    pos = end_pos + std::strlen(kPyExecSuffix);
    res.push_back(substr);
    MS_LOG(DEBUG) << "Found input: " << substr;
    start_pos = script.find(kPyExecPrefix, pos);
    if (start_pos == string::npos) {
      return res;
    }
    end_pos = script.find(kPyExecSuffix, start_pos + std::strlen(kPyExecPrefix));
  }
  return res;
}

AnfNodePtr GeneratePyInterpretNodeWithScriptSrc(const FuncGraphPtr &func_graph, const TypePtrList &types,
                                                const AnfNodePtrList &node_inputs, std::string script_str) {
  // Pack local parameters keys.
  auto input_str_list = GetPyExecuteInputFromUnicodeStr(script_str);
  if (input_str_list.empty()) {
    MS_LOG(ERROR) << "Not found PyExecute input. script: " << script_str;
    return nullptr;
  }
  if (input_str_list.size() != node_inputs.size()) {
    if (script_str.find(kPyExecuteSlice) == string::npos) {
      MS_LOG(ERROR) << "Input string size is " << input_str_list.size()
                    << " and input node size is: " << node_inputs.size() << ". Size not match.";
      return nullptr;
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
  bool has_list_dict = false;
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
    if (!has_list_dict && (types[i]->isa<List>() || types[i]->isa<Dictionary>())) {
      has_list_dict = true;
    }
  }
  const auto key_value_tuple = func_graph->NewCNode(key_value_list);

  if (has_list_dict) {
    // If the multitype-fg has list or dict input, they may include inplace operation.
    // So, we directly convert it to PyExecute.
    auto res = CreatePyExecuteCNodeInOrder(func_graph, NewValueNode(script_str), key_value_name_tuple, key_value_tuple,
                                           key_value_name_tuple->debug_info());
    MS_LOG(DEBUG) << "Generate PyExecute node: " << res->DebugString();
    return res;
  }

  // Generate PyInterpret node with
  auto local_dict = func_graph->NewCNode({NewValueNode(prim::kPrimMakeDict), key_value_name_tuple, key_value_tuple});
  auto res =
    CreatePyInterpretCNodeInOrder(func_graph, script_str, py::dict(), local_dict, key_value_name_tuple->debug_info());
  MS_LOG(DEBUG) << "Generate PyInterpret node: " << res->DebugString();
  return res;
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

std::string GeneratePyInterpretScriptForBinOrComp(const std::string &left, const std::string &right,
                                                  const std::string &op) {
  auto real_left = ConvertToRealStr(left);
  auto real_right = ConvertToRealStr(right);
  auto unicode_left = ConvertRealStrToUnicodeStr(real_left, 0);
  auto unicode_right = ConvertRealStrToUnicodeStr(real_right, 1);
  auto res = unicode_left + " " + op + " " + unicode_right;
  MS_LOG(DEBUG) << "Generate new script for BinOp/Compare: " << res;
  return res;
}

std::string GeneratePyInterpretScriptForUnary(const std::string &operand, const std::string &op) {
  auto real_operand = ConvertToRealStr(operand);
  auto unicode_operand = ConvertRealStrToUnicodeStr(real_operand, 0);
  auto res = op + " " + unicode_operand;
  MS_LOG(DEBUG) << "Generate new script for UnaryOp: " << res;
  return res;
}

std::string GeneratePyInterpretScriptForSubscript(const std::string &value, const std::string &slice, bool is_slice) {
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

std::string GeneratePyInterpretScriptForCallNode(const AnfNodePtr &call_node, const std::string &name_id) {
  auto call_cnode = call_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(call_cnode);
  std::string new_expr_src = name_id + "(";
  for (size_t index = 1; index < call_cnode->inputs().size(); ++index) {
    auto call_input = call_cnode->input(index);
    auto str = GetNodeExprSrc(call_input);
    auto real_str = ConvertToRealStr(str);
    auto unicode_str = ConvertRealStrToUnicodeStr(real_str, index);
    if (index != call_cnode->inputs().size() - 1) {
      new_expr_src += unicode_str + ", ";
    } else {
      new_expr_src += unicode_str + ")";
    }
  }
  return new_expr_src;
}

bool ContainsSequenceAnyType(const AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return false;
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    auto seq_abs = abs->cast_ptr<abstract::AbstractSequence>();
    MS_EXCEPTION_IF_NULL(seq_abs);
    if (seq_abs->dynamic_len()) {
      auto element_abs = seq_abs->dynamic_len_element_abs();
      if (ContainsSequenceAnyType(element_abs)) {
        return true;
      }
    } else {
      const auto &elements = seq_abs->elements();
      for (size_t item_index = 0; item_index < elements.size(); ++item_index) {
        const auto &item_abs = elements[item_index];
        if (ContainsSequenceAnyType(item_abs)) {
          return true;
        }
      }
    }
  }
  return abs->isa<abstract::AbstractAny>();
}

py::object GeneratePyObj(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractList>()) {
    auto abs_list = abs->cast<abstract::AbstractListPtr>();
    if (HasObjInExtraInfoHolder(abs_list)) {
      return GetObjFromExtraInfoHolder(abs_list);
    }
    py::list ret = py::list(abs_list->size());
    const auto &elements = abs_list->elements();
    for (size_t i = 0; i < elements.size(); ++i) {
      ret[i] = GeneratePyObj(elements[i]);
    }
    return ret;
  } else if (abs->isa<abstract::AbstractTuple>()) {
    auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
    py::tuple ret = py::tuple(abs_tuple->size());
    const auto &elements = abs_tuple->elements();
    for (size_t i = 0; i < elements.size(); ++i) {
      ret[i] = GeneratePyObj(elements[i]);
    }
    return ret;
  } else if (abs->isa<abstract::AbstractDictionary>()) {
    auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    py::dict ret = py::dict();
    const auto &key_value_pairs = abs_dict->elements();
    for (size_t i = 0; i < key_value_pairs.size(); ++i) {
      py::object key = GeneratePyObj(key_value_pairs[i].first);
      // The key should be unique.
      key = py::isinstance<py::none>(key) ? py::str(std::to_string(i)) : key;
      ret[key] = GeneratePyObj(key_value_pairs[i].second);
    }
    return ret;
  }
  return ValueToPyData(abs->BuildValue());
}

bool EnableFallbackListDictInplace() {
  const auto allow_fallback_runtime = (fallback::GetJitSyntaxLevel() >= kCompatible);
  static const auto allow_inplace_ops = common::GetEnv("MS_DEV_FALLBACK_SUPPORT_LIST_DICT_INPLACE") != "0";
  return allow_fallback_runtime && allow_inplace_ops;
}

void AttachPyObjToExtraInfoHolder(const abstract::AbstractBasePtr &abs, const py::object &obj, bool create_in_graph) {
  MS_EXCEPTION_IF_NULL(abs);
  constexpr auto py_object_key = "py_obj_key";
  constexpr auto create_in_graph_key = "create_in_graph_key";
  if (abs->isa<abstract::AbstractList>()) {
    auto abs_list = abs->cast<abstract::AbstractListPtr>();
    abs_list->SetData<py::object>(py_object_key, std::make_shared<py::object>(obj));
    abs_list->SetData<bool>(create_in_graph_key, std::make_shared<bool>(create_in_graph));
    return;
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    abs_dict->SetData<py::object>(py_object_key, std::make_shared<py::object>(obj));
    abs_dict->SetData<bool>(create_in_graph_key, std::make_shared<bool>(create_in_graph));
    return;
  }
  MS_INTERNAL_EXCEPTION(TypeError) << "The abstract should be a ExtraInfoHolder but got : " << abs->ToString();
}

py::object GetObjFromExtraInfoHolder(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  constexpr auto py_object_key = "py_obj_key";
  if (abs->isa<abstract::AbstractList>()) {
    auto abs_list = abs->cast<abstract::AbstractListPtr>();
    return *abs_list->GetData<py::object>(py_object_key);
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    return *abs_dict->GetData<py::object>(py_object_key);
  }
  MS_INTERNAL_EXCEPTION(TypeError) << "The abstract should be a ExtraInfoHolder but got : " << abs->ToString();
}

bool GetCreateInGraphFromExtraInfoHolder(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  constexpr auto create_in_graph_key = "create_in_graph_key";
  if (abs->isa<abstract::AbstractList>()) {
    auto abs_list = abs->cast<abstract::AbstractListPtr>();
    return *abs_list->GetData<bool>(create_in_graph_key);
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    return *abs_dict->GetData<bool>(create_in_graph_key);
  }
  MS_INTERNAL_EXCEPTION(TypeError) << "The abstract should be a ExtraInfoHolder but got : " << abs->ToString();
}

bool HasObjInExtraInfoHolder(const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  constexpr auto py_object_key = "py_obj_key";
  if (abs->isa<abstract::AbstractList>()) {
    auto abs_list = abs->cast<abstract::AbstractListPtr>();
    return abs_list->HasData(py_object_key);
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    return abs_dict->HasData(py_object_key);
  }
  return false;
}

// Nested attach list and dict object to corresponding abstract.
void AttachPyObjToAbs(const AbstractBasePtr &abs, const py::object &obj, bool create_in_graph) {
  if (!EnableFallbackListDictInplace()) {
    return;
  }
  if (abs->isa<abstract::AbstractNamedTuple>()) {
    return;
  }
  if (!abs->isa<abstract::AbstractSequence>() && !abs->isa<abstract::AbstractDictionary>()) {
    return;
  }
  constexpr auto cell_list_attr = "__cell_as_list__";
  constexpr auto cell_dict_attr = "__cell_as_dict__";
  if (py::hasattr(obj, cell_list_attr) || py::hasattr(obj, cell_dict_attr)) {
    // CellList and CellDict do not support inplace operations, do not need to attach python object.
    return;
  }
  if (abs->isa<abstract::AbstractList>()) {
    MS_LOG(DEBUG) << "Attach list python" << obj << " to abstract: " << abs->ToString();
    if (!py::isinstance<py::list>(obj)) {
      MS_INTERNAL_EXCEPTION(TypeError) << "Object should be list but got: " << py::str(obj);
    }
    auto abs_list = abs->cast<abstract::AbstractListPtr>();
    AttachPyObjToExtraInfoHolder(abs_list, obj, create_in_graph);
    auto list_obj = py::list(obj);
    for (size_t i = 0; i < abs_list->size(); ++i) {
      auto element_abs = abs_list->elements()[i];
      auto element_obj = list_obj[i];
      AttachPyObjToAbs(element_abs, element_obj, create_in_graph);
    }
    return;
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    if (!py::isinstance<py::dict>(obj)) {
      MS_INTERNAL_EXCEPTION(TypeError) << "Object should be dict but got: " << py::str(obj);
    }
    auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    MS_LOG(DEBUG) << "Attach dict python" << obj << " to abstract: " << abs->ToString();
    AttachPyObjToExtraInfoHolder(abs_dict, obj, create_in_graph);
    auto dict_obj = py::dict(obj);
    auto key_list_obj = py::list(obj);
    const auto &key_value_pairs = abs_dict->elements();
    for (size_t i = 0; i < key_value_pairs.size(); ++i) {
      auto value_abs = key_value_pairs[i].second;
      auto value_obj = dict_obj[key_list_obj[i]];
      AttachPyObjToAbs(value_abs, value_obj, create_in_graph);
    }
    return;
  }
  auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
  if (!py::isinstance<py::tuple>(obj)) {
    MS_INTERNAL_EXCEPTION(TypeError) << "Object should be tuple but got: " << py::str(obj);
  }
  auto tuple_obj = py::tuple(obj);
  for (size_t i = 0; i < abs_tuple->size(); ++i) {
    auto element_abs = abs_tuple->elements()[i];
    auto element_obj = tuple_obj[i];
    AttachPyObjToAbs(element_abs, element_obj, create_in_graph);
  }
}

std::string GetPyObjectPtrStr(const py::object &obj) {
  std::stringstream ss;
  ss << obj.ptr();
  return ss.str();
}

void SetPyObjectToNode(const AnfNodePtr &node, const py::object &obj) {
  MS_EXCEPTION_IF_NULL(node);
  if (!EnableFallbackListDictInplace()) {
    return;
  }
  constexpr auto py_obj_str = "__py_object__";
  if (py::isinstance<py::list>(obj)) {
    node->set_user_data<py::list>(py_obj_str, std::make_shared<py::list>(py::list(obj)));
  } else if (py::isinstance<py::tuple>(obj)) {
    node->set_user_data<py::tuple>(py_obj_str, std::make_shared<py::tuple>(py::tuple(obj)));
  } else if (py::isinstance<py::dict>(obj)) {
    node->set_user_data<py::dict>(py_obj_str, std::make_shared<py::dict>(py::dict(obj)));
  }
}

bool HasPyObjectInNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  constexpr auto py_obj_str = "__py_object__";
  return node->has_user_data(py_obj_str);
}

py::object GetPyObjectFromNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  constexpr auto py_obj_str = "__py_object__";
  return *node->user_data<py::object>(py_obj_str);
}

// Convert some CNode to PyExectue, eg:
// isinstance(xxx.asnumpy(), np.ndarray)  -- > PyExectue("isinstance(arg1, arg2)", local_keys, local_values)
AnfNodePtr ConvertCNodeToPyExecuteForPrim(const CNodePtr &cnode, const string &name) {
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &fg = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(fg);
  std::string script = name + "(";
  std::string internal_arg;
  size_t arg_nums = cnode->inputs().size() - 1;
  std::vector<AnfNodePtr> keys_tuple_node_inputs{NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AnfNodePtr> values_tuple_node_inputs{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t index = 1; index < arg_nums; ++index) {
    internal_arg = fallback::ConvertRealStrToUnicodeStr(name, index);
    script = script + internal_arg + ", ";
    auto key_node = NewValueNode(std::make_shared<StringImm>(internal_arg));
    auto value_node = cnode->input(index);
    (void)keys_tuple_node_inputs.emplace_back(key_node);
    (void)values_tuple_node_inputs.emplace_back(value_node);
  }
  string last_input = fallback::ConvertRealStrToUnicodeStr(name, arg_nums);
  script = script + last_input + ")";
  (void)keys_tuple_node_inputs.emplace_back(NewValueNode(std::make_shared<StringImm>(last_input)));
  (void)values_tuple_node_inputs.emplace_back(cnode->input(arg_nums));
  auto script_node = NewValueNode(std::make_shared<StringImm>(script));
  auto keys_tuple_node = fg->NewCNodeInOrder(keys_tuple_node_inputs);
  auto values_tuple_node = fg->NewCNodeInOrder(values_tuple_node_inputs);
  auto pyexecute_node =
    fallback::CreatePyExecuteCNodeInOrder(fg, script_node, keys_tuple_node, values_tuple_node, cnode->debug_info());
  MS_LOG(DEBUG) << "Convert: " << cnode->DebugString() << " -> " << pyexecute_node->DebugString();
  pyexecute_node->set_debug_info(cnode->debug_info());
  return pyexecute_node;
}

AnfNodePtr GenerateOnesOrZerosLikeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input,
                                       const std::string &type) {
  auto str_value = std::make_shared<StringImm>("__import__('mindspore').common._utils." + type + "(key)");
  auto script_node = NewValueNode(str_value);

  std::vector<ValuePtr> key_value{std::make_shared<StringImm>("key")};
  const auto key_tuple = std::make_shared<ValueTuple>(key_value);
  auto key_tuple_node = NewValueNode(key_tuple);
  auto values_tuple_node = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), input});
  AnfNodePtr template_node = fallback::CreatePyExecuteCNodeInOrder(func_graph, script_node, key_tuple_node,
                                                                   values_tuple_node, values_tuple_node->debug_info());
  return template_node;
}
}  // namespace fallback

namespace raiseutils {
namespace {
bool CheckIsStr(const AbstractBasePtr &abs) {
  auto scalar = abs->cast_ptr<abstract::AbstractScalar>();
  MS_EXCEPTION_IF_NULL(scalar);
  auto scalar_type = scalar->BuildType();
  MS_EXCEPTION_IF_NULL(scalar_type);
  if (scalar_type->IsSameTypeId(String::kTypeId)) {
    return true;
  }
  return false;
}

std::string GetScalarStringValue(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  auto scalar = abs->cast<abstract::AbstractScalarPtr>();
  MS_EXCEPTION_IF_NULL(scalar);
  auto scalar_value = scalar->BuildValue();
  return scalar_value->ToString();
}

std::string GetVariable(const AnfNodePtr &input, const std::shared_ptr<KeyValueInfo> &key_value,
                        const std::string &exception_str, bool need_symbol) {
  std::string key = MakeRaiseKey(key_value->num_str);
  std::stringstream script_buffer;
  key_value->num_str += 1;
  if (need_symbol) {
    script_buffer << exception_str << "'+f'{" << key << "}'+'";
  } else {
    script_buffer << exception_str << key;
  }
  (void)key_value->keys.emplace_back(NewValueNode(std::make_shared<StringImm>(key)));
  (void)key_value->values.emplace_back(input);
  return script_buffer.str();
}

std::string GetTupleOrListString(const AbstractBasePtr &arg, const AnfNodePtr &input,
                                 const std::shared_ptr<KeyValueInfo> &key_value, bool need_symbol, bool need_comma) {
  MS_EXCEPTION_IF_NULL(arg);
  bool has_variable = CheckHasVariable(arg);
  std::stringstream exception_str;
  bool is_tuple = arg->isa<abstract::AbstractTuple>();
  // Process raise ValueError("str")
  auto arg_tuple = arg->cast_ptr<abstract::AbstractSequence>();
  MS_EXCEPTION_IF_NULL(arg_tuple);
  const auto &arg_tuple_elements = arg_tuple->elements();
  if (!input->isa<CNode>() && has_variable) {
    return GetVariable(input, key_value, exception_str.str(), need_symbol);
  }
  if (arg_tuple_elements.size() > 1 && !IsPrimitiveCNode(input, prim::kPrimJoinedStr)) {
    if (is_tuple) {
      exception_str << "(";
    } else {
      exception_str << "[";
    }
  }
  if (has_variable) {
    auto cnode = input->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(cnode);
    bool not_variable = (arg->BuildValue() != kValueAny) || IsValueNode<prim::DoSignaturePrimitive>(cnode->input(0));
    for (size_t index = 0; index < arg_tuple_elements.size(); ++index) {
      auto &element = arg_tuple_elements[index];
      const auto &inputs = cnode->inputs();
      if (arg_tuple_elements.size() >= cnode->inputs().size()) {
        MS_LOG(EXCEPTION) << "Size of cnode should be greater than arg_tuple_elements, "
                          << "but got cnode size: " << cnode->inputs().size()
                          << " arg_tuple_elements size: " << arg_tuple_elements.size();
      }
      auto inputs_in_tuple = inputs[index + 1];
      exception_str << GetExceptionString(element, inputs_in_tuple, key_value, need_symbol, need_comma);
      if (index != arg_tuple_elements.size() - 1 && need_comma && not_variable) {
        exception_str << ", ";
      }
    }
  } else {
    for (size_t index = 0; index < arg_tuple_elements.size(); ++index) {
      auto &element = arg_tuple_elements[index];
      exception_str << GetExceptionString(element, input, key_value, need_symbol, need_comma);
      if (index != arg_tuple_elements.size() - 1 && need_comma) {
        exception_str << ", ";
      }
    }
  }
  if (arg_tuple_elements.size() > 1 && !IsPrimitiveCNode(input, prim::kPrimJoinedStr)) {
    if (is_tuple) {
      exception_str << ")";
    } else {
      exception_str << "]";
    }
  }
  return exception_str.str();
}
}  // namespace

std::string MakeRaiseKey(const int index) { return "__internal_error_value" + std::to_string(index) + "__"; }

bool CheckNeedSymbol(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  bool need_symbol = false;
  if (abs->isa<abstract::AbstractScalar>()) {
    need_symbol = CheckIsStr(abs);
  } else if (abs->isa<abstract::AbstractSequence>()) {
    auto abs_list = abs->cast_ptr<abstract::AbstractSequence>();
    MS_EXCEPTION_IF_NULL(abs_list);
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
    exception_str = GetVariable(input, key_value, exception_str, need_symbol);
  } else if (arg->isa<abstract::AbstractDictionary>()) {
    MS_LOG(EXCEPTION) << "Dictionary type is currently not supporting";
  } else if (arg->isa<abstract::AbstractScalar>()) {
    // Process raise ValueError
    exception_str += GetScalarStringValue(arg);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unexpected abstract: " << arg->ToString();
  }
  return exception_str;
}

bool CheckHasVariable(const AbstractBasePtr &arg) {
  if (arg->isa<abstract::AbstractSequence>()) {
    auto arg_tuple = arg->cast_ptr<abstract::AbstractSequence>();
    MS_EXCEPTION_IF_NULL(arg_tuple);
    const auto &arg_tuple_elements = arg_tuple->elements();
    if (arg_tuple_elements.size() == 0) {
      MS_LOG(INTERNAL_EXCEPTION) << "The arg_tuple_elements can't be empty.";
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

namespace {
bool HasVariableCondition(const FuncGraphPtr &cur_graph, std::vector<FuncGraphPtr> *prev_graph) {
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
    if (std::find(prev_graph->begin(), prev_graph->end(), cur_graph) != prev_graph->end()) {
      continue;
    }
    prev_graph->push_back(cur_graph);
    if (HasVariableCondition(temp_node->func_graph(), prev_graph)) {
      return true;
    }
  }
  if (HasVariableCondition(cur_graph->parent(), prev_graph)) {
    return true;
  }
  return false;
}
}  // namespace

bool HasVariableCondition(const FuncGraphPtr &cur_graph) {
  std::vector<FuncGraphPtr> prev_graph;
  return HasVariableCondition(cur_graph, &prev_graph);
}
}  // namespace raiseutils
}  // namespace mindspore
