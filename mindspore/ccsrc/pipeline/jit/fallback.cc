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

#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"
#include "ops/core_ops.h"
#include "pipeline/jit/debug/trace.h"
#include "pipeline/jit/parse/resolve.h"

namespace mindspore {
AnfNodePtr ConvertInterpretedObjectToPyExecute(const FuncGraphPtr &fg, const ValuePtr &value, const AnfNodePtr &node) {
  const auto &interpreted_value = dyn_cast<parse::InterpretedObject>(value);
  if (interpreted_value == nullptr) {
    return nullptr;
  }
  const auto &value_node_value = interpreted_value->obj();

  auto value_node_key = interpreted_value->name();
  (void)value_node_key.erase(
    std::remove_if(value_node_key.begin(), value_node_key.end(), [](char c) { return !std::isalnum(c); }),
    value_node_key.end());

  // Set the value node into dict firstly.
  py::module mod = python_adapter::GetPyModule(parse::PYTHON_MOD_PARSE_MODULE);
  constexpr auto set_local_variable = "set_local_variable";
  MS_LOG(DEBUG) << set_local_variable << "(" << value_node_key << ", " << value_node_value << ")";
  (void)python_adapter::CallPyModFn(mod, set_local_variable, value_node_key, value_node_value);

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
  return type_ptr;
}

TypePtr GetJitAnnotationTypeFromComment(const AnfNodePtr &node) {
  const auto &debug_info = trace::GetSourceCodeDebugInfo(node->debug_info());
  const auto &location = debug_info->location();
  if (location == nullptr) {
    MS_LOG(WARNING) << "Location info is null, node: " << node->DebugString();
    return nullptr;
  }
  const auto &comments = location->comments();
  if (comments.empty()) {
    return nullptr;
  }
  // Only use the last comment.
  const auto &comment = comments.back();
  std::regex regex("^#\\s*@jit.typing\\s*:\\s*\\(\\)\\s*->\\s*(tensor|tuple_|list_)+\\[?([a-zA-Z0-9]+)?\\]?$");
  std::smatch matched_results;
  if (std::regex_match(comment, matched_results, regex)) {
    constexpr auto container_match_count = 3;
    // Not match.
    if (matched_results.size() != container_match_count) {
      return nullptr;
    }
    const auto &container_type_str = matched_results[1];
    const auto &dtype_str = matched_results[container_match_count - 1];
    MS_LOG(INFO) << "matched_results: " << matched_results[0] << ", " << container_type_str << ", " << dtype_str;
    // Match nothing.
    if (container_type_str.str().empty()) {
      return nullptr;
    }
    // Handle base type only.
    if (dtype_str.str().empty()) {
      const auto &base_type_str = container_type_str;
      const auto &base_type = GetTypeFromString(base_type_str);
      return base_type;
    }
    // Handle container type: tensor, list_ and tuple_.
    const auto &container_type = GetTypeFromString(container_type_str);
    if (container_type == nullptr) {
      return nullptr;
    }
    const auto &dtype = GetTypeFromString(dtype_str);
    if (dtype == nullptr) {
      return nullptr;
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
}  // namespace mindspore
