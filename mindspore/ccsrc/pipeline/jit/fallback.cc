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
#include <string>

#include "include/common/utils/python_adapter.h"
#include "utils/log_adapter.h"
#include "ops/core_ops.h"
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
}  // namespace mindspore
