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

#include "frontend/optimizer/py_interpret_to_execute.h"

#include <memory>
#include <string>
#include <utility>
#include <unordered_map>

#include "abstract/abstract_function.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "utils/symbolic.h"
#include "pipeline/jit/parse/resolve.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
namespace {
py::object CallPythonPushGlobalParams(const py::object &dict) {
  constexpr auto python_mod_parse = "mindspore._extends.parse";  // The same as PYTHON_MOD_PARSE_MODULE[]
  py::module mod = python_adapter::GetPyModule(python_mod_parse);
  constexpr auto python_merge_dict = "merge_global_params";
  return python_adapter::CallPyModFn(mod, python_merge_dict, dict);
}
}  // namespace

// Convert PyInterpret into PyExecute:
//   PyInterpret(script, global_dict, local_dict)
//   -->
//   PyExecute(script, local_dict_keys, local_dict_values),
//   with side-effect operation:
//     Push global_dict into global parameters list.
//     (So it requires no same key name.)
bool PyInterpretToExecute(const pipeline::ResourcePtr &resource) {
  auto manager = resource->manager();
  const auto &all_nodes = manager->all_nodes();
  auto transact = manager->Transact();
  constexpr auto input_index_one = 1;
  constexpr auto input_index_two = 2;
  constexpr auto input_index_three = 3;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimPyInterpret)) {
      continue;
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_LOG(DEBUG) << "cnode: " << cnode->DebugString();
    auto new_cnode = std::make_shared<CNode>(*cnode);
    new_cnode->set_input(0, NewValueNode(prim::kPrimPyExecute));

    if (!IsValueNode<parse::Script>(cnode->input(input_index_one))) {
      MS_LOG(EXCEPTION) << "The first input should be a Script, but got "
                        << cnode->input(input_index_one)->DebugString();
    }
    const auto &script = GetValueNode<std::shared_ptr<parse::Script>>(cnode->input(input_index_one));
    const auto &script_str = script->script();
    const auto &script_strimm_node = NewValueNode(std::make_shared<StringImm>(script_str));
    new_cnode->set_input(input_index_one, script_strimm_node);

    if (!IsValueNode<ValueDictionary>(cnode->input(input_index_two))) {
      MS_LOG(EXCEPTION) << "The second input should be a dictionary, but got "
                        << cnode->input(input_index_two)->DebugString();
    }
    const auto &global_dict = GetValueNode<ValueDictionaryPtr>(cnode->input(input_index_two));
    py::object py_global_dict = ValueToPyData(global_dict);
    MS_LOG(DEBUG) << "py_global_dict: " << py::str(py_global_dict);
    CallPythonPushGlobalParams(py_global_dict);

    if (!IsPrimitiveCNode(cnode->input(input_index_three), prim::kPrimMakeDict)) {
      MS_LOG(EXCEPTION) << "The 3rd input should be a dictionary, but got "
                        << cnode->input(input_index_three)->DebugString();
    }
    const auto &local_dict_cnode = dyn_cast<CNode>(cnode->input(input_index_three));
    MS_EXCEPTION_IF_NULL(local_dict_cnode);
    const auto &local_dict_keys = local_dict_cnode->input(input_index_one);
    const auto &local_dict_values = local_dict_cnode->input(input_index_two);
    if (!IsValueNode<ValueTuple>(local_dict_keys) || !IsPrimitiveCNode(local_dict_values, prim::kPrimMakeTuple)) {
      MS_LOG(EXCEPTION) << "The dictionary's keys and values should be a tuple, but got "
                        << local_dict_cnode->DebugString();
    }
    new_cnode->set_input(input_index_two, local_dict_keys);
    new_cnode->set_input(input_index_three, local_dict_values);
    (void)transact.Replace(cnode, new_cnode);
  }
  transact.Commit();
  return true;
}
}  // namespace opt
}  // namespace mindspore
