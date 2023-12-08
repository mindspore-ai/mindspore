/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/abstract_function.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "utils/interpret_node_recorder.h"
#include "utils/symbolic.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/fallback.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
namespace {
CNodePtr Transform(const CNodePtr &cnode, const FuncGraphManagerPtr &manager);
AnfNodePtr NewValueNodeWithAbstract(const ValuePtr &value) {
  auto value_node = NewValueNode(value);
  value_node->set_abstract(value->ToAbstract());
  return value_node;
}

AnfNodePtr FuncGraphToPyData(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    return node;
  }
  auto value_node = node->cast_ptr<ValueNode>();
  auto value = value_node->value();
  if (value->IsFromTypeId(FuncGraph::kTypeId)) {
    auto fg = value->cast_ptr<FuncGraph>();
    MS_EXCEPTION_IF_NULL(fg);
    auto wrapper_obj = fg->python_obj();
    if (wrapper_obj != nullptr && wrapper_obj->isa<parse::PyObjectWrapper>()) {
      return NewValueNode(
        std::make_shared<parse::InterpretedObject>(wrapper_obj->cast_ptr<parse::PyObjectWrapper>()->obj()));
    }
  }
  return node;
}

std::vector<AnfNodePtr> ConvertValueTupleToList(const AnfNodePtr &node) {
  if ((!IsValueNode<ValueTuple>(node) && !IsPrimitiveCNode(node, prim::kPrimMakeTuple))) {
    MS_LOG(INTERNAL_EXCEPTION) << "The dictionary's keys and values should be a tuple, but got " << node->DebugString();
  }
  std::vector<AnfNodePtr> node_list;
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    auto cnode = node->cast_ptr<CNode>();
    auto inputs = cnode->inputs();
    std::copy(inputs.begin() + 1, inputs.end(), std::back_inserter(node_list));
    return node_list;
  }
  auto tuple_value = GetValueNode<ValueTuplePtr>(node);
  auto value_list = tuple_value->value();
  std::transform(value_list.begin(), value_list.end(), std::back_inserter(node_list),
                 [](const ValuePtr &value) -> AnfNodePtr { return NewValueNodeWithAbstract(value); });
  return node_list;
}

std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> UnzipGlobalDict(const AnfNodePtr &dict_node) {
  MS_EXCEPTION_IF_NULL(dict_node);
  if (!dict_node->isa<ValueNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "The PyInterpret global dict should be a InterpretedObject value node, but got "
                               << dict_node->DebugString();
  }
  auto interpreted_object = GetValueNode<parse::InterpretedObjectPtr>(dict_node);
  MS_EXCEPTION_IF_NULL(interpreted_object);
  ValuePtr converted_value = nullptr;
  if (!parse::ConvertData(interpreted_object->obj(), &converted_value)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Convert data failed";
  }
  MS_EXCEPTION_IF_NULL(converted_value);
  auto dict_value = dyn_cast<ValueDictionary>(converted_value);
  if (dict_value == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "The PyInterpret local dict or global dict should be a dictionary, but got "
                               << dict_value->ToString();
  }
  std::vector<AnfNodePtr> keys;
  std::vector<AnfNodePtr> values;
  for (auto item : dict_value->value()) {
    (void)keys.emplace_back(NewValueNodeWithAbstract(item.first));
    (void)values.emplace_back(NewValueNodeWithAbstract(item.second));
  }
  return std::make_pair(keys, values);
}

std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> UnzipLocalDict(const AnfNodePtr &dict_node) {
  MS_EXCEPTION_IF_NULL(dict_node);
  if (dict_node->isa<ValueNode>()) {
    auto dict_value = GetValueNode<ValueDictionaryPtr>(dict_node);
    if (dict_value == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "The PyInterpret local dict should be a dictionary, but got "
                                 << dict_node->DebugString();
    }
    std::vector<AnfNodePtr> keys;
    std::vector<AnfNodePtr> values;
    for (auto item : dict_value->value()) {
      (void)keys.emplace_back(NewValueNodeWithAbstract(item.first));
      (void)values.emplace_back(NewValueNodeWithAbstract(item.second));
    }
    return std::make_pair(keys, values);
  }

  if (!IsPrimitiveCNode(dict_node, prim::kPrimMakeDict)) {
    MS_LOG(INTERNAL_EXCEPTION) << "The PyInterpret local dict should be a dictionary, but got "
                               << dict_node->DebugString();
  }
  auto make_dict_node = dict_node->cast_ptr<CNode>();
  constexpr auto kMakeDictKeysInputIndex = 1;
  constexpr auto kMakeDictValueInputIndex = 2;
  auto keys_input = make_dict_node->input(kMakeDictKeysInputIndex);
  auto values_input = make_dict_node->input(kMakeDictValueInputIndex);

  auto keys_list = ConvertValueTupleToList(keys_input);
  auto values_list = ConvertValueTupleToList(values_input);
  return std::make_pair(keys_list, values_list);
}

std::set<std::string> GetLocalKeySet(const std::vector<AnfNodePtr> &key_node_list) {
  std::set<std::string> key_set;
  std::transform(key_node_list.begin(), key_node_list.end(), std::inserter(key_set, key_set.begin()),
                 [](const AnfNodePtr &node) -> std::string {
                   auto abs = node->abstract();
                   MS_EXCEPTION_IF_NULL(abs);
                   auto value = abs->BuildValue();
                   MS_EXCEPTION_IF_NULL(value);
                   return GetValue<std::string>(value);
                 });
  return key_set;
}

// Merge global dict to local dict and return merged key and value
std::pair<AnfNodePtr, AnfNodePtr> MergeGlobalDictToLocal(const AnfNodePtr &global_dict_node,
                                                         const AnfNodePtr &local_dict_node,
                                                         const FuncGraphPtr &func_graph,
                                                         const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(global_dict_node);
  MS_EXCEPTION_IF_NULL(local_dict_node);
  auto [global_keys, global_values] = UnzipGlobalDict(global_dict_node);
  auto [local_keys, local_values] = UnzipLocalDict(local_dict_node);

  auto local_dict_keys_set = GetLocalKeySet(local_keys);

  std::vector<AnfNodePtr> local_keys_inputs{NewValueNode(prim::kPrimMakeTuple)};
  std::vector<AnfNodePtr> local_value_inputs{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t index = 0; index < global_keys.size(); ++index) {
    auto global_key = global_keys.at(index);
    MS_EXCEPTION_IF_NULL(global_key);
    auto key = GetValueNode<StringImmPtr>(global_key);
    if (local_dict_keys_set.find(GetValue<std::string>(key)) != local_dict_keys_set.end()) {
      MS_LOG(INFO) << "The global dict has the same name with local dict.:" << key->ToString();
      continue;
    }
    MS_LOG(DEBUG) << "The global key " << global_key->DebugString() << ", value "
                  << global_values.at(index)->DebugString() << ". merged in local dict.";
    (void)local_keys_inputs.emplace_back(global_key);
    (void)local_value_inputs.emplace_back(FuncGraphToPyData(global_values.at(index)));
  }
  std::copy(local_keys.begin(), local_keys.end(), std::back_inserter(local_keys_inputs));
  std::transform(local_values.begin(), local_values.end(), std::back_inserter(local_value_inputs),
                 [&manager, &func_graph](const AnfNodePtr &node) -> AnfNodePtr {
                   if (!IsPrimitiveCNode(node, prim::kPrimPyInterpret)) {
                     return node;
                   }
                   auto trans_node = Transform(node->cast<CNodePtr>(), manager);
                   (void)manager->Replace(node, trans_node);
                   return trans_node;
                 });
  return std::make_pair(func_graph->NewCNode(local_keys_inputs), func_graph->NewCNode(local_value_inputs));
}

CNodePtr Transform(const CNodePtr &cnode, const FuncGraphManagerPtr &manager) {
  constexpr auto input_index_one = 1;
  constexpr auto input_index_two = 2;
  constexpr auto input_index_three = 3;
  auto new_cnode = std::make_shared<CNode>(*cnode);
  new_cnode->CloneUserData(cnode);
  new_cnode->set_input(0, NewValueNode(prim::kPrimPyExecute));

  if (!IsValueNode<parse::Script>(cnode->input(input_index_one))) {
    MS_LOG(INTERNAL_EXCEPTION) << "The first input should be a Script, but got "
                               << cnode->input(input_index_one)->DebugString();
  }
  const auto &script = GetValueNode<std::shared_ptr<parse::Script>>(cnode->input(input_index_one));
  const auto &script_str = script->script();
  const auto &script_strimm_node = NewValueNode(std::make_shared<StringImm>(script_str));
  new_cnode->set_input(input_index_one, script_strimm_node);
  auto global_dict_node = cnode->input(input_index_two);
  auto local_dict_node = cnode->input(input_index_three);

  auto [local_dict_keys, local_dict_values] =
    MergeGlobalDictToLocal(global_dict_node, local_dict_node, cnode->func_graph(), manager);

  new_cnode->set_input(input_index_two, local_dict_keys);
  new_cnode->set_input(input_index_three, local_dict_values);

  // Record the PyExecute node.
  InterpretNodeRecorder::GetInstance().PushPyExecuteNode(new_cnode);
  return new_cnode;
}
}  // namespace

// Convert PyInterpret into PyExecute:
//   PyInterpret(script, global_dict, local_dict)
//   -->
//   PyExecute(script, local_dict_keys, local_dict_values),
//   with side-effect operation:
//     Merge global_dict to local dict.
//     If there are arguments in global dict and local dict use local dict argument instead of global dict.
bool PyInterpretToExecute(const pipeline::ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto transact = manager->Transact();
  const auto all_nodes = manager->all_nodes();
  for (const auto &node : all_nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimPyInterpret)) {
      auto trans_node = Transform(node->cast<CNodePtr>(), manager);
      (void)transact.Replace(node, trans_node);
    }
  }
  transact.Commit();
  return true;
}
}  // namespace opt
}  // namespace mindspore
