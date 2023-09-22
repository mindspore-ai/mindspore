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

#include "frontend/operator/composite/dict_operation.h"

#include <vector>
#include <utility>
#include <algorithm>

#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "abstract/param_validator.h"
#include "frontend/optimizer/opt.h"
#include "pipeline/jit/ps/fallback.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
FuncGraphPtr DictSetItem::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  constexpr size_t dict_setitem_args_size = 3;
  abstract::CheckArgsSize("DictSetItem", args_list, dict_setitem_args_size);

  constexpr size_t dict_index = 0;
  auto data_abs = args_list[dict_index];
  MS_EXCEPTION_IF_NULL(data_abs);
  auto dict_abs = data_abs->cast<abstract::AbstractDictionaryPtr>();
  MS_EXCEPTION_IF_NULL(dict_abs);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("setitem");

  auto dict_param = ret->add_parameter();
  auto index_param = ret->add_parameter();
  auto target_param = ret->add_parameter();
  std::vector<AnfNodePtr> inputs = {dict_param, index_param, target_param};

  if (fallback::HasObjInExtraInfoHolder(dict_abs) && !fallback::GetCreateInGraphFromExtraInfoHolder(dict_abs)) {
    // The dict input has attached python object and the object is not created in graph.
    // Convert the DictSetItem to InplaceDictSetItem node.
    inputs.insert(inputs.begin(), NewValueNode(prim::kPrimDictInplaceSetItem));
    auto dict_inplace_setitem_node = ret->NewCNodeInOrder(inputs);
    dict_inplace_setitem_node->set_has_side_effect_node(true);
    ret->set_output(dict_inplace_setitem_node);
    ret->set_has_side_effect_node(true);
    return ret;
  }

  inputs.insert(inputs.begin(), NewValueNode(prim::kPrimDictSetItem));
  auto dict_setitem_node = ret->NewCNode(inputs);
  ret->set_output(dict_setitem_node);
  return ret;
}

FuncGraphPtr DictClear::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  constexpr size_t dict_clear_args_size = 1;
  abstract::CheckArgsSize("DictClear", args_list, dict_clear_args_size);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("clear");
  (void)ret->add_parameter();

  auto empty_dict = std::vector<std::pair<ValuePtr, ValuePtr>>();
  ret->set_output(NewValueNode(std::make_shared<ValueDictionary>(empty_dict)));
  return ret;
}

AnfNodePtr GeneratePyExecuteNodeHasKey(const FuncGraphPtr &fg) {
  auto dict_input = fg->add_parameter();
  auto value_input = fg->add_parameter();

  const std::string internal_dict = "__iternal_dict__";
  const std::string internal_target = "__internal_target__";

  std::stringstream script_buffer;
  script_buffer << internal_target << " in " << internal_dict;
  const std::string &script = script_buffer.str();
  const auto script_str = std::make_shared<StringImm>(script);

  std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
  (void)key_value_names_list.emplace_back(NewValueNode(internal_dict));
  (void)key_value_names_list.emplace_back(NewValueNode(internal_target));
  const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);
  std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
  (void)key_value_list.emplace_back(dict_input);
  (void)key_value_list.emplace_back(value_input);
  const auto key_value_tuple = fg->NewCNode(key_value_list);
  return fallback::CreatePyExecuteCNode(fg, NewValueNode(script_str), key_value_name_tuple, key_value_tuple, nullptr);
}

FuncGraphPtr DictHasKey::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  constexpr size_t dict_has_key_args_size = 2;
  abstract::CheckArgsSize("DictHasKey", args_list, dict_has_key_args_size);

  auto dict = dyn_cast<abstract::AbstractDictionary>(args_list[0]);
  ValuePtr key_value = args_list[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(dict);
  MS_EXCEPTION_IF_NULL(key_value);
  const auto &elems = dict->elements();
  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("has_key");
  // If key_value or value of dictionary has variable, then we convert has key operation to pyexecute.
  bool has_variable = (key_value == kValueAny) || std::any_of(elems.cbegin(), elems.cend(),
                                                              [&key_value](const abstract::AbstractElementPair &item) {
                                                                return item.first->BuildValue() == kValueAny;
                                                              });
  if (has_variable) {
    auto out = GeneratePyExecuteNodeHasKey(ret);
    ret->set_output(out);
    return ret;
  }
  bool has_key = false;
  auto it = std::find_if(elems.cbegin(), elems.cend(), [&key_value](const abstract::AbstractElementPair &item) {
    return *key_value == *item.first->BuildValue();
  });
  if (it != elems.cend()) {
    has_key = true;
  }

  (void)ret->add_parameter();
  (void)ret->add_parameter();

  auto out = NewValueNode(MakeValue(has_key));
  ret->set_output(out);
  return ret;
}

FuncGraphPtr DictUpdate::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  constexpr size_t dict_update_args_size = 2;
  abstract::CheckArgsSize("DictUpdate", args_list, dict_update_args_size);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("update");

  AnfNodePtrList key_inputs;
  AnfNodePtrList value_inputs;
  (void)key_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  (void)value_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));

  std::vector<std::pair<ValuePtr, size_t>> key_place_map;
  AddNodeToLists(args_list[0], ret, &key_inputs, &value_inputs, &key_place_map);
  AddNodeToLists(args_list[1], ret, &key_inputs, &value_inputs, &key_place_map);

  ret->set_output(ret->NewCNode(
    {NewValueNode(prim::kPrimMakeDict), ret->NewCNode(std::move(key_inputs)), ret->NewCNode(std::move(value_inputs))}));
  return ret;
}

void DictUpdate::AddNodeToLists(const AbstractBasePtr &arg, const FuncGraphPtr &ret, AnfNodePtrList *keys,
                                AnfNodePtrList *values, std::vector<std::pair<ValuePtr, size_t>> *key_place_map) const {
  auto dict = dyn_cast<abstract::AbstractDictionary>(arg);
  MS_EXCEPTION_IF_NULL(dict);
  auto &dict_elems = dict->elements();
  auto arg_node = ret->add_parameter();

  for (const auto &elem : dict_elems) {
    auto elem_key = elem.first->BuildValue();
    MS_EXCEPTION_IF_NULL(elem_key);
    AnfNodePtr dict_value = ret->NewCNode({NewValueNode(prim::kPrimDictGetItem), arg_node, NewValueNode(elem_key)});
    auto map_find =
      std::find_if(key_place_map->cbegin(), key_place_map->cend(),
                   [&elem_key](const std::pair<ValuePtr, size_t> &item) { return *elem_key == *item.first; });
    if (map_find == key_place_map->cend()) {
      (void)key_place_map->emplace_back(std::make_pair(elem_key, values->size()));
      (void)keys->emplace_back(NewValueNode(elem_key));
      (void)values->emplace_back(dict_value);
    } else {
      values->at(map_find->second) = dict_value;
    }
  }
}

FuncGraphPtr DictFromKeys::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  constexpr size_t dict_fromkeys_args_size = 3;
  abstract::CheckArgsSize("DictFromKeys", args_list, dict_fromkeys_args_size);
  const auto &values = ParseIterableObject(args_list[1]);
  auto value_node = args_list[2]->BuildValue();
  MS_EXCEPTION_IF_NULL(value_node);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("fromkeys");
  (void)ret->add_parameter();
  (void)ret->add_parameter();
  (void)ret->add_parameter();

  std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
  for (auto &value : values) {
    auto key_node = value->BuildValue();
    MS_EXCEPTION_IF_NULL(key_node);
    (void)key_values.emplace_back(std::make_pair(key_node, value_node));
  }

  ret->set_output(NewValueNode(std::make_shared<ValueDictionary>(key_values)));
  return ret;
}

abstract::AbstractBasePtrList DictFromKeys::ParseIterableObject(const abstract::AbstractBasePtr &arg_key) const {
  auto key_type = arg_key->BuildType();
  if (key_type->IsSameTypeId(List::kTypeId) || key_type->IsSameTypeId(Tuple::kTypeId)) {
    abstract::AbstractSequencePtr dict_keys = dyn_cast<abstract::AbstractSequence>(arg_key);
    MS_EXCEPTION_IF_NULL(dict_keys);
    return dict_keys->elements();
  }
  if (key_type->IsSameTypeId(Dictionary::kTypeId)) {
    auto dict_keys = dyn_cast<abstract::AbstractDictionary>(arg_key);
    MS_EXCEPTION_IF_NULL(dict_keys);
    AbstractBasePtrList keys;
    auto &dict_elems = dict_keys->elements();
    (void)std::transform(dict_elems.cbegin(), dict_elems.cend(), std::back_inserter(keys),
                         [](const abstract::AbstractElementPair &item) { return item.first; });
    return keys;
  }
  if (key_type->IsSameTypeId(String::kTypeId)) {
    string dict_keys = arg_key->BuildValue()->ToString();
    AbstractBasePtrList keys;
    (void)std::transform(dict_keys.cbegin(), dict_keys.cend(), std::back_inserter(keys), [](const char &item) {
      return std::make_shared<abstract::AbstractScalar>(std::string(1, item));
    });
    return keys;
  }

  MS_LOG(INTERNAL_EXCEPTION) << key_type->ToString() << " object is not iterable";
}
}  // namespace prim
}  // namespace mindspore
