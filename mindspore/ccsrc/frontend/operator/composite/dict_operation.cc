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

#include "abstract/param_validator.h"
#include "frontend/optimizer/opt.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
FuncGraphPtr DictClear::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  constexpr size_t dict_clear_args_size = 1;
  abstract::CheckArgsSize("DictClear", args_list, dict_clear_args_size);

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("clear");
  (void)ret->add_parameter();

  auto empty_dict = std::vector<std::pair<std::string, ValuePtr>>();
  ret->set_output(NewValueNode(std::make_shared<ValueDictionary>(empty_dict)));
  return ret;
}

FuncGraphPtr DictHasKey::GenerateFuncGraph(const abstract::AbstractBasePtrList &args_list) {
  constexpr size_t dict_has_key_args_size = 2;
  abstract::CheckArgsSize("DictHasKey", args_list, dict_has_key_args_size);

  auto dict = dyn_cast<abstract::AbstractDictionary>(args_list[0]);
  ValuePtr key_value = args_list[1]->BuildValue();
  MS_EXCEPTION_IF_NULL(dict);
  MS_EXCEPTION_IF_NULL(key_value);
  if (!key_value->isa<StringImm>()) {
    MS_LOG(EXCEPTION) << "The key should be string, but got " << key_value->ToString();
  }

  auto key_str = GetValue<std::string>(key_value);
  auto dict_elems = dict->elements();
  bool has_key = false;
  auto it = std::find_if(dict_elems.begin(), dict_elems.end(),
                         [&key_str](const abstract::AbstractAttribute &item) { return item.first == key_str; });
  if (it != dict_elems.end()) {
    has_key = true;
  }

  FuncGraphPtr ret = std::make_shared<FuncGraph>();
  ret->set_flag(FUNC_GRAPH_FLAG_CORE, true);
  ret->debug_info()->set_name("has_key");
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

  std::unordered_map<std::string, size_t> hash_map;
  AddNodeToLists(args_list[0], ret, &key_inputs, &value_inputs, &hash_map);
  AddNodeToLists(args_list[1], ret, &key_inputs, &value_inputs, &hash_map);

  ret->set_output(ret->NewCNode(
    {NewValueNode(prim::kPrimMakeDict), ret->NewCNode(std::move(key_inputs)), ret->NewCNode(std::move(value_inputs))}));
  return ret;
}

void DictUpdate::AddNodeToLists(const AbstractBasePtr &arg, const FuncGraphPtr &ret, AnfNodePtrList *keys,
                                AnfNodePtrList *values, std::unordered_map<std::string, size_t> *hash_map) {
  auto dict = dyn_cast<abstract::AbstractDictionary>(arg);
  MS_EXCEPTION_IF_NULL(dict);
  auto &dict_elems = dict->elements();
  auto arg_node = ret->add_parameter();

  for (const auto &elem : dict_elems) {
    AnfNodePtr dict_value = ret->NewCNode({NewValueNode(prim::kPrimDictGetItem), arg_node, NewValueNode(elem.first)});

    auto map_find = hash_map->find(elem.first);
    if (map_find == hash_map->end()) {
      hash_map->insert(std::make_pair(elem.first, values->size()));
      (void)keys->emplace_back(NewValueNode(elem.first));
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

  std::vector<std::pair<std::string, ValuePtr>> key_values;
  for (auto &value : values) {
    auto key = value->BuildValue();
    if (!key->IsSameTypeId(StringImm::kTypeId)) {
      MS_LOG(EXCEPTION) << "The key should be string, but got " << key->type_name();
    }

    std::string key_node = GetValue<std::string>(key);
    (void)key_values.emplace_back(std::make_pair(key_node, value_node));
  }

  ret->set_output(NewValueNode(std::make_shared<ValueDictionary>(key_values)));
  return ret;
}

abstract::AbstractBasePtrList DictFromKeys::ParseIterableObject(const abstract::AbstractBasePtr &arg_key) {
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
    (void)std::transform(
      dict_elems.begin(), dict_elems.end(), std::back_inserter(keys),
      [](const abstract::AbstractAttribute &item) { return std::make_shared<abstract::AbstractScalar>(item.first); });
    return keys;
  }
  if (key_type->IsSameTypeId(String::kTypeId)) {
    auto value = arg_key->BuildValue();
    MS_EXCEPTION_IF_NULL(value);
    string dict_keys = value->ToString();
    AbstractBasePtrList keys;
    (void)std::transform(dict_keys.begin(), dict_keys.end(), std::back_inserter(keys), [](const char &item) {
      return std::make_shared<abstract::AbstractScalar>(std::string(1, item));
    });
    return keys;
  }

  MS_LOG(EXCEPTION) << key_type->ToString() << " object is not iterable";
}
}  // namespace prim
}  // namespace mindspore
