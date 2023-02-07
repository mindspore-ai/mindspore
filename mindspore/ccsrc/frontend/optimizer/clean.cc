/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/clean.h"
#include <iterator>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <utility>
#include <memory>
#include "abstract/abstract_value.h"
#include "base/base.h"
#include "mindspore/core/ops/core_ops.h"
#include "pipeline/jit/debug/trace.h"
#include "frontend/optimizer/opt.h"
#include "frontend/operator/composite/composite.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "pipeline/jit/parse/resolve.h"
#include "utils/hash_map.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractBasePtr;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractDictionaryPtr;
using mindspore::abstract::AbstractElementPair;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractListPtr;
using mindspore::abstract::AbstractRowTensor;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractSequence;
using mindspore::abstract::AbstractSequencePtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;

namespace {
static constexpr size_t kMaxSeqRecursiveDepth = 6;
void CheckInputsSize(const CNodePtr &cnode, size_t expect_size) {
  if (cnode->size() != expect_size) {
    std::string op_name = GetCNodeFuncName(cnode);
    MS_LOG(EXCEPTION) << op_name << " should have " << expect_size << " inputs, but got " << cnode->size();
  }
}

template <typename T>
std::shared_ptr<T> GetAbstract(const AnfNodePtr &node) {
  return dyn_cast<T>(node->abstract());
}

bool CheckContainsDict(const AbstractBasePtr &abs) {
  if (abs == nullptr) {
    return false;
  }
  if (abs->isa<AbstractDictionary>()) {
    return true;
  }
  if (abs->isa<AbstractSequence>()) {
    auto abs_seq = abs->cast<AbstractSequencePtr>();
    const auto &elements = abs_seq->elements();
    if (std::any_of(elements.begin(), elements.end(),
                    [](const AbstractBasePtr &element) { return CheckContainsDict(element); })) {
      return true;
    }
  }
  return false;
}

// ===========================================================================
// BaseRewriter provides a common framework for data struct simplify.
// ===========================================================================
class BaseRewriter : protected SimpleRewriter {
 public:
  BaseRewriter(const FuncGraphPtr &root_graph, const FuncGraphManagerPtr &manager)
      : SimpleRewriter(root_graph, manager) {}
  ~BaseRewriter() override = default;

  bool Execute() {
    bool changed = Run();
    if (changed) {
      UpdateAbstracts();
    }
    return changed;
  }

 protected:
  virtual AnfNodePtr ConvertPrimitiveCNode(const CNodePtr &cnode, const PrimitivePtr &prim) = 0;
  virtual AnfNodePtr ConvertValueNode(const ValueNodePtr &value_node, const ValuePtr &value) = 0;
  virtual AbstractBasePtr ConvertAbstract(const AbstractBasePtr &abs) = 0;

  AnfNodePtr NodeRewrite(const AnfNodePtr &node) override {
    auto new_node = ConvertNode(node);
    if (new_node != nullptr) {
      new_node->set_abstract(node->abstract());
    }
    return new_node;
  }

  AnfNodePtr ConvertNode(const AnfNodePtr &node) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode != nullptr) {
      if (cnode->size() == 0) {
        return nullptr;
      }
      // Get primitive from cnode.
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      if (prim == nullptr) {
        return nullptr;
      }
      // Call primitive cnode converter.
      return ConvertPrimitiveCNode(cnode, prim);
    }
    auto value_node = node->cast<ValueNodePtr>();
    if (value_node != nullptr) {
      const auto &value = value_node->value();
      if (value == nullptr) {
        return nullptr;
      }
      // Call value node converter.
      return ConvertValueNode(value_node, value);
    }
    return nullptr;
  }

  void UpdateAbstracts() {
    const auto &nodes = manager_->all_nodes();
    for (const auto &node : nodes) {
      const auto &abs = node->abstract();
      if (abs == nullptr) {
        continue;
      }
      bool is_interpret_dict = false;
      // Do not convert the abstract of Interpret node(AbstractDictionary) to AbstractSequence.
      if (abs->isa<AbstractDictionary>()) {
        AbstractDictionaryPtr abs_dict = abs->cast<AbstractDictionaryPtr>();
        auto &dict_elements = abs_dict->elements();
        for (auto &element : dict_elements) {
          TypePtr type = element.second->GetTypeTrack();
          MS_EXCEPTION_IF_NULL(type);
          auto value = element.second->BuildValue();
          MS_EXCEPTION_IF_NULL(value);
          if (type->type_id() == kMetaTypeExternal && value->isa<parse::InterpretedObject>()) {
            is_interpret_dict = true;
            break;
          }
        }
      }
      if (is_interpret_dict) {
        continue;
      }
      // Call abstract converter.
      auto new_abs = ConvertAbstract(abs);
      if (new_abs != nullptr) {
        node->set_abstract(new_abs);
      }
    }
  }
};

// ===========================================================================
// SimplifyDataStructuresRewriter convert ObjectClass, Dictionary to Tuple.
// ===========================================================================
class SimplifyDataStructuresRewriter : public BaseRewriter {
 public:
  using ThisClass = SimplifyDataStructuresRewriter;
  SimplifyDataStructuresRewriter(const FuncGraphPtr &root_graph, const FuncGraphManagerPtr &manager)
      : BaseRewriter(root_graph, manager), is_dict_output_{HasDictOutput()} {}
  ~SimplifyDataStructuresRewriter() override = default;

 protected:
  static std::string GetStringValue(const AnfNodePtr &node) {
    auto str = GetValueNode<StringImmPtr>(node);
    if (str == nullptr) {
      return "";
    }
    return str->value();
  }

  static int64_t GetElementIndex(const std::vector<AbstractElementPair> &attrs, const AnfNodePtr &name) {
    auto n_attrs = attrs.size();
    auto name_abstract = GetAbstract<AbstractBase>(name);
    MS_EXCEPTION_IF_NULL(name_abstract);
    auto name_value = name_abstract->BuildValue();
    MS_EXCEPTION_IF_NULL(name_value);
    for (size_t i = 0; i < n_attrs; ++i) {
      if (*name_value == *attrs[i].first->BuildValue()) {
        return SizeToLong(i);
      }
    }
    return SizeToLong(n_attrs);
  }

  static CNodePtr NewTupleGetCNode(const AnfNodePtr &cnode, const AnfNodePtr &data_node,
                                   const std::vector<AbstractElementPair> &elements, const AnfNodePtr &name_node) {
    int64_t index = GetElementIndex(elements, name_node);
    auto index_node = NewValueNode(index);
    auto prim_node = NewValueNode(prim::kPrimTupleGetItem);
    return cnode->func_graph()->NewCNode({prim_node, data_node, index_node});
  }

  // From:
  //   DictGetItem(data:AbstractDictionary, key:AbstractBase)
  // To:
  //   TupleGetItem(data, index:Int64Imm)
  AnfNodePtr ConvertDictGetItemToTupleGetItem(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [dict_getitem, dict, item]
    const size_t expect_inputs_size = 3;
    CheckInputsSize(node, expect_inputs_size);

    constexpr size_t data_index = 1;
    constexpr size_t key_index = 2;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[key_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(key);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    return NewTupleGetCNode(node, data, abs_dict->elements(), key);
  }

  // DictGetItem --> PyExecute()
  AnfNodePtr RebuildDictGetItem(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    // Inputs should be [dict_setitem, dict, item]
    const size_t expect_inputs_size = 3;
    CheckInputsSize(node, expect_inputs_size);

    const size_t data_index = 1;
    const size_t item_key_index = 2;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[item_key_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(key);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    auto func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);

    // Script
    constexpr auto internal_dict_self_str = "__internal_dict_self__";
    constexpr auto internal_dict_key_str = "__internal_dict_key__";
    std::stringstream script_buffer;
    script_buffer << internal_dict_self_str << "[" << internal_dict_key_str << "]";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    // Pack local parameters keys.
    const auto script_dict_self_name = std::make_shared<StringImm>(internal_dict_self_str);
    const auto script_dict_key_name = std::make_shared<StringImm>(internal_dict_key_str);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_self_name));
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_key_name));
    const auto key_value_name_tuple = func_graph->NewCNode(key_value_names_list);

    // Pack the local parameters values, not support list, tuple, or dict.
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(data);
    (void)key_value_list.emplace_back(key);
    const auto key_value_tuple = func_graph->NewCNode(key_value_list);

    // Build the new dict node.
    const auto dict_getitem_node = func_graph->NewCNode(
      {NewValueNode(prim::kPrimPyExecute), NewValueNode(script_str), key_value_name_tuple, key_value_tuple});
    int64_t index = GetElementIndex(abs_dict->elements(), key);
    const auto &val = abs_dict->elements()[index].second;
    const auto &tensor_val = dyn_cast<abstract::AbstractTensor>(val);
    if (tensor_val != nullptr) {
      const auto &tensor_type = tensor_val->element()->BuildType();
      dict_getitem_node->set_user_data<Type>("__py_execute_tensor_type__", tensor_type);
      const auto &tensor_shape = dyn_cast<abstract::Shape>(tensor_val->BuildShape());
      MS_EXCEPTION_IF_NULL(tensor_shape);
      dict_getitem_node->set_user_data<abstract::Shape>("__py_execute_tensor_shape__", tensor_shape);
      MS_LOG(DEBUG) << "key: " << key->abstract()->BuildValue()->ToString() << ", type: " << tensor_type->ToString()
                    << ", shape: " << tensor_shape->ToString() << ", val: " << tensor_val->ToString();
    }
    MS_LOG(DEBUG) << "Made dict getitem node: " << dict_getitem_node->DebugString();
    dict_getitem_node->set_debug_info(node->debug_info());
    return dict_getitem_node;
  }

  AnfNodePtr ConvertDictGetItem(const CNodePtr &node) {
    static const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
    if (support_fallback_runtime && is_dict_output_) {
      return RebuildDictGetItem(node);
    }
    return ConvertDictGetItemToTupleGetItem(node);
  }

  // From:
  //   DictSetItem(data:AbstractDictionary, key:AbstractBase, value)
  // To:
  //   TupleSetItem(data, index:Int64Imm, value)
  // Or:
  //   tuple_add(data, value)
  AnfNodePtr ConvertDictSetItemToTupleSetItem(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [dict_setitem, dict, item, value]
    const size_t expect_inputs_size = 4;
    CheckInputsSize(node, expect_inputs_size);

    const size_t data_index = 1;
    const size_t cons_index = 2;
    const size_t item_value_index = 3;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[cons_index];
    auto &item_value = inputs[item_value_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(key);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    int64_t index = GetElementIndex(abs_dict->elements(), key);
    auto func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    if (index >= static_cast<int64_t>(abs_dict->elements().size())) {
      // For dictionary set, if the key does not exist, we should create a new item.
      std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
      for (size_t i = 0; i < abs_dict->elements().size(); ++i) {
        auto tuple_getitem_i =
          func_graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, NewValueNode(SizeToLong(i))});
        (void)make_tuple_inputs.emplace_back(tuple_getitem_i);
      }
      (void)make_tuple_inputs.emplace_back(item_value);
      auto new_node = func_graph->NewCNode(make_tuple_inputs);
      new_node->set_debug_info(node->debug_info());
      return new_node;
    }
    auto index_node = NewValueNode(index);
    auto new_node = func_graph->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, index_node, item_value});
    new_node->set_debug_info(node->debug_info());
    return new_node;
  }

  bool HasDictOutput() const {
    const AnfNodePtr &output = root_graph_->output();
    return CheckContainsDict(output->abstract());
  }

  // DictSetItem --> PyExecute()
  AnfNodePtr RebuidDictSetItem(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    // Inputs should be [dict_setitem, dict, item, value]
    const size_t expect_inputs_size = 4;
    CheckInputsSize(node, expect_inputs_size);

    const size_t data_index = 1;
    const size_t item_key_index = 2;
    const size_t item_value_index = 3;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[item_key_index];
    auto &item_value = inputs[item_value_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(key);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    auto func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);

    // Script
    constexpr auto internal_dict_self_str = "__internal_dict_self__";
    constexpr auto internal_dict_key_str = "__internal_dict_key__";
    constexpr auto internal_dict_value_str = "__internal_dict_value__";
    std::stringstream script_buffer;
    script_buffer << "__import__('mindspore').common._utils.dict_setitem(" << internal_dict_self_str << ", "
                  << internal_dict_key_str << ", " << internal_dict_value_str << ")";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    // Pack local parameters keys.
    const auto script_dict_self_name = std::make_shared<StringImm>(internal_dict_self_str);
    const auto script_dict_key_name = std::make_shared<StringImm>(internal_dict_key_str);
    const auto script_dict_value_name = std::make_shared<StringImm>(internal_dict_value_str);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_self_name));
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_key_name));
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_value_name));
    const auto key_value_name_tuple = func_graph->NewCNode(key_value_names_list);

    // Pack the local parameters values, not support list, tuple, or dict.
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(data);
    (void)key_value_list.emplace_back(key);
    (void)key_value_list.emplace_back(item_value);
    const auto key_value_tuple = func_graph->NewCNode(key_value_list);

    // Build the new dict node.
    const auto dict_setitem_node = func_graph->NewCNode(
      {NewValueNode(prim::kPrimPyExecute), NewValueNode(script_str), key_value_name_tuple, key_value_tuple});
    MS_LOG(DEBUG) << "Made dict setitem node: " << dict_setitem_node->DebugString();
    dict_setitem_node->set_debug_info(node->debug_info());
    return dict_setitem_node;
  }

  AnfNodePtr ConvertDictSetItem(const CNodePtr &node) {
    static const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
    if (support_fallback_runtime && is_dict_output_) {
      return RebuidDictSetItem(node);
    }
    return ConvertDictSetItemToTupleSetItem(node);
  }

  // From:
  //   MakeDict(name, input)
  // To:
  //   input
  AnfNodePtr EraseMakeDictNode(const CNodePtr &node) const {
    MS_EXCEPTION_IF_NULL(node);
    constexpr size_t expect_inputs_size = 3;
    constexpr size_t input_index = 2;
    CheckInputsSize(node, expect_inputs_size);
    return node->input(input_index);
  }

  // MakeDict(keys, values) --> PyExecute('dict(zip(keys, values))', ...)
  AnfNodePtr RebuildMakeDictNode(const CNodePtr &node) const {
    constexpr auto internal_tuple_keys_str = "__internal_tuple_keys__";
    constexpr auto internal_tuple_values_str = "__internal_tuple_values__";
    constexpr auto internal_dict_zip_keys_str = "__internal_dict_zip_keys__";
    constexpr auto internal_dict_zip_values_str = "__internal_dict_zip_values__";
    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);

    // Local parameters values.
    // Pack the key tuple.
    const auto script_key_tuple_str = std::make_shared<StringImm>(internal_tuple_keys_str);
    const auto make_key_tuple_node =
      fg->NewCNode({NewValueNode(prim::kPrimPyExecute), NewValueNode(script_key_tuple_str),
                    NewValueNode(script_key_tuple_str), node->input(1)});
    make_key_tuple_node->set_debug_info(node->input(1)->debug_info());
    // Pack the value tuple.
    constexpr size_t values_input_index = 2;
    const auto script_value_tuple_str = std::make_shared<StringImm>(internal_tuple_values_str);
    const auto make_value_tuple_node =
      fg->NewCNode({NewValueNode(prim::kPrimPyExecute), NewValueNode(script_value_tuple_str),
                    NewValueNode(script_value_tuple_str), node->input(values_input_index)});
    make_value_tuple_node->set_debug_info(node->input(values_input_index)->debug_info());
    // Pack the local parameters values
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(make_key_tuple_node);
    (void)key_value_list.emplace_back(make_value_tuple_node);
    const auto key_value_tuple = fg->NewCNode(key_value_list);

    // Pack local parameters keys.
    const auto script_dict_key_name = std::make_shared<StringImm>(internal_dict_zip_keys_str);
    const auto script_dict_value_name = std::make_shared<StringImm>(internal_dict_zip_values_str);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_key_name));
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_value_name));
    const auto key_value_name_tuple = fg->NewCNode(key_value_names_list);

    // Script
    std::stringstream script_buffer;
    script_buffer << "dict(zip(" << internal_dict_zip_keys_str << "," << internal_dict_zip_values_str << "),)";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    // Build the new dict node.
    const auto make_dict_node = fg->NewCNode(
      {NewValueNode(prim::kPrimPyExecute), NewValueNode(script_str), key_value_name_tuple, key_value_tuple});
    MS_LOG(DEBUG) << "Made dict node: " << make_dict_node->DebugString();
    make_dict_node->set_debug_info(node->debug_info());
    return make_dict_node;
  }

  AnfNodePtr ConvertMakeDict(const CNodePtr &node) {
    static const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
    if (support_fallback_runtime && is_dict_output_) {
      return RebuildMakeDictNode(node);
    }
    return EraseMakeDictNode(node);
  }

  // From:
  //   DictGetValues(dict:AbstractDictionary)
  // To:
  //   dict
  AnfNodePtr EraseDictGetValues(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    constexpr size_t expect_inputs_size = 2;
    CheckInputsSize(node, expect_inputs_size);
    return node->input(1);
  }

  // From:
  //   DictItems(dict:AbstractDictionary)
  // To:
  //   kPrimMakeList(MakeTuple(key0, TupleGetItem(dict, 0)), ...)
  AnfNodePtr EraseDictItems(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    auto fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    constexpr size_t expect_inputs_size = 2;
    CheckInputsSize(node, expect_inputs_size);

    const auto &input = node->input(1);
    auto abs_dict = GetAbstract<AbstractDictionary>(input);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    const auto &elements = abs_dict->elements();
    std::vector<AnfNodePtr> new_inputs;
    new_inputs.reserve(elements.size() + 1);
    (void)new_inputs.emplace_back(NewValueNode(prim::kPrimMakeList));
    for (size_t i = 0; i < elements.size(); ++i) {
      auto index_node = NewValueNode(static_cast<int64_t>(i));
      MS_EXCEPTION_IF_NULL(elements[i].first->BuildValue());
      auto key_node = NewValueNode(elements[i].first->BuildValue());
      auto value_node = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input, index_node});
      auto tuple_node = fg->NewCNode({NewValueNode(prim::kPrimMakeTuple), key_node, value_node});
      (void)new_inputs.emplace_back(tuple_node);
    }
    return fg->NewCNode(std::move(new_inputs));
  }

  // From:
  //   MakeKeywordArg(key, value)
  // To:
  //   value
  AnfNodePtr EraseMakeKeywordArgNode(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    // Inputs should be [make_keyword_arg, key, value]
    constexpr size_t expect_input_size = 3;
    constexpr size_t value_inputs_index = 2;
    CheckInputsSize(node, expect_input_size);
    return node->input(value_inputs_index);
  }

  // From:
  //   ExtractKeywordArg(arg, key)
  // To:
  //   key
  AnfNodePtr EraseExtractKeywordArg(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    // Inputs should be [extract_keyword_arg, arg, key]
    const size_t expect_inputs_size = 3;
    CheckInputsSize(node, expect_inputs_size);
    constexpr size_t key_index = 2;
    return node->input(key_index);
  }

  // dict(k0:v0, k1:v1, ...) --> tuple(v0, v1, ...)
  AnfNodePtr DictToTuple(const ValueDictionaryPtr &dict) const {
    const auto &keys_values = dict->value();
    std::vector<ValuePtr> value_list;
    value_list.reserve(keys_values.size());
    (void)std::transform(keys_values.begin(), keys_values.end(), std::back_inserter(value_list),
                         [](const auto &value) { return value.second; });
    return NewValueNode(std::make_shared<ValueTuple>(value_list));
  }

  // dict(k0:v0, k1:v1, ...) --> PyExecute('dict(zip(keys, values))', ...)
  AnfNodePtr RebuildValueDict(const ValueNodePtr &value_node, const ValueDictionaryPtr &dict) const {
    constexpr auto internal_tuple_keys_str = "__internal_tuple_keys__";
    constexpr auto internal_tuple_values_str = "__internal_tuple_values__";
    constexpr auto internal_dict_zip_keys_str = "__internal_dict_zip_keys__";
    constexpr auto internal_dict_zip_values_str = "__internal_dict_zip_values__";

    const auto &keys_values = dict->value();
    std::vector<ValuePtr> key_list;
    key_list.reserve(keys_values.size());
    std::vector<ValuePtr> value_list;
    value_list.reserve(keys_values.size());
    for (const auto &key_value : keys_values) {
      (void)key_list.emplace_back(key_value.first);
      (void)value_list.emplace_back(key_value.second);
    }

    // Local parameters values.
    // Pack the key tuple.
    const auto script_key_tuple_str = std::make_shared<StringImm>(internal_tuple_keys_str);
    const auto key_tuple = std::make_shared<ValueTuple>(key_list);
    const auto make_key_tuple_node =
      root_graph_->NewCNode({NewValueNode(prim::kPrimPyExecute), NewValueNode(script_key_tuple_str),
                             NewValueNode(script_key_tuple_str), NewValueNode(key_tuple)});
    // Pack the value tuple.
    const auto script_value_tuple_str = std::make_shared<StringImm>(internal_tuple_values_str);
    const auto value_tuple = std::make_shared<ValueTuple>(value_list);
    const auto make_value_tuple_node =
      root_graph_->NewCNode({NewValueNode(prim::kPrimPyExecute), NewValueNode(script_value_tuple_str),
                             NewValueNode(script_value_tuple_str), NewValueNode(value_tuple)});
    // Pack the local parameters values
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_list.emplace_back(make_key_tuple_node);
    (void)key_value_list.emplace_back(make_value_tuple_node);
    const auto key_value_tuple = root_graph_->NewCNode(key_value_list);

    // Pack local parameters keys.
    const auto script_dict_key_name = std::make_shared<StringImm>(internal_dict_zip_keys_str);
    const auto script_dict_value_name = std::make_shared<StringImm>(internal_dict_zip_values_str);
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_key_name));
    (void)key_value_names_list.emplace_back(NewValueNode(script_dict_value_name));
    const auto key_value_name_tuple = root_graph_->NewCNode(key_value_names_list);

    // Script
    std::stringstream script_buffer;
    script_buffer << "dict(zip(" << internal_dict_zip_keys_str << "," << internal_dict_zip_values_str << "),)";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    // Build the new dict node.
    const auto make_dict_node = root_graph_->NewCNode(
      {NewValueNode(prim::kPrimPyExecute), NewValueNode(script_str), key_value_name_tuple, key_value_tuple});
    MS_LOG(DEBUG) << "Made dict node: " << make_dict_node->DebugString();
    make_dict_node->set_debug_info(value_node->debug_info());
    return make_dict_node;
  }

  using Converter = AnfNodePtr (ThisClass::*)(const CNodePtr &);
  using ConverterMap = mindspore::HashMap<PrimitivePtr, Converter, PrimitiveHasher, PrimitiveEqual>;
  static inline const ConverterMap converters_{
    {prim::kPrimDictGetItem, &ThisClass::ConvertDictGetItem},
    {prim::kPrimDictSetItem, &ThisClass::ConvertDictSetItem},
    {prim::kPrimDictGetValues, &ThisClass::EraseDictGetValues},
    {prim::kPrimMakeDict, &ThisClass::ConvertMakeDict},
    {prim::kPrimMakeKeywordArg, &ThisClass::EraseMakeKeywordArgNode},
    {prim::kPrimExtractKeywordArg, &ThisClass::EraseExtractKeywordArg},
    {prim::kPrimDictItems, &ThisClass::EraseDictItems},
  };

  AnfNodePtr ConvertPrimitiveCNode(const CNodePtr &cnode, const PrimitivePtr &prim) override {
    // Find cnode converter by primitive.
    auto iter = converters_.find(prim);
    if (iter == converters_.end()) {
      return nullptr;
    }
    // Call converter.
    return (this->*(iter->second))(cnode);
  }

  AnfNodePtr ConvertValueNode(const ValueNodePtr &value_node, const ValuePtr &value) override {
    // Convert Dictionary value node.
    if (value->isa<ValueDictionary>()) {
      static const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
      if (support_fallback_runtime && is_dict_output_) {
        return RebuildValueDict(value_node, value->cast<ValueDictionaryPtr>());
      }
      return DictToTuple(value->cast<ValueDictionaryPtr>());
    }
    return nullptr;
  }

  static std::shared_ptr<AbstractTuple> MakeAbstractTuple(const std::vector<AbstractElementPair> &attrs) {
    std::vector<AbstractBasePtr> elements;
    elements.reserve(attrs.size());
    (void)std::transform(attrs.begin(), attrs.end(), std::back_inserter(elements),
                         [](const auto &item) { return item.second; });
    return std::make_shared<AbstractTuple>(std::move(elements));
  }

  // AbstractDictionary --> AbstractSequence.
  static AbstractSequencePtr ConvertToAbstractSequence(const AbstractBasePtr &abs, size_t depth) {
    if (depth > kMaxSeqRecursiveDepth) {
      MS_LOG(EXCEPTION) << "List or Dict nesting is not allowed more than " << kMaxSeqRecursiveDepth << " levels.";
    }
    auto abs_seq = abs->cast<AbstractSequencePtr>();
    if (abs_seq != nullptr) {
      const auto &seq_elements = abs_seq->elements();
      // First we check if elements should be converted,
      // changed_elements maps old element to new element.
      mindspore::HashMap<AbstractBasePtr, AbstractBasePtr> changed_elements;
      for (const auto &element : seq_elements) {
        auto new_element = ConvertToAbstractSequence(element, depth + 1);
        if (new_element != nullptr) {
          (void)changed_elements.emplace(element, new_element);
        }
      }
      if (changed_elements.empty()) {
        // Here the AbstractList don't need to convert to AbstractTuple.
        return nullptr;
      }
      // Always make new AbstractSequence when elements changed.
      std::vector<AbstractBasePtr> elements;
      elements.reserve(seq_elements.size());
      for (const auto &element : seq_elements) {
        auto iter = changed_elements.find(element);
        if (iter != changed_elements.end()) {
          (void)elements.emplace_back(iter->second);
        } else {
          (void)elements.emplace_back(element);
        }
      }
      // Here the AbstractList don't need to convert to AbstractTuple.
      if (abs_seq->isa<AbstractList>()) {
        return std::make_shared<AbstractList>(std::move(elements));
      } else {
        return std::make_shared<AbstractTuple>(std::move(elements));
      }
    }
    // AbstractDictionary --> AbstractTuple.
    auto abs_dict = abs->cast<AbstractDictionaryPtr>();
    if (abs_dict != nullptr) {
      const auto &dict_elements = abs_dict->elements();
      std::vector<AbstractBasePtr> elements;
      elements.reserve(dict_elements.size());
      for (const auto &element : dict_elements) {
        auto new_element = ConvertToAbstractSequence(element.second, depth + 1);
        if (new_element != nullptr) {
          (void)elements.emplace_back(new_element);
        } else {
          (void)elements.emplace_back(element.second);
        }
      }
      return std::make_shared<AbstractTuple>(elements);
    }
    return nullptr;
  }

  AbstractBasePtr ConvertAbstract(const AbstractBasePtr &abs) override {
    // AbstractDictionary --> AbstractSequence.
    return ConvertToAbstractSequence(abs, 0);
  }

 private:
  bool is_dict_output_{false};
};

// ==================================================================
// CleanAfterOptARewriter converts List, Sparse, RowTensor to Tuple.
// ==================================================================
class CleanAfterOptARewriter : public BaseRewriter {
 public:
  using ThisClass = CleanAfterOptARewriter;
  CleanAfterOptARewriter(const FuncGraphPtr &root_graph, const FuncGraphManagerPtr &manager)
      : BaseRewriter(root_graph, manager) {}
  ~CleanAfterOptARewriter() override = default;

 protected:
  // From:
  //   MakeList(arg1, arg2, ...)
  // To:
  //   MakeTuple(arg1, arg2, ...)
  AnfNodePtr ConvertMakeListToMakeTuple(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    std::vector<AnfNodePtr> inputs;
    inputs.reserve(node->size());
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    // Inputs of node should be [make_list, item1, item2, ...], so offset by 1 to get items;
    (void)inputs.insert(inputs.cend(), node->inputs().cbegin() + 1, node->inputs().cend());
    return node->func_graph()->NewCNode(std::move(inputs));
  }

  // From:
  //   list_getitem(list, key)
  // To:
  //   TupleGetItem(list, key)
  AnfNodePtr ConvertListGetItemToTupleGetItem(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [list_getitem, list, item]
    constexpr size_t expect_input_size = 3;
    CheckInputsSize(node, expect_input_size);
    constexpr size_t data_index = 1;
    constexpr size_t cons_index = 2;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[cons_index];
    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, key});
  }

  // From:
  //   ListSetItem(list, index, item)
  // To:
  //   TupleSetItem(list, index, item)
  AnfNodePtr ConvertListSetItemToTupleSetItem(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [list_setitem, list, index, item]
    const size_t expect_inputs_size = 4;
    CheckInputsSize(node, expect_inputs_size);

    const size_t data_index = 1;
    const size_t cons_index = 2;
    const size_t value_index = 3;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &key = inputs[cons_index];
    auto &value = inputs[value_index];
    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, key, value});
  }

  // From:
  //   MakeSparseTensor(indices, values, dense_shape)
  // To:
  //   MakeTuple(indices, values, dense_shape)
  AnfNodePtr ConvertMakeSparseToMakeTuple(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    std::vector<AnfNodePtr> inputs;
    inputs.reserve(node->size());
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    // Inputs of node should be [make_sparse, indices, values, dense_shape], so offset by 1 to get items.
    (void)inputs.insert(inputs.cend(), node->inputs().cbegin() + 1, node->inputs().cend());
    auto new_node = node->func_graph()->NewCNode(std::move(inputs));
    new_node->set_abstract(node->abstract());
    return new_node;
  }

  static inline const mindspore::HashMap<std::string, int64_t> sparse_attr_map = {
    {prim::kCSRTensorGetIndptr, 0},     {prim::kCSRTensorGetIndices, 1}, {prim::kCSRTensorGetValues, 2},
    {prim::kCSRTensorGetDenseShape, 3}, {prim::kCOOTensorGetIndices, 0}, {prim::kCOOTensorGetValues, 1},
    {prim::kCOOTensorGetDenseShape, 2}, {prim::kRowTensorGetIndices, 0}, {prim::kRowTensorGetValues, 1},
    {prim::kRowTensorGetDenseShape, 2}};

  // From:
  //   SparseTensorGetXXX(sparse) # index
  // To:
  //   TupleGetItem(sparse, index)
  AnfNodePtr ConvertSparseGetAttrToTupleGetItem(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    constexpr size_t kExpectInputSize = 2;
    constexpr size_t kSparseAttrIndex = 1;
    CheckInputsSize(node, kExpectInputSize);

    auto prim = GetValueNode<PrimitivePtr>(node->input(0));
    if (prim != nullptr) {
      auto iter = sparse_attr_map.find(prim->name());
      if (iter != sparse_attr_map.end()) {
        const auto &sparse = node->input(kSparseAttrIndex);
        auto index_node = NewValueNode(iter->second);
        auto new_node = node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), sparse, index_node});
        new_node->set_abstract(node->abstract());
        return new_node;
      }
    }
    return nullptr;
  }

  using Converter = AnfNodePtr (ThisClass::*)(const CNodePtr &);
  using ConverterMap = mindspore::HashMap<PrimitivePtr, Converter, PrimitiveHasher, PrimitiveEqual>;
  static inline const ConverterMap converters_{
    {prim::kPrimMakeList, &ThisClass::ConvertMakeListToMakeTuple},
    {prim::kPrimListGetItem, &ThisClass::ConvertListGetItemToTupleGetItem},
    {prim::kPrimListSetItem, &ThisClass::ConvertListSetItemToTupleSetItem},
    // SparseProcess: 1.MakeSparse->MakeTuple 2.SparseGetAttr->TupleGetItem
    {prim::kPrimMakeRowTensor, &ThisClass::ConvertMakeSparseToMakeTuple},
    {prim::kPrimRowTensorGetIndices, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimRowTensorGetValues, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimRowTensorGetDenseShape, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimMakeCSRTensor, &ThisClass::ConvertMakeSparseToMakeTuple},
    {prim::kPrimCSRTensorGetIndptr, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCSRTensorGetIndices, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCSRTensorGetValues, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCSRTensorGetDenseShape, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimMakeCOOTensor, &ThisClass::ConvertMakeSparseToMakeTuple},
    {prim::kPrimCOOTensorGetIndices, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCOOTensorGetValues, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
    {prim::kPrimCOOTensorGetDenseShape, &ThisClass::ConvertSparseGetAttrToTupleGetItem},
  };

  AnfNodePtr ConvertPrimitiveCNode(const CNodePtr &cnode, const PrimitivePtr &prim) override {
    // Find cnode converter by primitive.
    auto iter = converters_.find(prim);
    if (iter == converters_.end()) {
      return nullptr;
    }
    // Call converter.
    return (this->*(iter->second))(cnode);
  }

  static ValuePtr ConvertValueSequenceToValueTuple(const ValuePtr &value, size_t depth, bool *need_convert) {
    MS_EXCEPTION_IF_NULL(need_convert);
    MS_EXCEPTION_IF_NULL(value);
    if (depth > kMaxSeqRecursiveDepth) {
      MS_LOG(EXCEPTION) << "List nesting is not allowed more than " << kMaxSeqRecursiveDepth << " levels.";
    }

    if (value->isa<ValueSequence>()) {
      std::vector<ValuePtr> elements;
      auto value_seq = value->cast<ValueSequencePtr>();
      (void)std::transform(value_seq->value().begin(), value_seq->value().end(), std::back_inserter(elements),
                           [&](const ValuePtr &value) -> ValuePtr {
                             bool is_convert = false;
                             auto convert_value = ConvertValueSequenceToValueTuple(value, depth + 1, &is_convert);
                             *need_convert |= is_convert;
                             return convert_value;
                           });
      *need_convert |= value->isa<ValueList>();
      if (*need_convert) {
        return std::make_shared<ValueTuple>(elements);
      }
    }

    return value;
  }

  AnfNodePtr ConvertValueNode(const ValueNodePtr &, const ValuePtr &value) override {
    bool need_convert = false;
    auto convert_value = ConvertValueSequenceToValueTuple(value, 0, &need_convert);
    if (need_convert) {
      return std::make_shared<ValueNode>(convert_value);
    }
    return nullptr;
  }

  // AbstractSequence, AbstractDict, AbstractRowTensor --> AbstractTuple.
  static AbstractBasePtr ConvertToAbstractTuple(const AbstractBasePtr &abs, size_t depth) {
    if (depth > kMaxSeqRecursiveDepth) {
      MS_LOG(EXCEPTION) << "List or Dict nesting is not allowed more than " << kMaxSeqRecursiveDepth << " levels.";
    }
    // AbstractList --> AbstractTuple.
    auto abs_seq = abs->cast<AbstractSequencePtr>();
    if (abs_seq != nullptr) {
      // Dynamic length sequence do not convert.
      if (abs_seq->dynamic_len()) {
        return abs->Clone();
      }
      const auto &seq_elements = abs_seq->elements();
      // First we check if elements should be converted,
      // changed_elements maps old element to new element.
      mindspore::HashMap<AbstractBasePtr, AbstractBasePtr> changed_elements;
      for (const auto &element : seq_elements) {
        auto new_element = ConvertToAbstractTuple(element, depth + 1);
        if (new_element != nullptr) {
          (void)changed_elements.emplace(element, new_element);
        }
      }
      if (changed_elements.empty()) {
        if (abs->isa<AbstractTuple>()) {
          // If no elements changed and it is an AbstractTuple, do not convert.
          return nullptr;
        }
        // If no elements changed but it is not an AbstractTuple, convert it by copy elements.
        return std::make_shared<AbstractTuple>(seq_elements);
      }
      // Always make new AbstractTuple when elements changed.
      std::vector<AbstractBasePtr> elements;
      elements.reserve(seq_elements.size());
      for (const auto &element : seq_elements) {
        auto iter = changed_elements.find(element);
        if (iter != changed_elements.end()) {
          (void)elements.emplace_back(iter->second);
        } else {
          (void)elements.emplace_back(element);
        }
      }
      return std::make_shared<AbstractTuple>(std::move(elements));
    }
    // AbstractDict --> AbstractTuple.
    auto abs_dict = abs->cast<AbstractDictionaryPtr>();
    if (abs_dict != nullptr) {
      const auto &dict_elements = abs_dict->elements();
      std::vector<AbstractBasePtr> elements;
      elements.reserve(dict_elements.size());
      for (const auto &element : dict_elements) {
        auto new_element = ConvertToAbstractTuple(element.second, depth + 1);
        if (new_element != nullptr) {
          (void)elements.emplace_back(new_element);
        } else {
          (void)elements.emplace_back(element.second);
        }
      }
      return std::make_shared<AbstractTuple>(elements);
    }
    // AbstractRowTensor --> AbstractTuple.
    auto abs_row_tensor = abs->cast<std::shared_ptr<AbstractRowTensor>>();
    if (abs_row_tensor != nullptr) {
      std::vector<AbstractBasePtr> elements{abs_row_tensor->indices(), abs_row_tensor->values(),
                                            abs_row_tensor->dense_shape()};
      return std::make_shared<AbstractTuple>(std::move(elements));
    }
    return nullptr;
  }

  AbstractBasePtr ConvertAbstract(const AbstractBasePtr &abs) override {
    // AbstractSequence, AbstractDict, AbstractRowTensor --> AbstractTuple.
    return ConvertToAbstractTuple(abs, 0);
  }
};
}  // namespace

bool SimplifyDataStructures(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);
  SimplifyDataStructuresRewriter rewriter(root, manager);
  return rewriter.Execute();
}

bool CleanAfterOptA(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);
  CleanAfterOptARewriter rewriter(root, manager);
  return rewriter.Execute();
}
}  // namespace opt
}  // namespace mindspore
