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

#include "frontend/optimizer/fallback_rewriter.h"
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
#include "pipeline/jit/action.h"
#include "frontend/optimizer/opt.h"
#include "frontend/operator/composite/composite.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "pipeline/jit/fallback.h"
#include "pipeline/jit/parse/resolve.h"
#include "utils/hash_map.h"
#include "utils/anf_utils.h"
#include "utils/tensor_construct_utils.h"

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

  bool need_renormalized() const { return need_renormalized_; }

  void set_need_renormalized(bool need_renormalized) { need_renormalized_ = need_renormalized; }

  virtual bool Execute() {
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
    if (IsPrimitiveCNode(new_node, prim::kPrimPyExecute)) {
      need_renormalized_ = true;
      return new_node;
    }
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

 private:
  bool need_renormalized_{false};
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

  bool Execute() override {
    bool changed = Run();
    if (changed) {
      UpdateAbstracts();
    }
    ConvertParameter();
    return changed;
  }

 protected:
  void ConvertParameter() {
    const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
    if (!support_fallback_runtime || !is_dict_output_) {
      return;
    }
    for (const auto &para : root_graph_->parameters()) {
      auto new_node_and_abs = ConvertParameterDictAbstract(para, para->abstract());
      if (new_node_and_abs.first == para) {
        continue;
      }
      manager_->Replace(para, new_node_and_abs.first);
      para->set_abstract(new_node_and_abs.second);
    }
  }

  std::pair<AnfNodePtr, AbstractBasePtr> ConvertParameterDictAbstract(const AnfNodePtr &cur_node,
                                                                      const AbstractBasePtr &cur_abs) {
    MS_EXCEPTION_IF_NULL(cur_abs);
    auto seq_abs = cur_abs->cast_ptr<AbstractSequence>();
    if (seq_abs != nullptr) {
      bool is_tuple = seq_abs->isa<AbstractTuple>();
      auto seq_prim = is_tuple ? prim::kPrimMakeTuple : prim::kPrimMakeList;
      std::vector<AnfNodePtr> seq_inputs{NewValueNode(seq_prim)};
      AbstractBasePtrList abs_list;
      for (size_t i = 0; i < seq_abs->elements().size(); ++i) {
        auto getitem_prim = is_tuple ? prim::kPrimTupleGetItem : prim::kPrimListGetItem;
        auto next_node =
          root_graph_->NewCNodeInOrder({NewValueNode(getitem_prim), cur_node, NewValueNode(SizeToLong(i))});
        auto node_and_abs = ConvertParameterDictAbstract(next_node, seq_abs->elements()[i]);
        (void)seq_inputs.emplace_back(node_and_abs.first);
        (void)abs_list.emplace_back(node_and_abs.second);
      }
      if (is_tuple) {
        return std::make_pair(root_graph_->NewCNodeInOrder(seq_inputs), std::make_shared<AbstractTuple>(abs_list));
      }
      return std::make_pair(root_graph_->NewCNodeInOrder(seq_inputs), std::make_shared<AbstractList>(abs_list));
    }
    auto dict_abs = cur_abs->cast_ptr<AbstractDictionary>();
    if (dict_abs != nullptr) {
      std::vector<AnfNodePtr> key_inputs{NewValueNode(prim::kPrimMakeTuple)};
      std::vector<AnfNodePtr> value_inputs{NewValueNode(prim::kPrimMakeTuple)};
      AbstractBasePtrList abs_list;
      for (size_t i = 0; i < dict_abs->elements().size(); ++i) {
        auto next_node =
          root_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), cur_node, NewValueNode(SizeToLong(i))});
        auto node_and_abs = ConvertParameterDictAbstract(next_node, dict_abs->elements()[i].second);
        (void)key_inputs.emplace_back(NewValueNode(dict_abs->elements()[i].first->BuildValue()));
        (void)value_inputs.emplace_back(node_and_abs.first);
        (void)abs_list.emplace_back(node_and_abs.second);
      }
      auto make_dict =
        root_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimMakeDict), root_graph_->NewCNodeInOrder(key_inputs),
                                      root_graph_->NewCNodeInOrder(value_inputs)});
      return std::make_pair(make_dict, std::make_shared<AbstractTuple>(abs_list));
    }
    return std::make_pair(cur_node, cur_abs);
  }

  static std::string GetStringValue(const AnfNodePtr &node) {
    auto str = GetValueNode<StringImmPtr>(node);
    if (str == nullptr) {
      return "";
    }
    return str->value();
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

  AnfNodePtr ConvertDictGetItem(const CNodePtr &node) {
    const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
    if (!support_fallback_runtime || !is_dict_output_) {
      return ConvertDictGetItemToTupleGetItem(node);
    }
    return nullptr;
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

  AnfNodePtr ConvertDictSetItem(const CNodePtr &node) {
    const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
    if (!support_fallback_runtime || !is_dict_output_) {
      return ConvertDictSetItemToTupleSetItem(node);
    }
    return nullptr;
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

  AnfNodePtr ConvertMakeDict(const CNodePtr &node) {
    const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
    if (!support_fallback_runtime || !is_dict_output_) {
      return EraseMakeDictNode(node);
    }
    return nullptr;
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
      const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
      if (!support_fallback_runtime || !is_dict_output_) {
        return DictToTuple(value->cast<ValueDictionaryPtr>());
      }
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

  void UpdateAbstracts() {
    const auto &nodes = manager_->all_nodes();
    for (const auto &node : nodes) {
      const auto &abs = node->abstract();
      if (abs == nullptr) {
        continue;
      }
      // Set flag for convert AbstractNone(PyExecute) to AbstractTensor in next renormalize.
      if (IsPrimitiveCNode(node, prim::kPrimPyExecute) && abs->isa<abstract::AbstractNone>()) {
        constexpr auto data_type = "__py_execute_no_return_type__";
        if (node->has_user_data(data_type)) {
          auto type = std::make_shared<TypeAnything>();
          node->set_user_data<Type>(data_type, type);
          set_need_renormalized(true);
        }
      }
    }
  }

 protected:
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

  // DictGetItem --> PyExecute()
  AnfNodePtr ConvertDictGetItem(const CNodePtr &node) {
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
    const auto dict_getitem_node = func_graph->NewCNodeInOrder(
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

  // DictSetItem --> PyExecute()
  AnfNodePtr ConvertDictSetItem(const CNodePtr &node) {
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
    const auto dict_setitem_node = func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimPyExecute), NewValueNode(script_str), key_value_name_tuple, key_value_tuple});
    MS_LOG(DEBUG) << "Made dict setitem node: " << dict_setitem_node->DebugString();
    dict_setitem_node->set_debug_info(node->debug_info());
    return dict_setitem_node;
  }

  AnfNodePtr ConstructInternalTupleKeysNode(const FuncGraphPtr &fg, const AnfNodePtr &keys_node) {
    constexpr auto internal_tuple_keys_str = "__internal_tuple_keys__";
    MS_EXCEPTION_IF_NULL(fg);
    const auto script_key_tuple_str = std::make_shared<StringImm>(internal_tuple_keys_str);
    auto dict_py_exec_key = std::make_shared<ValueTuple>(std::vector<ValuePtr>{script_key_tuple_str});
    auto dict_tuple_key_value = fg->NewCNode({std::make_shared<ValueNode>(prim::kPrimMakeTuple), keys_node});
    const auto make_key_tuple_node =
      fg->NewCNode({NewValueNode(prim::kPrimPyExecute), NewValueNode(script_key_tuple_str),
                    NewValueNode(dict_py_exec_key), dict_tuple_key_value});
    return make_key_tuple_node;
  }

  AnfNodePtr ConstructInternalTupleValueNode(const FuncGraphPtr &fg, const AnfNodePtr &values_node) {
    constexpr auto internal_tuple_values_str = "__internal_tuple_values__";
    MS_EXCEPTION_IF_NULL(fg);
    const auto script_value_tuple_str = std::make_shared<StringImm>(internal_tuple_values_str);
    auto dict_py_exec_value = std::make_shared<ValueTuple>(std::vector<ValuePtr>{script_value_tuple_str});
    auto dict_tuple_node = fg->NewCNode({std::make_shared<ValueNode>(prim::kPrimMakeTuple), values_node});
    const auto make_value_tuple_node =
      fg->NewCNode({NewValueNode(prim::kPrimPyExecute), NewValueNode(script_value_tuple_str),
                    NewValueNode(dict_py_exec_value), dict_tuple_node});
    return make_value_tuple_node;
  }

  AnfNodePtr ConstructNewDictNode(const FuncGraphPtr &fg, const AnfNodePtr &make_key_tuple_node,
                                  const AnfNodePtr &make_value_tuple_node) {
    constexpr auto internal_dict_zip_keys_str = "__internal_dict_zip_keys__";
    constexpr auto internal_dict_zip_values_str = "__internal_dict_zip_values__";
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

    // Construct Script Node
    std::stringstream script_buffer;
    script_buffer << "dict(zip(" << internal_dict_zip_keys_str << "," << internal_dict_zip_values_str << "),)";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);

    // Build the new dict node.
    const auto make_dict_node = fg->NewCNodeInOrder(
      {NewValueNode(prim::kPrimPyExecute), NewValueNode(script_str), key_value_name_tuple, key_value_tuple});
    MS_LOG(DEBUG) << "Made dict node: " << make_dict_node->DebugString();
    return make_dict_node;
  }

  // MakeDict(keys, values) --> PyExecute('dict(zip(keys, values))', ...)
  AnfNodePtr ConvertMakeDict(const CNodePtr &node) {
    const auto &fg = node->func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    // Local parameters values.
    // Get the key tuple.
    constexpr size_t keys_input_index = 1;
    auto keys_node = node->input(keys_input_index);
    const auto make_key_tuple_node = ConstructInternalTupleKeysNode(fg, keys_node);
    make_key_tuple_node->set_debug_info(node->input(keys_input_index)->debug_info());
    // Get the value tuple.
    constexpr size_t values_input_index = 2;
    auto values_node = node->input(values_input_index);
    const auto make_value_tuple_node = ConstructInternalTupleValueNode(fg, values_node);
    make_value_tuple_node->set_debug_info(node->input(values_input_index)->debug_info());

    auto new_dict_node = ConstructNewDictNode(fg, make_key_tuple_node, make_value_tuple_node);
    new_dict_node->set_debug_info(node->debug_info());
    return new_dict_node;
  }

  using Converter = AnfNodePtr (ThisClass::*)(const CNodePtr &);
  using ConverterMap = mindspore::HashMap<PrimitivePtr, Converter, PrimitiveHasher, PrimitiveEqual>;
  static inline const ConverterMap converters_{
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
    {prim::kPrimDictGetItem, &ThisClass::ConvertDictGetItem},
    {prim::kPrimDictSetItem, &ThisClass::ConvertDictSetItem},
    {prim::kPrimMakeDict, &ThisClass::ConvertMakeDict},
  };

  // Convert ValueNode<None> to PyExecute("None", ("None"), ("None")).
  AnfNodePtr NoneConvertPyExecute(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto str_value = std::make_shared<StringImm>("None");
    auto script_node = NewValueNode(str_value);

    std::vector<ValuePtr> none_value{str_value};
    const auto none_tuple = std::make_shared<ValueTuple>(none_value);
    auto none_tuple_node = NewValueNode(none_tuple);

    AnfNodePtr none_execute_node =
      func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimPyExecute), script_node, none_tuple_node, none_tuple_node});
    MS_LOG(DEBUG) << "none_execute_node:" << none_execute_node->DebugString();

    // Keep AbstractNone for PyExecute, because the control flow join problem.
    auto none_type = std::make_shared<TypeNone>();
    none_execute_node->set_user_data<Type>("__py_execute_no_return_type__", none_type);
    AbstractBasePtr res = std::make_shared<abstract::AbstractNone>();
    res->set_value(kAnyValue);
    none_execute_node->set_abstract(res);
    return none_execute_node;
  }

  void CheckCNodeInputsHasNone(const CNodePtr &cnode) {
    MS_EXCEPTION_IF_NULL(cnode);
    const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
    if (!support_fallback_runtime) {
      return;
    }
    if (AnfUtils::IsRealKernel(cnode)) {
      return;
    }
    const auto &inputs = cnode->inputs();
    const auto &cur_func = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cur_func);
    for (auto &input : inputs) {
      if (!IsValueNode<None>(input)) {
        continue;
      }
      auto none_py_execute = NoneConvertPyExecute(cur_func);
      manager_->Replace(input, none_py_execute);
      set_need_renormalized(true);
    }
  }

  AnfNodePtr ConvertPrimitiveCNode(const CNodePtr &cnode, const PrimitivePtr &prim) override {
    // Process None in CNode with JIT Fallback: convert ValueNode<None> to PyExecute("None", (), ()).
    CheckCNodeInputsHasNone(cnode);
    // Find cnode converter by primitive.
    auto iter = converters_.find(prim);
    if (iter == converters_.end()) {
      return nullptr;
    }
    // Call converter.
    return (this->*(iter->second))(cnode);
  }

  AnfNodePtr ValueListConvertPyExecute(const ValuePtr &value, const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(value);
    MS_EXCEPTION_IF_NULL(func_graph);
    auto value_list = value->cast<ValueListPtr>();
    MS_EXCEPTION_IF_NULL(value_list);
    auto values = value_list->value();

    // Script and local parameters keys
    constexpr auto internal_element_str_prefix = "__internal_list_element_";
    std::stringstream script_buffer;
    script_buffer << "list((";
    std::vector<AnfNodePtr> key_value_names_list{NewValueNode(prim::kPrimMakeTuple)};
    for (size_t i = 0; i < values.size(); ++i) {
      std::string element_name_str = internal_element_str_prefix + std::to_string(i) + "__";
      script_buffer << element_name_str << ", ";
      const auto element_name = std::make_shared<StringImm>(element_name_str);
      (void)key_value_names_list.emplace_back(NewValueNode(element_name));
    }
    script_buffer << "))";
    const std::string &script = script_buffer.str();
    const auto script_str = std::make_shared<StringImm>(script);
    const auto key_value_name_tuple = func_graph->NewCNode(key_value_names_list);

    // Pack the local parameters values, not support list, tuple, or dict.
    std::vector<AnfNodePtr> key_value_list{NewValueNode(prim::kPrimMakeTuple)};
    for (auto element : values) {
      auto converted_element = ProcessValueSequence(element);
      (void)key_value_list.emplace_back(converted_element);
    }
    const auto key_value_tuple = func_graph->NewCNode(key_value_list);

    // Build the new dict node.
    const auto list_value_node = func_graph->NewCNodeInOrder(
      {NewValueNode(prim::kPrimPyExecute), NewValueNode(script_str), key_value_name_tuple, key_value_tuple});
    MS_LOG(DEBUG) << "List value node convert to PyExecute node: " << list_value_node->DebugString();
    return list_value_node;
  }

  AnfNodePtr ProcessValueSequence(const ValuePtr &value) {
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueTuple>()) {
      auto value_seq = value->cast<ValueTuplePtr>();
      MS_EXCEPTION_IF_NULL(value_seq);
      auto values = value_seq->value();
      std::vector<AnfNodePtr> value_seq_inputs;
      (void)value_seq_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
      for (auto inner_value : values) {
        auto inner_value_seq = ProcessValueSequence(inner_value);
        (void)value_seq_inputs.emplace_back(inner_value_seq);
      }
      auto iter_value = root_graph_->NewCNode(value_seq_inputs);
      return iter_value;
    } else if (value->isa<ValueList>()) {
      return ValueListConvertPyExecute(value, root_graph_);
    }
    if (value->isa<None>()) {
      return NoneConvertPyExecute(root_graph_);
    }
    return NewValueNode(value);
  }

  AnfNodePtr PackDictValue(const ValueDictionaryPtr &dict) {
    const auto &keys_values = dict->value();
    std::vector<AnfNodePtr> value_list{NewValueNode(prim::kPrimMakeTuple)};
    for (const auto &key_value : keys_values) {
      auto iter_value = ProcessValueSequence(key_value.second);
      (void)value_list.emplace_back(iter_value);
    }
    auto value_tuple_node = root_graph_->NewCNode(value_list);
    return value_tuple_node;
  }

  // dict(k0:v0, k1:v1, ...) --> PyExecute('dict(zip(keys, values))', ...)
  AnfNodePtr RebuildValueDict(const ValueNodePtr &value_node, const ValueDictionaryPtr &dict) {
    const auto &keys_values = dict->value();

    // Local parameters values.
    // Pack the key tuple.
    std::vector<ValuePtr> key_list;
    key_list.reserve(keys_values.size());
    for (const auto &key_value : keys_values) {
      (void)key_list.emplace_back(key_value.first);
    }
    const auto key_tuple = std::make_shared<ValueTuple>(key_list);
    auto key_tuple_node = NewValueNode(key_tuple);

    // Pack the value tuple.
    auto value_tuple_node = PackDictValue(dict);

    // Generate Make Dict PyExecute Node value
    auto make_key_tuple_node = ConstructInternalTupleKeysNode(root_graph_, key_tuple_node);
    auto make_value_tuple_node = ConstructInternalTupleValueNode(root_graph_, value_tuple_node);

    auto make_dict_node = ConstructNewDictNode(root_graph_, make_key_tuple_node, make_value_tuple_node);
    make_dict_node->set_debug_info(value_node->debug_info());
    return make_dict_node;
  }

  AnfNodePtr ConvertInterpretedObjectValue(const ValueNodePtr &node, const parse::InterpretedObjectPtr &value) {
    // Convert InterpretedObject value node to PyExecute CNode.
    return ConvertInterpretedObjectToPyExecute(root_graph_, value, node);
  }

  AnfNodePtr ConvertValueNode(const ValueNodePtr &value_node, const ValuePtr &value) override {
    const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") != "0");
    if (support_fallback_runtime) {
      if (value->isa<ValueDictionary>()) {
        return RebuildValueDict(value_node, value->cast<ValueDictionaryPtr>());
      } else if (value->isa<parse::InterpretedObject>()) {
        return ConvertInterpretedObjectValue(value_node, value->cast<parse::InterpretedObjectPtr>());
      }
    }
    return nullptr;
  }

  // AbstractRowTensor --> AbstractTuple.
  static AbstractBasePtr ConvertToAbstractTuple(const AbstractBasePtr &abs, size_t depth) {
    if (depth > kMaxSeqRecursiveDepth) {
      MS_LOG(EXCEPTION) << "List or Dict nesting is not allowed more than " << kMaxSeqRecursiveDepth << " levels.";
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

bool CleanAfterOptA(const FuncGraphPtr &root, const pipeline::ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(resource);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);
  CleanAfterOptARewriter rewriter(root, manager);
  bool change = rewriter.Execute();
  // Renormalize for new PyExecute node.
  rewriter.UpdateAbstracts();
  if (rewriter.need_renormalized()) {
    abstract::AbstractBasePtrList new_args_spec;
    std::transform(root->parameters().begin(), root->parameters().end(), std::back_inserter(new_args_spec),
                   [](const AnfNodePtr &param) -> AbstractBasePtr { return param->abstract(); });
    (void)pipeline::Renormalize(resource, root, new_args_spec);
  }
  return change;
}
}  // namespace opt
}  // namespace mindspore
