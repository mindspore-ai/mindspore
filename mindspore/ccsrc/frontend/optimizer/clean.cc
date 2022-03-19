/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "abstract/abstract_value.h"
#include "base/base.h"
#include "base/core_ops.h"
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
using mindspore::abstract::AbstractAttribute;
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractBasePtr;
using mindspore::abstract::AbstractClass;
using mindspore::abstract::AbstractCOOTensor;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractListPtr;
using mindspore::abstract::AbstractRowTensor;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;

namespace {
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
      : BaseRewriter(root_graph, manager) {}
  ~SimplifyDataStructuresRewriter() override = default;

 protected:
  static std::string GetStringValue(const AnfNodePtr &node) {
    auto str = GetValueNode<StringImmPtr>(node);
    if (str == nullptr) {
      return "";
    }
    return str->value();
  }

  static int64_t GetAttrIndex(const std::vector<AbstractAttribute> &attrs, const std::string &name) {
    auto n_attrs = attrs.size();
    for (size_t i = 0; i < n_attrs; ++i) {
      if (attrs[i].first == name) {
        return SizeToLong(i);
      }
    }
    return SizeToLong(n_attrs);
  }

  static CNodePtr NewTupleGetCNode(const AnfNodePtr &cnode, const AnfNodePtr &data_node,
                                   const std::vector<AbstractAttribute> &attributes, const AnfNodePtr &name_node) {
    int64_t index = GetAttrIndex(attributes, GetStringValue(name_node));
    auto index_node = NewValueNode(index);
    auto prim_node = NewValueNode(prim::kPrimTupleGetItem);
    return cnode->func_graph()->NewCNode({prim_node, data_node, index_node});
  }

  // From:
  //   GetAttr(data:AbstractClass, attr:StringImm)
  // To:
  //   TupleGetItem(data, index:Int64Imm)
  AnfNodePtr ConvertGetAttrToTupleGetItem(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [getattr, data, attribute].
    const size_t expect_inputs_size = 3;
    CheckInputsSize(node, expect_inputs_size);

    // Input arguments.
    constexpr size_t data_index = 1;
    constexpr size_t attr_index = 2;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &attr = inputs[attr_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(attr);

    auto abs_class = GetAbstract<AbstractClass>(data);
    if (abs_class == nullptr) {
      return nullptr;
    }
    return NewTupleGetCNode(node, data, abs_class->attributes(), attr);
  }

  // From:
  //   DictGetItem(data:AbstractDictionary, cons:StringImm)
  // To:
  //   TupleGetItem(data, index:Int64Imm)
  AnfNodePtr ConvertDictGetItemToTupleGetItem(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [dict_getitem, dict, item]
    const size_t expect_inputs_size = 3;
    CheckInputsSize(node, expect_inputs_size);

    constexpr size_t data_index = 1;
    constexpr size_t attr_index = 2;
    const auto &inputs = node->inputs();
    auto &data = inputs[data_index];
    auto &attr = inputs[attr_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(attr);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    return NewTupleGetCNode(node, data, abs_dict->elements(), attr);
  }

  // From:
  //   DictSetItem(data:AbstractDictionary, cons:StringImm, value)
  // To:
  //   TupleSetItem(data, index:Int64Imm, value)
  // Or:
  //   tuple_add(data, value)
  AnfNodePtr ConvertDictSetItemToTupleSetItem(const CNodePtr &node) {
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
    auto &cons = inputs[cons_index];
    auto &item_value = inputs[item_value_index];
    MS_EXCEPTION_IF_NULL(data);
    MS_EXCEPTION_IF_NULL(cons);

    auto abs_dict = GetAbstract<AbstractDictionary>(data);
    if (abs_dict == nullptr) {
      return nullptr;
    }
    int64_t index = GetAttrIndex(abs_dict->elements(), GetStringValue(cons));
    if (index >= static_cast<int64_t>(abs_dict->elements().size())) {
      // For dictionary set, if the key does not exist, we should create a new item.
      auto tuple_add_op = std::make_shared<prim::TupleAdd>("tuple_add");
      auto make_tuple_node = node->func_graph()->NewCNode({NewValueNode(prim::kPrimMakeTuple), item_value});
      return node->func_graph()->NewCNode({NewValueNode(tuple_add_op), data, make_tuple_node});
    }
    auto index_node = NewValueNode(index);
    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, index_node, item_value});
  }

  // From:
  //   MakeRecord(klass, attr1, attr2, ...)
  // To:
  //   MakeTuple(attr1, attr2, ...)
  AnfNodePtr ConvertMakeRecordToMakeTuple(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());
    std::vector<AnfNodePtr> inputs;
    inputs.reserve(node->size() - 1);
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    // Inputs of node should be [make_record, klass, attr1, attr2, ...], so offset by 2 to get attr.
    constexpr auto attr_start_index = 2;
    auto &old_inputs = node->inputs();
    (void)inputs.insert(inputs.end(), old_inputs.begin() + attr_start_index, old_inputs.end());
    return node->func_graph()->NewCNode(std::move(inputs));
  }

  // From:
  //   Partial(MakeRecord, arg1, arg2, ...)
  // To:
  //   Partial(MakeTuple, arg1, arg2, ...)
  // Or:
  //   MakeTuple  # not args
  AnfNodePtr ConvertPartialMakeRecord(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    const auto &inputs = node->inputs();
    // Inputs should be [partial, fn, arg1, ...], so offset by 2 to get arg;
    constexpr auto min_inputs_size = 2;
    if (inputs.size() < min_inputs_size) {
      MS_LOG(EXCEPTION) << "Partial should have at least 2 inputs, but got " << inputs.size();
    }
    if (!IsPrimitive(inputs[1], prim::kPrimMakeRecord)) {
      return nullptr;
    }
    if (inputs.size() == min_inputs_size) {
      return NewValueNode(prim::kPrimMakeTuple);
    }
    std::vector<AnfNodePtr> new_inputs;
    new_inputs.reserve(inputs.size());
    constexpr auto first_arg_idx = 2;
    (void)new_inputs.emplace_back(inputs[0]);
    (void)new_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    (void)new_inputs.insert(new_inputs.end(), inputs.begin() + first_arg_idx, inputs.end());
    return node->func_graph()->NewCNode(std::move(new_inputs));
  }

  // From:
  //   MakeDict(name, input)
  // To:
  //   input
  AnfNodePtr EraseMakeDictNode(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    constexpr size_t expect_inputs_size = 3;
    constexpr size_t input_index = 2;
    CheckInputsSize(node, expect_inputs_size);
    return node->input(input_index);
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
      auto key_node = NewValueNode(elements[i].first);
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
  ValueTuplePtr DictToTuple(const ValueDictionaryPtr &dict) {
    const auto &elements = dict->value();
    std::vector<ValuePtr> values;
    values.reserve(elements.size());
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(values),
                         [](const auto &element) { return element.second; });
    return std::make_shared<ValueTuple>(values);
  }

  using Converter = AnfNodePtr (ThisClass::*)(const CNodePtr &);
  using ConverterMap = mindspore::HashMap<PrimitivePtr, Converter, PrimitiveHasher, PrimitiveEqual>;
  static inline const ConverterMap converters_{
    {prim::kPrimGetAttr, &ThisClass::ConvertGetAttrToTupleGetItem},
    {prim::kPrimMakeRecord, &ThisClass::ConvertMakeRecordToMakeTuple},
    {prim::kPrimPartial, &ThisClass::ConvertPartialMakeRecord},
    {prim::kPrimDictGetItem, &ThisClass::ConvertDictGetItemToTupleGetItem},
    {prim::kPrimDictSetItem, &ThisClass::ConvertDictSetItemToTupleSetItem},
    {prim::kPrimDictGetValues, &ThisClass::EraseDictGetValues},
    {prim::kPrimMakeDict, &ThisClass::EraseMakeDictNode},
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

  AnfNodePtr ConvertValueNode(const ValueNodePtr &, const ValuePtr &value) override {
    // Convert ClassObject value node.
    if (value->isa<parse::ClassObject>()) {
      return NewValueNode(prim::kPrimMakeTuple);
    }
    // Convert Dictionary value node.
    if (value->isa<ValueDictionary>()) {
      return NewValueNode(DictToTuple(value->cast<ValueDictionaryPtr>()));
    }
    return nullptr;
  }

  static std::shared_ptr<AbstractTuple> MakeAbstractTuple(const std::vector<AbstractAttribute> &attrs) {
    std::vector<AbstractBasePtr> elements;
    elements.reserve(attrs.size());
    (void)std::transform(attrs.begin(), attrs.end(), std::back_inserter(elements),
                         [](const auto &item) { return item.second; });
    return std::make_shared<AbstractTuple>(std::move(elements));
  }

  AbstractBasePtr ConvertAbstract(const AbstractBasePtr &abs) override {
    // AbstractDictionary --> AbstractTuple.
    auto abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    if (abs_dict != nullptr) {
      return MakeAbstractTuple(abs_dict->elements());
    }
    // AbstractClass --> AbstractTuple.
    auto abs_class = abs->cast<abstract::AbstractClassPtr>();
    if (abs_class != nullptr) {
      return MakeAbstractTuple(abs_class->attributes());
    }
    return nullptr;
  }
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
    (void)inputs.insert(inputs.end(), node->inputs().begin() + 1, node->inputs().end());
    return node->func_graph()->NewCNode(std::move(inputs));
  }

  // From:
  //   ListGetItem(list, cons)
  // To:
  //   TupleGetItem(list, cons)
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
    auto &cons = inputs[cons_index];
    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, cons});
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
    auto &cons = inputs[cons_index];
    auto &value = inputs[value_index];
    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, cons, value});
  }

  // From:
  //   MakeSparse(indices, values, dense_shape)
  // To:
  //   MakeTuple(indices, values, dense_shape)
  AnfNodePtr ConvertMakeSparseToMakeTuple(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    std::vector<AnfNodePtr> inputs;
    inputs.reserve(node->size());
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    // Inputs of node should be [make_sparse, indices, values, dense_shape], so offset by 1 to get items.
    (void)inputs.insert(inputs.end(), node->inputs().begin() + 1, node->inputs().end());
    return node->func_graph()->NewCNode(std::move(inputs));
  }

  // From:
  //   RowTensorGetXXX(sparse) # index
  // To:
  //   TupleGetItem(sparse, index)
  AnfNodePtr ConvertSparseGetAttr(const CNodePtr &node, const int64_t index) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(node->func_graph());

    // Inputs should be [sparse_getattr, sparse]
    constexpr size_t expect_input_index = 2;
    CheckInputsSize(node, expect_input_index);

    const auto &sparse = node->input(1);
    auto cons_node = NewValueNode(index);
    return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), sparse, cons_node});
  }

  AnfNodePtr ConvertSparseGetIndices(const CNodePtr &node) { return ConvertSparseGetAttr(node, 0); }

  AnfNodePtr ConvertSparseGetValues(const CNodePtr &node) { return ConvertSparseGetAttr(node, 1); }

  AnfNodePtr ConvertSparseGetDenseShape(const CNodePtr &node) {
    constexpr int64_t dense_shape_index = 2;
    return ConvertSparseGetAttr(node, dense_shape_index);
  }

  using Converter = AnfNodePtr (ThisClass::*)(const CNodePtr &);
  using ConverterMap = mindspore::HashMap<PrimitivePtr, Converter, PrimitiveHasher, PrimitiveEqual>;
  static inline const ConverterMap converters_{
    {prim::kPrimMakeList, &ThisClass::ConvertMakeListToMakeTuple},
    {prim::kPrimListGetItem, &ThisClass::ConvertListGetItemToTupleGetItem},
    {prim::kPrimListSetItem, &ThisClass::ConvertListSetItemToTupleSetItem},
    {prim::kPrimMakeRowTensor, &ThisClass::ConvertMakeSparseToMakeTuple},
    {prim::kPrimRowTensorGetIndices, &ThisClass::ConvertSparseGetIndices},
    {prim::kPrimRowTensorGetValues, &ThisClass::ConvertSparseGetValues},
    {prim::kPrimRowTensorGetDenseShape, &ThisClass::ConvertSparseGetDenseShape},
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

  static constexpr size_t kMaxListRecursiveDepth = 5;

  // ValueList --> ValueTuple
  static ValueTuplePtr ConvertValueListToValueTuple(const ValueListPtr &value_list, size_t depth) {
    if (depth > kMaxListRecursiveDepth) {
      MS_LOG(EXCEPTION) << "List nesting is not allowed more than " << kMaxListRecursiveDepth << " levels.";
    }
    const auto &list_elements = value_list->value();
    std::vector<ValuePtr> elements;
    elements.reserve(list_elements.size());
    for (const auto &element : list_elements) {
      if (element->isa<ValueList>()) {
        (void)elements.emplace_back(ConvertValueListToValueTuple(element->cast<ValueListPtr>(), depth + 1));
      } else {
        (void)elements.emplace_back(element);
      }
    }
    return std::make_shared<ValueTuple>(elements);
  }

  AnfNodePtr ConvertValueNode(const ValueNodePtr &, const ValuePtr &value) override {
    auto value_list = dyn_cast<ValueList>(value);
    if (value_list != nullptr) {
      return std::make_shared<ValueNode>(ConvertValueListToValueTuple(value_list, 0));
    }
    return nullptr;
  }

  // AbstractList --> AbstractTuple
  static AbstractTuplePtr ConvertAbstractListToAbstractTuple(const AbstractListPtr &abs_list, size_t depth) {
    if (depth > kMaxListRecursiveDepth) {
      MS_LOG(EXCEPTION) << "List nesting is not allowed more than " << kMaxListRecursiveDepth << " levels.";
    }
    const auto &list_elements = abs_list->elements();
    std::vector<AbstractBasePtr> elements;
    elements.reserve(list_elements.size());
    for (const auto &element : list_elements) {
      if (element->isa<AbstractList>()) {
        (void)elements.emplace_back(ConvertAbstractListToAbstractTuple(element->cast<AbstractListPtr>(), depth + 1));
      } else {
        (void)elements.emplace_back(element);
      }
    }
    return std::make_shared<AbstractTuple>(std::move(elements));
  }

  AbstractBasePtr ConvertAbstract(const AbstractBasePtr &abs) override {
    // AbstractList --> AbstractTuple.
    auto abs_list = abs->cast<AbstractListPtr>();
    if (abs_list != nullptr) {
      return ConvertAbstractListToAbstractTuple(abs_list, 0);
    }
    // AbstractCOOTensor --> AbstractTuple.
    auto abs_sparse = abs->cast<abstract::AbstractCOOTensorPtr>();
    if (abs_sparse != nullptr) {
      std::vector<AbstractBasePtr> elements{abs_sparse->indices(), abs_sparse->values(), abs_sparse->dense_shape()};
      return std::make_shared<AbstractTuple>(std::move(elements));
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
