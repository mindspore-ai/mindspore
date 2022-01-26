/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <string>
#include <vector>
#include <algorithm>
#include "debug/trace.h"
#include "frontend/operator/composite/composite.h"
#include "pipeline/jit/parse/resolve.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
using mindspore::abstract::AbstractAttribute;
using mindspore::abstract::AbstractClass;
using mindspore::abstract::AbstractCOOTensor;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractJTagged;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractRowTensor;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractUndetermined;

inline void CheckInputsSize(size_t actual_size, size_t expect_size, const std::string &op_name) {
  if (actual_size != expect_size) {
    MS_LOG(EXCEPTION) << op_name << " should have " << expect_size << " inputs, but got " << actual_size;
  }
}

static AbstractBasePtr Reabs(const AbstractBasePtr &t) {
  if (t == nullptr) {
    return nullptr;
  }

  if (t->isa<AbstractClass>()) {
    auto abs_class = dyn_cast<AbstractClass>(t);
    AbstractBasePtrList baselist;
    auto attributes = abs_class->attributes();
    (void)std::transform(attributes.begin(), attributes.end(), std::back_inserter(baselist),
                         [](const AbstractAttribute &item) { return item.second; });
    return std::make_shared<AbstractTuple>(baselist);
  }
  if (t->isa<AbstractDictionary>()) {
    auto abs_dict = dyn_cast<AbstractDictionary>(t);
    AbstractBasePtrList baselist;
    auto elements = abs_dict->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(baselist),
                         [](const AbstractAttribute &item) { return item.second; });
    return std::make_shared<AbstractTuple>(baselist);
  }

  return nullptr;
}

static AbstractBasePtr AdaptAbs(const AbstractBasePtr &t) {
  if (t == nullptr) {
    return nullptr;
  }

  if (t->isa<AbstractList>()) {
    auto abs_list = dyn_cast<AbstractList>(t);
    return std::make_shared<AbstractTuple>(abs_list->elements());
  }

  if (t->isa<AbstractCOOTensor>()) {
    auto abs_sparse = dyn_cast<AbstractCOOTensor>(t);
    std::vector<AbstractBasePtr> abstract_list{abs_sparse->indices(), abs_sparse->values(), abs_sparse->dense_shape()};
    return std::make_shared<AbstractTuple>(abstract_list);
  }

  if (t->isa<AbstractRowTensor>()) {
    auto abs_row_tensor = dyn_cast<AbstractRowTensor>(t);
    std::vector<AbstractBasePtr> abstract_list{abs_row_tensor->indices(), abs_row_tensor->values(),
                                               abs_row_tensor->dense_shape()};
    return std::make_shared<AbstractTuple>(abstract_list);
  }

  return nullptr;
}

AnfNodePtr ConvertGetAttrToTupleGetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  const auto &inputs = node->inputs();
  // Inputs should be [getattr, data, attribute]
  const size_t expect_inputs_size = 3;
  CheckInputsSize(inputs.size(), expect_inputs_size, GetCNodeFuncName(node));

  constexpr size_t data_index = 1;
  constexpr size_t attribute_index = 2;
  AnfNodePtr data = inputs[data_index];
  AnfNodePtr cons = inputs[attribute_index];
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(cons);

  auto dt = data->abstract();
  if (dt == nullptr || dt->BuildType()->type_id() == kObjectTypeUndeterminedType) {
    return nullptr;
  }

  if (!dt->isa<AbstractClass>()) {
    MS_LOG(EXCEPTION) << "First parameter of getattr is not AbstractClass, but " << dt->type_name() << ".";
  }

  auto cons_is_str = IsValueNode<StringImm>(cons);
  auto cons_str = cons_is_str ? GetValue<std::string>(GetValueNode(cons)) : "";

  auto ct = dyn_cast<AbstractClass>(dt);
  const auto &cmap = ct->attributes();
  int64_t count = 0;
  for (auto &item : cmap) {
    if (cons_is_str && item.first == cons_str) {
      break;
    }
    count++;
  }

  auto idx_c = NewValueNode(count);
  AbstractBasePtr aptr = std::make_shared<AbstractScalar>(std::make_shared<Int64Imm>(count));
  idx_c->set_abstract(aptr);

  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, idx_c});
}

AnfNodePtr ConvertDictGetItemToTupleGetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  // Inputs should be [dict_getitem, dict, item]
  const auto &inputs = node->inputs();
  const size_t expect_inputs_size = 3;
  CheckInputsSize(inputs.size(), expect_inputs_size, GetCNodeFuncName(node));

  constexpr size_t data_index = 1;
  constexpr size_t cons_index = 2;
  AnfNodePtr data = inputs[data_index];
  AnfNodePtr cons = inputs[cons_index];
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(cons);

  auto dt = data->abstract();
  MS_EXCEPTION_IF_NULL(dt);
  if (!dt->isa<abstract::AbstractDictionary>()) {
    MS_LOG(EXCEPTION) << "first parameter of dict_getitem is not AbstractDictionary, but " << dt->type_name();
  }
  auto cons_is_str = IsValueNode<StringImm>(cons);
  auto cons_str = cons_is_str ? GetValue<std::string>(GetValueNode(cons)) : "";

  auto ct = dyn_cast<abstract::AbstractDictionary>(dt);
  const auto &cmap = ct->elements();
  int64_t count = 0;
  for (auto &item : cmap) {
    if (cons_is_str && item.first == cons_str) {
      break;
    }
    count++;
  }

  auto idx_c = NewValueNode(count);
  AbstractBasePtr aptr = std::make_shared<AbstractScalar>(std::make_shared<Int64Imm>(count));
  idx_c->set_abstract(aptr);
  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, idx_c});
}

AnfNodePtr ConvertDictSetItemToTupleSetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  // Inputs should be [dict_setitem, dict, item, value]
  const auto &inputs = node->inputs();
  const size_t expect_inputs_size = 4;
  CheckInputsSize(inputs.size(), expect_inputs_size, GetCNodeFuncName(node));

  const size_t data_index = 1;
  const size_t cons_index = 2;
  const size_t item_value_index = 3;
  AnfNodePtr data = inputs[data_index];
  AnfNodePtr cons = inputs[cons_index];
  AnfNodePtr item_value = inputs[item_value_index];
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(cons);

  auto dt = data->abstract();
  MS_EXCEPTION_IF_NULL(dt);
  if (!dt->isa<abstract::AbstractDictionary>()) {
    MS_LOG(EXCEPTION) << "first parameter of dict_setitem is not AbstractDictionary, but " << dt->type_name();
  }
  auto cons_is_str = IsValueNode<StringImm>(cons);
  auto cons_str = cons_is_str ? GetValue<std::string>(GetValueNode(cons)) : "";

  auto ct = dyn_cast<abstract::AbstractDictionary>(dt);
  const auto &cmap = ct->elements();
  int64_t count = 0;
  for (auto &item : cmap) {
    if (cons_is_str && item.first == cons_str) {
      break;
    }
    count++;
  }
  if (LongToSize(count) >= cmap.size()) {
    // for dictionary set, if the key does not exist, we should create a new item
    auto tuple_add_op = std::make_shared<prim::TupleAdd>("tuple_add");
    auto tuple_new_item = node->func_graph()->NewCNode({NewValueNode(prim::kPrimMakeTuple), item_value});
    return node->func_graph()->NewCNode({NewValueNode(tuple_add_op), data, tuple_new_item});
  }
  auto idx_c = NewValueNode(count);
  AbstractBasePtr aptr = std::make_shared<AbstractScalar>(std::make_shared<Int64Imm>(count));
  idx_c->set_abstract(aptr);
  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, idx_c, item_value});
}

AnfNodePtr ConvertMakeRecordToMakeTuple(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  // Inputs of node should be [make_record, klass, attr1, attr2, ...], so offset by 2 to get attr;
  (void)inputs.insert(inputs.end(), node->inputs().begin() + 2, node->inputs().end());
  return node->func_graph()->NewCNode(inputs);
}

AnfNodePtr ErasePartialNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  const auto &inputs = node->inputs();
  // Inputs should be [partial, fn, arg1, ...], so offset by 2 to get arg;
  const size_t min_inputs_size = 2;
  if (inputs.size() < min_inputs_size) {
    MS_LOG(EXCEPTION) << "Partial should have at least 2 inputs, but got " << inputs.size();
  }

  std::vector<AnfNodePtr> args(inputs.begin() + 2, inputs.end());
  auto oper = inputs[1];
  if (IsPrimitive(oper, prim::kPrimMakeRecord)) {
    if (args.size() == 1) {
      return NewValueNode(prim::kPrimMakeTuple);
    }

    if (args.size() > 1) {
      std::vector<AnfNodePtr> new_inputs;
      new_inputs.emplace_back(NewValueNode(prim::kPrimPartial));
      new_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
      (void)new_inputs.insert(new_inputs.end(), args.begin() + 1, args.end());

      MS_EXCEPTION_IF_NULL(node->func_graph());
      return node->func_graph()->NewCNode(new_inputs);
    }
  }
  return nullptr;
}

AnfNodePtr ConvertMakeListToMakeTuple(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  // Inputs of node should be [make_list, item1, item2, ...], so offset by 1 to get items;
  (void)inputs.insert(inputs.end(), node->inputs().begin() + 1, node->inputs().end());
  return node->func_graph()->NewCNode(inputs);
}

AnfNodePtr ConvertListGetItemToTupleGetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  const auto &inputs = node->inputs();
  // Inputs should be [list_getitem, list, item]
  constexpr size_t expect_input_size = 3;
  CheckInputsSize(inputs.size(), expect_input_size, GetCNodeFuncName(node));
  constexpr size_t real_input_index = 1;
  constexpr size_t index_input_index = 2;
  AnfNodePtr data = inputs[real_input_index];
  AnfNodePtr cons = inputs[index_input_index];
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(cons);

  auto cons_node = cons->cast<ValueNodePtr>();
  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, cons_node});
}

AnfNodePtr ConvertListSetItemToTupleSetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  const auto &inputs = node->inputs();
  // Inputs should be [list_setitem, list, index, item]
  const size_t expect_inputs_size = 4;
  CheckInputsSize(inputs.size(), expect_inputs_size, GetCNodeFuncName(node));

  const size_t data_index = 1;
  const size_t cons_index = 2;
  const size_t value_index = 3;
  AnfNodePtr data = inputs[data_index];
  AnfNodePtr cons = inputs[cons_index];
  AnfNodePtr value = inputs[value_index];

  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, cons, value});
}

AnfNodePtr EraseMakeDictNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &inputs = node->inputs();
  const size_t expect_inputs_size = 3;
  CheckInputsSize(inputs.size(), expect_inputs_size, GetCNodeFuncName(node));
  return inputs[2];
}

AnfNodePtr EraseDictGetValues(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &inputs = node->inputs();
  const size_t expect_inputs_size = 2;
  CheckInputsSize(inputs.size(), expect_inputs_size, GetCNodeFuncName(node));
  return inputs[1];
}

AnfNodePtr EraseDictItems(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &inputs = node->inputs();
  const size_t expect_inputs_size = 2;
  CheckInputsSize(inputs.size(), expect_inputs_size, GetCNodeFuncName(node));
  const auto &tmp = inputs[0]->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(tmp);
  MS_EXCEPTION_IF_NULL(tmp->value()->cast<ValueTuplePtr>());
  ValuePtrList keys = tmp->value()->cast<ValueTuplePtr>()->value();
  std::vector<AnfNodePtr> outer_node{NewValueNode(prim::kPrimMakeList)};
  for (size_t i = 0; i < keys.size(); ++i) {
    std::vector<AnfNodePtr> inner_node;
    inner_node.push_back(NewValueNode(prim::kPrimMakeTuple));
    inner_node.push_back(NewValueNode(keys[i]));
    inner_node.push_back(NewCNode(
      std::vector<AnfNodePtr>{NewValueNode(prim::kPrimTupleGetItem), inputs[1], NewValueNode(i)}, node->func_graph()));
    outer_node.push_back(NewCNode(inner_node, node->func_graph()));
  }
  return NewCNode(outer_node, node->func_graph());
}

AnfNodePtr EraseMakeKeywordArgNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &inputs = node->inputs();
  // Inputs should be [make_keyword_arg, key, value]
  constexpr size_t expect_input_size = 3;
  constexpr size_t value_inputs_index = 2;
  CheckInputsSize(inputs.size(), expect_input_size, GetCNodeFuncName(node));
  return inputs[value_inputs_index];
}

AnfNodePtr EraseExtractKeywordArg(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &inputs = node->inputs();
  // Inputs should be [extract_keyword_arg, arg, key]
  const size_t expect_inputs_size = 3;
  CheckInputsSize(inputs.size(), expect_inputs_size, GetCNodeFuncName(node));
  constexpr size_t key_index = 2;
  return inputs[key_index];
}

ValueTuplePtr ConvertValueListToValueTuple(const ValueListPtr &value_list, int64_t depth) {
  const int64_t DEPTH_MAX = 5;
  if (depth > DEPTH_MAX) {
    MS_LOG(EXCEPTION) << "List nesting is not allowed more than 6 levels.";
  }
  std::vector<ValuePtr> elements;
  for (const auto &it : value_list->value()) {
    ValuePtr value = nullptr;
    if (it->isa<ValueList>()) {
      value = ConvertValueListToValueTuple(it->cast<ValueListPtr>(), depth + 1);
    } else {
      value = it;
    }
    elements.push_back(value);
  }
  return std::make_shared<ValueTuple>(elements);
}

AnfNodePtr ConvertValueListNodeToValueTupleNode(const ValueNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  ValuePtr value = node->value();
  auto value_list = value->cast<ValueListPtr>();
  MS_EXCEPTION_IF_NULL(value_list);
  int64_t depth = 0;
  return std::make_shared<ValueNode>(ConvertValueListToValueTuple(value_list, depth));
}

// Convert class to Tuple
// Convert getattr to getitem
// Convert make_record to make_tuple
bool SimplifyDataStructures(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);

  bool changed = false;

  // Since `manager->Replace(...);` will modify member `all_nodes_`, so `all_node` can't be a ref var
  AnfNodeSet all_node = manager->all_nodes();
  for (auto &node : all_node) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    AnfNodePtr new_node = nullptr;
    if (IsValueNode<parse::ClassObject>(node)) {
      new_node = NewValueNode(prim::kPrimMakeTuple);
    } else if (IsPrimitiveCNode(node, prim::kPrimGetAttr)) {
      new_node = ConvertGetAttrToTupleGetItem(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeRecord)) {
      new_node = ConvertMakeRecordToMakeTuple(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimPartial)) {
      new_node = ErasePartialNode(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimDictGetItem)) {
      new_node = ConvertDictGetItemToTupleGetItem(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimDictSetItem)) {
      new_node = ConvertDictSetItemToTupleSetItem(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimDictGetValues)) {
      new_node = EraseDictGetValues(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeDict)) {
      new_node = EraseMakeDictNode(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeKeywordArg)) {
      new_node = EraseMakeKeywordArgNode(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimExtractKeywordArg)) {
      new_node = EraseExtractKeywordArg(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimDictItems)) {
      new_node = EraseDictItems(cnode);
    }

    if (new_node != nullptr) {
      new_node->set_abstract(node->abstract());
      MS_LOG(DEBUG) << "Replace node: " << node->DebugString() << " with new_node: " << new_node->DebugString();
      (void)manager->Replace(node, new_node);
      changed = true;
    }
  }

  for (auto &node : manager->all_nodes()) {
    auto ret = Reabs(node->abstract());
    if (ret) {
      MS_LOG(DEBUG) << "Replace " << node->DebugString() << "'s abstract " << node->abstract()->ToString() << " with "
                    << ret->ToString();
      node->set_abstract(ret);
      if (ret->cast<abstract::AbstractTuplePtr>()->size() > 0) {
        changed = true;
      }
    }
  }
  return changed;
}

AnfNodePtr ConvertMakeSparseToMakeTuple(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  // Inputs of node should be [make_sparse, indices, values, dense_shape], so offset by 1 to get items;
  (void)inputs.insert(inputs.end(), node->inputs().begin() + 1, node->inputs().end());
  return node->func_graph()->NewCNode(inputs);
}

AnfNodePtr ConvertSparseGetAttrToTupleGetItem(const CNodePtr &node, const int64_t &index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  const auto &inputs = node->inputs();
  // Inputs should be [sparse_getattr, sparse]
  constexpr size_t expect_input_index = 2;
  CheckInputsSize(inputs.size(), expect_input_index, GetCNodeFuncName(node));
  constexpr size_t sparse_index = 1;
  AnfNodePtr sparse = inputs[sparse_index];
  MS_EXCEPTION_IF_NULL(sparse);
  auto cons_node = NewValueNode(index);
  AbstractBasePtr aptr = std::make_shared<AbstractScalar>(std::make_shared<Int64Imm>(index));
  cons_node->set_abstract(aptr);

  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), sparse, cons_node});
}

bool CleanAfterOptA(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);

  bool changed = false;
  // Since `manager->Replace(...);` will modify member `all_nodes_`, so `all_node` can't be a ref var
  auto all_node = manager->all_nodes();
  for (auto &node : all_node) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    AnfNodePtr new_node = nullptr;
    if (IsPrimitiveCNode(node, prim::kPrimMakeList)) {
      new_node = ConvertMakeListToMakeTuple(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimListGetItem)) {
      new_node = ConvertListGetItemToTupleGetItem(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimListSetItem)) {
      new_node = ConvertListSetItemToTupleSetItem(cnode);
    } else if (IsValueNode<ValueList>(node)) {
      new_node = ConvertValueListNodeToValueTupleNode(node->cast<ValueNodePtr>());
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeRowTensor)) {
      new_node = ConvertMakeSparseToMakeTuple(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimRowTensorGetIndices)) {
      constexpr int64_t indices_index = 0;
      new_node = ConvertSparseGetAttrToTupleGetItem(cnode, indices_index);
    } else if (IsPrimitiveCNode(node, prim::kPrimRowTensorGetValues)) {
      constexpr int64_t value_index = 1;
      new_node = ConvertSparseGetAttrToTupleGetItem(cnode, value_index);
    } else if (IsPrimitiveCNode(node, prim::kPrimRowTensorGetDenseShape)) {
      constexpr int64_t shape_index = 2;
      new_node = ConvertSparseGetAttrToTupleGetItem(cnode, shape_index);
    }

    if (new_node != nullptr) {
      new_node->set_abstract(node->abstract());
      MS_LOG(DEBUG) << "Replace node: " << node->DebugString() << " with new_node: " << new_node->DebugString();
      (void)manager->Replace(node, new_node);
      changed = true;
    }
  }

  for (auto &node : manager->all_nodes()) {
    auto ret = AdaptAbs(node->abstract());
    if (ret) {
      MS_LOG(DEBUG) << "Replace " << node->DebugString() << "'s abstract " << node->abstract()->ToString() << " with "
                    << ret->ToString();
      node->set_abstract(ret);
      changed = true;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
