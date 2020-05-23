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

#include "optimizer/clean.h"
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include "./common.h"
#include "debug/trace.h"
#include "operator/composite/composite.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
using mindspore::abstract::AbstractAttribute;
using mindspore::abstract::AbstractClass;
using mindspore::abstract::AbstractDictionary;
using mindspore::abstract::AbstractJTagged;
using mindspore::abstract::AbstractList;
using mindspore::abstract::AbstractScalar;
using mindspore::abstract::AbstractTuple;

static AbstractBasePtr Reabs(const AbstractBasePtr &t) {
  if (t == nullptr) {
    return nullptr;
  }

  AbstractBasePtr res = t;
  if (t->isa<AbstractClass>()) {
    auto abs_class = dyn_cast<AbstractClass>(t);
    AbstractBasePtrList baselist;
    auto attributes = abs_class->attributes();
    (void)std::transform(attributes.begin(), attributes.end(), std::back_inserter(baselist),
                         [](const AbstractAttribute &item) { return item.second; });
    res = std::make_shared<AbstractTuple>(baselist);
  } else if (t->isa<AbstractDictionary>()) {
    auto abs_dict = dyn_cast<AbstractDictionary>(t);
    AbstractBasePtrList baselist;
    auto elements = abs_dict->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(baselist),
                         [](const AbstractAttribute &item) { return item.second; });
    res = std::make_shared<AbstractTuple>(baselist);
  } else if (t->isa<AbstractList>()) {
    auto abs_dict = dyn_cast<AbstractList>(t);
    res = std::make_shared<AbstractTuple>(abs_dict->elements());
  }
  return res;
}

AnfNodePtr ConvertGetAttrToTupleGetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  const auto &inputs = node->inputs();
  // Inputs should be [getattr, data, attribute]
  MS_ASSERT(inputs.size() == 3 && "GetAttr should have three inputs.");

  AnfNodePtr data = inputs[1];
  AnfNodePtr cons = inputs[2];
  MS_EXCEPTION_IF_NULL(data);
  MS_EXCEPTION_IF_NULL(cons);

  auto dt = data->abstract();
  MS_EXCEPTION_IF_NULL(dt);
  if (!dt->isa<AbstractClass>()) {
    MS_LOG(EXCEPTION) << "First parameter of getattr is not AbstractClass, but " << dt->type_name() << ".";
  }

  auto cons_is_str = IsValueNode<StringImm>(cons);
  auto cons_str = cons_is_str ? GetValue<std::string>(GetValueNode(cons)) : "";

  auto ct = dyn_cast<AbstractClass>(dt);
  const auto &cmap = ct->attributes();
  int count = 0;
  for (auto &item : cmap) {
    if (cons_is_str && item.first == cons_str) {
      break;
    }
    count++;
  }

  auto idx_c = NewValueNode(count);
  AbstractBasePtr aptr = std::make_shared<AbstractScalar>(std::make_shared<Int32Imm>(count));
  idx_c->set_abstract(aptr);

  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, idx_c});
}

AnfNodePtr ConvertDictGetItemToTupleGetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  // Inputs should be [dict_getitem, dict, item]
  const auto &inputs = node->inputs();
  MS_ASSERT(inputs.size() == 3 && "DictGetItem should have three inputs.");

  AnfNodePtr data = inputs[1];
  AnfNodePtr cons = inputs[2];
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
  int count = 0;
  for (auto &item : cmap) {
    if (cons_is_str && item.first == cons_str) {
      break;
    }
    count++;
  }

  auto idx_c = NewValueNode(count);
  AbstractBasePtr aptr = std::make_shared<AbstractScalar>(std::make_shared<Int32Imm>(count));
  idx_c->set_abstract(aptr);
  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), data, idx_c});
}

AnfNodePtr ConvertDictSetItemToTupleSetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  // Inputs should be [dict_setitem, dict, item, value]
  const auto &inputs = node->inputs();
  MS_ASSERT(inputs.size() == 4 && "DictSetItem should have three inputs.");

  AnfNodePtr data = inputs[1];
  AnfNodePtr cons = inputs[2];
  AnfNodePtr item_value = inputs[3];
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
  int count = 0;
  for (auto &item : cmap) {
    if (cons_is_str && item.first == cons_str) {
      break;
    }
    count++;
  }
  if (IntToSize(count) >= cmap.size()) {
    // for dictionary set, if the key does not exist, we should create a new item
    auto tuple_add_op = std::make_shared<prim::TupleAdd>("tuple_add");
    auto tuple_new_item = node->func_graph()->NewCNode({NewValueNode(prim::kPrimMakeTuple), item_value});
    return node->func_graph()->NewCNode({NewValueNode(tuple_add_op), data, tuple_new_item});
  }
  auto idx_c = NewValueNode(count);
  AbstractBasePtr aptr = std::make_shared<AbstractScalar>(std::make_shared<Int32Imm>(count));
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
  MS_ASSERT(inputs.size() >= 2 && "Partial should have more than two inputs.");

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
  if (inputs.size() < 3) {
    MS_LOG(EXCEPTION) << "Node's input number < 3.";
  }

  AnfNodePtr data = inputs[1];
  AnfNodePtr cons = inputs[2];
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
  if (inputs.size() < 4) {
    MS_LOG(EXCEPTION) << "Node's input number < 4.";
  }

  AnfNodePtr data = inputs[1];
  AnfNodePtr cons = inputs[2];
  AnfNodePtr value = inputs[3];

  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, cons, value});
}

AnfNodePtr EraseMakeDictNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &inputs = node->inputs();
  MS_ASSERT(inputs.size() >= 3 && "MakeDict should have three inputs");
  return inputs[2];
}

AnfNodePtr EraseMakeKeywordArgNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &inputs = node->inputs();
  // Inputs should be [make_keyword_arg, key, value]
  MS_ASSERT(inputs.size() == 3 && "MakeKeyword should have three inputs");
  return inputs[2];
}

AnfNodePtr EraseExtractKeywordArg(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &inputs = node->inputs();
  // Inputs should be [extract_keyword_arg, arg, key]
  MS_ASSERT(inputs.size() == 3 && "ExtractKeyword should have three inputs");
  return inputs[2];
}

ValueTuplePtr ConvertValueListToValueTuple(const ValueListPtr &value_list, int depth) {
  const int DEPTH_MAX = 5;
  if (depth > DEPTH_MAX) {
    MS_LOG(EXCEPTION) << "List nesting is not allowed more than 5 levels.";
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
  int depth = 0;
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
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeDict)) {
      new_node = EraseMakeDictNode(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeKeywordArg)) {
      new_node = EraseMakeKeywordArgNode(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimExtractKeywordArg)) {
      new_node = EraseExtractKeywordArg(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimMakeList)) {
      new_node = ConvertMakeListToMakeTuple(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimListGetItem)) {
      new_node = ConvertListGetItemToTupleGetItem(cnode);
    } else if (IsPrimitiveCNode(node, prim::kPrimListSetItem)) {
      new_node = ConvertListSetItemToTupleSetItem(cnode);
    } else if (IsValueNode<ValueList>(node)) {
      new_node = ConvertValueListNodeToValueTupleNode(node->cast<ValueNodePtr>());
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
    node->set_abstract(ret);
  }
  return changed;
}

// expand tuples in graph parameters
static std::vector<AnfNodePtr> ExpandTuplesP(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph,
                                             const std::vector<AnfNodePtr> &params) {
  MS_EXCEPTION_IF_NULL(mng);
  MS_EXCEPTION_IF_NULL(func_graph);

  std::vector<AnfNodePtr> new_params;
  for (const auto &param : params) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_abs = param->abstract();
    MS_EXCEPTION_IF_NULL(param_abs);

    if (param_abs->isa<AbstractJTagged>()) {
      MS_LOG(EXCEPTION) << "Not Implemented Error NodeInfo: " << trace::GetDebugInfo(param->debug_info());
    }

    if (!param_abs->isa<AbstractTuple>()) {
      new_params.emplace_back(param);
      continue;
    }

    std::vector<AnfNodePtr> new_param;
    std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple)};
    auto abs_tuple = dyn_cast<AbstractTuple>(param_abs);
    for (auto &elem : abs_tuple->elements()) {
      auto np = std::make_shared<Parameter>(func_graph);
      np->set_abstract(elem);
      new_param.emplace_back(np);
    }
    (void)inputs.insert(inputs.end(), new_param.begin(), new_param.end());
    auto new_tuple = func_graph->NewCNode(inputs);
    (void)mng->Replace(param, new_tuple);

    auto expand_param = ExpandTuplesP(mng, func_graph, new_param);
    (void)new_params.insert(new_params.end(), expand_param.begin(), expand_param.end());
  }
  return new_params;
}

// expand tuples in graph applies
static std::vector<AnfNodePtr> ExpandTuplesC(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &inputs) {
  MS_EXCEPTION_IF_NULL(graph);

  std::vector<AnfNodePtr> new_inputs;
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);

    auto input_abs = input->abstract();
    MS_EXCEPTION_IF_NULL(input_abs);

    if (input_abs->isa<AbstractJTagged>()) {
      auto abstract_tag = dyn_cast<AbstractJTagged>(input_abs);
      if (abstract_tag->element()->isa<AbstractTuple>()) {
        MS_LOG(EXCEPTION) << "Not Implemented Error JTagged NodeInfo: " << trace::GetDebugInfo(input->debug_info());
      }
    }

    if (!input_abs->isa<AbstractTuple>()) {
      new_inputs.emplace_back(input);
      continue;
    }

    int idx = 0;
    std::vector<AnfNodePtr> new_input;
    auto abs_tuple = dyn_cast<AbstractTuple>(input_abs);
    for (auto &elem : abs_tuple->elements()) {
      auto c_node = graph->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input, NewValueNode(idx)});
      AbstractBasePtr aptr = std::make_shared<AbstractScalar>(std::make_shared<Int32Imm>(idx));
      c_node->input(2)->set_abstract(aptr);
      c_node->set_abstract(elem);
      new_input.emplace_back(c_node);
      idx++;
    }

    auto expand_tuple = ExpandTuplesC(graph, new_input);
    (void)new_inputs.insert(new_inputs.end(), expand_tuple.begin(), expand_tuple.end());
  }

  return new_inputs;
}

// remove most uses of tuples from the graph parameters & apply inputs
// tuples that are returned will be kept
// tuples in CNode's inputs: AbstractTuple (a, b ,c) -->
//         CNode("tuple_getitem", (a,b,c), 0)
//         CNode("tuple_getitem", (a,b,c), 1)
//         CNode("tuple_getitem", (a,b,c), 2)
// tuples in Graph's parameters: AbstractTuple (a, b, c) -->
//         CNode("make_tuple", Parameter(a), Parameter(b), Parameter(c))
// cppcheck-suppress unusedFunction
void EraseTuple(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(root);

  // NOTICE: since `manager->Replace(...);` will modify member `all_nodes_`, so `all_node` can't be a ref var
  AnfNodeSet all_node = manager->all_nodes();
  for (auto &node : all_node) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }

    const auto &inputs = cnode->inputs();

    // Bypass the first input in inputs as it's fn.
    if (!IsValueNode<Primitive>(inputs[0])) {
      std::vector<AnfNodePtr> expand_inputs;
      (void)expand_inputs.insert(expand_inputs.end(), inputs.begin() + 1, inputs.end());

      auto new_inputs = ExpandTuplesC(cnode->func_graph(), expand_inputs);
      if (new_inputs != expand_inputs) {
        std::vector<AnfNodePtr> cnode_inputs{inputs[0]};
        (void)cnode_inputs.insert(cnode_inputs.end(), new_inputs.begin(), new_inputs.end());

        MS_EXCEPTION_IF_NULL(node->func_graph());
        auto new_node = node->func_graph()->NewCNode(cnode_inputs);
        new_node->set_abstract(node->abstract());

        (void)manager->Replace(node, new_node);
      }
      // Bypass the first 2 inputs in inputs as it's [partial, fn].
    } else if (cnode->IsApply(prim::kPrimPartial) && !IsValueNode<Primitive>(inputs[1])) {
      std::vector<AnfNodePtr> expand_inputs;
      (void)expand_inputs.insert(expand_inputs.end(), inputs.begin() + 2, inputs.end());

      auto new_inputs = ExpandTuplesC(cnode->func_graph(), expand_inputs);
      if (new_inputs != expand_inputs) {
        std::vector<AnfNodePtr> cnode_inputs{inputs[0], inputs[1]};
        (void)cnode_inputs.insert(cnode_inputs.end(), new_inputs.begin(), new_inputs.end());

        MS_EXCEPTION_IF_NULL(cnode->func_graph());
        auto new_node = cnode->func_graph()->NewCNode(cnode_inputs);
        new_node->set_abstract(cnode->abstract());

        (void)manager->Replace(node, new_node);
      }
    }
  }

  FuncGraphSet all_graph = manager->func_graphs();
  for (auto &func_graph : all_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    auto expand_p = ExpandTuplesP(manager, func_graph, func_graph->parameters());
    manager->SetParameters(func_graph, expand_p);
  }
}
}  // namespace opt
}  // namespace mindspore
