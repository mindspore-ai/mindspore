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
#include <string>
#include <algorithm>
#include <vector>
#include <deque>
#include <utility>
#include <memory>
#include "tools/optimizer/graph/make_list_pass.h"
#include "mindspore/core/abstract/ops/primitive_infer_map.h"
#include "mindspore/core/utils/anf_utils.h"
#include "ops/sequence_ops.h"

namespace mindspore::opt {
// From:
//   MakeList(arg1, arg2, ...)
// To:
//   MakeTuple(arg1, arg2, ...)
AnfNodePtr MakeListPass::ConvertMakeListToMakeTuple(const CNodePtr &node) {
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
AnfNodePtr MakeListPass::ConvertListGetItemToTupleGetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  // Inputs should be [list_getitem, list, item]
  constexpr size_t expect_inputs_size = 3;
  if (node->size() != expect_inputs_size) {
    std::string op_name = GetCNodeFuncName(node);
    MS_LOG(EXCEPTION) << op_name << " should have " << expect_inputs_size << " inputs, but got " << node->size();
    return nullptr;
  }
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
AnfNodePtr MakeListPass::ConvertListSetItemToTupleSetItem(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());

  // Inputs should be [list_setitem, list, index, item]
  const size_t expect_inputs_size = 4;
  if (node->size() != expect_inputs_size) {
    std::string op_name = GetCNodeFuncName(node);
    MS_LOG(EXCEPTION) << op_name << " should have " << expect_inputs_size << " inputs, but got " << node->size();
    return nullptr;
  }

  const size_t data_index = 1;
  const size_t cons_index = 2;
  const size_t value_index = 3;
  const auto &inputs = node->inputs();
  auto &data = inputs[data_index];
  auto &key = inputs[cons_index];
  auto &value = inputs[value_index];
  return node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleSetItem), data, key, value});
}

AnfNodePtr MakeListPass::ConvertMakeListPrimitiveCNode(const CNodePtr &cnode, const PrimitivePtr &prim) {
  if (prim->name() == prim::kPrimMakeList->name()) {
    return ConvertMakeListToMakeTuple(cnode);
  } else if (prim->name() == prim::kPrimListGetItem->name()) {
    return ConvertListGetItemToTupleGetItem(cnode);
  } else if (prim->name() == prim::kPrimListSetItem->name()) {
    return ConvertListSetItemToTupleSetItem(cnode);
  }
  return nullptr;
}

static constexpr size_t kMaxSeqRecursiveDepth = 6;
ValuePtr MakeListPass::ConvertValueSequenceToValueTuple(const ValuePtr &value, size_t depth, bool *need_convert) {
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

AnfNodePtr MakeListPass::ConvertMakeListValueNode(const ValueNodePtr &value_node, const ValuePtr &value) {
  bool need_convert = false;
  auto convert_value = ConvertValueSequenceToValueTuple(value, 0, &need_convert);
  if (need_convert) {
    return std::make_shared<ValueNode>(convert_value);
  }
  return nullptr;
}

AnfNodePtr MakeListPass::ConvertMakeListNode(const AnfNodePtr &node) {
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
    return ConvertMakeListPrimitiveCNode(cnode, prim);
  }
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node != nullptr) {
    const auto &value = value_node->value();
    if (value == nullptr) {
      return nullptr;
    }
    // Call value node converter.
    return ConvertMakeListValueNode(value_node, value);
  }
  return nullptr;
}

// AbstractRowTensor --> AbstractTuple.
AbstractBasePtr MakeListPass::ConvertToAbstractTuple(const AbstractBasePtr &abs, size_t depth) {
  if (depth > kMaxSeqRecursiveDepth) {
    MS_LOG(EXCEPTION) << "List, tuple and dict nesting is not allowed more than " << kMaxSeqRecursiveDepth
                      << " levels.";
  }
  // Convert RowTensor in AbstractSequence to AbstractTuple.
  auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
  if (abs_seq != nullptr) {
    if (abs_seq->dynamic_len() && abs_seq->isa<abstract::AbstractList>()) {
      auto converted_abs_tuple =
        std::make_shared<abstract::AbstractTuple>(abs_seq->elements(), abs_seq->sequence_nodes());
      converted_abs_tuple->set_dynamic_len(true);
      converted_abs_tuple->set_dynamic_len_element_abs(abs_seq->dynamic_len_element_abs());
      return converted_abs_tuple;
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
      if (abs->isa<abstract::AbstractTuple>()) {
        // If no elements changed and it is an AbstractTuple, do not convert.
        return nullptr;
      }
      // If no elements changed but it is not an AbstractTuple, convert it by copy elements.
      return std::make_shared<abstract::AbstractTuple>(seq_elements);
    }
    // Make new abstract sequence.
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
    if (abs_seq->isa<abstract::AbstractList>()) {
      return std::make_shared<abstract::AbstractList>(std::move(elements));
    }
    return std::make_shared<abstract::AbstractTuple>(std::move(elements));
  }
  // AbstractRowTensor --> AbstractTuple.
  auto abs_row_tensor = abs->cast<std::shared_ptr<abstract::AbstractRowTensor>>();
  if (abs_row_tensor != nullptr) {
    std::vector<AbstractBasePtr> elements{abs_row_tensor->indices(), abs_row_tensor->values(),
                                          abs_row_tensor->dense_shape()};
    return std::make_shared<abstract::AbstractTuple>(std::move(elements));
  }
  return nullptr;
}

AnfNodePtr MakeListPass::MakeListNodeRewrite(const AnfNodePtr &node) {
  auto new_node = ConvertMakeListNode(node);
  if (new_node != nullptr) {
    new_node->set_abstract(node->abstract());
  }
  return new_node;
}

STATUS MakeListPass::UpdateMakeListAbstracts(const FuncGraphPtr &func_graph) {
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    return lite::RET_ERROR;
  }
  const auto &nodes = manager->all_nodes();
  for (const auto &node : nodes) {
    const auto &abs = node->abstract();
    if (abs == nullptr) {
      continue;
    }
    // Call abstract converter.
    auto new_abs = ConvertToAbstractTuple(abs, 0);
    if (new_abs != nullptr) {
      node->set_abstract(new_abs);
    }
  }
  return lite::RET_OK;
}

STATUS MakeListPass::MakeListToMakeTuple(const FuncGraphPtr &func_graph) {
  bool changed = false;
  auto seen = NewSeenGeneration();
  std::deque<AnfNodePtr> todo;
  auto add_todo = [&seen, &todo](const AnfNodePtr &node) {
    if (node != nullptr && node->seen_ != seen) {
      (void)todo.emplace_back(node);
    }
  };
  (void)todo.emplace_back(func_graph->return_node());
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_ERROR, "Manager is nullptr.");
  auto &all_nodes = manager->all_nodes();
  while (!todo.empty()) {
    AnfNodePtr node = std::move(todo.front());
    todo.pop_front();
    if (node == nullptr || node->seen_ == seen || !all_nodes.contains(node)) {
      continue;
    }
    node->seen_ = seen;
    auto cnode = node->cast_ptr<CNode>();
    if (cnode != nullptr) {
      for (auto &input : cnode->inputs()) {
        add_todo(input);
      }
    } else {
      auto fg = GetValuePtr<FuncGraph>(node);
      if (fg != nullptr) {
        add_todo(fg->return_node());
      }
    }
    TraceGuard trace_guard(std::make_shared<TraceOpt>(node->debug_info()));
    ScopeGuard scope_guard(node->scope());
    auto new_node = MakeListNodeRewrite(node);
    if (new_node != nullptr) {
      (void)manager->Replace(node, new_node);
      changed = true;
    }
  }
  if (changed) {
    return UpdateMakeListAbstracts(func_graph);
  }
  return lite::RET_OK;
}

bool MakeListPass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  if (MakeListToMakeTuple(func_graph) != lite::RET_OK) {
    return false;
  }
  return true;
}
}  // namespace mindspore::opt
