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
#include "backend/common/pass/convert_list_to_tuple.h"
#include <map>
#include <utility>
#include <algorithm>
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
static const std::map<std::string, std::string> kOpListToTupleNames = {{prim::kMakeListNew, prim::kMakeTuple},
                                                                       {prim::kListGetItem, prim::kTupleGetItem},
                                                                       {prim::kListSetItem, prim::kTupleSetItem}};
static const size_t kMaxRecursiveDepth = 6;

const AnfNodePtr ConvertListToTuple::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  // Value list --> Value tuple.
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    bool need_convert = false;
    auto convert_value = ConvertValueSequenceToValueTuple(value_node->value(), &need_convert);
    if (need_convert) {
      return std::make_shared<ValueNode>(convert_value);
    }
    return nullptr;
  }

  if (!node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  // List name --> tuple name.
  auto old_full_name = cnode->fullname_with_scope();
  auto old_name = common::AnfAlgo::GetCNodeName(cnode);
  auto iter = kOpListToTupleNames.find(old_name);
  if (iter != kOpListToTupleNames.end()) {
    auto primitive = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(primitive);
    primitive->set_name(iter->second);
    // Reset full scope name.
    cnode->set_fullname_with_scope("");
    common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), cnode);
    MS_LOG(DEBUG) << "Rename op from " << old_name << " to " << iter->second << " for op " << old_full_name << " to "
                  << cnode->fullname_with_scope();
  }

  // List abstract --> tuple abstract.
  auto new_abs = ConvertSequenceAbsToTupleAbs(node->abstract());
  if (new_abs != nullptr) {
    node->set_abstract(new_abs);
    common::AnfAlgo::SetNodeAttr(kAttrAbstractAdaptationProcessed, MakeValue(true), cnode);
    MS_LOG(DEBUG) << "Convert sequence abstract to tuple abstract for op " << old_full_name << ", new op name "
                  << cnode->fullname_with_scope();
  }

  return nullptr;
}

// ValueSequence --> ValueTuple.
ValuePtr ConvertListToTuple::ConvertValueSequenceToValueTuple(const ValuePtr &value, bool *need_convert,
                                                              size_t depth) const {
  MS_EXCEPTION_IF_NULL(need_convert);
  MS_EXCEPTION_IF_NULL(value);
  if (depth > kMaxRecursiveDepth) {
    MS_LOG(EXCEPTION) << "List nesting is not allowed more than " << kMaxRecursiveDepth << " levels.";
  }

  if (value->isa<ValueSequence>()) {
    std::vector<ValuePtr> elements;
    auto value_seq = value->cast<ValueSequencePtr>();
    (void)std::transform(value_seq->value().begin(), value_seq->value().end(), std::back_inserter(elements),
                         [&](const ValuePtr &value) -> ValuePtr {
                           bool is_convert = false;
                           auto convert_value = ConvertValueSequenceToValueTuple(value, &is_convert, depth + 1);
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

// AbstractSequence --> AbstractTuple.
AbstractBasePtr ConvertListToTuple::ConvertSequenceAbsToTupleAbs(const AbstractBasePtr &abs, size_t depth) const {
  if (abs == nullptr) {
    return nullptr;
  }

  if (depth > kMaxRecursiveDepth) {
    MS_LOG(EXCEPTION) << "List nesting is not allowed more than " << kMaxRecursiveDepth << " levels.";
  }

  auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
  if (abs_seq != nullptr) {
    // Dynamic length sequence convert by the dynamic abs.
    if (abs_seq->dynamic_len() && abs_seq->isa<abstract::AbstractList>()) {
      auto converted_dynamic_abs_tuple =
        std::make_shared<abstract::AbstractTuple>(abs_seq->elements(), abs_seq->sequence_nodes());
      converted_dynamic_abs_tuple->set_dynamic_len(true);
      converted_dynamic_abs_tuple->set_dynamic_len_element_abs(abs_seq->dynamic_len_element_abs());
      return converted_dynamic_abs_tuple;
    }
    const auto &seq_elements = abs_seq->elements();
    // First we check if elements should be converted,
    // changed_elements maps old element to new element.
    mindspore::HashMap<AbstractBasePtr, AbstractBasePtr> changed_elements;
    for (const auto &element : seq_elements) {
      auto new_element = ConvertSequenceAbsToTupleAbs(element, depth + 1);
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
    return std::make_shared<abstract::AbstractTuple>(std::move(elements));
  }

  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
