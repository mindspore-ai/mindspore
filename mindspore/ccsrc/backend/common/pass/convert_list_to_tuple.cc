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
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
static const size_t kMaxRecursiveDepth = 6;

bool ConvertListToTuple::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  for (auto node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    // List abstract --> tuple abstract.
    auto new_abs = ConvertSequenceAbsToTupleAbs(node->abstract());
    if (new_abs != nullptr) {
      node->set_abstract(new_abs);
      if (node->isa<CNode>()) {
        common::AnfAlgo::SetNodeAttr(kAttrAbstractAdaptationProcessed, MakeValue(true), node);
      }
      MS_LOG(INFO) << "Convert sequence abstract to tuple abstract for op:" << node->fullname_with_scope()
                   << ",debug name:" << node->DebugString();
    }
  }
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto node : graph->parameters()) {
    MS_EXCEPTION_IF_NULL(node);
    // Convert unused list parameter to tuple.
    if (manager->node_users().find(node) != manager->node_users().end()) {
      continue;
    }
    auto new_abs = ConvertSequenceAbsToTupleAbs(node->abstract());
    if (new_abs != nullptr) {
      node->set_abstract(new_abs);
      MS_LOG(INFO) << "Convert sequence abstract to tuple abstract for op:" << node->fullname_with_scope()
                   << ",debug name:" << node->DebugString();
    }
  }
  return true;
}

// AbstractSequence --> AbstractTuple.
AbstractBasePtr ConvertListToTuple::ConvertSequenceAbsToTupleAbs(const AbstractBasePtr &abs, size_t depth) const {
  if (abs == nullptr || !abs->isa<abstract::AbstractSequence>()) {
    return nullptr;
  }

  if (depth > kMaxRecursiveDepth) {
    MS_LOG(EXCEPTION) << "List nesting is not allowed more than " << kMaxRecursiveDepth << " levels.";
  }

  auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(abs_seq);
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
}  // namespace opt
}  // namespace mindspore
