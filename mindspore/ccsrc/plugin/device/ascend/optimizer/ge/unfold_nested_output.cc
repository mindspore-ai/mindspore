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

#include "plugin/device/ascend/optimizer/ge/unfold_nested_output.h"

#include <algorithm>
#include <map>
#include <memory>
#include <deque>
#include <vector>
#include <utility>
#include "ops/sequence_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
bool IsNestedTuple(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple) || IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  if (!abs || !abs->isa<abstract::AbstractTuple>()) {
    return false;
  }
  auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abs_tuple);
  for (const auto element : abs_tuple->elements()) {
    MS_EXCEPTION_IF_NULL(element);
    if (element->isa<abstract::AbstractTuple>()) {
      return true;
    }
  }
  return false;
}

void GetUnfoldElements(const abstract::AbstractTuplePtr &abs_tuple, std::vector<AbstractBasePtr> *unfold_abs_elements) {
  MS_EXCEPTION_IF_NULL(abs_tuple);
  auto tuple_elements = abs_tuple->elements();
  for (size_t i = 0; i < tuple_elements.size(); ++i) {
    std::vector<AbstractBasePtr> elements;
    auto tuple_element = tuple_elements[i];
    if (tuple_element->isa<abstract::AbstractTuple>()) {
      auto abs_tuple_1 = tuple_element->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(abs_tuple_1);
      GetUnfoldElements(abs_tuple_1, &elements);
    } else {
      elements.push_back(tuple_element);
    }
    unfold_abs_elements->insert(unfold_abs_elements->end(), elements.begin(), elements.end());
  }
}

size_t GetElementsSize(const CNodePtr &cnode, int64_t origin_index) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto abs = cnode->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "Node: " << cnode->fullname_with_scope() << " must be a nested tuple.";
  }
  auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abs_tuple);
  auto tuple_elements = abs_tuple->elements();
  auto tuple_element = tuple_elements[origin_index];
  if (!tuple_element->isa<abstract::AbstractTuple>()) {
    return 1;
  }
  auto abs_tuple_1 = tuple_element->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abs_tuple_1);
  std::vector<AbstractBasePtr> unfold_abs_elements;
  GetUnfoldElements(abs_tuple_1, &unfold_abs_elements);
  return unfold_abs_elements.size();
}

int64_t GetUnfoldIndex(const CNodePtr &cnode, int64_t origin_index) {
  // cnode: the input node of TupleGetItem, origin_index: the input index of TupleGetItem.
  MS_EXCEPTION_IF_NULL(cnode);
  if (origin_index < 0) {
    MS_LOG(EXCEPTION) << "index: " << origin_index << " cannot be less than 0.";
  }
  int64_t begin_idx = 0;
  if (!IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem) && origin_index == 0) {
    return begin_idx;
  } else if (origin_index == 0) {
    auto input = cnode->input(kRealInputNodeIndexInTupleGetItem);
    auto input_cnode = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    auto input_idx_node = cnode->input(kInputNodeOutputIndexInTupleGetItem);
    auto input_idx = AnfUtils::GetIntValue(input_idx_node);
    begin_idx = GetUnfoldIndex(input_cnode, input_idx);
  } else {
    begin_idx = GetUnfoldIndex(cnode, origin_index - 1) + GetElementsSize(cnode, origin_index - 1);
  }
  return begin_idx;
}

void UnfoldNestedOutputNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AbstractBasePtr> unfold_abs_elements;
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractTuple>()) {
    MS_LOG(ERROR) << "Node: " << node->fullname_with_scope() << "'s output is not a tuple.";
    return;
  }
  auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abs_tuple);
  // Unfold the nested tuple.
  GetUnfoldElements(abs_tuple, &unfold_abs_elements);
  // Unfold the output, because GE converter cannot process the nested output.
  // If input is neated MakeTuple, the next unfold_maketuple pass will unfold the input.
  node->set_abstract(std::make_shared<abstract::AbstractTuple>(unfold_abs_elements));
}

void ProcessSucceedTupleGetItem(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                const AnfNodePtr &tuplegetitem_node, int64_t unfold_idx) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(tuplegetitem_node);
  auto tuplegetitem_cnode = tuplegetitem_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuplegetitem_cnode);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto abs = tuplegetitem_node->abstract();
  MS_EXCEPTION_IF_NULL(abs);

  if (IsNestedTuple(tuplegetitem_node)) {
    // TupleGetItem node with nested tuple output is not used currently.
    MS_LOG(INFO) << "No need to process nested TupleGetItem node currently.";
  } else if (abs->isa<abstract::AbstractTuple>()) {
    auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abs_tuple);
    std::vector<AnfNodePtr> unfold_tuplegetitem_nodes{NewValueNode(std::make_shared<Primitive>(kMakeTupleOpName))};
    for (const auto &element : abs_tuple->elements()) {
      std::vector<AnfNodePtr> tuplegetitem_node_inputs{NewValueNode(std::make_shared<Primitive>(kTupleGetItemOpName))};
      tuplegetitem_node_inputs.push_back(node);
      auto new_axis_node = NewValueNode(unfold_idx);
      ++unfold_idx;
      tuplegetitem_node_inputs.push_back(new_axis_node);
      AnfNodePtr unfold_tuplegetitem_node = func_graph->NewCNode(tuplegetitem_node_inputs);
      unfold_tuplegetitem_node->set_abstract(element);
      unfold_tuplegetitem_nodes.push_back(unfold_tuplegetitem_node);
    }
    AnfNodePtr new_maketuple_node = func_graph->NewCNode(unfold_tuplegetitem_nodes);
    new_maketuple_node->set_abstract(abs);
    manager->Replace(tuplegetitem_node, new_maketuple_node);
  } else {
    if (tuplegetitem_cnode->input(kRealInputNodeIndexInTupleGetItem) != node) {
      MS_LOG(DEBUG) << "The function only process the tuplegetitem node used by origin maketuple node directly.";
      return;
    }
    tuplegetitem_cnode->set_input(kRealInputNodeIndexInTupleGetItem, node);
    auto new_axis_node = NewValueNode(unfold_idx);
    tuplegetitem_cnode->set_input(kInputNodeOutputIndexInTupleGetItem, new_axis_node);
    manager->SetEdge(tuplegetitem_node, kRealInputNodeIndexInTupleGetItem, node);
  }
}

void ProcessSucceedNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &users = manager->node_users();
  std::deque<AnfNodePtr> todo{node};
  std::vector<std::pair<AnfNodePtr, size_t>> need_process_tuplegetitem_nodes;
  while (!todo.empty()) {
    auto process_node = todo.front();
    todo.pop_front();
    auto iter = users.find(process_node);
    if (iter == users.end()) {
      return;
    }
    for (auto &user : iter->second) {
      auto user_node = user.first;
      if (!user_node) {
        continue;
      }
      if (IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
        auto user_cnode = user_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(user_cnode);
        auto input = user_cnode->input(kRealInputNodeIndexInTupleGetItem);
        auto input_cnode = input->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(input_cnode);
        auto idx_node = user_cnode->input(kInputNodeOutputIndexInTupleGetItem);
        auto tuplegetitem_idx = AnfUtils::GetIntValue(idx_node);
        auto unfold_idx = GetUnfoldIndex(input_cnode, tuplegetitem_idx);
        need_process_tuplegetitem_nodes.push_back({user_node, unfold_idx});
        todo.push_back(user_node);
      }
    }
  }

  for (const auto &nodes_pair : need_process_tuplegetitem_nodes) {
    auto tuplegetitem_node = nodes_pair.first;
    auto unfold_idx = nodes_pair.second;
    MS_LOG(DEBUG) << "TupleGetUtem_node: " << tuplegetitem_node->fullname_with_scope()
                  << ", unfold_idx: " << unfold_idx;
    ProcessSucceedTupleGetItem(func_graph, node, tuplegetitem_node, unfold_idx);
  }
}
}  // namespace

bool UnfoldNestedOutput::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (auto node : node_list) {
    if (node != nullptr && node->isa<CNode>() && IsNestedTuple(node)) {
      UnfoldNestedOutputNode(node);
      ProcessSucceedNode(func_graph, node);
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
