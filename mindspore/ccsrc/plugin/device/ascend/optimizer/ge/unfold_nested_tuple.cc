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

#include "plugin/device/ascend/optimizer/ge/unfold_nested_tuple.h"

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
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractTuple>()) {
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
  std::vector<AbstractBasePtr> elements;
  for (size_t i = 0; i < tuple_elements.size(); ++i) {
    auto tuple_element = tuple_elements[i];
    if (tuple_element->isa<abstract::AbstractTuple>()) {
      auto abs_tuple_1 = tuple_element->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(abs_tuple_1);
      GetUnfoldElements(abs_tuple_1, &elements);
    } else {
      unfold_abs_elements->push_back(tuple_element);
    }
  }
  unfold_abs_elements->insert(unfold_abs_elements->end(), elements.begin(), elements.end());
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
  MS_EXCEPTION_IF_NULL(cnode);
  auto input = cnode->input(kRealInputNodeIndexInTupleGetItem);
  auto input_cnode = input->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  if (origin_index < 0) {
    MS_LOG(EXCEPTION) << "index: " << origin_index << " cannot be less than 0.";
  }
  int64_t begin_idx = 0;
  if (!IsPrimitiveCNode(input_cnode, prim::kPrimTupleGetItem) && origin_index == 0) {
    return begin_idx;
  } else if (origin_index == 0) {
    auto idx_node = cnode->input(kInputNodeOutputIndexInTupleGetItem);
    auto idx = AnfUtils::GetIntValue(idx_node);
    begin_idx = GetUnfoldIndex(input_cnode, idx);
  } else {
    begin_idx = GetUnfoldIndex(cnode, origin_index - 1) + GetElementsSize(input_cnode, origin_index - 1);
  }
  return begin_idx;
}

AnfNodePtr UnfoldNestedOutputNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  AnfNodePtr new_node = func_graph->NewCNode(inputs);
  std::vector<AbstractBasePtr> unfold_abs_elements;
  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (!abs->isa<abstract::AbstractTuple>()) {
    MS_LOG(ERROR) << "Node: " << node->fullname_with_scope() << "'s output is not a tuple.";
    return nullptr;
  }
  auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abs_tuple);
  // Unfold the nested tuple.
  GetUnfoldElements(abs_tuple, &unfold_abs_elements);
  new_node->set_abstract(std::make_shared<abstract::AbstractTuple>(unfold_abs_elements));
  return new_node;
}

void ProcessSucceedTupleGetItem(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &new_node,
                                const AnfNodePtr &tuple_node, int64_t unfold_idx) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_node);
  MS_EXCEPTION_IF_NULL(tuple_node);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto tuple_cnode = tuple_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_cnode);
  auto idx_node = tuple_cnode->input(kInputNodeOutputIndexInTupleGetItem);
  auto tuplegetitem_idx = AnfUtils::GetIntValue(idx_node);
  auto abs = tuple_node->abstract();
  MS_EXCEPTION_IF_NULL(abs);

  if (IsNestedTuple(tuple_node)) {
    MS_LOG(INFO) << "No need to process nested TupleGetItem node.";
  } else if (!abs->isa<abstract::AbstractTuple>()) {
    if (tuplegetitem_idx == unfold_idx) {
      MS_LOG(DEBUG) << "The unfold index of TupleGetItem is same as the origin index.";
      return;
    }
    auto new_axis_node = NewValueNode(unfold_idx);
    tuple_cnode->set_input(kRealInputNodeIndexInTupleGetItem, new_axis_node);
  } else {
    auto abs_tuple = abs->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abs_tuple);
    std::vector<AnfNodePtr> unfold_tuplegetitem_nodes{NewValueNode(std::make_shared<Primitive>(kMakeTupleOpName))};
    for (const auto &element : abs_tuple->elements()) {
      std::vector<AnfNodePtr> tuplegetitem_node_inputs{NewValueNode(std::make_shared<Primitive>(kTupleGetItemOpName))};
      tuplegetitem_node_inputs.push_back(new_node);
      auto new_axis_node = NewValueNode(unfold_idx);
      ++unfold_idx;
      tuplegetitem_node_inputs.push_back(new_axis_node);
      AnfNodePtr unfold_tuplegetitem_node = func_graph->NewCNode(tuplegetitem_node_inputs);
      unfold_tuplegetitem_node->set_abstract(element);
      unfold_tuplegetitem_nodes.push_back(unfold_tuplegetitem_node);
    }
    AnfNodePtr new_maketuple_node = func_graph->NewCNode(unfold_tuplegetitem_nodes);
    new_maketuple_node->set_abstract(abs);
    manager->Replace(tuple_node, new_maketuple_node);
  }
}

void ProcessSucceedNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &new_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_node);
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
        auto idx_node = user_cnode->input(kInputNodeOutputIndexInTupleGetItem);
        auto tuplegetitem_idx = AnfUtils::GetIntValue(idx_node);
        auto unfold_idx = GetUnfoldIndex(user_cnode, tuplegetitem_idx);
        need_process_tuplegetitem_nodes.push_back({user_node, unfold_idx});
        todo.push_back(user_node);
      }
    }
  }

  for (const auto &nodes_pair : need_process_tuplegetitem_nodes) {
    auto tuple_node = nodes_pair.first;
    auto unfold_idx = nodes_pair.second;
    ProcessSucceedTupleGetItem(func_graph, node, new_node, tuple_node, unfold_idx);
  }
}
}  // namespace

const BaseRef UnfoldNestedTuple::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(UnVisited);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr UnfoldNestedTuple::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                            const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>() || !IsNestedTuple(node)) {
    return nullptr;
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  AnfNodePtr new_node = UnfoldNestedOutputNode(func_graph, node);
  ProcessSucceedNode(func_graph, node, new_node);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->Replace(node, new_node);
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
