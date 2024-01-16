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

#include "plugin/device/ascend/optimizer/ge/unfold_maketuple.h"

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <vector>
#include <unordered_map>
#include <utility>
#include "ops/sequence_ops.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
void ReplaceNodeWithNewMakeTupleNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                     const AnfNodePtr &new_maketuple_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(new_maketuple_node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto iter = manager->node_users().find(node);
  if (iter == manager->node_users().end()) {
    return;
  }
  for (const auto &used_node_info : iter->second) {
    auto used_node = used_node_info.first;
    auto used_index = used_node_info.second;
    MS_EXCEPTION_IF_NULL(used_node);
    if (!used_node->isa<CNode>()) {
      continue;
    }
    utils::cast<CNodePtr>(used_node)->set_input(IntToSize(used_index), new_maketuple_node);
  }
}

bool IsTupleInput(const AnfNodePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  auto abs = input->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTuple>()) {
    return true;
  }
  return false;
}

void GetUnfoldInputs(const AnfNodePtr &node, std::vector<AnfNodePtr> *unfold_nodes,
                     std::vector<AnfNodePtr> *maketuple_inputs = nullptr) {
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (size_t idx = 1; idx < inputs.size(); ++idx) {
    std::vector<AnfNodePtr> nodes;
    auto input = inputs[idx];
    if (IsTupleInput(input)) {
      if (IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
        if (maketuple_inputs) {
          maketuple_inputs->push_back(input);
        }
        GetUnfoldInputs(input, &nodes, maketuple_inputs);
      } else if (IsPrimitiveCNode(input, prim::kPrimTupleGetItem)) {
        auto input_cnode = input->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(input_cnode);
        const auto &idx_node = input_cnode->input(kInputNodeOutputIndexInTupleGetItem);
        int64_t tuplegetitem_idx = AnfUtils::GetIntValue(idx_node);
        // real_node is a MakeTuple node, need collect its all inputs.
        auto real_node = common::AnfAlgo::VisitKernelWithReturnType(input, tuplegetitem_idx).first;
        auto real_cnode = real_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(real_cnode);
        const auto real_cnode_inputs = real_cnode->inputs();
        nodes.insert(nodes.end(), real_cnode_inputs.begin() + kIndex1, real_cnode_inputs.end());
      } else {
        MS_LOG(INFO) << "Unknown tuple input.";
        nodes.push_back(input);
      }
    } else {
      nodes.push_back(input);
    }
    unfold_nodes->insert(unfold_nodes->end(), nodes.begin(), nodes.end());
  }
}

bool IsNestedMaketuple(const AnfNodePtr &node, std::vector<AnfNodePtr> *unfold_nodes,
                       std::vector<AnfNodePtr> *maketuple_inputs,
                       std::unordered_map<AnfNodePtr, int64_t> *unfold_nodes_index) {
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  GetUnfoldInputs(node, unfold_nodes, maketuple_inputs);
  if (unfold_nodes->size() + kIndex1 > inputs.size() || !maketuple_inputs->empty()) {
    for (size_t idx = 0; idx < unfold_nodes->size(); ++idx) {
      unfold_nodes_index->insert({unfold_nodes->at(idx), SizeToLong(idx)});
    }
    return true;
  }
  return false;
}

std::vector<AnfNodePtr> GetRealInputNode(const AnfNodePtr &node) {
  std::vector<AnfNodePtr> real_nodes;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &idx_node = cnode->input(kInputNodeOutputIndexInTupleGetItem);
  int64_t tuplegetitem_idx = AnfUtils::GetIntValue(idx_node);
  auto real_node = common::AnfAlgo::VisitKernelWithReturnType(node, tuplegetitem_idx).first;
  if (IsPrimitiveCNode(real_node, prim::kPrimMakeTuple)) {
    GetUnfoldInputs(real_node, &real_nodes);
  } else {
    real_nodes.push_back(real_node);
  }
  return real_nodes;
}

void ProcessSucceedTupleGetItem(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                const AnfNodePtr &tuplegetitem_node, const AnfNodePtr &unfold_maketuple_node,
                                const std::unordered_map<AnfNodePtr, int64_t> &unfold_nodes_index,
                                const std::vector<AnfNodePtr> &real_nodes) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(tuplegetitem_node);
  auto tuplegetitem_cnode = tuplegetitem_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuplegetitem_cnode);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto tuplegetitem_node_abs = tuplegetitem_node->abstract();
  MS_EXCEPTION_IF_NULL(tuplegetitem_node_abs);

  if (tuplegetitem_node_abs->isa<abstract::AbstractTuple>()) {
    AbstractBasePtrList make_tuple_abstract;
    std::vector<AnfNodePtr> unfold_tuplegetitem_nodes{NewValueNode(std::make_shared<Primitive>(kMakeTupleOpName))};
    for (auto real_node : real_nodes) {
      std::vector<AnfNodePtr> tuplegetitem_node_inputs{NewValueNode(std::make_shared<Primitive>(kTupleGetItemOpName))};
      tuplegetitem_node_inputs.push_back(unfold_maketuple_node);
      auto iter = unfold_nodes_index.find(real_node);
      if (iter == unfold_nodes_index.end()) {
        MS_LOG(EXCEPTION) << "Node: " << real_node->fullname_with_scope() << " cannot be found in unfold_nodes_index.";
      }
      int64_t real_idx = iter->second;
      auto new_idx_node = NewValueNode(real_idx);
      tuplegetitem_node_inputs.push_back(new_idx_node);
      AnfNodePtr unfold_tuplegetitem_node = func_graph->NewCNode(tuplegetitem_node_inputs);
      unfold_tuplegetitem_node->set_abstract(real_node->abstract());
      make_tuple_abstract.push_back(real_node->abstract());
      unfold_tuplegetitem_nodes.push_back(unfold_tuplegetitem_node);
    }
    AnfNodePtr new_maketuple_node = func_graph->NewCNode(unfold_tuplegetitem_nodes);
    new_maketuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(make_tuple_abstract));
    ReplaceNodeWithNewMakeTupleNode(func_graph, tuplegetitem_node, new_maketuple_node);
  } else {
    if (tuplegetitem_cnode->input(kRealInputNodeIndexInTupleGetItem) != node) {
      MS_LOG(DEBUG) << "The function only process the tuplegetitem node used by origin maketuple node directly.";
      return;
    }
    if (real_nodes.size() > kSizeOne) {
      MS_LOG(ERROR) << "The size of real_nodes must equal to 1 when tuplegetitem node has single output, but it's: "
                    << real_nodes.size();
    }
    auto real_node = real_nodes[kIndex0];
    auto iter = unfold_nodes_index.find(real_node);
    if (iter == unfold_nodes_index.end()) {
      MS_LOG(EXCEPTION) << "Node: " << real_node->fullname_with_scope() << " cannot be found in unfold_nodes_index.";
    }
    int64_t real_idx = iter->second;
    auto new_idx_node = NewValueNode(real_idx);
    tuplegetitem_cnode->set_input(kRealInputNodeIndexInTupleGetItem, unfold_maketuple_node);
    tuplegetitem_cnode->set_input(kInputNodeOutputIndexInTupleGetItem, new_idx_node);
  }
}

AnfNodePtr ProcessSucceedNodes(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                               const std::vector<AnfNodePtr> &unfold_nodes,
                               const std::vector<AnfNodePtr> &maketuple_inputs,
                               const std::unordered_map<AnfNodePtr, int64_t> &unfold_nodes_index) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> unfold_maketuple_inputs{NewValueNode(std::make_shared<Primitive>(kMakeTupleOpName))};
  unfold_maketuple_inputs.insert(unfold_maketuple_inputs.end(), unfold_nodes.begin(), unfold_nodes.end());
  std::vector<abstract::AbstractBasePtr> unfold_maketuple_abs;
  for (const auto &unfold_node : unfold_nodes) {
    auto abs = unfold_node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    unfold_maketuple_abs.push_back(abs);
  }
  auto unfold_maketuple_node = func_graph->NewCNode(unfold_maketuple_inputs);
  unfold_maketuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(unfold_maketuple_abs));

  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &users = manager->node_users();
  std::deque<AnfNodePtr> todo{node};
  todo.insert(todo.end(), maketuple_inputs.begin(), maketuple_inputs.end());
  std::vector<std::pair<AnfNodePtr, std::vector<AnfNodePtr>>> need_process_tuplegetitem_nodes;
  while (!todo.empty()) {
    auto process_node = todo.front();
    todo.pop_front();
    auto iter = users.find(process_node);
    if (iter == users.end()) {
      continue;
    }
    for (auto user : iter->second) {
      auto user_node = user.first;
      if (user_node == nullptr) {
        continue;
      }
      if (IsPrimitiveCNode(user_node, prim::kPrimMakeTuple)) {
        MS_LOG(DEBUG) << "Maketuple node with nested tuple cannot be connected by another Maketuple node.";
        continue;
      } else if (IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
        std::vector<AnfNodePtr> real_nodes = GetRealInputNode(user_node);
        need_process_tuplegetitem_nodes.push_back({user_node, real_nodes});
        todo.push_back(user_node);
      }
    }
  }
  for (const auto &tuplegetitem_node_pair : need_process_tuplegetitem_nodes) {
    ProcessSucceedTupleGetItem(func_graph, node, tuplegetitem_node_pair.first, unfold_maketuple_node,
                               unfold_nodes_index, tuplegetitem_node_pair.second);
  }
  return unfold_maketuple_node;
}
}  // namespace

bool UnfoldMaketuple::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (const auto node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (node == nullptr || !node->isa<CNode>() || !IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      continue;
    }
    std::vector<AnfNodePtr> unfold_nodes;
    std::vector<AnfNodePtr> maketuple_inputs;
    std::unordered_map<AnfNodePtr, int64_t> unfold_nodes_index;
    if (IsNestedMaketuple(node, &unfold_nodes, &maketuple_inputs, &unfold_nodes_index)) {
      auto unfold_maketuple_node =
        ProcessSucceedNodes(func_graph, node, unfold_nodes, maketuple_inputs, unfold_nodes_index);
      ReplaceNodeWithNewMakeTupleNode(func_graph, node, unfold_maketuple_node);
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
