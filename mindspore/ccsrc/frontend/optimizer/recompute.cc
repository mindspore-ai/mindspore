/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/recompute.h"
#include <memory>
#include <queue>
#include <utility>
#include <list>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "ir/func_graph.h"
#include "mindspore/core/base/core_ops.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kGradientsFlag = "Gradients";
bool IsBpropNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return node->fullname_with_scope().find(kGradientsFlag) == 0;
}

bool WithRecomputedScope(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto full_name_with_scope = node->fullname_with_scope();
  return full_name_with_scope.find(kAttrRecompute) == 0;
}

bool HasRecomputeCNodeAttr(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return false;
  }
  auto cnode_recompute_val = cnode->GetAttr(kAttrRecompute);
  return cnode_recompute_val != nullptr && cnode_recompute_val->isa<BoolImm>() && GetValue<bool>(cnode_recompute_val);
}

bool IsCandidateRecomputedNode(const CNodePtr &node) { return !IsBpropNode(node) && HasRecomputeCNodeAttr(node); }

std::vector<CNodePtr> FindCandidateRecomputedNodes(const FuncGraphManagerPtr &mng,
                                                   const std::vector<CNodePtr> &cnodes) {
  MS_EXCEPTION_IF_NULL(mng);
  std::vector<CNodePtr> candidate_recomputed_nodes;
  for (const auto &cnode : cnodes) {
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsCandidateRecomputedNode(cnode)) {
      continue;
    }
    // Check outputs.
    const auto &node_users = mng->node_users();
    auto output_set_iter = node_users.find(cnode);
    if (output_set_iter == node_users.end()) {
      continue;
    }
    const auto &node_index_set = output_set_iter->second;
    if (!std::any_of(node_index_set.begin(), node_index_set.end(),
                     [](const std::pair<AnfNodePtr, int> &node_index) { return IsBpropNode(node_index.first); })) {
      continue;
    }
    // Check inputs.
    const auto &inputs = cnode->inputs();
    if (std::any_of(inputs.begin(), inputs.end(), [](const AnfNodePtr &node) { return IsBpropNode(node); })) {
      continue;
    }
    candidate_recomputed_nodes.emplace_back(cnode);
  }
  return candidate_recomputed_nodes;
}

void GetMaxSubGraph(const FuncGraphManagerPtr &mng, std::unordered_set<CNodePtr> *recomputed_nodes, bool get_inputs,
                    bool get_outputs) {
  MS_EXCEPTION_IF_NULL(mng);
  MS_EXCEPTION_IF_NULL(recomputed_nodes);
  std::queue<CNodePtr> nodes_to_visit;
  for (const auto &node : *recomputed_nodes) {
    nodes_to_visit.push(node);
  }
  recomputed_nodes->clear();
  while (!nodes_to_visit.empty()) {
    auto current_node = nodes_to_visit.front();
    nodes_to_visit.pop();
    recomputed_nodes->insert(current_node);
    if (get_inputs) {
      for (const auto &input : current_node->inputs()) {
        MS_EXCEPTION_IF_NULL(input);
        if (input->isa<CNode>()) {
          auto input_cnode = input->cast<CNodePtr>();
          if (recomputed_nodes->find(input_cnode) == recomputed_nodes->end() &&
              IsCandidateRecomputedNode(input_cnode)) {
            nodes_to_visit.push(input_cnode);
          }
        }
      }
    }

    if (get_outputs) {
      const auto &node_users = mng->node_users();
      auto output_set_iter = node_users.find(current_node);
      if (output_set_iter == node_users.end()) {
        continue;
      }
      for (const auto &node_index_set : output_set_iter->second) {
        auto output_node = node_index_set.first;
        MS_EXCEPTION_IF_NULL(output_node);
        if (output_node->isa<CNode>()) {
          auto output_cnode = output_node->cast<CNodePtr>();
          if (recomputed_nodes->find(output_cnode) == recomputed_nodes->end() &&
              IsCandidateRecomputedNode(output_cnode)) {
            nodes_to_visit.push(output_cnode);
          }
        }
      }
    }
  }
}

void GetOriginRecomputeAndTargetNodes(const FuncGraphManagerPtr &mng,
                                      const std::unordered_set<CNodePtr> &max_recomputed_sub_graph,
                                      std::unordered_set<CNodePtr> *recompute_nodes,
                                      std::unordered_set<CNodePtr> *target_nodes) {
  MS_EXCEPTION_IF_NULL(mng);
  MS_EXCEPTION_IF_NULL(recompute_nodes);
  MS_EXCEPTION_IF_NULL(target_nodes);
  const auto &node_users = mng->node_users();
  for (const auto &node : max_recomputed_sub_graph) {
    bool inserted = false;
    auto output_set_iter = node_users.find(node);
    if (output_set_iter == node_users.end()) {
      continue;
    }
    for (const auto &node_index_set : output_set_iter->second) {
      auto output_node = node_index_set.first;
      MS_EXCEPTION_IF_NULL(output_node);
      if (!IsBpropNode(output_node)) {
        continue;
      }
      target_nodes->insert(output_node->cast<CNodePtr>());
      if (!inserted) {
        recompute_nodes->insert(node);
        inserted = true;
      }
    }
  }
}

std::vector<AnfNodePtr> GetFirstTargetInputs(const std::vector<CNodePtr> &origin_nodes_topological,
                                             const std::unordered_set<CNodePtr> &recomputed_origin_nodes,
                                             const std::unordered_set<CNodePtr> &target_nodes) {
  std::vector<AnfNodePtr> first_target_inputs;
  for (const auto &node : origin_nodes_topological) {
    MS_EXCEPTION_IF_NULL(node);
    if (target_nodes.find(node) != target_nodes.end()) {
      for (size_t i = 1; i < node->size(); ++i) {
        auto input = node->input(i);
        if (!input->isa<CNode>()) {
          continue;
        }
        MS_EXCEPTION_IF_NULL(input);
        if (recomputed_origin_nodes.find(input->cast<CNodePtr>()) != recomputed_origin_nodes.end()) {
          continue;
        }
        first_target_inputs.emplace_back(input);
      }
      break;
    }
  }
  return first_target_inputs;
}

bool HasGradInputs(const AnfNodePtr &node, std::unordered_map<AnfNodePtr, bool> *has_grad_inputs_map) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(has_grad_inputs_map);
  if (has_grad_inputs_map->find(node) != has_grad_inputs_map->end()) {
    return has_grad_inputs_map->find(node)->second;
  }
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    has_grad_inputs_map->insert(std::make_pair(node, false));
    return false;
  }
  const auto &inputs = cnode->inputs();
  if (std::any_of(inputs.begin(), inputs.end(), [&has_grad_inputs_map](const AnfNodePtr &input) {
        return IsBpropNode(input) || HasGradInputs(input, has_grad_inputs_map);
      })) {
    has_grad_inputs_map->insert(std::make_pair(node, true));
    return true;
  }
  has_grad_inputs_map->insert(std::make_pair(node, false));
  return false;
}

bool HasForwardOutput(const FuncGraphManagerPtr &mng, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(mng);
  const auto &node_users = mng->node_users();
  auto output_set_iter = node_users.find(node);
  if (output_set_iter == node_users.end()) {
    return false;
  }
  for (const auto &node_index_set : output_set_iter->second) {
    if (!IsBpropNode(node_index_set.first) && !IsPrimitiveCNode(node_index_set.first, prim::kPrimControlDepend)) {
      return true;
    }
  }
  return false;
}

void GetTupleGetItemOutputNodes(const FuncGraphManagerPtr &mng, const AnfNodePtr &node,
                                std::vector<AnfNodePtr> *tuple_getitem_output_nodes) {
  MS_EXCEPTION_IF_NULL(mng);
  MS_EXCEPTION_IF_NULL(tuple_getitem_output_nodes);
  const auto &node_users = mng->node_users();
  auto output_set_iter = node_users.find(node);
  if (output_set_iter == node_users.end()) {
    return;
  }
  for (const auto &node_index_set : output_set_iter->second) {
    if (IsPrimitiveCNode(node_index_set.first, prim::kPrimTupleGetItem)) {
      tuple_getitem_output_nodes->emplace_back(node_index_set.first);
    }
  }
}

// Set 'recompute' cnode attr for the nodes according to its scope.
// A node set 'recompute' cnode attr can become the candidate recomputed node.
void SetRecomputedAttr(const FuncGraphPtr &graph, const std::vector<CNodePtr> &origin_nodes_topological) {
  MS_EXCEPTION_IF_NULL(graph);
  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  std::unordered_map<AnfNodePtr, bool> has_grad_inputs_map;
  for (const auto &node : origin_nodes_topological) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsBpropNode(node)) {
      continue;
    }
    // Do not recompute the communicate op.
    if (IsPrimitiveCNode(node, prim::kPrimAllGather)) {
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    if (!HasForwardOutput(mng, node) || HasGradInputs(node, &has_grad_inputs_map)) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      continue;
    }
    auto prim_recompute_attr = prim->GetAttr(kAttrRecompute);
    int prim_recompute_val = -1;
    if (prim_recompute_attr != nullptr && prim_recompute_attr->isa<BoolImm>()) {
      prim_recompute_val = GetValue<bool>(prim_recompute_attr);
    }
    if ((WithRecomputedScope(node) && prim_recompute_val != 0) || prim_recompute_val == 1) {
      cnode->AddAttr(kAttrRecompute, MakeValue(true));
    }
    if (!HasRecomputeCNodeAttr(node)) {
      continue;
    }
    // Set attr for the tuple_getitem outputs.
    std::vector<AnfNodePtr> tuple_getitem_output_nodes;
    GetTupleGetItemOutputNodes(mng, node, &tuple_getitem_output_nodes);
    for (const auto &output_node : tuple_getitem_output_nodes) {
      auto output_cnode = output_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(output_cnode);
      output_cnode->AddAttr(kAttrRecompute, MakeValue(true));
    }
  }
}

CNodePtr CreateNewRecomputedNode(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                                 const std::vector<AnfNodePtr> &new_inputs) {
  auto recomputed_node = graph->NewCNode(new_inputs);
  MS_EXCEPTION_IF_NULL(recomputed_node);
  recomputed_node->AddAttr("duplicated", MakeValue(true));
  recomputed_node->AddAttr(kAttrNeedCseAfterRecompute, MakeValue(true));
  recomputed_node->set_abstract(origin_node->abstract());
  recomputed_node->set_scope(origin_node->scope());
  return recomputed_node;
}

CNodePtr NewRecomputedNode(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                           const std::vector<AnfNodePtr> &first_target_inputs,
                           const std::unordered_set<CNodePtr> &recomputed_origin_nodes,
                           std::unordered_map<CNodePtr, CNodePtr> *origin_to_recomputed_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  MS_EXCEPTION_IF_NULL(origin_to_recomputed_nodes);
  auto iter = origin_to_recomputed_nodes->find(origin_node);
  if (iter != origin_to_recomputed_nodes->end()) {
    return iter->second;
  }
  MS_LOG(DEBUG) << "Begin to Duplicating origin recomputed node: " << origin_node->DebugString();
  std::vector<AnfNodePtr> new_inputs;
  bool has_recomputed_inputs = false;
  for (size_t i = 0; i < origin_node->size(); ++i) {
    auto input = origin_node->input(i);
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>()) {
      new_inputs.emplace_back(input);
      continue;
    }
    auto input_cnode = input->cast<CNodePtr>();
    if (recomputed_origin_nodes.find(input_cnode) == recomputed_origin_nodes.end()) {
      new_inputs.emplace_back(input);
    } else {
      has_recomputed_inputs = true;
      new_inputs.emplace_back(NewRecomputedNode(graph, input_cnode, first_target_inputs, recomputed_origin_nodes,
                                                origin_to_recomputed_nodes));
    }
  }
  // Add the execution dependency.
  if (!has_recomputed_inputs && new_inputs.size() > 1) {
    std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
    std::copy(first_target_inputs.begin(), first_target_inputs.end(), std::back_inserter(make_tuple_inputs));
    auto first_input = new_inputs[1];
    MS_EXCEPTION_IF_NULL(first_input);
    std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), first_input,
                                          graph->NewCNode(make_tuple_inputs)};
    auto depend_node = graph->NewCNode(depend_inputs);
    MS_EXCEPTION_IF_NULL(depend_node);
    depend_node->set_abstract(first_input->abstract());
    new_inputs[1] = depend_node;
  }
  auto recomputed_node = CreateNewRecomputedNode(graph, origin_node, new_inputs);
  origin_to_recomputed_nodes->insert(std::make_pair(origin_node, recomputed_node));
  return recomputed_node;
}

void DuplicateRecomputedNodes(const FuncGraphPtr &graph, const std::unordered_set<CNodePtr> &target_nodes,
                              const std::unordered_set<CNodePtr> &origin_recomputed_nodes,
                              const std::vector<AnfNodePtr> &first_target_inputs,
                              std::unordered_map<CNodePtr, CNodePtr> *origin_to_recomputed_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (const auto &target_node : target_nodes) {
    MS_EXCEPTION_IF_NULL(target_node);
    MS_LOG(DEBUG) << "Rebuild a new target_node " << target_node->DebugString() << " with the new recomputed input";
    auto target_cnode = target_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(target_cnode);
    std::vector<AnfNodePtr> new_target_inputs;
    for (const auto &input : target_cnode->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>()) {
        new_target_inputs.emplace_back(input);
      } else {
        auto input_cnode = input->cast<CNodePtr>();
        if (origin_recomputed_nodes.find(input_cnode) != origin_recomputed_nodes.end()) {
          new_target_inputs.emplace_back(NewRecomputedNode(graph, input_cnode, first_target_inputs,
                                                           origin_recomputed_nodes, origin_to_recomputed_nodes));
        } else {
          new_target_inputs.emplace_back(input_cnode);
        }
      }
    }
    auto new_target_node = graph->NewCNode(new_target_inputs);
    new_target_node->AddAttr("target_grad", MakeValue(true));
    new_target_node->set_abstract(target_node->abstract());
    new_target_node->set_scope(target_node->scope());
    mng->Replace(target_node, new_target_node);
  }
}
}  // namespace

void InsertRecomputedNodes(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.begin(), orders.end());
  SetRecomputedAttr(graph, origin_nodes_topological);
  // Get candidate origin recomputed nodes which have no grad inputs and output to at least one grad node directly.
  std::vector<CNodePtr> candidate_recomputed_nodes = FindCandidateRecomputedNodes(mng, origin_nodes_topological);
  std::unordered_set<CNodePtr> visited_nodes;
  for (const auto &candidate_recomputed_node : candidate_recomputed_nodes) {
    if (visited_nodes.find(candidate_recomputed_node) != visited_nodes.end()) {
      continue;
    }
    std::unordered_set<CNodePtr> max_recomputed_sub_graph = {candidate_recomputed_node};
    // Get max continuous recomputed sub-graph.
    GetMaxSubGraph(mng, &max_recomputed_sub_graph, true, true);
    visited_nodes.insert(max_recomputed_sub_graph.begin(), max_recomputed_sub_graph.end());
    // Get the origin recomputed nodes which directly output to the grad nodes.
    std::unordered_set<CNodePtr> origin_recomputed_nodes;
    std::unordered_set<CNodePtr> target_nodes;
    GetOriginRecomputeAndTargetNodes(mng, max_recomputed_sub_graph, &origin_recomputed_nodes, &target_nodes);
    // Also get the inputs of origin recomputed nodes which eventually output to the grad nodes.
    GetMaxSubGraph(mng, &origin_recomputed_nodes, true, false);

    // Get the inputs of the first target node in the topological sequence. The duplicated recomputed nodes should
    // not be executed until these inputs are ready.
    std::vector<AnfNodePtr> first_target_inputs =
      GetFirstTargetInputs(origin_nodes_topological, origin_recomputed_nodes, target_nodes);
    std::unordered_map<CNodePtr, CNodePtr> origin_to_recomputed_nodes;
    // Begin duplicate origin recomputed nodes with each target node.
    DuplicateRecomputedNodes(graph, target_nodes, origin_recomputed_nodes, first_target_inputs,
                             &origin_to_recomputed_nodes);
  }
  // Set need cse attr for doing cse after recompute.
  for (const auto &node : orders) {
    if (WithRecomputedScope(node)) {
      node->AddAttr(kAttrNeedCseAfterRecompute, MakeValue(true));
    }
  }
}
}  // namespace opt
}  // namespace mindspore
