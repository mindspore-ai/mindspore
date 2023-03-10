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

#include "include/common/utils/recompute_helper.h"
#include <memory>
#include <queue>
#include <list>
#include <string>
#include <algorithm>
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/utils.h"

namespace mindspore {
constexpr auto kGradientsFlag = "Gradients";
const int64_t fusion_id_increasement_size = 2000;
bool CanNotRecomputed(const CNodePtr &node) {
  static mindspore::HashSet<PrimitivePtr> not_recomputed_op_list{
    prim::kPrimDropoutGenMask, prim::kPrimLoad, prim::kPrimTupleGetItem, prim::kPrimSend, prim::kPrimReceive};

  return std::any_of(not_recomputed_op_list.begin(), not_recomputed_op_list.end(),
                     [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
}

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

ValuePtr GetRecomputeCNodeAttr(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast_ptr<CNode>();
  if (cnode == nullptr) {
    return nullptr;
  }
  return cnode->GetAttr(kAttrRecompute);
}

bool IsSetNoRecomputeCNodeAttr(const AnfNodePtr &node) {
  auto cnode_recompute_val = GetRecomputeCNodeAttr(node);
  return cnode_recompute_val != nullptr && cnode_recompute_val->isa<BoolImm>() && !GetValue<bool>(cnode_recompute_val);
}

bool IsSetRecomputeCNodeAttr(const AnfNodePtr &node) {
  auto cnode_recompute_val = GetRecomputeCNodeAttr(node);
  return cnode_recompute_val != nullptr && cnode_recompute_val->isa<BoolImm>() && GetValue<bool>(cnode_recompute_val);
}

bool IsCandidateRecomputedNode(const CNodePtr &node) {
  // The tuple_getitem in the bprop function should also be recomputed.
  return (!IsBpropNode(node) || IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) && IsSetRecomputeCNodeAttr(node);
}

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
                     [](const auto &node_index) { return IsBpropNode(node_index.first); })) {
      continue;
    }
    // Check inputs.
    const auto &inputs = cnode->inputs();
    if (std::any_of(inputs.begin(), inputs.end(), [](const AnfNodePtr &node) { return IsBpropNode(node); })) {
      continue;
    }
    (void)candidate_recomputed_nodes.emplace_back(cnode);
  }
  return candidate_recomputed_nodes;
}

void GetMaxSubGraph(const FuncGraphManagerPtr &mng, mindspore::HashSet<CNodePtr> *recomputed_nodes, bool get_inputs,
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
    // No need to find nodes through side-effect dependency.
    if (IsPrimitiveCNode(current_node, prim::kPrimUpdateState)) {
      continue;
    }
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
                                      const mindspore::HashSet<CNodePtr> &max_recomputed_sub_graph,
                                      mindspore::HashSet<CNodePtr> *recompute_nodes,
                                      mindspore::HashSet<CNodePtr> *target_nodes) {
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
      // The tuple_getitem to be recomputed can be in the bprop function.
      if (!IsBpropNode(output_node) || IsPrimitiveCNode(output_node, prim::kPrimTupleGetItem)) {
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
                                             const mindspore::HashSet<CNodePtr> &recomputed_origin_nodes,
                                             const mindspore::HashSet<CNodePtr> &target_nodes) {
  std::vector<AnfNodePtr> first_target_inputs;
  for (const auto &node : origin_nodes_topological) {
    MS_EXCEPTION_IF_NULL(node);
    if (target_nodes.find(node) != target_nodes.end()) {
      for (size_t i = 1; i < node->size(); ++i) {
        auto input = node->input(i);
        MS_EXCEPTION_IF_NULL(input);
        if (!input->isa<CNode>()) {
          continue;
        }
        if (recomputed_origin_nodes.find(input->cast<CNodePtr>()) != recomputed_origin_nodes.end()) {
          continue;
        }
        (void)first_target_inputs.emplace_back(input);
      }
      break;
    }
  }
  return first_target_inputs;
}

bool HasGradInputs(const AnfNodePtr &node, mindspore::HashMap<AnfNodePtr, bool> *has_grad_inputs_map) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(has_grad_inputs_map);
  if (has_grad_inputs_map->find(node) != has_grad_inputs_map->end()) {
    return has_grad_inputs_map->find(node)->second;
  }
  auto cnode = node->cast_ptr<CNode>();
  if (cnode == nullptr) {
    (void)has_grad_inputs_map->emplace(node, false);
    return false;
  }
  const auto &inputs = cnode->inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    // For the pipeline split case, the forward pass may depend on the backward pass.
    if (cnode->IsApply(prim::kPrimDepend) && i == kDependAttachNodeIndex) {
      continue;
    }
    if (IsBpropNode(inputs[i]) || HasGradInputs(inputs[i], has_grad_inputs_map)) {
      (void)has_grad_inputs_map->emplace(node, true);
      return true;
    }
  }
  (void)has_grad_inputs_map->emplace(node, false);
  return false;
}

bool HasForwardOutput(const FuncGraphManagerPtr &mng, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(mng);
  const auto &node_users = mng->node_users();
  auto output_set_iter = node_users.find(node);
  if (output_set_iter == node_users.end()) {
    return false;
  }

  return std::any_of(output_set_iter->second.begin(), output_set_iter->second.end(),
                     [](const auto &node_index_set) { return !IsBpropNode(node_index_set.first); });
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
      (void)tuple_getitem_output_nodes->emplace_back(node_index_set.first);
    }
  }
}

bool SetRecomputedScope(const CNodePtr &node) {
  return WithRecomputedScope(node) ||
         (IsPrimitiveCNode(node, prim::kPrimDepend) && WithRecomputedScope(node->input(kRealInputIndexInDepend)));
}

// Set 'recompute' cnode attr for the nodes according to its scope.
// A node set 'recompute' cnode attr can become the candidate recomputed node.
void SetRecomputedAttr(const FuncGraphPtr &graph, const std::vector<CNodePtr> &origin_nodes_topological) {
  MS_EXCEPTION_IF_NULL(graph);
  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  mindspore::HashMap<AnfNodePtr, bool> has_grad_inputs_map;
  for (const auto &node : origin_nodes_topological) {
    MS_EXCEPTION_IF_NULL(node);
    // The node may be set the non-recomputed before such as the cell outputs.
    if (IsSetNoRecomputeCNodeAttr(node)) {
      continue;
    }
    if (IsBpropNode(node)) {
      continue;
    }
    // Filter some unrecomputable operators.
    if (CanNotRecomputed(node)) {
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
      prim_recompute_val = static_cast<int>(GetValue<bool>(prim_recompute_attr));
    }
    if ((SetRecomputedScope(cnode) && prim_recompute_val != 0) || prim_recompute_val == 1) {
      cnode->AddAttr(kAttrRecompute, MakeValue(true));
    }
    if (!IsSetRecomputeCNodeAttr(node)) {
      continue;
    }
    // Set attr for the tuple_getitem outputs.
    std::vector<AnfNodePtr> tuple_getitem_output_nodes;
    GetTupleGetItemOutputNodes(mng, node, &tuple_getitem_output_nodes);
    for (const auto &output_node : tuple_getitem_output_nodes) {
      auto output_cnode = output_node->cast_ptr<CNode>();
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
  if (origin_node->HasPrimalAttr(kAttrMicro)) {
    recomputed_node->AddPrimalAttr(kAttrMicro, origin_node->GetPrimalAttr(kAttrMicro));
  }
  if (origin_node->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
    recomputed_node->AddPrimalAttr(kAttrMicro, origin_node->GetPrimalAttr(kPrimalAttrForwardCommNodeUniqueId));
  }
  static int64_t recompute_id = 0;
  ++recompute_id;
  recomputed_node->AddAttr(kAttrRecomputeId, MakeValue(recompute_id));
  origin_node->AddAttr(kAttrRecomputeId, MakeValue(recompute_id));
  static const PrimitiveSet dropout_prims = {prim::kPrimDropout, prim::kPrimDropoutDoMask, prim::kPrimDropoutDoMaskV3};
  static const std::vector<std::string> need_primal_attr = {kAttrFusion, kPrimalAttrUniqueId,
                                                            kPrimalAttrForwardUniqueId};
  if (IsOneOfPrimitiveCNode(origin_node, dropout_prims)) {
    for (auto &primal_attr : need_primal_attr) {
      if (origin_node->HasPrimalAttr(primal_attr)) {
        recomputed_node->AddPrimalAttr(primal_attr, origin_node->GetPrimalAttr(primal_attr));
      }
    }
  }
  return recomputed_node;
}

CNodePtr NewRecomputedNode(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                           const std::vector<AnfNodePtr> &first_target_inputs,
                           const mindspore::HashSet<CNodePtr> &recomputed_origin_nodes,
                           mindspore::HashMap<CNodePtr, CNodePtr> *origin_to_recomputed_nodes) {
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
    if (i == 0 && IsPrimitive(input, prim::kPrimAllGather)) {
      auto prim = GetValuePtr<Primitive>(input);
      auto instance_name = prim->instance_name();
      bool is_from_parallel_optimizer = instance_name.find("parallel_optimizer") != std::string::npos;
      int64_t fusion_id = prim->HasAttr(kAttrFusion) ? GetValue<int64_t>(prim->GetAttr(kAttrFusion)) : 0;
      if (is_from_parallel_optimizer && fusion_id > 0) {
        auto new_prim = std::make_shared<Primitive>(prim::kPrimAllGather->name());
        (void)new_prim->SetAttrs(prim->attrs());
        new_prim->set_attr(kAttrFusion, MakeValue(fusion_id + fusion_id_increasement_size));
        new_prim->set_prim_type(prim->prim_type());
        new_prim->set_instance_name(instance_name);
        auto value_node = NewValueNode(new_prim);
        (void)new_inputs.emplace_back(value_node);
        continue;
      }
    }
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>()) {
      (void)new_inputs.emplace_back(input);
      continue;
    }
    auto input_cnode = input->cast<CNodePtr>();
    if (recomputed_origin_nodes.find(input_cnode) == recomputed_origin_nodes.end()) {
      if (IsPrimitiveCNode(input_cnode, prim::kPrimUpdateState)) {
        auto u = NewValueNode(kUMonad);
        u->set_abstract(kUMonad->ToAbstract());
        (void)new_inputs.emplace_back(u);
      } else {
        (void)new_inputs.emplace_back(input);
      }
    } else {
      has_recomputed_inputs = true;
      (void)new_inputs.emplace_back(NewRecomputedNode(graph, input_cnode, first_target_inputs, recomputed_origin_nodes,
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
    depend_node->AddAttr("recompute_depend", MakeValue(true));
    new_inputs[1] = depend_node;
  }
  auto recomputed_node = CreateNewRecomputedNode(graph, origin_node, new_inputs);
  (void)origin_to_recomputed_nodes->emplace(origin_node, recomputed_node);
  return recomputed_node;
}

void DuplicateRecomputedNodes(const FuncGraphPtr &graph, const mindspore::HashSet<CNodePtr> &target_nodes,
                              const mindspore::HashSet<CNodePtr> &origin_recomputed_nodes,
                              const std::vector<AnfNodePtr> &first_target_inputs,
                              mindspore::HashMap<CNodePtr, CNodePtr> *origin_to_recomputed_nodes) {
  MS_EXCEPTION_IF_NULL(graph);
  auto mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  for (const auto &target_node : target_nodes) {
    MS_EXCEPTION_IF_NULL(target_node);
    MS_LOG(DEBUG) << "Rebuild a new target_node " << target_node->DebugString() << " with the new recomputed input";
    auto target_cnode = target_node->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(target_cnode);
    std::vector<AnfNodePtr> new_target_inputs;
    for (const auto &input : target_cnode->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>()) {
        (void)new_target_inputs.emplace_back(input);
      } else {
        auto input_cnode = input->cast<CNodePtr>();
        if (origin_recomputed_nodes.find(input_cnode) != origin_recomputed_nodes.end()) {
          (void)new_target_inputs.emplace_back(NewRecomputedNode(graph, input_cnode, first_target_inputs,
                                                                 origin_recomputed_nodes, origin_to_recomputed_nodes));
        } else {
          (void)new_target_inputs.emplace_back(input_cnode);
        }
      }
    }
    auto new_target_node = graph->NewCNode(new_target_inputs);
    new_target_node->CloneCNodeInfo(target_node);
    new_target_node->AddAttr("target_grad", MakeValue(true));
    new_target_node->set_scope(target_node->scope());
    mng->Replace(target_node, new_target_node);
  }
}
}  // namespace mindspore
