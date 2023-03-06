/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "include/common/utils/cse.h"

#include <vector>
#include <set>

#include "ir/anf.h"
#include "utils/hash_map.h"
#include "abstract/abstract_function.h"
#include "utils/flags.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "mindspore/core/ops/core_ops.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
using mindspore::abstract::AbstractBase;
using mindspore::abstract::AbstractFunction;
using mindspore::abstract::AbstractFunctionPtr;

bool WithRecomputedScope(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto full_name_with_scope = node->fullname_with_scope();
  return full_name_with_scope.find(kAttrRecompute) == 0;
}

bool IsSetRecomputed(const CNodePtr &a, const CNodePtr &b) {
  return (WithRecomputedScope(a) && !a->HasAttr(kAttrNeedCseAfterRecompute)) ||
         (WithRecomputedScope(b) && !b->HasAttr(kAttrNeedCseAfterRecompute));
}

bool IsHiddenSideEffectNode(const AnfNodePtr &node) {
  auto prim = GetCNodePrimitive(node);
  if (prim == nullptr) {
    return false;
  }
  return prim->HasAttr(GRAPH_FLAG_SIDE_EFFECT_HIDDEN);
}

void UpdateDebugInfoAndDumpFlag(const AnfNodePtr &main, const AnfNodePtr &node) {
  if (main == nullptr || !main->isa<CNode>()) {
    return;
  }
  if (AnfUtils::GetDumpFlag(node) && !AnfUtils::GetDumpFlag(main)) {
    AnfUtils::SetDumpFlag(main);
  }
  auto main_cnode = main->cast<CNodePtr>();
  main_cnode->AddFusedDebugInfo(node);
}

BasePtr AbsOf(const AnfNodePtr &node, bool ignore_fg_abs_tracking_id) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_abs = node->abstract();
  // In testcase: TestOptOpt.CSE, node->abstract() is null.
  if (node_abs == nullptr) {
    return kAnyValue;
  }
  if (node_abs->isa<abstract::PrimitiveAbstractClosure>()) {
    // Ignore the tracking_id and prim pointer hash.
    auto prim_abs = node_abs->cast_ptr<abstract::PrimitiveAbstractClosure>();
    return prim_abs->prim();
  } else if (ignore_fg_abs_tracking_id && node_abs->isa<abstract::FuncGraphAbstractClosure>()) {
    // Ignore the tracking_id.
    return node_abs->cast_ptr<abstract::AbstractFunction>()->CopyWithoutTrackingId();
  }
  return node_abs;
}

bool CSE::BuildOrderGroupForOneGraph(const FuncGraphPtr &fg, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(fg);
  std::vector<std::size_t> order_group;
  mindspore::HashMap<std::size_t, std::vector<AnfNodePtr>> groups;
  mindspore::HashMap<AnfNodePtr, std::size_t> hashes;

  std::vector<AnfNodePtr> toposet = TopoSort(fg->get_return());
  for (const auto &node : toposet) {
    MS_EXCEPTION_IF_NULL(node);
    if (hashes.find(node) != hashes.end()) {
      continue;
    }
    if (IsHiddenSideEffectNode(node) && node->func_graph() != nullptr) {
      MS_LOG(DEBUG) << "Add hidden func graph:" << node->func_graph()->ToString();
      (void)hidden_side_effect_func_graphs_.insert(node->func_graph());
    }
    std::size_t h = 0;
    if (node->isa<ValueNode>()) {
      auto prim = GetValueNode<PrimitivePtr>(node);
      if (IsPrimitiveEquals(prim, prim::kPrimUpdateState)) {
        continue;
      }
      ValueNodePtr value_node = node->cast<ValueNodePtr>();
      auto value = value_node->value();
      MS_EXCEPTION_IF_NULL(value);
      h = hash_combine(value->hash(), (AbsOf(value_node, true)->hash()));
    } else if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      auto &inputs = cnode->inputs();
      size_t init = 0;
      h = std::accumulate(inputs.begin(), inputs.end(), init, [&hashes](std::size_t hash, const AnfNodePtr &node_in) {
        return hash_combine(hash, hashes[node_in]);
      });
    } else if (node->isa<Parameter>()) {
      h = node->hash();
    } else {
      MS_LOG(ERROR) << "Unknown node type";
    }

    hashes[node] = h;
    if (groups.find(h) == groups.end()) {
      std::vector<AnfNodePtr> innervec({node});
      groups[h] = innervec;
      order_group.emplace_back(h);
    } else {
      groups[h].push_back(node);
    }
  }
  return CalReplaceNodes(manager, order_group, &groups);
}

void CSE::DoReplace(const FuncGraphManagerPtr &manager) {
  auto transact = manager->Transact();
  // if A is a hidden_side_effect node, then A's user B can't be replaced by main, then B's user C can't be replaced by
  // main.
  HashSet<AnfNodePtr> cannot_replace_nodes;
  for (const auto &[node, main] : replicated_nodes_) {
    bool main_input_cannot_replace = false;
    if (main->isa<CNode>()) {
      auto c_main = main->cast<CNodePtr>();
      const auto &c_main_inputs = c_main->inputs();
      auto input_can_not_replace = [&cannot_replace_nodes](const AnfNodePtr &node) {
        return cannot_replace_nodes.find(node) != cannot_replace_nodes.cend();
      };
      main_input_cannot_replace = std::any_of(c_main_inputs.cbegin(), c_main_inputs.cend(), input_can_not_replace);
    }
    if (HasHiddenSideEffect(main) || main_input_cannot_replace) {
      (void)cannot_replace_nodes.insert(main);
      continue;
    }
    // We don't merge primitive cnodes with random effect.
    MS_LOG(DEBUG) << "CSE replace, node:" << node->DebugString() << ", main:" << main->DebugString();
    (void)transact.Replace(node, main);
  }
  transact.Commit();
}

bool CSE::BuildOrderGroupAndDoReplace(const FuncGraphManagerPtr manager) {
  bool changed = false;
  for (const auto &fg : manager->func_graphs()) {
    changed = BuildOrderGroupForOneGraph(fg, manager) || changed;
  }
  DoReplace(manager);
  return changed;
}

// Check whether is a func graph call node and func graph has hidden side effect node.
bool CSE::IsHiddenSideEffectCall(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  // Check weather it is a func graph call.
  if (IsValueNode<Primitive>(cnode->input(kAnfPrimitiveIndex))) {
    return false;
  }
  // If it is a func graph call node, get all graphs  from abstract.
  auto func_graphs = abstract::GetFuncGraphsFromCallNode(cnode);
  auto is_hidden_side_effect_graph = [this](const FuncGraphPtr &fg) -> bool {
    return hidden_side_effect_func_graphs_.find(fg) != hidden_side_effect_func_graphs_.end();
  };
  return std::any_of(func_graphs.cbegin(), func_graphs.cend(), is_hidden_side_effect_graph);
}

bool CSE::HasHiddenSideEffect(const AnfNodePtr &node) {
  if (IsHiddenSideEffectNode(node)) {
    return true;
  }
  if (IsHiddenSideEffectCall(node)) {
    return true;
  }
  return false;
}

AnfNodePtr CSE::GetReplicatedNode(const AnfNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  HashSet<AnfNodePtr> visited_nodes;
  auto it = replicated_nodes_.find(node);
  if (it != replicated_nodes_.cend()) {
    return it->second;
  }
  return node;
}

void CSE::AddReplicatedNode(const AnfNodePtr &node, const AnfNodePtr &main) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(main);
  if (node == main) {
    MS_LOG(WARNING) << "Can't replace node by itself, node:" << node->DebugString();
    return;
  }
  (void)replicated_nodes_.emplace(node, main);
}

bool CSE::CheckReplace(const AnfNodePtr &main, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(main);
  MS_EXCEPTION_IF_NULL(node);
  if (main->isa<ValueNode>() && node->isa<ValueNode>()) {
    auto main_value = GetValueNode(main);
    auto node_value = GetValueNode(node);
    return (AbsOf(main, true) == AbsOf(node, true)) && (*main_value == *node_value);
  } else if (main->isa<CNode>() && node->isa<CNode>()) {
    auto c_main = main->cast<CNodePtr>();
    auto c_node = node->cast<CNodePtr>();
    // Not do cse for the node set recompute before the recompute pass.
    if (IsSetRecomputed(c_main, c_node)) {
      return false;
    }
    // Can not merge J because the J user size should be 1.
    if (IsPrimitiveCNode(c_main, prim::kPrimJ)) {
      return false;
    }
    const auto &inputs1 = c_main->inputs();
    const auto &inputs2 = c_node->inputs();
    if (inputs1.size() != inputs2.size()) {
      return false;
    }
    // Check inputs, all inputs should equal.
    for (size_t i = 0; i < inputs1.size(); i++) {
      auto input1 = GetReplicatedNode(inputs1[i]);
      auto input2 = GetReplicatedNode(inputs2[i]);
      MS_EXCEPTION_IF_NULL(input1);
      MS_EXCEPTION_IF_NULL(input2);
      if ((input1 == input2) || (*input1 == *input2)) {
        continue;
      }
      // Handle the case of two different Tensor, but with the same value.
      if (IsValueNode<tensor::Tensor>(input1) && IsValueNode<tensor::Tensor>(input2)) {
        auto tensor1 = GetValueNode<tensor::TensorPtr>(input1);
        auto tensor2 = GetValueNode<tensor::TensorPtr>(input2);
        if (tensor1->ValueEqual(*tensor2)) {
          continue;
        }
      }
      return false;
    }
    return true;
  }
  // a parameter node.
  return false;
}

bool CSE::CalReplaceNodes(const FuncGraphManagerPtr manager, const std::vector<std::size_t> &order_group,
                          mindspore::HashMap<std::size_t, std::vector<AnfNodePtr>> *groups) {
  bool changes = false;
  std::set<size_t> clear_set;
  for (auto &h : order_group) {
    std::vector<AnfNodePtr> &group = (*groups)[h];
    // If there are more than 2 node in that group, they may be same common expression can be eliminated.
    if (group.size() > 1) {
      for (size_t k = 0; k < group.size() - 1; k++) {
        AnfNodePtr main = group[k];
        MS_EXCEPTION_IF_NULL(main);

        // When all node in group has been replaced
        // or a valuenode node, skip compare in group
        if ((k + 1 + clear_set.size() == group.size()) || (k > 0 && main->isa<ValueNode>())) {
          break;
        }

        // skip node has been replaced
        if (clear_set.find(k) != clear_set.end()) {
          continue;
        }

        // Compare with rest elements in this group.
        for (size_t i = k + 1; i < group.size(); i++) {
          auto node = group[i];
          MS_EXCEPTION_IF_NULL(node);

          if (clear_set.find(i) != clear_set.end()) {
            continue;
          }
          if (main->func_graph() != node->func_graph()) {
            continue;
          }
          if (CheckReplace(node, main)) {
            changes = true;
            UpdateDebugInfoAndDumpFlag(main, node);
            MS_LOG(DEBUG) << "Add replicated_nodes_, node:" << node->DebugString() << ", main:" << main->DebugString();
            AddReplicatedNode(node, main);
            (void)clear_set.insert(i);
          }
        }
      }
      clear_set.clear();
    }
  }
  return changes;
}

void CSE::Init() {
  hidden_side_effect_func_graphs_.clear();
  replicated_nodes_.clear();
}

bool CSE::Cse(const FuncGraphPtr root, const FuncGraphManagerPtr manager) {
  MS_EXCEPTION_IF_NULL(manager);
  Init();
  manager->AddFuncGraph(root);
  return BuildOrderGroupAndDoReplace(manager);
}
}  // namespace opt
}  // namespace mindspore
