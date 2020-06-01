/**
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

#include "optimizer/opt.h"

#include <memory>
#include <unordered_set>
#include <deque>
#include <algorithm>

#include "ir/anf.h"
#include "ir/manager.h"
#include "utils/ordered_set.h"

#include "utils/log_adapter.h"
#include "optimizer/optimizer.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
SubstitutionPtr MakeSubstitution(const TransformFuncType &transform, const std::string &name, const PrimitivePtr &prim,
                                 const RenormAction &renorm_action) {
  auto fn = [prim](const AnfNodePtr &node) -> bool { return IsPrimitiveCNode(node, prim); };
  return std::make_shared<Substitution>(transform, name, fn, renorm_action);
}

SubstitutionPtr MakeSubstitution(const TransformFuncType &transform, const std::string &name,
                                 const std::vector<PrimitivePtr> &prims, const RenormAction &renorm_action) {
  auto fn = [prims](const AnfNodePtr &node) -> bool {
    if (!node->isa<CNode>()) {
      return false;
    }

    for (auto &prim : prims) {
      if (IsPrimitiveCNode(node, prim)) {
        return true;
      }
    }
    return false;
  };

  return std::make_shared<Substitution>(transform, name, fn, renorm_action);
}

SubstitutionPtr MakeSubstitution(const TransformFuncType &transform, const std::string &name,
                                 const PredicateFuncType &predicate, const RenormAction &renorm_action) {
  return std::make_shared<Substitution>(transform, name, predicate, renorm_action);
}

AnfNodePtr Substitution::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) const {
#ifdef ENABLE_PROFILE
  double t = GetTime();
#endif
  AnfNodePtr result = transform_(optimizer, node);
#ifdef ENABLE_PROFILE
  if (optimizer != nullptr) {
    auto time = GetTime();
    MsProfile::StatTime("substitution." + name_, time - t);
    if (result != nullptr) {
      MsProfile::StatTime("match." + name_, time - t);
    }
  }
#endif
  if (optimizer != nullptr && optimizer->is_watch_renormalize() && result != nullptr) {
    if (renorm_action_ == FORCE_RENORM) {
      optimizer->add_node_to_renormalize(result);
    } else {
      // renorm_action_ is CHECK_RENORM
      if (result->abstract() == nullptr) {
        optimizer->add_node_to_renormalize(result);
      }
    }
  }

  return result;
}

static bool isTraversable(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  if (node->isa<CNode>() || node->isa<Parameter>()) {
    return true;
  }
  if (IsValueNode<FuncGraph>(node) || IsValueNode<RefKey>(node)) {
    return true;
  }
  return false;
}

bool SubstitutionList::ApplyTransform(const OptimizerPtr &optimizer, const AnfNodePtr &root_node,
                                      const SubstitutionPtr &transform) const {
#ifdef ENABLE_PROFILE
  double start = GetTime();
#endif
  FuncGraphManagerPtr manager = optimizer->manager();
  auto seen = NewSeenGeneration();
  // 1024 is for the initial capacity of deque
  std::deque<AnfNodePtr> todo(1024);
  todo.push_back(root_node);
  bool changes = false;

  auto &all_nodes = manager->all_nodes();
  while (!todo.empty()) {
    AnfNodePtr node = todo.front();
    todo.pop_front();

    // check whether this node has been matched.
    if (node == nullptr || node->seen_ == seen || !isTraversable(node) || !all_nodes.contains(node)) {
      continue;
    }
    node->seen_ = seen;

    // select nodes that this transform can be applied.
    bool is_match = transform->predicate_(node);

    // apply transform on this node
    bool change = false;
    if (is_match) {
      auto ret = (*transform)(optimizer, node);
      if (ret != nullptr && ret != node) {
        change = true;
        changes = true;
#ifdef ENABLE_PROFILE
        double t = GetTime();
#endif
        (void)manager->Replace(node, ret);
#ifdef ENABLE_PROFILE
        MsProfile::StatTime("replace." + transform->name_, GetTime() - t);
#endif
        node = ret;
      }
    }

    // find success, and add them to todo list
    if (IsValueNode<FuncGraph>(node)) {
      todo.push_back(GetValueNode<FuncGraphPtr>(node)->output());
    }

    if (node->isa<CNode>()) {
      auto &inputs = node->cast<CNodePtr>()->inputs();
      (void)std::copy(inputs.begin(), inputs.end(), std::back_inserter(todo));
    }

    auto &node_users = manager->node_users();
    if (change && node_users.find(node) != node_users.end()) {
      for (auto &use : node_users[node]) {
        auto use_node = use.first;
        if (use_node == nullptr) {
          continue;
        }
        todo.push_back(use_node);
        if (use_node->seen_ == seen) {
          use_node->seen_--;
        }
      }
    }
  }

#ifdef ENABLE_PROFILE
  MsProfile::StatTime("opt.transform", GetTime() - start);
#endif
  return changes;
}

bool SubstitutionList::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const {
  MS_EXCEPTION_IF_NULL(optimizer);
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = optimizer->manager();
  manager->AddFuncGraph(func_graph);

  bool loop = false;
  bool changes = false;

  do {
    loop = false;
    for (auto const &transform : list_) {
      auto change = ApplyTransform(optimizer, func_graph->output(), transform);
      changes = changes || change;
      loop = loop || change;
    }

    if (is_once_) {
      break;
    }
  } while (loop);

  return changes;
}
}  // namespace opt
}  // namespace mindspore
