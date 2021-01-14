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

#include "frontend/optimizer/opt.h"

#include <deque>
#include <memory>
#include <unordered_map>
#include <algorithm>

#include "ir/anf.h"
#include "ir/manager.h"
#include "frontend/optimizer/optimizer.h"
#include "utils/log_adapter.h"

namespace mindspore {
/* namespace to support opt */
namespace opt {
SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name, const PrimitivePtr &prim,
                                 const RenormAction &renorm_action) {
  auto fn = [prim](const AnfNodePtr &node) -> bool { return IsPrimitiveCNode(node, prim); };
  return std::make_shared<Substitution>(transform, name, fn, renorm_action);
}

SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name,
                                 const std::vector<PrimitivePtr> &prims, const RenormAction &renorm_action) {
  auto fn = [prims](const AnfNodePtr &node) -> bool {
    if (!node->isa<CNode>()) {
      return false;
    }

    auto cnode = node->cast<CNodePtr>();
    auto inp0 = cnode->input(0);
    auto prim0 = GetValueNode<PrimitivePtr>(inp0);
    if (prim0 == nullptr) {
      return false;
    }

    auto hash = prim0->Hash();
    auto const &name = prim0->name();
    for (auto &prim : prims) {
      if (hash == prim->Hash() && name == prim->name()) {
        return true;
      }
    }
    return false;
  };

  return std::make_shared<Substitution>(transform, name, fn, renorm_action);
}

SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name,
                                 const PredicateFuncType &predicate, const RenormAction &renorm_action) {
  return std::make_shared<Substitution>(transform, name, predicate, renorm_action);
}

AnfNodePtr Substitution::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
#ifdef ENABLE_PROFILE
  double t = GetTime();
#endif
  AnfNodePtr result = (*transform_)(optimizer, node);
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
    if ((renorm_action_ == FORCE_RENORM) || (result->abstract() == nullptr)) {
      optimizer->set_is_untyped_generated();
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
  todo.clear();
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
      TraceGuard trace_guard(std::make_shared<TraceOpt>(node->debug_info()));
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
  MsProfile::StatTime("opt.transform." + optimizer->name(), GetTime() - start);
#endif
  return changes;
}

bool SubstitutionList::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const {
  MS_EXCEPTION_IF_NULL(optimizer);
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = optimizer->manager();
  manager->AddFuncGraph(func_graph);

  // for transform status counting
  size_t space = 0;
  std::unordered_map<std::string, std::vector<bool>> status;
  if (optimizer->is_on_debug_) {
    for (size_t i = 0; i < list_.size(); i++) {
      status[list_[i]->name_ + std::to_string(i)] = {};
    }
  }

  bool loop = false;
  bool changes = false;

  do {
    loop = false;
    for (size_t i = 0; i < list_.size(); i++) {
      auto change = ApplyTransform(optimizer, func_graph->output(), list_[i]);
      changes = changes || change;
      loop = loop || change;

      // record the status of each transform
      static const auto enable_dump_pass_ir = (common::GetEnv("ENV_DUMP_PASS_IR") == "1");
      if (enable_dump_pass_ir && MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
        auto fg_name = optimizer->name() + "_" + std::to_string(optimizer->CurPass_.counter) + "_" +
                       optimizer->CurPass_.name + "_" + list_[i]->name_;
        DumpIR(fg_name + ".ir", func_graph);
        if (MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
          func_graph->DumpFuncGraph(fg_name);
          ExportIR(fg_name + ".dat", "", func_graph);
        }
      }
      if (optimizer->is_on_debug_) {
        status[list_[i]->name_ + std::to_string(i)].push_back(change);
        space = std::max(list_[i]->name_.size(), space);
      }
    }

    if (is_once_) {
      break;
    }
  } while (loop);

  // display the status of each transform
  if (optimizer->is_on_debug_) {
    std::stringstream ss;
    ss << std::endl
       << "Pass: " << optimizer->name() << "(" << optimizer->CurPass_.counter << ")_" << optimizer->CurPass_.name
       << std::endl;
    for (size_t i = 0; i < list_.size(); i++) {
      auto name = list_[i]->name_;
      ss << std::left << std::setw(space + 4) << name << "\t";
      for (auto change : status[name + std::to_string(i)]) {
        ss << change << " ";
      }
      ss << std::endl;
    }
    MS_LOG(DEBUG) << ss.str();
  }

  return changes;
}
}  // namespace opt
}  // namespace mindspore
