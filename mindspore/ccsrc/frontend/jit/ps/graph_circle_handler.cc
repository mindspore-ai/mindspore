/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "frontend/jit/ps/graph_circle_handler.h"

#include <string>
#include <deque>
#include <utility>
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "primitive/framework_ops.h"

namespace mindspore {
namespace circle_handler {
namespace {
AnfNodePtrList CollectSortingCircleList(const std::deque<AnfNodePtr> &todo, const AnfNodePtr &next, SeenNum seen) {
  AnfNodePtrList ret;
  auto circle_node_it = std::find(todo.begin(), todo.end(), next);
  for (; circle_node_it != todo.end(); ++circle_node_it) {
    auto circle_node = *circle_node_it;
    if (circle_node->seen_ == seen) {
      (void)ret.emplace_back(circle_node);
    }
  }
  return ret;
}

std::string GenerateCircleDebugString(const AnfNodePtrList &circle, const std::string &pass_name,
                                      const std::string &switch_name) {
  if (circle.empty()) {
    return "";
  }
  std::stringstream buffer;
  buffer << "Encounter graph circle for pass: " << pass_name;
  if (switch_name != "") {
    buffer << ". It can be disabled by turning off: " << switch_name;
  }
  buffer << ". circles are:\n";
  for (size_t i = 0; i < circle.size(); ++i) {
    buffer << std::to_string(i) << ": " << circle[i]->DebugString() << "\n";
  }
  return buffer.str();
}
}  // namespace

AnfNodePtrList FindGraphCircle(const FuncGraphPtr &fg) {
  auto root = fg->output();
  MS_EXCEPTION_IF_NULL(root);
  auto seen = NewSeenGeneration();
  std::deque<AnfNodePtr> todo;
  (void)todo.emplace_back(root);
  while (!todo.empty()) {
    AnfNodePtr &node = todo.back();
    if (node->extra_seen_ == seen) {  // We use extra_seen_ as finish flag
      todo.pop_back();
      continue;
    }
    auto incl = AlwaysInclude(node);
    if (node->seen_ == seen) {  // We use seen_ as checking flag
      node->extra_seen_ = seen;
      todo.pop_back();
      continue;
    }
    node->seen_ = seen;
    if (incl == FOLLOW) {
      for (auto &weak_next : SuccDeeperSimple(node)) {
        auto next = weak_next.lock();
        if (next == nullptr || next->extra_seen_ == seen) {
          continue;
        }
        if (next->seen_ != seen) {
          (void)todo.emplace_back(std::move(next));
          continue;
        }
        auto next_fg = next->func_graph();
        if (next_fg != nullptr && next_fg->return_node() == next) {
          continue;
        }
        return CollectSortingCircleList(todo, next, seen);
      }
    } else if (incl > EXCLUDE) {  // Not NOFOLLOW or EXCLUDE
      MS_LOG(INTERNAL_EXCEPTION) << "The result of include(node) must be one of: \"follow\", \"nofollow\", \"exclude\"";
    }
  }
  return AnfNodePtrList{};
}

void SetAttrToDepend(const FuncGraphPtr &fg) {
  std::string enable_recovery = common::GetEnv("MS_DEV_ENABLE_PASS_CIRCLE_RECOVERY");
  if (enable_recovery != "1") {
    return;
  }
  const auto &all_nodes = TopoSort(fg->output(), SuccDeeperSimple);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
      continue;
    }
    node->cast<CNodePtr>()->AddAttr(kCircleDetect, MakeValue(true));
  }
}

bool RevertDependNode(const FuncGraphPtr &fg, const FuncGraphManagerPtr &mng) {
  const auto &all_nodes = TopoSort(fg->output(), SuccDeeperSimple, AlwaysInclude, true);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimDepend)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode->HasAttr(kCircleDetect)) {
      continue;
    }
    auto replace_node = cnode->input(1);
    mng->Replace(cnode, replace_node);
  }
  // Recheck if the graph has circle.
  const auto &circle_nodes = FindGraphCircle(fg);
  return circle_nodes.empty();
}

void DetectAndRevertGraphCircle(const FuncGraphPtr &fg, const FuncGraphManagerPtr &mng, const std::string &pass_name,
                                const std::string &switch_name) {
  const auto &circle_nodes = FindGraphCircle(fg);
  if (circle_nodes.empty()) {
    MS_LOG(INFO) << "No graph circle for pass: " << pass_name;
    return;
  }
  const auto &debug_str = GenerateCircleDebugString(circle_nodes, pass_name, switch_name);
  std::string enable_recovery = common::GetEnv("MS_DEV_ENABLE_PASS_CIRCLE_RECOVERY");
  if (enable_recovery != "1") {
    MS_LOG(ERROR) << debug_str
                  << "You can set MS_DEV_ENABLE_PASS_CIRCLE_RECOVERY=1 to skip the pass that encounter graph cycle";
    return;
  } else {
    MS_LOG(WARNING) << debug_str;
  }

  bool succ = RevertDependNode(fg, mng);
  if (!succ) {
    MS_LOG(EXCEPTION) << "Failed to recover circle graph for pass: " << pass_name;
  }
}
}  // namespace circle_handler
}  // namespace mindspore
