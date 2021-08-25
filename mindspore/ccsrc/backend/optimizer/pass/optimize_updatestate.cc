/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/pass/optimize_updatestate.h"
#include <memory>
#include <vector>
#include <string>
#include "base/core_ops.h"
#include "utils/utils.h"

namespace mindspore {
namespace opt {
constexpr size_t kInputIndex = 1;
constexpr size_t kAttachIndex = 2;
constexpr size_t kAdditionalAttachIndex = 3;

const BaseRef OptimizeUpdateState::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimUpdateState, Xs});
}

const AnfNodePtr OptimizeUpdateState::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto update_state = dyn_cast<CNode>(node);
  MS_EXCEPTION_IF_NULL(update_state);
  if (update_state->size() <= kAdditionalAttachIndex) {
    // Skip UpdateState nodes with no additional attaches.
    return nullptr;
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  std::vector<AnfNodePtr> new_inputs;
  (void)new_inputs.emplace_back(update_state->input(0));
  (void)new_inputs.emplace_back(update_state->input(kInputIndex));
  (void)new_inputs.emplace_back(update_state->input(kAttachIndex));
  for (size_t i = kAdditionalAttachIndex; i < update_state->size(); ++i) {
    auto &attach = update_state->input(i);
    auto &users = node_users[attach];
    // In heterogeneous, parameters in subgraphs may only be used by UpdateState and should not be eliminated.
    if ((users.size() == 1) && (users.front().first == update_state) && !attach->isa<Parameter>()) {
      // If the only user of attach is the UpdateState node, drop the attach node.
      continue;
    }
    (void)new_inputs.emplace_back(attach);
  }
  if (new_inputs.size() == update_state->size()) {
    // Attaches not changed.
    return nullptr;
  }
  // Attaches changed, make a new UpdateState.
  auto new_update_state = func_graph->NewCNode(new_inputs);
  new_update_state->set_abstract(update_state->abstract());
  new_update_state->set_scope(update_state->scope());
  return new_update_state;
}
}  // namespace opt
}  // namespace mindspore
