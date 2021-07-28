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

#include "pipeline/jit/static_analysis/remove_monad.h"
#include <algorithm>
#include <map>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "base/core_ops.h"

namespace mindspore::pipeline {
namespace {
class RemoveMonad {
 public:
  explicit RemoveMonad(const FuncGraphPtr &func_graph) : func_graph_(func_graph), manager_(func_graph->manager()) {
    MS_EXCEPTION_IF_NULL(func_graph_);
    MS_EXCEPTION_IF_NULL(manager_);
  }
  ~RemoveMonad() = default;

  void Run() {
    auto nodes = TopoSort(func_graph_->get_return());
    for (auto &node : nodes) {
      if (node->isa<CNode>()) {
        auto prim = GetCNodePrimitive(node);
        if (prim != nullptr && CheckPrimRandomEffect(prim)) {
          // Remove monad input
          RemoveMonadFromRandomNodes(node);
        }
      }
      // Remove random nodes from monad chain
      if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
        RemoveRandomNodesFromMonadChain(node);
      }
    }
  }

 private:
  bool CheckPrimRandomEffect(const PrimitivePtr &prim) {
    bool has_random_effect = false;
    MS_EXCEPTION_IF_NULL(prim);
    auto effect_val = prim->GetAttr(GRAPH_FLAG_RANDOM_EFFECT);
    if (effect_val != nullptr && effect_val->isa<BoolImm>()) {
      has_random_effect = GetValue<bool>(effect_val);
    }
    return has_random_effect;
  }

  void RemoveMonadFromRandomNodes(const AnfNodePtr &node) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto &inputs = cnode->inputs();
    std::vector<AnfNodePtr> new_random_node_inputs;
    // Remove monad input, in order to parallel execution of random number operators
    (void)std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(new_random_node_inputs),
                       [](const AnfNodePtr &input) { return !HasAbstractMonad(input); });
    auto new_random_node = func_graph_->NewCNode(new_random_node_inputs);
    MS_EXCEPTION_IF_NULL(node->abstract());
    new_random_node->set_abstract(node->abstract());
    new_random_node->set_scope(node->scope());
    (void)manager_->Replace(node, new_random_node);
  }

  void RemoveRandomNodesFromMonadChain(const AnfNodePtr &node) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    const size_t first_index = 1;
    const size_t attach_index = 2;
    auto monad_input = cnode->input(first_index);
    auto attach_input = cnode->input(attach_index);
    if (attach_input->isa<CNode>()) {
      auto prim = GetCNodePrimitive(attach_input);
      if (prim != nullptr && CheckPrimRandomEffect(prim)) {
        (void)manager_->Replace(cnode, monad_input);
      }
    }
  }

  const FuncGraphPtr &func_graph_;
  FuncGraphManagerPtr manager_;
};
}  // namespace

// Remove monad from random operator of the given graph.
void RemoveRandomOpMonad(const FuncGraphPtr &func_graph) {
  RemoveMonad remover(func_graph);
  remover.Run();
}
}  // namespace mindspore::pipeline
