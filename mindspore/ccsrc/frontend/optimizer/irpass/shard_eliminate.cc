/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/shard_eliminate.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
AnfNodePtr ExpandShard(const CNodePtr &node) {
  auto vnode = node->input(1)->cast<ValueNodePtr>();
  auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  return NewValueNode(func_graph);
}
}  // namespace internal

bool ExpandShardPrim::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  bool change = false;
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &shard_node : prim_nodes_) {
    auto expanded_shard = internal::ExpandShard(shard_node);
    (void)manager->Replace(shard_node, expanded_shard);
    change = true;
  }
  return change;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
