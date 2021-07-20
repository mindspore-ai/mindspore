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

#include "frontend/optimizer/irpass/bool_scalar_eliminate.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
AnfNodePtr BoolScalarEliminate::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return nullptr;
  }

  if (!cnode->IsApply(prim::kPrimGetAttr)) {
    return nullptr;
  }

  auto vnode = cnode->input(1)->cast<ValueNodePtr>();
  if (vnode == nullptr) {
    return nullptr;
  }

  if (!vnode->value()->isa<BoolImm>()) {
    return nullptr;
  }

  auto res = optimizer->resource();
  auto manager = res->manager();
  auto &node_users = manager->node_users();
  auto iter = node_users.find(node);
  if (iter == node_users.end()) {
    return nullptr;
  }

  AnfNodeIndexSet node_idx_set = iter->second;
  for (auto &item : node_idx_set) {
    (void)manager->Replace(item.first, vnode);
  }
  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
