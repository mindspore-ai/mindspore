/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/utils.h"
#include "primitive/sequence_ops.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"
#include "utils/compile_config.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
AnfNodePtr GetBpropGetter(const FuncGraphManagerPtr &manager, const CNodePtr &node) {
  const auto &user_nodes = manager->node_users()[node];
  for (const auto &iter : user_nodes) {
    if (IsPrimitiveCNode(iter.first, prim::kPrimTupleGetItem)) {
      auto idx = GetValueNode<Int64ImmPtr>(iter.first->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
      if (idx != nullptr && idx->value() == 1) {
        return iter.first;
      }
    }
  }
  return nullptr;
}

AnfNodePtr GetBpropCaller(const FuncGraphManagerPtr &manager, const AnfNodePtr &bprop_getter) {
  MS_EXCEPTION_IF_NULL(manager);
  if (bprop_getter == nullptr) {
    return nullptr;
  }
  const auto &node_users = manager->node_users();
  auto iter = node_users.find(bprop_getter);
  if (iter == node_users.end()) {
    return nullptr;
  }
  if (iter->second.size() != 1) {
    MS_LOG_WITH_NODE(EXCEPTION, bprop_getter)
      << "The number of bprop caller should be 1, but got " << iter->second.size()
      << ", bprop_getter: " << bprop_getter->DebugString();
  }
  auto user_node_idx = iter->second.begin();
  if (user_node_idx->second != 0) {
    MS_LOG_WITH_NODE(EXCEPTION, bprop_getter)
      << "The bprop_getter should be used in input 0, but got " << user_node_idx->second;
  }
  return user_node_idx->first;
}

bool RecomputeBeforeInline() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  const auto enable_recompute_before_inline = common::GetCompileConfig("ENABLE_RECOMPUTE_BEFORE_INLINE") == "1";
  return cell_reuse || enable_recompute_before_inline;
}
}  // namespace opt
}  // namespace mindspore
