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
#include "backend/common/graph_kernel/core/split_schemer.h"
#include "ops/core_ops.h"

namespace mindspore::graphkernel {
bool SplitSchemer::NeedInline(size_t group_id) const {
  if (group_id >= need_inline_.size()) {
    MS_LOG(EXCEPTION) << "The group_id " << group_id << " is out of range of group num " << need_inline_.size();
  }
  return need_inline_[group_id] != 0;
}

size_t CommonSplitSchemer::AddGroup(AnfNodePtrList &&nodes, bool need_inline) {
  auto group_id = split_plan_.size();
  (void)split_plan_.emplace_back(nodes);
  need_inline_.push_back(need_inline ? 1 : 0);
  for (const auto &node : split_plan_.back()) {
    node_group_[node] = group_id;
  }
  return group_id;
}

void CommonSplitSchemer::GroupReturnNode(const FuncGraphPtr &func_graph) {
  auto ret_node = func_graph->get_return();
  MS_EXCEPTION_IF_NULL(ret_node);
  auto output = func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  // set the make_tuple node to a new group.
  if (IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    (void)AddGroup(AnfNodePtrList{output, ret_node}, true);
  } else {
    auto group_id = node_group_[output];
    node_group_[ret_node] = group_id;
    (void)split_plan_[group_id].emplace_back(ret_node);
  }
}
}  // namespace mindspore::graphkernel
