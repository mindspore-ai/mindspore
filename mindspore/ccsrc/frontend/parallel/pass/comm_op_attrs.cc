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

#include "frontend/parallel/pass/comm_op_attrs.h"
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "include/utils/anfalgo.h"
#include "include/utils/utils.h"
#include "ir/graph_utils.h"

namespace mindspore {
namespace opt {
void CommOpAttrs(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  AnfNodePtr return_node = graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(return_node);
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto primitive = GetCNodePrimitive(node);
    if (primitive == nullptr) {
      continue;
    }
    if (!common::AnfAlgo::IsCommunicationOp(node)) {
      continue;
    }
    auto comm_prim = common::AnfAlgo::GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(comm_prim);
    std::string group_name = "";
    if (comm_prim->HasAttr(parallel::GROUP)) {
      group_name = GetValue<std::string>(comm_prim->GetAttr(parallel::GROUP));
    }
    std::vector<unsigned int> rank_list = {};
    auto long_rank_list = parallel::g_device_manager->FindRankListByHashName(group_name);
    (void)std::transform(long_rank_list.begin(), long_rank_list.end(), std::back_inserter(rank_list),
                         [](int64_t d) -> unsigned int { return IntToUint(LongToInt(d)); });
    (void)comm_prim->AddAttr(kAttrGroupRankIds, MakeValue<std::vector<unsigned int>>(rank_list));
  }
}
}  // namespace opt
}  // namespace mindspore
