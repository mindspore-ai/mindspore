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

#include "frontend/optimizer/comm_op_reuse_tag.h"
#include <memory>
#include <vector>
#include <string>
#include <algorithm>
#include "ir/func_graph.h"
#include "mindspore/core/ops/core_ops.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/device_manager.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/comm_manager.h"

namespace mindspore {
namespace opt {
namespace {
inline bool is_comm_ops(const AnfNodePtr &node) {
  static const std::vector<PrimitivePtr> kCommunicationOpsPrim = {prim::kPrimAllReduce,
                                                                  prim::kPrimAllGather,
                                                                  prim::kPrimReduceScatter,
                                                                  prim::kPrimAllToAll,
                                                                  prim::kPrimAllSwap,
                                                                  prim::kPrimAllToAllv,
                                                                  prim::kPrimNeighborExchange,
                                                                  prim::kPrimNeighborExchangeV2,
                                                                  prim::kPrimNeighborExchangeV2Grad};

  for (const auto &prim : kCommunicationOpsPrim) {
    if (IsPrimitiveCNode(node, prim)) {
      return true;
    }
  }

  return false;
}
}  // namespace

void AddCommOpReuseTag(const FuncGraphPtr &graph) {
  if (parallel::g_device_manager == nullptr) {
    MS_LOG(INFO) << "parallel::g_device_manager is not initialized.";
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);

  if (!parallel::IsAutoParallelCareGraph(graph)) {
    return;
  }
  AnfNodePtr return_node = graph->get_return();
  MS_EXCEPTION_IF_NULL(return_node);
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(return_node);
  for (auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!is_comm_ops(node)) {
      continue;
    }
    auto comm_prim = GetCNodePrimitive(node);
    MS_EXCEPTION_IF_NULL(comm_prim);
    if (comm_prim->HasAttr(parallel::FUSION) && GetValue<int64_t>(comm_prim->GetAttr(parallel::FUSION)) != 0) {
      continue;
    }
    (void)comm_prim->AddAttr(parallel::COMM_REUSE, MakeValue(true));

    std::string group_name = "";
    if (comm_prim->HasAttr(parallel::GROUP)) {
      group_name = GetValue<std::string>(comm_prim->GetAttr(parallel::GROUP));
    }
    std::vector<unsigned int> rank_list = {};
    auto long_rank_list = parallel::g_device_manager->FindRankListByHashName(group_name);
    (void)std::transform(long_rank_list.begin(), long_rank_list.end(), std::back_inserter(rank_list),
                         [](int64_t d) -> unsigned int { return IntToUint(LongToInt(d)); });
    (void)comm_prim->AddAttr(kAttrRankList, MakeValue<std::vector<unsigned int>>(rank_list));
  }
}
}  // namespace opt
}  // namespace mindspore
