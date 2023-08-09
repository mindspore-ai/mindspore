/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/dropout_gen_mask_fusion.h"

#include <memory>

#include "mindspore/core/ops/nn_ops.h"
#include "ir/graph_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace opt {
bool DropoutGenMaskFusion::DoFusion(const std::vector<CNodePtr> &genmasks, const std::set<int64_t> fusion_set,
                                    const FuncGraphManagerPtr &manager) const {
  // Fusion of masks with the same fusion_id
  auto &node_users_map = manager->node_users();
  std::vector<CNodePtr> temp;
  for (auto &fusion : fusion_set) {
    temp.clear();
    for (auto &mask : genmasks) {
      auto cur_fusion = GetValue<int64_t>(mask->GetPrimalAttr(kAttrFusion));
      if (cur_fusion == fusion) {
        (void)temp.emplace_back(mask);
      }
    }
    auto mask_first = temp.front();
    for (size_t i = 1; i < temp.size(); ++i) {
      auto node_users = node_users_map[temp[i]];
      for (auto user_pair : node_users) {
        auto user_node = user_pair.first->cast<CNodePtr>();
        manager->SetEdge(user_node, user_pair.second, mask_first);
      }
    }
  }
  return !fusion_set.empty();
}

bool DropoutGenMaskFusion::Run(const FuncGraphPtr &func_graph) {
  // Only pipeline parallel need run this pass
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto stages = parallel_context->pipeline_stage_split_num();
  auto grad_accu_num = parallel_context->grad_accumulation_step();
  if (stages <= 1 && grad_accu_num <= 1) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<CNodePtr> genmasks;
  std::set<int64_t> fusion_set;
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());

  // Get all GenMasks with fusion attr
  for (auto &node : node_list) {
    if (IsOneOfPrimitiveCNode(
          node, {prim::kPrimDropoutGenMask, prim::kPrimDropoutGenMaskV3, prim::kPrimStatelessDropOutGenMask})) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->HasPrimalAttr(kAttrFusion)) {
        genmasks.push_back(cnode);
        auto fusion_id = GetValue<int64_t>(cnode->GetPrimalAttr(kAttrFusion));
        (void)fusion_set.insert(fusion_id);
      }
    }
  }
  MS_LOG(INFO) << "DropoutGenMask's Num: " << genmasks.size() << " with fusion_id's Num: " << fusion_set.size();

  return DoFusion(genmasks, fusion_set, manager);
}
}  // namespace opt
}  // namespace mindspore
