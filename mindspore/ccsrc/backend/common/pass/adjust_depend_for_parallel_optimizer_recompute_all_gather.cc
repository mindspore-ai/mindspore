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

#include "backend/common/pass/adjust_depend_for_parallel_optimizer_recompute_all_gather.h"
#include <algorithm>
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
constexpr const int64_t kFusionGap = 2;
bool AdjustDependForParallelOptimizerRecomputeAllGather::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  mindspore::HashMap<int64_t, bool> forward_allgather_recompute_value_in_fusion_group;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::vector<int64_t> parallel_optimizer_recompute_allgather_fusion_ids;
  std::vector<AnfNodePtr> parallel_optimizer_recompute_allgathers;
  std::vector<AnfNodePtr> parallel_optimizer_recompute_first_fusion_allgathers;
  int64_t unrecompute_max_fusion_id = -1;
  int64_t recompute_min_fusion_id = 0;
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!common::AnfAlgo::IsAllgather(cnode) || !common::AnfAlgo::IsFusion(cnode) ||
        !common::AnfAlgo::IsFromParallelOptimizer(cnode)) {
      continue;
    }
    if (common::AnfAlgo::IsRecompute(cnode)) {
      int64_t fusion_id = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
      if (std::find(parallel_optimizer_recompute_allgather_fusion_ids.begin(),
                    parallel_optimizer_recompute_allgather_fusion_ids.end(),
                    fusion_id) == parallel_optimizer_recompute_allgather_fusion_ids.end()) {
        parallel_optimizer_recompute_allgather_fusion_ids.push_back(fusion_id);
        if (recompute_min_fusion_id == 0 || fusion_id < recompute_min_fusion_id) {
          recompute_min_fusion_id = fusion_id;
        }
        parallel_optimizer_recompute_first_fusion_allgathers.push_back(node);
      } else {
        parallel_optimizer_recompute_allgathers.push_back(node);
      }
    } else {
      int64_t unrecompute_fusion_id = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
      unrecompute_max_fusion_id = std::max(unrecompute_fusion_id, unrecompute_max_fusion_id);
      bool would_be_recomputed = common::AnfAlgo::HasNodeAttr(kAttrRecompute, cnode) &&
                                 common::AnfAlgo::GetNodeAttr<bool>(cnode, kAttrRecompute);
      auto [iter, inserted] =
        forward_allgather_recompute_value_in_fusion_group.emplace(unrecompute_fusion_id, would_be_recomputed);
      if (!inserted && iter->second != would_be_recomputed) {
        MS_LOG(EXCEPTION) << "In same fusion group, the allgather recompute attribute should be equal. "
                             "The normal node is:"
                          << cnode->fullname_with_scope();
      }
    }
  }
  IncreaseAllgatherFusionId(parallel_optimizer_recompute_allgathers,
                            parallel_optimizer_recompute_first_fusion_allgathers, unrecompute_max_fusion_id,
                            recompute_min_fusion_id);
  return AdjustAllgatherDepend(graph, parallel_optimizer_recompute_allgathers);
}

void AdjustDependForParallelOptimizerRecomputeAllGather::IncreaseAllgatherFusionId(
  const std::vector<AnfNodePtr> &parallel_optimizer_recompute_allgathers,
  const std::vector<AnfNodePtr> &parallel_optimizer_recompute_first_fusion_allgathers,
  int64_t unrecompute_max_fusion_id, int64_t recompute_min_fusion_id) const {
  // means that there may some forward allgather and duplicated allgather would be fused.
  if (recompute_min_fusion_id <= unrecompute_max_fusion_id) {
    MS_LOG(WARNING) << "Increase the duplicated allgather fusion id";
    for (auto &adjust_node : parallel_optimizer_recompute_first_fusion_allgathers) {
      MS_EXCEPTION_IF_NULL(adjust_node);
      int64_t current_fusion_id = common::AnfAlgo::GetNodeAttr<int64_t>(adjust_node, kAttrFusion);
      int64_t destination_fusion_id =
        (kFusionGap + current_fusion_id + unrecompute_max_fusion_id) - recompute_min_fusion_id;
      common::AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(destination_fusion_id), adjust_node);
    }
    for (auto &adjust_node : parallel_optimizer_recompute_allgathers) {
      MS_EXCEPTION_IF_NULL(adjust_node);
      int64_t current_fusion_id = common::AnfAlgo::GetNodeAttr<int64_t>(adjust_node, kAttrFusion);
      int64_t destination_fusion_id =
        (kFusionGap + current_fusion_id + unrecompute_max_fusion_id) - recompute_min_fusion_id;
      common::AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(destination_fusion_id), adjust_node);
    }
  }
}

bool AdjustDependForParallelOptimizerRecomputeAllGather::AdjustAllgatherDepend(
  const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &parallel_optimizer_recompute_allgathers) const {
  MS_EXCEPTION_IF_NULL(graph);
  FuncGraphManagerPtr manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  bool changed = false;
  for (auto &node : parallel_optimizer_recompute_allgathers) {
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto depend_node = common::AnfAlgo::GetInputNode(cnode, 0);
    MS_EXCEPTION_IF_NULL(depend_node);
    auto set_edge_node = node;
    if (IsPrimitiveCNode(depend_node, prim::kPrimTensorMove)) {
      auto tensormove_cnode = depend_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(tensormove_cnode);
      set_edge_node = depend_node;
      depend_node = common::AnfAlgo::GetInputNode(tensormove_cnode, 0);
    }
    if (IsPrimitiveCNode(depend_node, prim::kPrimDepend)) {
      auto depend_cnode = depend_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(depend_cnode);
      AnfNodeIndexSet allgather_node_set = manager->node_users()[cnode];
      for (auto &node_pair : allgather_node_set) {
        auto allgather_next_node = node_pair.first;
        CNodePtr allgather_next_cnode = node_pair.first->cast<CNodePtr>();
        if (allgather_next_cnode == nullptr || !IsValueNode<Primitive>(allgather_next_cnode->input(0))) {
          continue;
        }
        std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          allgather_next_node, common::AnfAlgo::GetInputNode(depend_cnode, 1)};
        auto new_depend = graph->NewCNode(inputs);
        new_depend->set_abstract(depend_node->abstract());
        manager->SetEdge(set_edge_node, 1, common::AnfAlgo::GetInputNode(depend_cnode, 0));
        (void)manager->Replace(allgather_next_node, new_depend);
        changed = true;
      }
    } else if (IsPrimitiveCNode(depend_node, prim::kPrimCast) &&
               IsPrimitiveCNode(common::AnfAlgo::GetInputNode(depend_node->cast<CNodePtr>(), 0), prim::kPrimDepend)) {
      auto cast_cnode = depend_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cast_cnode);
      auto cast_depend_node = common::AnfAlgo::GetInputNode(cast_cnode, 0);
      auto cast_depend_cnode = cast_depend_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cast_depend_cnode);
      AnfNodeIndexSet allgather_node_set = manager->node_users()[cnode];
      for (auto &node_pair : allgather_node_set) {
        auto allgather_next_node = node_pair.first;
        CNodePtr allgather_next_cnode = node_pair.first->cast<CNodePtr>();
        if (allgather_next_cnode == nullptr || !IsValueNode<Primitive>(allgather_next_cnode->input(0))) {
          continue;
        }
        std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          allgather_next_node, common::AnfAlgo::GetInputNode(cast_depend_cnode, 1)};
        auto new_depend = graph->NewCNode(inputs);
        new_depend->set_abstract(cast_depend_node->abstract());
        manager->SetEdge(depend_node, 1, common::AnfAlgo::GetInputNode(cast_depend_cnode, 0));
        (void)manager->Replace(allgather_next_node, new_depend);
        changed = true;
      }
    } else {
      MS_LOG(WARNING) << "The parallel optimizer recompute allgather has no depend edge";
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
