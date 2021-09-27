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

#include "backend/optimizer/pass/adjust_depend_for_parallel_optimizer_recompute_all_gather.h"
#include "utils/utils.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
bool AdjustDependForParallelOptimizerRecomputeAllGather::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::unordered_map<int64_t, bool> forward_allgather_recompute_value_in_fusion_group;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::vector<int64_t> parallel_optimizer_recompute_allgather_fusion_ids;
  std::vector<AnfNodePtr> parallel_optimizer_recompute_allgathers;
  std::vector<AnfNodePtr> parallel_optimizer_recompute_first_fusion_allgathers;
  int64_t unrecompute_max_fusion_id = -1;
  int64_t recompute_min_fusion_id = 0;
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfAlgo::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto primitive = AnfAlgo::GetCNodePrimitive(cnode);
    auto instance_name = primitive->instance_name();
    bool is_allgather = AnfAlgo::GetCNodeName(cnode) == kAllGatherOpName;
    bool is_fusion = AnfAlgo::HasNodeAttr(kAttrFusion, cnode) && AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion) > 0;
    bool is_recompute = cnode->GetAttr(kAttrDuplicated) != nullptr && GetValue<bool>(cnode->GetAttr(kAttrDuplicated));
    bool is_from_parallel_optimizer = instance_name.find("parallel_optimizer") != std::string::npos;
    if (is_allgather && is_fusion && is_recompute && is_from_parallel_optimizer) {
      int64_t fusion_id = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
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
    }
    if (!is_recompute && is_fusion && is_allgather && is_from_parallel_optimizer) {
      int64_t unrecompute_fusion_id = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion);
      unrecompute_max_fusion_id = std::max(unrecompute_fusion_id, unrecompute_max_fusion_id);
      bool would_be_recomputed =
        AnfAlgo::HasNodeAttr(kAttrRecompute, cnode) && AnfAlgo::GetNodeAttr<bool>(cnode, kAttrRecompute);
      if (forward_allgather_recompute_value_in_fusion_group.find(unrecompute_fusion_id) ==
          forward_allgather_recompute_value_in_fusion_group.end()) {
        forward_allgather_recompute_value_in_fusion_group[unrecompute_fusion_id] = would_be_recomputed;
      } else if (forward_allgather_recompute_value_in_fusion_group[unrecompute_fusion_id] != would_be_recomputed) {
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
  int64_t unrecompute_max_fusion_id, int64_t recompute_min_fusion_id) {
  // means that there may some forward allgather and duplicated allgather would be fused.
  if (recompute_min_fusion_id <= unrecompute_max_fusion_id) {
    MS_LOG(WARNING) << "Increase the duplicated allgather fusion id";
    for (auto &adjust_node : parallel_optimizer_recompute_first_fusion_allgathers) {
      int64_t current_fusion_id = AnfAlgo::GetNodeAttr<int64_t>(adjust_node, kAttrFusion);
      int64_t destination_fusion_id = current_fusion_id + unrecompute_max_fusion_id - recompute_min_fusion_id + 2;
      AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(destination_fusion_id), adjust_node);
    }
    for (auto &adjust_node : parallel_optimizer_recompute_allgathers) {
      int64_t current_fusion_id = AnfAlgo::GetNodeAttr<int64_t>(adjust_node, kAttrFusion);
      int64_t destination_fusion_id = current_fusion_id + unrecompute_max_fusion_id - recompute_min_fusion_id + 2;
      AnfAlgo::SetNodeAttr(kAttrFusion, MakeValue(destination_fusion_id), adjust_node);
    }
  }
}

bool AdjustDependForParallelOptimizerRecomputeAllGather::AdjustAllgatherDepend(
  const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &parallel_optimizer_recompute_allgathers) {
  FuncGraphManagerPtr manager = graph->manager();
  bool changed = false;
  for (auto &node : parallel_optimizer_recompute_allgathers) {
    auto cnode = node->cast<CNodePtr>();
    auto depend_node = AnfAlgo::GetInputNode(cnode, 0);
    if (IsPrimitiveCNode(depend_node, prim::kPrimDepend)) {
      auto depend_cnode = depend_node->cast<CNodePtr>();
      AnfNodeIndexSet allgather_node_set = manager->node_users()[cnode];
      for (auto &node_pair : allgather_node_set) {
        auto allgather_next_node = node_pair.first;
        CNodePtr allgather_next_cnode = node_pair.first->cast<CNodePtr>();
        if (allgather_next_cnode == nullptr || !IsValueNode<Primitive>(allgather_next_cnode->input(0))) {
          continue;
        }
        std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          allgather_next_node, AnfAlgo::GetInputNode(depend_cnode, 1)};
        auto new_depend = graph->NewCNode(inputs);
        new_depend->set_abstract(depend_node->abstract());
        manager->SetEdge(node, 1, AnfAlgo::GetInputNode(depend_cnode, 0));
        (void)manager->Replace(allgather_next_node, new_depend);
        changed = true;
      }
    } else if (IsPrimitiveCNode(depend_node, prim::kPrimCast) &&
               IsPrimitiveCNode(AnfAlgo::GetInputNode(depend_node->cast<CNodePtr>(), 0), prim::kPrimDepend)) {
      auto cast_cnode = depend_node->cast<CNodePtr>();
      auto cast_depend_node = AnfAlgo::GetInputNode(cast_cnode, 0);
      auto cast_depend_cnode = cast_depend_node->cast<CNodePtr>();
      AnfNodeIndexSet allgather_node_set = manager->node_users()[cnode];
      for (auto &node_pair : allgather_node_set) {
        auto allgather_next_node = node_pair.first;
        CNodePtr allgather_next_cnode = node_pair.first->cast<CNodePtr>();
        if (allgather_next_cnode == nullptr || !IsValueNode<Primitive>(allgather_next_cnode->input(0))) {
          continue;
        }
        std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          allgather_next_node, AnfAlgo::GetInputNode(cast_depend_cnode, 1)};
        auto new_depend = graph->NewCNode(inputs);
        new_depend->set_abstract(cast_depend_node->abstract());
        manager->SetEdge(depend_node, 1, AnfAlgo::GetInputNode(cast_depend_cnode, 0));
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
