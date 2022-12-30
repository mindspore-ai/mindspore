/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/allreduce_fusion/allreduce_fusion.h"
#include <memory>
#include <queue>
#include <string>
#include <functional>
#include <utility>
#include "utils/hash_set.h"
#include "ir/func_graph.h"
#include "frontend/parallel/costmodel_context.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
void SetMirrorFusion(const CNodePtr &mirror_cnode, int64_t fusion, const std::string &parameter_name) {
  MS_EXCEPTION_IF_NULL(mirror_cnode);
  MS_LOG(DEBUG) << "Set Mirror " << mirror_cnode->DebugString() << " fusion " << fusion;
  auto node_prim = GetValueNode<PrimitivePtr>(mirror_cnode->input(0));
  (void)node_prim->AddAttr(FUSION, MakeValue(std::make_shared<Int64Imm>(fusion)));
  (void)node_prim->AddAttr(PARAMETER, MakeValue(std::make_shared<StringImm>(parameter_name)));
}

void AdjustRelatedFusionNode(const CNodePtr &ret, const std::unordered_map<std::string, CNodePtr> &comm_node_map) {
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  for (const auto &related_node : all_nodes) {
    if (!IsPrimitiveCNode(related_node)) {
      continue;
    }
    auto related_cnode = related_node->cast<CNodePtr>();
    if (!related_cnode->HasAttr(kRelatedCommNodeId)) {
      continue;
    }
    auto related_comm_node_id = GetValue<std::string>(related_cnode->GetAttr(kRelatedCommNodeId));
    if (comm_node_map.find(related_comm_node_id) == comm_node_map.end()) {
      continue;
    }
    auto comm_cnode = comm_node_map.at(related_comm_node_id);
    if (!IsPrimitiveCNode(comm_cnode)) {
      continue;
    }
    auto node_prim = GetValueNode<PrimitivePtr>(comm_cnode->input(0));
    if (!node_prim->HasAttr(FUSION)) {
      continue;
    }
    if (!related_cnode->HasPrimalAttr(kRelatedNodeId) || !related_cnode->HasPrimalAttr(kRelatedFusionKey)) {
      continue;
    }
    auto related_fusion_key = GetValue<std::string>(related_cnode->GetPrimalAttr(kRelatedFusionKey));
    auto fusion_id_pos = related_fusion_key.rfind("_");
    if (fusion_id_pos != std::string::npos) {
      auto sub_str = related_fusion_key.substr(0, fusion_id_pos);
      auto auto_fusion_id = GetValue<int64_t>(node_prim->GetAttr(FUSION));
      auto new_related_fusion_key = sub_str + "_" + std::to_string(auto_fusion_id);
      MS_LOG(INFO) << "replace related fusion key to: " << new_related_fusion_key;
      related_cnode->AddPrimalAttr(kRelatedFusionKey, MakeValue<std::string>(new_related_fusion_key));
    }
  }
}

Status AllCommFusion::SetFusionBySize(const CNodePtr &ret, int64_t threshold, const PrimitivePtr &primp) const {
  auto filter = [primp](const AnfNodePtr &node) { return !IsPrimitiveCNode(node, primp); };
  auto todo = DeepScopedGraphSearchWithFilter(ret, AlwaysInclude, filter);
  auto temp = threshold;
  int64_t fusion = 1;
  bool init = true;
  std::string parameter_name;
  std::string name;
  std::unordered_map<std::string, CNodePtr> comm_node_map;
  for (auto &node : todo) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode->input(1)->Shape() == nullptr) {
      continue;
    }
    auto input_shapes = GetNodeShape(cnode->input(1));
    int64_t input_size = std::accumulate(input_shapes[0].begin(), input_shapes[0].end(), 1, std::multiplies<int64_t>());
    FuncGraphPtr func_graph = cnode->func_graph();
    if (IsPrimitiveEquals(primp, prim::kPrimMirror)) {
      name = ALL_REDUCE;
      std::pair<AnfNodePtr, bool> param_node_pair = FindParameter(cnode->input(1), func_graph);
      if (!param_node_pair.first) {
        continue;
      }
      parameter_name = ParameterName(param_node_pair.first);
    }

    if (IsPrimitiveEquals(primp, prim::kPrimMicroStepAllGather) || IsPrimitiveEquals(primp, prim::kPrimAllGather)) {
      name = ALL_GATHER;
      if (!cnode->input(0) || !cnode->input(1)) {
        continue;
      }
      PrimitivePtr primp1 = GetValueNode<PrimitivePtr>(cnode->input(0));
      if (!primp1->HasAttr(RECOMPUTE) || GetValue<bool>(primp1->GetAttr(RECOMPUTE))) {
        continue;
      }
      std::pair<AnfNodePtr, bool> param_node_pair = FindParameterWithAllgather(cnode->input(1), func_graph, name);
      if (!param_node_pair.first) {
        continue;
      }
      parameter_name = ParameterName(param_node_pair.first);
    }

    if (init || input_size < temp) {
      temp -= input_size;
      init = false;
    } else {
      temp = threshold;
      fusion++;
    }
    SetMirrorFusion(cnode, fusion, parameter_name);
    comm_node_map[cnode->UniqueId()] = cnode;
  }
  AdjustRelatedFusionNode(ret, comm_node_map);
  MS_LOG(INFO) << name << " fusion by size succeed.";
  return SUCCESS;
}

Status AllCommFusion::SetFusionBySizeReduceScatter(const CNodePtr &ret, int64_t threshold,
                                                   const PrimitivePtr &primp) const {
  auto filter = [primp](const AnfNodePtr &node) { return !IsPrimitiveCNode(node, primp); };
  auto todo = DeepScopedGraphSearchWithFilter(ret, AlwaysInclude, filter);
  auto temp = threshold;
  int64_t fusion = 1;
  bool init = true;
  std::unordered_map<std::string, CNodePtr> comm_node_map;
  for (auto &node : todo) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode->input(1) == nullptr) {
      continue;
    }
    FuncGraphPtr func_graph = cnode->func_graph();
    std::pair<AnfNodePtr, bool> param_node_pair =
      FindParameterWithAllgather(cnode->input(1), func_graph, REDUCE_SCATTER);
    if (!param_node_pair.first) {
      continue;
    }
    auto parameter_name = ParameterName(param_node_pair.first);
    auto input_shapes = GetNodeShape(param_node_pair.first);
    int64_t input_size = std::accumulate(input_shapes[0].begin(), input_shapes[0].end(), 1, std::multiplies<int64_t>());
    if (init || input_size < temp) {
      temp -= input_size;
      init = false;
    } else {
      temp = threshold;
      fusion++;
    }
    SetMirrorFusion(cnode, fusion, parameter_name);
    comm_node_map[cnode->UniqueId()] = cnode;
  }
  AdjustRelatedFusionNode(ret, comm_node_map);
  MS_LOG(INFO) << "Reduce_Scatter fusion by size succeed.";
  return SUCCESS;
}

Status AllCommFusion::ProcessCommOpsFusion(const CNodePtr &ret, const std::string &comm_name) {
  if (ret == nullptr) {
    MS_LOG(ERROR) << "ret is nullptr.";
    return FAILED;
  }
  ret_ = ret;
  root_graph_ = ret_->func_graph();
  MS_EXCEPTION_IF_NULL(root_graph_);
  auto graph_set = ForwardGraph(root_graph_);
  if (graph_set.size() > 1) {
    MS_LOG(INFO) << comm_name << " fusion don't support multiple subgraphs now.";
    return SUCCESS;
  }
  auto forward_graph = *(graph_set.begin());
  MS_EXCEPTION_IF_NULL(forward_graph);
  forward_ret_ = forward_graph->get_return();
  MS_EXCEPTION_IF_NULL(forward_ret_);
  if (allreduce_graph_.set_head_cnode(forward_ret_) != SUCCESS) {
    MS_LOG(ERROR) << comm_name << "Graph set_head_cnode failed.";
    return FAILED;
  }
  int64_t threshold = 0;
  if (comm_name == ALL_REDUCE) {
    threshold = ParallelContext::GetInstance()->fusion_threshold_mb();
  } else if (comm_name == ALL_GATHER) {
    threshold = ParallelContext::GetInstance()->allgather_fusion_threshold_mb();
  } else if (comm_name == REDUCE_SCATTER) {
    threshold = ParallelContext::GetInstance()->reducescatter_fusion_threshold_mb();
  } else {
    MS_LOG(ERROR) << " Comm Ops must be ALL_REDUCE, ALL_GATHER or REDUCE_SCATTER, but got " << comm_name;
  }
  threshold *= DEFAULT_THRESHOLD_MB_TO_BYTE;
  if (threshold <= 0) {
    MS_LOG(ERROR) << "The threshold of" << comm_name << "fusion must be larger than 0, but got " << threshold << ".";
    return FAILED;
  }
  if (comm_name == REDUCE_SCATTER) {
    (void)SetFusionBySizeReduceScatter(ret, threshold, prim::kPrimVirtualAssignAdd);
  }
  if (comm_name == ALL_REDUCE) {
    (void)SetFusionBySize(ret, threshold, prim::kPrimMirror);
  }
  if (comm_name == ALL_GATHER) {
    (void)SetFusionBySize(ret, threshold, prim::kPrimMicroStepAllGather);
    (void)SetFusionBySize(ret, threshold, prim::kPrimAllGather);
  }

  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
