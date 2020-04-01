/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "parallel/allreduce_fusion/allreduce_fusion.h"
#include <queue>
#include <unordered_set>
#include <string>
#include <memory>
#include "utils/log_adapter.h"
#include "parallel/status.h"
#include "ir/func_graph.h"
#include "parallel/step_parallel.h"
#include "parallel/graph_util/node_info.h"
#include "parallel/costmodel_context.h"

namespace mindspore {
namespace parallel {
std::unordered_set<CNodePtr> FindCNodesWithPara(const AnfNodePtr& para, uint32_t recursive_times = 0) {
  if (recursive_times > MAX_RECURSIVE_CALL_TIMES) {
    MS_LOG(EXCEPTION) << "FindCNodesWithPara exceeds max recursive call times! Max recursive call times is "
                      << MAX_RECURSIVE_CALL_TIMES;
  }
  MS_EXCEPTION_IF_NULL(para);
  MS_EXCEPTION_IF_NULL(para->func_graph());
  FuncGraphManagerPtr manager = para->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_set = manager->node_users()[para];
  std::unordered_set<CNodePtr> cnode_set;
  for (auto& node_pair : node_set) {
    auto cnode = node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto node_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(node_prim);
    if (node_prim->name() == DEPEND && node_pair.second != 1) {
      continue;
    }
    if (IsParallelCareNode(cnode) && cnode->operator_info() != nullptr) {
      (void)cnode_set.emplace(cnode);
    } else {
      auto cnode_set_sub = FindCNodesWithPara(node_pair.first, recursive_times + 1);
      for (auto& cnode_sub : cnode_set_sub) {
        (void)cnode_set.emplace(cnode_sub);
      }
    }
  }
  return cnode_set;
}

Status AllreduceFusion::AddNodeToGraph() {
  const auto& parameters = root_graph_->parameters();
  for (auto& parameter : parameters) {
    if (!ParameterRequireGrad(parameter)) {
      continue;
    }
    auto cnode_set = FindCNodesWithPara(parameter);
    if (cnode_set.empty()) {
      continue;
    }
    for (auto& cnode : cnode_set) {
      MS_LOG(DEBUG) << "AddNode " << cnode->DebugString();
      if (allreduce_graph_.AddNode(cnode, parameter) != SUCCESS) {
        MS_LOG(ERROR) << "AddNode failed! cnode: " << cnode->DebugString();
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

CNodeCostMap AllreduceFusion::FindCNode(const AnfNodePtr& from, uint32_t recursive_times) const {
  if (recursive_times > MAX_RECURSIVE_CALL_TIMES) {
    MS_LOG(EXCEPTION) << "FindCNode exceeds max recursive call times! Max recursive call times is "
                      << MAX_RECURSIVE_CALL_TIMES;
  }
  MS_EXCEPTION_IF_NULL(from);
  std::unordered_map<CNodePtr, double> cnode_dist;
  if (!from->isa<CNode>()) {
    return cnode_dist;
  }
  auto cnode = from->cast<CNodePtr>();
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return cnode_dist;
  }

  MS_LOG(DEBUG) << "cnode " << cnode->ToString() << " IsParallelCareNode: " << IsParallelCareNode(cnode)
                << " operator_info: " << (cnode->operator_info() != nullptr);

  if (IsParallelCareNode(cnode) && (cnode->operator_info() != nullptr)) {
    auto cost = cnode->operator_info()->GetForwardMemoryCostFromCNode();
    MS_LOG(DEBUG) << "cnode " << cnode->DebugString() << " cost: " << cost;

    if (allreduce_graph_.NodeInGraph(cnode)) {
      cnode_dist[cnode] = cost;
      return cnode_dist;
    } else {
      auto cnode_dist_next = FindNextCNodes(cnode, recursive_times + 1);
      for (auto& ele : cnode_dist_next) {
        cnode_dist[ele.first] = cost + ele.second;
      }
    }
  } else {
    auto cnode_dist_next = FindNextCNodes(cnode);
    for (auto& ele : cnode_dist_next) {
      cnode_dist[ele.first] = ele.second;
    }
  }
  return cnode_dist;
}

CNodeCostMap AllreduceFusion::FindNextCNodes(const CNodePtr& from, uint32_t recursive_times) const {
  if (recursive_times > MAX_RECURSIVE_CALL_TIMES) {
    MS_LOG(EXCEPTION) << "FindNextCNodes exceeds max recursive call times! Max recursive call times is "
                      << MAX_RECURSIVE_CALL_TIMES;
  }
  const auto& from_inputs = from->inputs();
  std::unordered_map<CNodePtr, double> dist_map;
  MS_LOG(DEBUG) << "from cnode " << from->DebugString() << " has " << from_inputs.size() << " inputs";
  for (auto& input_node : from_inputs) {
    auto cnode_dist = FindCNode(input_node, recursive_times + 1);
    for (auto& ele : cnode_dist) {
      (void)dist_map.emplace(ele);
    }
  }
  return dist_map;
}

Status AllreduceFusion::AddEdgeToGraph() {
  std::unordered_map<CNodePtr, int32_t> cnode_state_map;
  const auto& cnodes = allreduce_graph_.cnode_set();
  for (auto& cnode : cnodes) {
    cnode_state_map[cnode] = 0;
  }
  const auto& head_cnode = allreduce_graph_.head_cnode();
  std::queue<CNodePtr> cnode_queue;
  cnode_queue.emplace(head_cnode);
  cnode_state_map[head_cnode] = 1;

  while (!cnode_queue.empty()) {
    const auto cur_cnode = cnode_queue.front();
    cnode_queue.pop();
    cnode_state_map[cur_cnode] = 2;
    auto next = FindNextCNodes(cur_cnode);
    for (auto& ele : next) {
      auto& cnode = ele.first;
      auto& dist = ele.second;
      if (cnode_state_map[cnode] == 0) {
        cnode_queue.emplace(cnode);
        cnode_state_map[cnode] = 1;
      }
      if (allreduce_graph_.AddEdge(cur_cnode, cnode, dist) != SUCCESS) {
        MS_LOG(ERROR) << "AddEdge error";
        return FAILED;
      }
      MS_LOG(DEBUG) << "from " << cur_cnode->DebugString() << ", to " << cnode->DebugString() << " dist " << dist;
    }
  }
  return SUCCESS;
}

std::vector<CNodePtr> FindMirror(const AnfNodePtr& para, uint32_t recursive_times = 0) {
  if (recursive_times > MAX_RECURSIVE_CALL_TIMES) {
    MS_LOG(EXCEPTION) << "FindMirror exceeds max recursive call times! Max recursive call times is "
                      << MAX_RECURSIVE_CALL_TIMES;
  }
  MS_EXCEPTION_IF_NULL(para);
  MS_EXCEPTION_IF_NULL(para->func_graph());
  FuncGraphManagerPtr manager = para->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[para];
  std::vector<CNodePtr> cnode_list;
  for (auto& node_pair : node_set) {
    auto cnode = node_pair.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    auto node_prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(node_prim);
    if (node_prim->name() == CAST) {
      auto mirror_cnodes = FindMirror(node_pair.first, recursive_times + 1);
      if (mirror_cnodes.empty()) {
        MS_LOG(WARNING) << "mirror node after cast not found";
        continue;
      }
      if (mirror_cnodes.size() > 1) {
        MS_LOG(EXCEPTION) << "mirror node after cast number is not 1";
      }
      cnode_list.emplace_back(mirror_cnodes[0]);
    }
    if (node_prim->name() == MIRROR_OPERATOR) {
      cnode_list.emplace_back(cnode);
    }
  }
  return cnode_list;
}

void SetMirrorFusion(const CNodePtr& mirror_cnode, int32_t fusion, const std::string& parameter_name) {
  MS_EXCEPTION_IF_NULL(mirror_cnode);
  MS_LOG(DEBUG) << "Set Mirror " << mirror_cnode->DebugString() << " fusion " << fusion;
  auto node_prim = GetValueNode<PrimitivePtr>(mirror_cnode->input(0));
  auto old_value_ptr = node_prim->GetAttr(FUSION);
  if (old_value_ptr != nullptr) {
    if (old_value_ptr->isa<Int32Imm>()) {
      int32_t old_value = old_value_ptr->cast<Int32ImmPtr>()->value();
      if (old_value < fusion) {
        return;
      }
    }
  }
  (void)node_prim->AddAttr(FUSION, MakeValue(std::make_shared<Int32Imm>(fusion)));
  (void)node_prim->AddAttr(PARAMETER, MakeValue(std::make_shared<StringImm>(parameter_name)));
}

Status FindMirrorAndSetFusion(const AnfNodePtr& para, int32_t fusion) {
  auto mirror_cnodes = FindMirror(para);
  if (mirror_cnodes.empty()) {
    MS_LOG(WARNING) << para->ToString() << " 0 Mirror CNode found.";
    return SUCCESS;
  }
  if (mirror_cnodes.size() > 2) {
    for (auto& mirror_cnode : mirror_cnodes) {
      MS_EXCEPTION_IF_NULL(mirror_cnode);
      MS_LOG(INFO) << mirror_cnode->DebugString();
    }
    MS_EXCEPTION_IF_NULL(para);
    MS_LOG(ERROR) << para->ToString() << " FindMirror is more than 2. " << mirror_cnodes.size()
                  << "Mirror CNode found.";
    return FAILED;
  }
  for (auto& mirror_cnode : mirror_cnodes) {
    auto parameter_name = ParameterName(para);
    SetMirrorFusion(mirror_cnode, fusion, parameter_name);
  }
  return SUCCESS;
}

Status FindMirrorAndSetFusion(const std::vector<AnfNodePtr>& paras, int32_t fusion) {
  for (auto& param_node : paras) {
    if (FindMirrorAndSetFusion(param_node, fusion) != SUCCESS) {
      MS_LOG(ERROR) << "FindMirrorAndSetFusion failed";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status AllreduceFusion::SetFusion(const std::vector<double>& cost_map) {
  if (cost_map.size() < 2) {
    MS_LOG(ERROR) << "cost_map must has at least 2 items, cost_map size is " << cost_map.size();
    return FAILED;
  }
  int32_t fusion = 1;
  for (auto cost_iter = cost_map.end() - 1; cost_iter != cost_map.begin(); --cost_iter) {
    auto paras = allreduce_graph_.GetParaByCost(*(cost_iter - 1), *cost_iter);
    if (FindMirrorAndSetFusion(paras, fusion) != SUCCESS) {
      MS_LOG(ERROR) << "FindMirrorAndSetFusion failed";
      return FAILED;
    }
    fusion++;
  }
  return SUCCESS;
}

std::vector<double> AllreduceFusion::GenerateCostMap(int32_t fusion_times, double tail_percent) const {
  double offset = allreduce_graph_.max() * (1 - tail_percent) / (fusion_times - 1);
  MS_LOG(DEBUG) << "max = " << allreduce_graph_.max() << ", offset = " << offset;
  std::vector<double> cost_map;
  double begin = 0;
  for (auto i = 0; i < fusion_times - 1; i++) {
    cost_map.push_back(begin);
    begin += offset;
  }
  cost_map.push_back(allreduce_graph_.max() * (1 - tail_percent));
  cost_map.push_back(allreduce_graph_.max());
  MS_LOG(DEBUG) << "cost_map = " << cost_map;
  return cost_map;
}

Status AllreduceFusion::SetFusionByBackwardCompTime() {
  auto fusion_times = CostModelContext::GetInstance()->costmodel_allreduce_fusion_times();
  if (fusion_times < 2) {
    MS_LOG(INFO) << "'costmodel_allreduce_fusion_times' is " << fusion_times << ". Bypass ProcessAllreduceFusion";
    return SUCCESS;
  }
  auto tail_percent = CostModelContext::GetInstance()->costmodel_allreduce_fusion_tail_percent();
  if (tail_percent < 0 || tail_percent >= 1) {
    MS_LOG(INFO) << "'costmodel_allreduce_fusion_tail_percent' is " << tail_percent
                 << ". Bypass ProcessAllreduceFusion";
    return SUCCESS;
  }
  const auto cost_map = GenerateCostMap(fusion_times, tail_percent);
  MS_LOG(DEBUG) << "AllreduceGraph GenerateCostMap succeed.";
  if (SetFusion(cost_map) != SUCCESS) {
    MS_LOG(ERROR) << "SetFusion failed.";
    return FAILED;
  }
  MS_LOG(DEBUG) << "AllreduceGraph SetFusion succeed.";
  return SUCCESS;
}

Status AllreduceFusion::GetSetFusionByBackwardCompAndAllreduceTimeParams() {
  tail_time_ = CostModelContext::GetInstance()->costmodel_allreduce_fusion_tail_time();
  if (tail_time_ <= 0) {
    MS_LOG(INFO) << "'costmodel_allreduce_tail_time' is " << tail_time_ << ". Bypass ProcessAllreduceFusion";
    return FAILED;
  }
  allreduce_inherent_time_ = CostModelContext::GetInstance()->costmodel_allreduce_fusion_allreduce_inherent_time();
  if (allreduce_inherent_time_ <= 0) {
    MS_LOG(INFO) << "'costmodel_allreduce_fusion_allreduce_inherent_time' is " << allreduce_inherent_time_
                 << ". Bypass ProcessAllreduceFusion";
    return FAILED;
  }
  if (tail_time_ <= allreduce_inherent_time_) {
    MS_LOG(INFO) << "'costmodel_allreduce_tail_time' is " << tail_time_
                 << "'costmodel_allreduce_fusion_allreduce_inherent_time' is " << allreduce_inherent_time_
                 << ".tail_time is not more than allreduce_inherent_time. Bypass ProcessAllreduceFusion";
    return FAILED;
  }
  allreduce_bandwidth_ = CostModelContext::GetInstance()->costmodel_allreduce_fusion_allreduce_bandwidth();
  if (allreduce_bandwidth_ <= 0) {
    MS_LOG(INFO) << "'costmodel_allreduce_fusion_allreduce_bandwidth' is " << allreduce_bandwidth_
                 << ". Bypass ProcessAllreduceFusion";
    return FAILED;
  }
  computation_time_parameter_ =
    CostModelContext::GetInstance()->costmodel_allreduce_fusion_computation_time_parameter();
  if (computation_time_parameter_ <= 0) {
    MS_LOG(INFO) << "'costmodel_allreduce_fusion_computation_time_parameter' is " << computation_time_parameter_
                 << ". Bypass ProcessAllreduceFusion";
    return FAILED;
  }
  return SUCCESS;
}

Status AllreduceFusion::SetFusionByBackwardCompAndAllreduceTime() {
  if (GetSetFusionByBackwardCompAndAllreduceTimeParams() != SUCCESS) {
    MS_LOG(ERROR) << "GetSetFusionByBackwardCompAndAllreduceTimeParams failed!";
    return FAILED;
  }
  allreduce_graph_.SortArnode();
  if (allreduce_graph_.RemoveExtraParas() != SUCCESS) {
    MS_LOG(ERROR) << "RemoveExtraParas failed!";
    return FAILED;
  }
  double para_size = (tail_time_ - allreduce_inherent_time_) / allreduce_bandwidth_;
  double to_cost = allreduce_graph_.max();
  int32_t fusion = 1;
  while (to_cost != 0) {
    MS_LOG(INFO) << "to_cost: " << to_cost << " para_size: " << para_size;
    auto node_cost_pair = allreduce_graph_.GetParaByParaSize(to_cost, para_size);
    MS_LOG(INFO) << "para size: " << node_cost_pair.first.size() << " from_cost: " << node_cost_pair.second;
    auto paras = node_cost_pair.first;
    if (FindMirrorAndSetFusion(paras, fusion) != SUCCESS) {
      MS_LOG(ERROR) << "FindMirrorAndSetFusion failed";
      return FAILED;
    }
    fusion++;
    para_size = ((to_cost - node_cost_pair.second) * computation_time_parameter_ - allreduce_inherent_time_) /
                allreduce_bandwidth_;
    to_cost = node_cost_pair.second;
  }
  MS_LOG(DEBUG) << "AllreduceGraph SetFusionByBackwardCompAndAllreduceTime succeed.";
  return SUCCESS;
}

Status AllreduceFusion::SetFusionByAlgorithm(int32_t algorithm) {
  if (algorithm == 1) {
    return SetFusionByBackwardCompTime();
  }
  return SetFusionByBackwardCompAndAllreduceTime();
}

Status AllreduceFusion::ProcessAllreduceFusion(const CNodePtr& ret) {
  if (ret == nullptr) {
    MS_LOG(ERROR) << "ret is nullptr.";
    return FAILED;
  }
  auto algorithm = CostModelContext::GetInstance()->costmodel_allreduce_fusion_algorithm();
  if (algorithm < 1 || algorithm > 2) {
    MS_LOG(INFO) << "'costmodel_allreduce_fusion_algorithm' is " << algorithm << ". Bypass ProcessAllreduceFusion";
    return SUCCESS;
  }
  ret_ = ret;
  root_graph_ = ret_->func_graph();
  MS_EXCEPTION_IF_NULL(root_graph_);
  auto forward_graph = ForwardGraph(root_graph_);
  MS_EXCEPTION_IF_NULL(forward_graph);
  forward_ret_ = forward_graph->get_return();
  MS_EXCEPTION_IF_NULL(forward_ret_);

  if (allreduce_graph_.set_head_cnode(forward_ret_) != SUCCESS) {
    MS_LOG(ERROR) << "AllreduceGraph set_head_cnode failed.";
    return FAILED;
  }
  MS_LOG(DEBUG) << "AllreduceGraph set_head_cnode succeed.";
  if (AddNodeToGraph() != SUCCESS) {
    MS_LOG(ERROR) << "AddNodeToGraph failed.";
    return FAILED;
  }
  MS_LOG(DEBUG) << "AllreduceGraph AddNodeToGraph succeed.";
  if (AddEdgeToGraph() != SUCCESS) {
    MS_LOG(ERROR) << "AddNodeToGraph failed.";
    return FAILED;
  }
  MS_LOG(DEBUG) << "AllreduceGraph AddEdgeToGraph succeed.";
  if (SetFusionByAlgorithm(algorithm) != SUCCESS) {
    MS_LOG(ERROR) << "SetFusionByAlgorithm failed.";
    return FAILED;
  }
  MS_LOG(DEBUG) << "AllreduceGraph SetFusionByAlgorithm succeed.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
