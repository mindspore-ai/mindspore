/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <set>

#include "frontend/parallel/ops_info/reshape_info.h"
#include "frontend/parallel/step_auto_parallel.h"
#include "include/frontend/parallel/graph_costmodel.h"

namespace mindspore {
namespace parallel {
namespace {
CostCacheKey TensorLayoutsToCacheKey(const TensorLayout &left, const TensorLayout &right) {
  return std::make_pair(std::make_pair(left.device_arrangement(), left.tensor_map()),
                        std::make_pair(right.device_arrangement(), right.tensor_map()));
}
}  // namespace

void CostGraph::CreateStarEliminationSubCostList(const StrategyPtr &first_succ_node_stra,
                                                 const CostPtrList &first_succ_node_clist,
                                                 const CostPtrList &first_succ_edge_clist,
                                                 const StrategyPtr &merged_op_stra, const CostPtrList &merged_op_clist,
                                                 std::vector<StrategyPtr> succ_nodes_stras,
                                                 CostPtrList &succ_edges_costs, CostPtrList &succ_nodes_costs,
                                                 CostPtrList *first_succ_node_clist_new) const {
  for (auto &first_succ_node_cost : first_succ_node_clist) {
    for (auto &first_succ_edge_cost : first_succ_edge_clist) {
      for (auto &merged_node_cost : merged_op_clist) {
        MS_EXCEPTION_IF_NULL(merged_node_cost);
        succ_nodes_stras[0] = first_succ_node_stra;
        succ_edges_costs[0] = first_succ_edge_cost;
        succ_nodes_costs[0] = first_succ_node_cost;

        double computation_cost = merged_node_cost->computation_cost_,
               memory_cost = merged_node_cost->memory_with_reuse_, commu_cost = merged_node_cost->communication_cost_,
               commu_without = merged_node_cost->communication_without_parameter_,
               commu_forward = merged_node_cost->communication_forward_;
        const auto triangle_star_stra_overwrite = CostModelContext::GetInstance()->triangle_star_strategy_overwrite();
        const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
        for (size_t i = 0; i < succ_nodes_stras.size(); ++i) {
          MS_EXCEPTION_IF_NULL(succ_edges_costs[i]);
          if (i == 0) {
            computation_cost += succ_edges_costs[i]->computation_cost_ + succ_nodes_costs[i]->computation_cost_;
            memory_cost += succ_edges_costs[i]->memory_with_reuse_ + succ_nodes_costs[i]->memory_with_reuse_;
            commu_cost += succ_edges_costs[i]->communication_cost_ + succ_nodes_costs[i]->communication_cost_;
            commu_forward += succ_edges_costs[i]->communication_forward_ + succ_nodes_costs[i]->communication_forward_;
            commu_without += succ_edges_costs[i]->communication_without_parameter_ +
                             succ_nodes_costs[i]->communication_without_parameter_;
          } else {
            computation_cost += succ_edges_costs[i]->computation_cost_;
            memory_cost += succ_edges_costs[i]->memory_with_reuse_;
            commu_cost += succ_edges_costs[i]->communication_cost_;
            commu_forward += succ_edges_costs[i]->communication_forward_;
            commu_without += succ_edges_costs[i]->communication_without_parameter_;
            if (triangle_star_stra_overwrite) {
              computation_cost += succ_nodes_costs[i]->computation_cost_;
              memory_cost += succ_nodes_costs[i]->memory_with_reuse_;
              commu_cost += succ_nodes_costs[i]->communication_cost_;
              commu_forward += succ_nodes_costs[i]->communication_forward_;
              commu_without += succ_nodes_costs[i]->communication_without_parameter_;
            }
          }
        }

        auto decision = std::make_shared<StarEliminationDecision>(merged_op_stra, merged_node_cost, succ_edges_costs,
                                                                  succ_nodes_stras, succ_nodes_costs);
        auto new_cost = std::make_shared<Cost>(computation_cost, commu_cost, decision);
        new_cost->communication_without_parameter_ = commu_without;
        new_cost->communication_with_partial_para_ = commu_without + gamma * (commu_cost - commu_without);
        new_cost->memory_with_reuse_ = memory_cost;
        new_cost->communication_forward_ = commu_forward;
        first_succ_node_clist_new->push_back(std::move(new_cost));
      }
    }
  }
}

void CostGraph::CreateStarEliminationCostList(std::vector<std::shared_ptr<Edge>> &succ_edges,
                                              const StrategyPtr &first_succ_node_stra,
                                              const CostPtrList &first_succ_node_clist,
                                              const CostPtrList &first_succ_edge_clist,
                                              const StrategyPtr &merged_op_stra, const CostPtrList &merged_op_clist,
                                              CostPtrList *first_succ_node_clist_new) const {
  std::vector<StrategyPtr> succ_nodes_stras(succ_edges.size(), nullptr);
  CostPtrList succ_edges_costs(succ_edges.size(), nullptr);
  CostPtrList succ_nodes_costs(succ_edges.size(), nullptr);
  std::function<void(size_t)> recursive = [&first_succ_node_stra, &first_succ_node_clist, &first_succ_edge_clist,
                                           &merged_op_stra, &merged_op_clist, &succ_nodes_stras, &succ_edges_costs,
                                           &succ_nodes_costs, &first_succ_node_clist_new, &succ_edges, &recursive,
                                           this](size_t k) {
    if (k == succ_edges.size()) {
      CreateStarEliminationSubCostList(first_succ_node_stra, first_succ_node_clist, first_succ_edge_clist,
                                       merged_op_stra, merged_op_clist, succ_nodes_stras, succ_edges_costs,
                                       succ_nodes_costs, first_succ_node_clist_new);
      return;
    }
    MS_LOG(DEBUG) << "The size of first_succ_node_clist: " << first_succ_node_clist.size()
                  << ", first_succ_edge_clist: " << first_succ_edge_clist.size()
                  << ", merged_op_clist: " << merged_op_clist.size()
                  << ", first_succ_node_clist_new: " << first_succ_node_clist_new->size() << ".";
    auto succ_edge = succ_edges[k];
    MS_EXCEPTION_IF_NULL(succ_edge);
    auto succ_node = succ_edge->next_operator();
    MS_EXCEPTION_IF_NULL(succ_node);
    for (auto &succ_node_stra_cost : succ_node->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(succ_node_stra_cost);
      auto succ_node_stra = succ_node_stra_cost->strategy_ptr;
      auto succ_node_clist = succ_node_stra_cost->cost_list;
      auto succ_edge_clist = succ_edge->GetCostList(merged_op_stra, succ_node_stra);

      for (auto &succ_node_cost : succ_node_clist) {
        MS_EXCEPTION_IF_NULL(succ_node_cost);
        for (auto &succ_edge_cost : succ_edge_clist) {
          MS_EXCEPTION_IF_NULL(succ_edge_cost);
          succ_nodes_stras[k] = succ_node_stra;
          succ_edges_costs[k] = succ_edge_cost;
          succ_nodes_costs[k] = succ_node_cost;
          recursive(k + 1);
        }
      }
    }
  };

  recursive(1);
}

std::vector<std::shared_ptr<Edge>> CostGraph::EliminationStar(const OperatorInfoPtr &merged_op) const {
  MS_EXCEPTION_IF_NULL(merged_op);
  auto succ_edges = merged_op->GetAliveSuccEdges();
  MS_LOG(INFO) << "Now eliminating star centered at: " << merged_op->name() << ".";
  for (auto &succ_edge : succ_edges) {
    MS_EXCEPTION_IF_NULL(succ_edge->next_operator());
    MS_LOG(INFO) << "The successive operator is: " << succ_edge->next_operator()->name() << ".";
  }

  MS_EXCEPTION_IF_NULL(succ_edges[0]);
  auto first_succ_node = succ_edges[0]->next_operator();
  auto first_succ_edge = succ_edges[0];
  bool valid = false;

  // 'merged_op' is merged into first_node
  MS_EXCEPTION_IF_NULL(first_succ_node);
  for (auto &first_succ_node_stra_cost : first_succ_node->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(first_succ_node_stra_cost);
    auto first_succ_node_stra = first_succ_node_stra_cost->strategy_ptr;
    auto first_succ_node_clist = first_succ_node_stra_cost->cost_list;
    CostPtrList first_succ_node_clist_new;

    for (auto &merged_op_stra_cost : merged_op->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(merged_op_stra_cost);
      auto merged_op_stra = merged_op_stra_cost->strategy_ptr;
      auto merged_op_clist = merged_op_stra_cost->cost_list;
      auto first_succ_edge_clist = first_succ_edge->GetCostList(merged_op_stra, first_succ_node_stra);

      CreateStarEliminationCostList(succ_edges, first_succ_node_stra, first_succ_node_clist, first_succ_edge_clist,
                                    merged_op_stra, merged_op_clist, &first_succ_node_clist_new);
    }
    Simplify(&first_succ_node_clist_new);
    // Set the new costlist w.r.t the strategy
    first_succ_node_stra_cost->cost_list = first_succ_node_clist_new;
    if ((!valid) && (!first_succ_node_clist_new.empty())) {
      valid = true;
    }
  }

  if (!valid) {
    MS_LOG(EXCEPTION) << "Eliminating star centered at: " << merged_op->name()
                      << " failed. It may be caused by "
                         "configuring inconsistent strategies for operators.";
  }

  merged_op->SetNotAlive();
  MS_LOG(INFO) << "Eliminating star centered at: " << merged_op->name() << " succeeded.";
  return succ_edges;
}

size_t CostGraph::GetNumEdges() const {
  size_t sum = 0;
  for (const auto &kv : edges_) {
    auto &edges = kv.second;
    sum += edges.size();
  }
  return sum;
}

Status CostGraph::InitReshapeStrategy() {
  // reshape init should be apply after the init of it's previous node and next node.
  for (size_t i = 0; i < ops_.size(); ++i) {
    if (ops_[i]->IsReshape()) {
      auto reshape_info = std::dynamic_pointer_cast<ReshapeInfo>(ops_[i]);
      auto in_edges = GetOriginalPrevEdges(ops_[i]);
      auto out_edges = GetOriginalNextEdges(ops_[i]);
      // deal with consecutive reshape op
      auto pre_reshape_iter = std::find_if(in_edges.begin(), in_edges.end(), [&](const std::shared_ptr<Edge> &edge) {
        return edge->prev_operator()->IsReshape();
      });
      auto next_reshape_iter = std::find_if(out_edges.begin(), out_edges.end(), [&](const std::shared_ptr<Edge> &edge) {
        return edge->next_operator()->IsReshape();
      });
      if (pre_reshape_iter != in_edges.end() || next_reshape_iter != out_edges.end()) {
        reshape_info->set_strategy(nullptr);
        continue;
      }
      auto pre_iter = std::find_if(in_edges.begin(), in_edges.end(), [&](const std::shared_ptr<Edge> &edge) {
        return edge->prev_operator()->name() == reshape_info->pre_operator_name();
      });
      auto next_iter = std::find_if(out_edges.begin(), out_edges.end(), [&](const std::shared_ptr<Edge> &edge) {
        return edge->next_operator()->name() == reshape_info->next_operator_name();
      });
      bool reshape_is_first_op = reshape_info->pre_operator_name() == reshape_info->name();
      if (reshape_is_first_op) {
        (void)reshape_info->InitSelectedStrategy(reshape_info->selected_strategy(), nullptr);
      }
      if (pre_iter != in_edges.end() || reshape_is_first_op) {
        MS_LOG(DEBUG) << "Set reshape input layout by " << reshape_info->pre_operator_name();
        int64_t pre_index = reshape_info->pre_operator_index();
        TensorInfo pre_info;
        std::shared_ptr<OperatorInfo> pre_op_info;
        if (reshape_is_first_op) {
          pre_op_info = reshape_info;
          pre_info = pre_op_info->inputs_tensor_info()[LongToSize(pre_index)];
        } else {
          pre_op_info = (*pre_iter)->prev_operator();
          pre_info = pre_op_info->outputs_tensor_info()[LongToSize(pre_index)];
        }
        reshape_info->SetInputLayout(pre_info.tensor_layout());
        if (pre_iter != in_edges.end()) {
          Dimensions stra = pre_info.InferStrategy();
          if (stra.empty()) {
            MS_LOG(EXCEPTION) << "Infer strategy by tensor_info failed";
          }
          Strategies stra_inputs = {stra};
          StrategyPtr reshape_stra =
            std::make_shared<Strategy>((*pre_iter)->prev_operator()->strategy()->GetInputStage(), stra_inputs);
          reshape_info->set_strategy(reshape_stra);
        }
      }
      if (next_iter != out_edges.end()) {
        MS_LOG(DEBUG) << "Set reshape output layout by " << reshape_info->next_operator_name();
        int64_t next_index = reshape_info->next_operator_index();
        reshape_info->SetOutputLayout(
          (*next_iter)->next_operator()->inputs_tensor_info()[LongToSize(next_index)].tensor_layout());
      }
      if (reshape_info->Init(nullptr, nullptr) != SUCCESS) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CostGraph::InitSelectedStrategy() {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    if (op->IsReshape()) {
      continue;
    }
    if (op->is_config_by_layout()) {
      op->set_auto_parallel(false);
      continue;
    }
    StrategyPtr out_strategy = nullptr;
    if (op->out_strategy() != nullptr) {
      out_strategy = op->out_strategy();
    }
    auto result_op = op->InitSelectedStrategy(op->selected_strategy(), out_strategy);
    if (result_op != SUCCESS) {
      return result_op;
    }
  }

  return SUCCESS;
}

Status CostGraph::ComputeOpsAndEdgesParameterInvolved() {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    const auto &output_parameter = op->ComputeOpAndPrevEdgeParameterInvolved();
    if ((output_parameter != 0) && (output_parameter != 1)) {
      MS_LOG(ERROR) << "Computing parameter_involved for " << op->name() << " failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}

void CostGraph::DFSForTopoOrder(const OperatorInfoPtr &current_op, std::vector<OperatorInfoPtr> *topo_order) {
  MS_EXCEPTION_IF_NULL(current_op);
  MS_EXCEPTION_IF_NULL(topo_order);

  visited.at(current_op) = true;
  for (const auto &s_edge : current_op->succ_edges()) {
    if (!visited.at(s_edge->next_operator())) {
      DFSForTopoOrder(s_edge->next_operator(), topo_order);
    }
  }
  topo_order->push_back(current_op);
}

// Compute a topological order of the costgraph
void CostGraph::TopologyOrder(std::vector<OperatorInfoPtr> *topo_order) {
  for (auto &op : ops_) {
    visited[op] = false;
  }

  for (auto &op : ops_) {
    if (!visited[op]) {
      DFSForTopoOrder(op, topo_order);
    }
  }
}
void CostGraph::MarkCriticalOpsAndEdges(const std::map<OperatorInfoPtr, int64_t> &candidate_ops) {
  for (auto &op : ops_) {
    auto search = candidate_ops.find(op);
    if (search != candidate_ops.end()) {
      // Mark the critical operators
      op->mark_output_critical();
      // Mark the successive edges
      for (auto &s_edge : op->succ_edges()) {
        s_edge->mark_output_critical();
      }
    } else {
      op->mark_output_not_critical();
    }
  }
}

Status CostGraph::DetermineCriticalOps(const std::vector<OperatorInfoPtr> &topo_order) {
  if (topo_order.size() == 0) {
    MS_LOG(ERROR) << "0 operator in costgraph.";
    return FAILED;
  }
  auto &first_op = topo_order[0];
  if (first_op->prev_edges().size() > 0) {
    MS_LOG(ERROR) << "The first operator in the first of topological order of "
                     "costgraph should have 0 incoming edge, but has "
                  << first_op->prev_edges() << "edges.";
    return FAILED;
  }
  // The 'curr_memory_state' records <OperatorInfo, remaining_output_cnt>, where remaining_output_cnt is the number
  // of the output of OperatorInfo that currently has not been used
  std::map<OperatorInfoPtr, int64_t> curr_memory_state;
  (void)curr_memory_state.emplace(std::make_pair(first_op, SizeToLong(first_op->succ_edges().size())));
  std::map<OperatorInfoPtr, int64_t> max_memory_state = curr_memory_state;
  // The 'curr_memory_size' records the current total memory size, which is the sum of outputs of operators that has
  // not been used
  double curr_memory_size = first_op->GetOutputsTotalSize();
  double max_memory_size = curr_memory_size;

  for (size_t finished = 1; finished < topo_order.size(); ++finished) {
    // Produce
    (void)curr_memory_state.emplace(
      std::make_pair(topo_order[finished], SizeToLong(topo_order[finished]->succ_edges().size())));
    curr_memory_size += topo_order[finished]->GetOutputsTotalSize();
    // Consume
    for (const auto &prev_edge : topo_order[finished]->prev_edges()) {
      const auto &prev_op = prev_edge->prev_operator();
      curr_memory_state[prev_op]--;
    }
    for (const auto &prev_edge : topo_order[finished]->prev_edges()) {
      const auto &prev_op = prev_edge->prev_operator();
      if (curr_memory_state[prev_op] < 0) {
        MS_LOG(ERROR) << "Failure: " << prev_op->name() << "'s current output count: " << curr_memory_state[prev_op];
        return FAILED;
      } else if (curr_memory_state[prev_op] == 0) {
        (void)curr_memory_state.erase(prev_op);
        curr_memory_size -= prev_op->GetOutputsTotalSize();
      }
    }

    if (curr_memory_size < 0) {
      MS_LOG(ERROR) << "Memory size calculation failed: " << curr_memory_size;
    }
    // Modify the max
    if (curr_memory_size > max_memory_size) {
      max_memory_size = curr_memory_size;
      max_memory_state = curr_memory_state;
    }
  }
  // Mark those critical operators
  MarkCriticalOpsAndEdges(max_memory_state);
  return SUCCESS;
}

Status CostGraph::ComputeOpsAndEdgesOutputCritical() {
  // Two steps to do:
  // 1. Compute a topological order of the costgraph
  // 2. Determine and mark the operators (and necessary edges) that are critical
  std::vector<OperatorInfoPtr> topo_order;
  TopologyOrder(&topo_order);
  std::reverse(std::begin(topo_order), std::end(topo_order));

  if (DetermineCriticalOps(topo_order) != SUCCESS) {
    MS_LOG(ERROR) << "Determining critical operators failed.";
    return FAILED;
  }
  return SUCCESS;
}

Status CostGraph::CalculateOpsMemoryCost() {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    if (op->CalculateMemoryCost() != SUCCESS) {
      MS_LOG(ERROR) << "Calculate Operator: " << op->name() << " cost for memory usage failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status CostGraph::CalculateOpsMemoryCostForInference() {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    if (op->CalculateMemoryCostForInference() != SUCCESS) {
      MS_LOG(ERROR) << "Calculate Operator: " << op->name() << " cost for memory usage failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status CostGraph::CalculateEdgesMemoryCost() {
  for (const auto &edge_pair : edges_) {
    const auto &edges = edge_pair.second;
    for (auto &one_edge : edges) {
      if (one_edge->CalculateMemoryCost() != SUCCESS) {
        MS_LOG(ERROR) << "Calculate Edge: " << one_edge->edge_name() << " cost for memory usage failed.";
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status CostGraph::CalculateEdgesMemoryCostForInference() {
  for (const auto &edge_pair : edges_) {
    const auto &edges = edge_pair.second;
    for (auto &one_edge : edges) {
      if (one_edge->CalculateMemoryCostForInference() != SUCCESS) {
        MS_LOG(ERROR) << "Calculate Edge: " << one_edge->edge_name() << " cost for memory usage failed.";
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

OperatorInfoPtr CostGraph::FindTmpIdentityByParameterName(const std::string &p_name) const {
  for (auto one_op : ops_) {
    if (one_op->name().find(IDENTITY_INFO) != std::string::npos) {
      if (one_op->refkey_parameter_name() == p_name) {
        return one_op;
      }
    }
  }
  return nullptr;
}
Status CostGraph::CorrectOpsMemoryCost() {
  for (auto &one_op : ops_) {
    if ((one_op->name().find(IDENTITY_INFO) != std::string::npos) && (one_op->is_output_parameter_involve() == 1)) {
      if (one_op->GetAliveSuccEdges().size() > 1) {
        // Filter out the case when the TmpIdentity being used by multiple operators
        std::map<size_t, int64_t> output_count;
        for (size_t i = 0; i < one_op->GetAliveSuccEdges().size(); ++i) {
          auto output_index = one_op->GetAliveSuccEdges()[i]->prev_op_output_index();
          output_count[output_index]++;
        }
        for (size_t i = 0; i < one_op->GetAliveSuccEdges().size(); ++i) {
          auto output_index = one_op->GetAliveSuccEdges()[i]->prev_op_output_index();
          if (output_count[output_index] <= 1) {
            continue;
          }
          auto next_op = one_op->GetAliveSuccEdges()[i]->next_operator();
          MS_EXCEPTION_IF_NULL(next_op);
          auto input_index = one_op->GetAliveSuccEdges()[i]->next_op_input_index();
          if (next_op->CorrectMemoryCost(input_index) != SUCCESS) {
            MS_LOG(ERROR) << "The operator name: " << one_op->name() << ", the next operator name: " << next_op->name()
                          << ", the output_index: " << output_index << ", the input_index: " << input_index << ".";
            return FAILED;
          }
          output_count[output_index]--;
        }
      }
    }
  }
  return SUCCESS;
}

Status CostGraph::CalculateMemoryCost() {
  const auto phase = CostModelContext::GetInstance()->run_phase();
  if (phase == TRAINING_PHASE) {
    // training phase
    if (ComputeOpsAndEdgesParameterInvolved() == SUCCESS) {
      // Calculate operators' memory usage
      if (CalculateOpsMemoryCost() != SUCCESS) {
        MS_LOG(ERROR) << "Calculating operators' cost for memory cost failed.";
        return FAILED;
      }
      // Calculate edges' memory usage
      if (CalculateEdgesMemoryCost() != SUCCESS) {
        MS_LOG(ERROR) << "Calculating edges' cost for memory cost failed.";
        return FAILED;
      }
      // Correct memory usage caused by TmpIdentity
      if (CorrectOpsMemoryCost() != SUCCESS) {
        MS_LOG(ERROR) << "Correcting operators' cost for memory cost failed.";
        return FAILED;
      }
    } else {
      MS_LOG(ERROR) << "Computing operators' parameter_involved failed.";
      return FAILED;
    }
  } else {
    // inference phase
    if (ComputeOpsAndEdgesOutputCritical() == SUCCESS) {
      // Calculate operators' memory usage
      if (CalculateOpsMemoryCostForInference() != SUCCESS) {
        MS_LOG(ERROR) << "Calculating operators' memory cost for inference failed.";
        return FAILED;
      }
      // Calculate edges's memory usage
      if (CalculateEdgesMemoryCostForInference() != SUCCESS) {
        MS_LOG(ERROR) << "Calculating operators' memory cost for inference failed.";
        return FAILED;
      }
    } else {
      MS_LOG(ERROR) << "Computing operators' critical flag failed.";
      return FAILED;
    }
  }
  return SUCCESS;
}

void CostGraph::CheckApproximateCostGraphEdges() {
  auto approximation = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (!approximation) {
    return;
  }
  for (const auto &s_edge : edges_) {
    auto &edges_vector = s_edge.second;
    for (auto &edge_ptr : edges_vector) {
      MS_EXCEPTION_IF_NULL(edge_ptr);
      if (edge_ptr->CheckStrategyCostPossibility()) {
        continue;
      }
      MS_LOG(INFO) << "Checking StrategyCost for edge: " << edge_ptr->edge_name()
                   << " impossible, re-initing the operators and edges";
      auto prev_op = edge_ptr->prev_operator();
      MS_EXCEPTION_IF_NULL(prev_op);
      auto next_op = edge_ptr->next_operator();
      MS_EXCEPTION_IF_NULL(next_op);
      // Check the 'prev_op'
      prev_op->ExactStrategiesAndRelatedEdges();
      // Check the 'next_op'
      next_op->ExactStrategiesAndRelatedEdges();
    }
  }
}

bool CostGraph::FindReshapeCostInCache(const TensorLayout &from, const TensorLayout &to, CostPtr *result) {
  auto iter = reshape_cost_cache_.find(TensorLayoutsToCacheKey(from, to));
  if (iter != reshape_cost_cache_.end()) {
    *result = iter->second;
    return true;
  }
  return false;
}

void CostGraph::SaveReshapeCostToCache(const TensorLayout &from, const TensorLayout &to, const CostPtr &result) {
  reshape_cost_cache_.emplace(TensorLayoutsToCacheKey(from, to), result);
}

bool CostGraph::FindEdgeCostPtrInCache(const TensorLayout &prev_layout, const TensorLayout &next_layout,
                                       CostPtrList *cost) {
  auto iter = cost_map_cache_.find(TensorLayoutsToCacheKey(prev_layout, next_layout));
  if (iter != cost_map_cache_.end()) {
    *cost = iter->second;
    return true;
  }
  return false;
}

void CostGraph::SaveEdgeCostPtrToCache(const TensorLayout &prev_layout, const TensorLayout &next_layout,
                                       const CostPtrList &cost) {
  cost_map_cache_.emplace(TensorLayoutsToCacheKey(prev_layout, next_layout), cost);
}
}  // namespace parallel
}  // namespace mindspore
