/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "parallel/auto_parallel/graph_costmodel.h"

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace mindspore {
namespace parallel {
CostGraphPtr entire_costgraph = nullptr;
size_t TOTAL_OPS = 0;
double COST_MODEL_GAMMA = DEFAULT_COST_MODEL_GAMMA;
bool COST_MODEL_SIMPLIFY_CALCULATION = DEFAULT_COST_MODEL_SIMPLIFY_CALCULATION;
double DEVICE_MEMORY_CAPACITY = DEFAULT_DEVICE_MEMORY_CAPACITY;
double COST_MODEL_COMMUNI_THRESHOLD = DEFAULT_COST_MODEL_COMMUNI_THRESHOLD;
double COST_MODEL_COMMUNI_CONST = DEFAULT_COST_MODEL_COMMUNI_CONST;
double COST_MODEL_COMMUNI_BIAS = DEFAULT_COST_MODEL_COMMUNI_BIAS;
bool TENSOR_SLICE_ALIGNMENT_ENABLE = DEFAULT_TENSOR_SLICE_ALIGNMENT_ENABLE;
size_t TENSOR_SLICE_ALIGNMENT_SIZE = DEFAULT_TENSOR_SLICE_ALIGNMENT_SIZE;
bool FULLY_USE_DEVICES = DEFAULT_FULLY_USE_DEVICES;
bool ELEMENTWISE_OP_STRA_FOLLOW = DEFAULT_ELEMENTWISE_OP_STRA_FOLLOW;
bool MULTI_SUBGRAPHS = DEFAULT_IS_MULTI_SUBGRAPHS;
int32_t RUN_PHASE = DEFAULT_RUN_PHASE;

void CostGraph::SetDeviceMemoryAndCostParameter() {
  MS_EXCEPTION_IF_NULL(CostModelContext::GetInstance());

  // DEVICE_MEMORY_CAPACITY
  auto device_memory = CostModelContext::GetInstance()->device_memory_capacity();
  if (device_memory <= 0) {
    MS_LOG(EXCEPTION) << "'device_memory_capacity' must be positive.";
  }
  dev_memory_ = device_memory;
  DEVICE_MEMORY_CAPACITY = device_memory;
  MS_LOG(INFO) << "device_memory_capacity: " << DEVICE_MEMORY_CAPACITY << ".";

  // COST_MODEL_ALPHA
  auto alpha = CostModelContext::GetInstance()->costmodel_alpha();
  if (alpha <= 0) {
    MS_LOG(EXCEPTION) << "'costmodel_alpha' must be positive.";
  }
  costmodel_alpha_ = alpha;
  MS_LOG(INFO) << "costmodel_alpha: " << costmodel_alpha_ << ".";

  // COST_MODEL_BETA
  auto beta = CostModelContext::GetInstance()->costmodel_beta();
  if (beta <= 0) {
    MS_LOG(EXCEPTION) << "'costmodel_beta' must be positive.";
  }
  costmodel_beta_ = beta;
  MS_LOG(INFO) << "costmodel_beta: " << costmodel_beta_ << ".";

  // COST_MODEL_GAMMA
  auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
  if ((gamma < 0) || (gamma > 1)) {
    MS_LOG(EXCEPTION) << "'costmodel_gamma' must in [0, 1].";
  }
  COST_MODEL_GAMMA = gamma;
  MS_LOG(INFO) << "costmodel_gamma: " << COST_MODEL_GAMMA << ".";

  // COST_MODEL_SIMPLIFY_CALCULATION
  auto simplify = CostModelContext::GetInstance()->costmodel_simplify_cal();
  COST_MODEL_SIMPLIFY_CALCULATION = simplify;
  if (COST_MODEL_SIMPLIFY_CALCULATION) {
    MS_LOG(INFO) << "costmodel_simplify_cal: true.";
  } else {
    MS_LOG(INFO) << "costmodel_simplify_cal: false.";
  }

  // COST_MODEL_COMMUNI_THRESHOLD
  auto communi_threshold = CostModelContext::GetInstance()->costmodel_communi_threshold();
  if (communi_threshold < 0) {
    MS_LOG(EXCEPTION) << "'costmodel_communi_threshold' must be non-zero.";
  }
  COST_MODEL_COMMUNI_THRESHOLD = communi_threshold;
  MS_LOG(INFO) << "costmodel_communi_threshold: " << COST_MODEL_COMMUNI_THRESHOLD << ".";

  // COST_MODEL_COMMUNI_CONST
  auto communi_const = CostModelContext::GetInstance()->costmodel_communi_const();
  if (communi_const < 0) {
    MS_LOG(EXCEPTION) << "'costmodel_communi_const' must be non-zero.";
  }
  COST_MODEL_COMMUNI_CONST = communi_const;
  MS_LOG(INFO) << "costmodel_communi_const: " << COST_MODEL_COMMUNI_CONST << ".";

  // COST_MODEL_COMMUNI_BIAS
  auto communi_bias = CostModelContext::GetInstance()->costmodel_communi_bias();
  if (communi_bias < 0) {
    MS_LOG(EXCEPTION) << "'costmodel_communi_bias' must be non-zero.";
  }
  COST_MODEL_COMMUNI_BIAS = communi_bias;
  MS_LOG(INFO) << "costmodel_communi_bias: " << COST_MODEL_COMMUNI_BIAS << ".";

  // TENSOR_SLICE_ALIGNMENT_ENABLE
  auto align_enable = CostModelContext::GetInstance()->tensor_slice_alignment_enable();
  TENSOR_SLICE_ALIGNMENT_ENABLE = align_enable;
  if (TENSOR_SLICE_ALIGNMENT_ENABLE) {
    MS_LOG(INFO) << "tensor_slice_align_enable: true.";
  } else {
    MS_LOG(INFO) << "tensor_slice_align_enable: false.";
  }

  // TENSOR_SLICE_ALIGNMENT_SIZE
  auto align_size = CostModelContext::GetInstance()->tensor_slice_alignment_size();
  if (align_size == 0) {
    MS_LOG(EXCEPTION) << "'tensor_slice_align_size' must be positive.";
  }
  TENSOR_SLICE_ALIGNMENT_SIZE = align_size;
  MS_LOG(INFO) << "tensor_slice_align_size: " << TENSOR_SLICE_ALIGNMENT_SIZE << ".";

  // FULLY_USE_DEVICES
  auto fully_devices = CostModelContext::GetInstance()->fully_use_device();
  FULLY_USE_DEVICES = fully_devices;
  if (FULLY_USE_DEVICES) {
    MS_LOG(INFO) << "fully_use_devices: true.";
  } else {
    MS_LOG(INFO) << "fully_use_devices: false.";
  }

  // ELEMENTWISE_OP_STRA_FOLLOW
  auto is_ele_op_follow = CostModelContext::GetInstance()->elementwise_stra_follow();
  ELEMENTWISE_OP_STRA_FOLLOW = is_ele_op_follow;
  if (ELEMENTWISE_OP_STRA_FOLLOW) {
    MS_LOG(INFO) << "elementwise_op_strategy_follow: true.";
  } else {
    MS_LOG(INFO) << "elementwise_op_strategy_follow: false.";
  }

  // MULTI_SUBGRAPHS
  auto multi_subgraphs = CostModelContext::GetInstance()->is_multi_subgraphs();
  MULTI_SUBGRAPHS = multi_subgraphs;
  if (MULTI_SUBGRAPHS) {
    MS_LOG(INFO) << "multi_subgraphs: true.";
  } else {
    MS_LOG(INFO) << "multi_subgraphs: false.";
  }

  // RUN_PHASE
  auto phase = CostModelContext::GetInstance()->run_phase();
  if (phase != 0 && phase != 1) {
    MS_LOG(EXCEPTION) << "'run_phase' must be in {0, 1}";
  }
  RUN_PHASE = phase;
  MS_LOG(INFO) << "run_phase: " << RUN_PHASE << ".";
}

void CostGraph::RemoveOperator(const OperatorInfoPtr &op) {
  for (auto it = ops_.begin(); it != ops_.end();) {
    if ((*it) == op) {
      it = ops_.erase(it);
    } else {
      ++it;
    }
  }
}

bool CostGraph::IsOperatorInCostGraph(const OperatorInfoPtr &op_test) {
  struct IsInGraph {
    const OperatorInfoPtr test_;
    explicit IsInGraph(const OperatorInfoPtr &n) : test_(n) {}
    bool operator()(const OperatorInfoPtr &in) const { return (test_ == in); }
  };
  return std::any_of(ops_.begin(), ops_.end(), IsInGraph(op_test));
}

bool CostGraph::IsEdgeInCostGraph(const std::string &test_edge_name, size_t output_index, size_t input_index) {
  for (auto &edge_pair : edges_) {
    auto edges = edge_pair.second;
    for (auto &edge : edges) {
      MS_EXCEPTION_IF_NULL(edge);
      bool bool_result = (edge->edge_name() == test_edge_name) && (edge->prev_op_output_index() == output_index) &&
                         (edge->next_op_input_index() == input_index);
      if (bool_result) {
        return true;
      }
    }
  }
  return false;
}

std::vector<std::shared_ptr<CostGraph>> CostGraph::ConstructConnectedComponents(
  std::vector<OperatorInfoPtr> alive_ops) {
  std::map<OperatorInfoPtr, bool> visited;

  for (auto &op : alive_ops) {
    visited[op] = false;
  }

  MS_LOG(INFO) << "visited: " << visited.size() << ".";
  for (auto &op : alive_ops) {
    if ((!visited[op]) && op->is_alive()) {
      std::shared_ptr<CostGraph> new_component = std::make_shared<CostGraph>();
      MS_EXCEPTION_IF_NULL(new_component);
      new_component->SetDeviceMemoryAndCostParameter();
      DFS(op, &visited, new_component);
      connected_compoents_.push_back(new_component);
    }
  }
  return connected_compoents_;
}

void CostGraph::DFS(const OperatorInfoPtr &current_op, std::map<OperatorInfoPtr, bool> *visited,
                    const std::shared_ptr<CostGraph> &component) {
  MS_EXCEPTION_IF_NULL(visited);
  MS_EXCEPTION_IF_NULL(component);
  visited->at(current_op) = true;
  component->AddOperator(current_op);

  for (auto &edge : current_op->succ_edges()) {
    bool bool_test = (visited->find(edge->next_operator()) != visited->end()) &&
                     (!visited->at(edge->next_operator())) && edge->next_operator()->is_alive();
    if (bool_test) {
      component->AddEdge(current_op, edge->next_operator(), edge);
      DFS(edge->next_operator(), visited, component);
    }
  }

  for (auto &edge : current_op->prev_edges()) {
    bool bool_test = (visited->find(edge->prev_operator()) != visited->end()) &&
                     (!visited->at(edge->prev_operator())) && edge->prev_operator()->is_alive();
    if (bool_test) {
      component->AddEdge(edge->prev_operator(), current_op, edge);
      DFS(edge->prev_operator(), visited, component);
    }
  }
}

// Create final cost list for the graph: u --> v
CostPtrList CostGraph::CreateFinalCostList(const OperatorInfoPtr &u, const std::shared_ptr<Edge> &e,
                                           const OperatorInfoPtr &v) {
  MS_EXCEPTION_IF_NULL(u);
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(e);
  CostPtrList ret;
  for (const auto &u_strategy : u->GetStrategyCost()) {
    for (const auto &v_strategy : v->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(u_strategy);
      MS_EXCEPTION_IF_NULL(v_strategy);
      auto u_strategy_ptr = u_strategy->strategy_ptr;
      auto v_strategy_ptr = v_strategy->strategy_ptr;
      CostPtrList clist1 = u_strategy->cost_list;
      CostPtrList clist2 = e->GetCostList(u_strategy_ptr, v_strategy_ptr);
      CostPtrList clist3 = v_strategy->cost_list;
      for (const auto &cost1 : clist1) {
        for (const auto &cost2 : clist2) {
          for (const auto &cost3 : clist3) {
            MS_EXCEPTION_IF_NULL(cost1);
            MS_EXCEPTION_IF_NULL(cost2);
            MS_EXCEPTION_IF_NULL(cost3);
            double computation = cost1->computation_cost_ + cost2->computation_cost_ + cost3->computation_cost_;
            double memory = cost1->memory_with_reuse_ + cost2->memory_with_reuse_ + cost3->memory_with_reuse_;
            double communication = cost1->communication_cost_ + cost2->communication_cost_ + cost3->communication_cost_;
            double communication_forward =
              cost1->communication_forward_ + cost2->communication_forward_ + cost3->communication_forward_;
            double communication_without_para = cost1->communication_without_parameter_ +
                                                cost2->communication_without_parameter_ +
                                                cost3->communication_without_parameter_;
            auto decision =
              std::make_shared<FinalDecision>(u_strategy->strategy_ptr, v_strategy->strategy_ptr, cost1, cost2, cost3);
            auto cost = std::make_shared<Cost>(computation, communication, decision);
            MS_EXCEPTION_IF_NULL(cost);
            cost->communication_without_parameter_ = communication_without_para;
            cost->communication_with_partial_para_ =
              communication_without_para + COST_MODEL_GAMMA * (communication - communication_without_para);
            cost->memory_with_reuse_ = memory;
            cost->communication_forward_ = communication_forward;
            ret.push_back(cost);
          }
        }
      }
    }
  }

  Simplify(&ret);
  return ret;
}

// Create final cost list for the graph containing a signle node: u
CostPtrList CostGraph::CreateFinalSingleCostList(const OperatorInfoPtr &u) {
  MS_EXCEPTION_IF_NULL(u);
  CostPtrList ret;
  for (const auto &u_strategy : u->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(u_strategy);
    auto u_strategy_ptr = u_strategy->strategy_ptr;
    CostPtrList clist1 = u_strategy->cost_list;
    for (const auto &cost1 : clist1) {
      MS_EXCEPTION_IF_NULL(cost1);
      auto decision = std::make_shared<FinalSingleDecision>(u_strategy_ptr, cost1);
      auto new_cost = std::make_shared<Cost>(cost1->computation_cost_, cost1->communication_cost_, decision);
      MS_EXCEPTION_IF_NULL(new_cost);
      new_cost->communication_without_parameter_ = cost1->communication_without_parameter_;
      new_cost->communication_with_partial_para_ =
        cost1->communication_without_parameter_ +
        COST_MODEL_GAMMA * (cost1->communication_cost_ - cost1->communication_without_parameter_);
      new_cost->memory_with_reuse_ = cost1->memory_with_reuse_;
      new_cost->communication_forward_ = cost1->communication_forward_;
      ret.push_back(new_cost);
    }
  }

  Simplify(&ret);
  return ret;
}

CostPtr CostGraph::SelectCostWithMinInferenceTime(const CostPtrList &cost_list, double memory) {
  // Select the cost with minimum inference time. Currently, the inference time is modeled as =
  // costmodel_alpha_ * computation_cost + costmodel_beta_ * communication_forward_
  if (cost_list.empty()) {
    MS_LOG(ERROR) << "Final cost list is null.";
    return nullptr;
  }
  CostPtrList after_mem_filter;
  double minimum_memory = DBL_MAX;
  // Filter out the valid costs.
  for (auto &a_cost : cost_list) {
    if (a_cost->memory_with_reuse_ <= memory) {
      after_mem_filter.emplace_back(std::move(a_cost));
    } else if (a_cost->memory_with_reuse_ < minimum_memory) {
      minimum_memory = a_cost->memory_with_reuse_;
    }
  }
  if (after_mem_filter.empty()) {
    MS_LOG(ERROR) << "No available cost. The minimum memory cost is: " << minimum_memory
                  << ", the memory capacity is: " << memory << ".";
    return nullptr;
  }
  // Init the returned value with first cost.
  CostPtr ret = after_mem_filter[0];

  double minimum = costmodel_alpha_ * ret->computation_cost_ + costmodel_beta_ * ret->communication_forward_;
  MS_LOG(INFO) << "Cost 0: "
               << "memory_cost: " << ret->memory_with_reuse_ << ", computation_cost_: " << ret->computation_cost_
               << ", communication_forward_: " << ret->communication_forward_
               << ", communication_with_partial_para_: " << ret->communication_with_partial_para_
               << ", communication_cost_: " << ret->communication_cost_
               << ", communication_without_parameter_: " << ret->communication_without_parameter_ << ".";
  MS_LOG(INFO) << "Cost 0: totoal_cost: " << minimum;
  for (size_t i = 1; i < after_mem_filter.size(); ++i) {
    MS_EXCEPTION_IF_NULL(after_mem_filter[i]);
    MS_LOG(INFO) << "Cost " << i << ": memory_cost: " << after_mem_filter[i]->memory_with_reuse_
                 << ", computation_cost_: " << after_mem_filter[i]->computation_cost_
                 << ", communication_forward_: " << after_mem_filter[i]->communication_forward_
                 << ", communication_with_partial_para_: " << after_mem_filter[i]->communication_with_partial_para_
                 << ", communication_cost_: " << after_mem_filter[i]->communication_cost_
                 << ", communication_without_parameter_: " << after_mem_filter[i]->communication_without_parameter_
                 << ".";
    auto tmp = costmodel_alpha_ * after_mem_filter[i]->computation_cost_ +
               costmodel_beta_ * after_mem_filter[i]->communication_forward_;
    MS_LOG(INFO) << "Cost " << i << ": total_cost: " << tmp;
    if (minimum > tmp) {
      minimum = tmp;
      ret = after_mem_filter[i];
      MS_LOG(INFO) << "Selected: " << i;
    }
  }
  return ret;
}

CostPtr CostGraph::SelectCostWithMinTrainingTime(const CostPtrList &cost_list, double memory) {
  // Select the cost with minimum training time. Currently, the training time is modeled as =
  // costmodel_alpha_ * computation_cost + costmodel_beta_ * communication_with_partial_para_
  if (cost_list.empty()) {
    MS_LOG(ERROR) << "Final cost list is null.";
    return nullptr;
  }
  CostPtrList after_mem_filter;
  double minimum_memory = DBL_MAX;
  // Filter out the valid costs.
  for (auto &a_cost : cost_list) {
    if (a_cost->memory_with_reuse_ <= memory) {
      after_mem_filter.emplace_back(std::move(a_cost));
    } else if (a_cost->memory_with_reuse_ < minimum_memory) {
      minimum_memory = a_cost->memory_with_reuse_;
    }
  }
  if (after_mem_filter.empty()) {
    MS_LOG(ERROR) << "No available cost. The minimum memory cost is: " << minimum_memory
                  << ", the memory capacity is: " << memory << ".";
    return nullptr;
  }
  // Init the returned value with first cost.
  CostPtr ret = after_mem_filter[0];

  double minimum = costmodel_alpha_ * ret->computation_cost_ + costmodel_beta_ * ret->communication_with_partial_para_;
  MS_LOG(INFO) << "Cost 0: "
               << "memory_cost: " << ret->memory_with_reuse_ << ", computation_cost_: " << ret->computation_cost_
               << ", communication_with_partial_para_: " << ret->communication_with_partial_para_
               << ", communication_cost_: " << ret->communication_cost_
               << ", communication_without_parameter_: " << ret->communication_without_parameter_ << ".";
  MS_LOG(INFO) << "Cost 0: totoal_cost: " << minimum;
  for (size_t i = 1; i < after_mem_filter.size(); ++i) {
    MS_EXCEPTION_IF_NULL(after_mem_filter[i]);
    MS_LOG(INFO) << "Cost " << i << ": memory_cost: " << after_mem_filter[i]->memory_with_reuse_
                 << ", computation_cost_: " << after_mem_filter[i]->computation_cost_
                 << ", communication_with_partial_para_: " << after_mem_filter[i]->communication_with_partial_para_
                 << ", communication_cost_: " << after_mem_filter[i]->communication_cost_
                 << ", communication_without_parameter_: " << after_mem_filter[i]->communication_without_parameter_
                 << ".";
    auto tmp = costmodel_alpha_ * after_mem_filter[i]->computation_cost_ +
               costmodel_beta_ * after_mem_filter[i]->communication_with_partial_para_;
    MS_LOG(INFO) << "Cost " << i << ": total_cost: " << tmp;
    if (minimum > tmp) {
      minimum = tmp;
      ret = after_mem_filter[i];
      MS_LOG(INFO) << "Selected: " << i;
    }
  }
  return ret;
}

CostPtrList CostGraph::SelectCostListWithMinTrainingTimeMultiple(const std::vector<CostPtrList> &all_cost_list,
                                                                 double available_memory) {
  CostPtrList selected_cost_list(all_cost_list.size(), nullptr);
  double minimum = DBL_MAX, total_memory = 0.0;
  CostPtrList ret(all_cost_list.size(), nullptr);
  // Check whether valid costs exist.
  for (size_t i = 0; i < all_cost_list.size(); ++i) {
    if (all_cost_list[i][0] == nullptr) {
      MS_LOG(ERROR) << "The cost list " << i << " is empty.";
      return ret;
    } else {
      double memory_i_cost = DBL_MAX;
      for (size_t j = 0; j < all_cost_list[i].size(); ++j) {
        if (all_cost_list[i][j]->memory_with_reuse_ < memory_i_cost) {
          memory_i_cost = all_cost_list[i][j]->memory_with_reuse_;
        }
      }
      total_memory += memory_i_cost;
    }
  }
  if (total_memory >= available_memory) {
    MS_LOG(ERROR) << "No strategy can be found under current memory: " << available_memory
                  << ", minimum strategy cost: " << total_memory << ".";
    return selected_cost_list;
  }

  std::function<void(size_t)> recursive = [&all_cost_list, &selected_cost_list, &minimum, &ret, &recursive,
                                           &available_memory, this](size_t k) {
    if (k == all_cost_list.size()) {
      double tmp_memory = 0.0, tmp_minimum = 0.0;
      for (size_t i = 0; i < selected_cost_list.size(); ++i) {
        MS_EXCEPTION_IF_NULL(selected_cost_list[i]);
        tmp_memory += selected_cost_list[i]->memory_with_reuse_;
        tmp_minimum += costmodel_alpha_ * selected_cost_list[i]->computation_cost_ +
                       costmodel_beta_ * selected_cost_list[i]->communication_with_partial_para_;
      }
      MS_LOG(INFO) << "tmp_memory: " << tmp_memory << ", tmp_minimum: " << tmp_minimum << ", minimum: " << minimum
                   << ".";
      if (tmp_memory < available_memory && tmp_minimum < minimum) {
        ret = selected_cost_list;
        minimum = tmp_minimum;
        MS_LOG(INFO) << "selected tmp_memory: " << tmp_memory << ", tmp_minimum: " << tmp_minimum << ".";
      }
      return;
    }

    MS_LOG(DEBUG) << "The value minimum: " << minimum << ", available_memory: " << available_memory << ".";
    for (auto &c : all_cost_list[k]) {
      selected_cost_list[k] = c;
      recursive(k + 1);
    }
  };
  recursive(0);
  return ret;
}

Status CostGraph::SearchStrategyForMultiNodeFinalGraph(const std::vector<OperatorInfoPtr> &alive_ops) {
  MS_LOG(INFO) << "There are " << alive_ops.size() << " nodes in the final graph.";
  auto connected_components = ConstructConnectedComponents(alive_ops);
  MS_LOG(INFO) << "There are " << connected_components.size() << " components in the final graph.";
  std::vector<CostPtrList> all_list;
  for (size_t j = 0; j < connected_components.size(); ++j) {
    auto one_component = connected_components[j];
    MS_EXCEPTION_IF_NULL(one_component);
    if (one_component->GetOperators().size() == 1) {
      MS_LOG(INFO) << "There are 1 operator in a component in the final graph.";
      auto cost_list = one_component->CreateFinalSingleCostList(one_component->GetOperators()[0]);
      all_list.push_back(cost_list);
    } else if (one_component->GetOperators().size() == 2) {
      MS_LOG(INFO) << "There are 2 operators in a component in the final graph.";
      OperatorInfoPtr u, v;
      auto first_op = one_component->GetOperators()[0];
      auto second_op = one_component->GetOperators()[1];
      MS_EXCEPTION_IF_NULL(first_op);
      MS_EXCEPTION_IF_NULL(second_op);
      if (!first_op->GetAliveSuccEdges().empty() &&
          first_op->GetAliveSuccEdges()[0]->next_operator().get() == second_op.get()) {
        u = first_op;
        v = second_op;
      } else if (!second_op->GetAliveSuccEdges().empty() &&
                 second_op->GetAliveSuccEdges()[0]->next_operator().get() == first_op.get()) {
        u = second_op;
        v = first_op;
      } else {
        MS_LOG(EXCEPTION) << "The final graph is not the case of u --> v, " << first_op->GetAliveSuccEdges().size()
                          << ", " << second_op->GetAliveSuccEdges().size() << ".";
      }
      MS_EXCEPTION_IF_NULL(u);
      auto e = u->GetAliveSuccEdges()[0];
      auto cost_list = one_component->CreateFinalCostList(u, e, v);
      all_list.push_back(cost_list);
    } else {
      MS_LOG(EXCEPTION) << "There are " << one_component->GetOperators().size()
                        << " operators in a component in the final graph.";
    }
  }
  //
  auto selected_cost_list = SelectCostListWithMinTrainingTimeMultiple(all_list, dev_memory_);
  for (size_t k = 0; k < selected_cost_list.size(); ++k) {
    auto selected_cost = selected_cost_list[k];
    if (selected_cost == nullptr) {
      MS_LOG(ERROR) << "No vaild strategy can be found under the current device memory: " << dev_memory_ << ".";
      return FAILED;
    }
    MS_EXCEPTION_IF_NULL(connected_components[k]);
    if (connected_components[k]->GetOperators().size() == 1) {
      auto u = connected_components[k]->GetOperators()[0];
      auto decision = selected_cost->decision_ptr_->cast<FinalSingleDecisionPtr>();
      u->SetSelectedStrategyAndCost(decision->u_strategy_, decision->u_cost_);
      MS_LOG(INFO) << "Searching the strategy for the component " << k << " final graph ended.";
    } else if (connected_components[k]->GetOperators().size() == 2) {
      OperatorInfoPtr u = nullptr, v = nullptr;
      auto first_op = connected_components[k]->GetOperators()[0];
      auto second_op = connected_components[k]->GetOperators()[1];
      MS_EXCEPTION_IF_NULL(first_op);
      MS_EXCEPTION_IF_NULL(second_op);
      if (!first_op->GetAliveSuccEdges().empty() &&
          first_op->GetAliveSuccEdges()[0]->next_operator().get() == second_op.get()) {
        u = first_op;
        v = second_op;
      } else if (!second_op->GetAliveSuccEdges().empty() &&
                 second_op->GetAliveSuccEdges()[0]->next_operator().get() == first_op.get()) {
        u = second_op;
        v = first_op;
      }
      MS_EXCEPTION_IF_NULL(u);
      auto e = u->GetAliveSuccEdges()[0];
      MS_EXCEPTION_IF_NULL(v);
      MS_EXCEPTION_IF_NULL(e);
      MS_EXCEPTION_IF_NULL(selected_cost->decision_ptr_);
      auto decision = selected_cost->decision_ptr_->cast<FinalDecisionPtr>();
      MS_EXCEPTION_IF_NULL(decision);
      u->SetSelectedStrategyAndCost(decision->u_strategy_, decision->left_cost_);
      v->SetSelectedStrategyAndCost(decision->v_strategy_, decision->right_cost_);
      e->set_selected_cost(decision->middle_cost_);
      MS_LOG(INFO) << "Searching the strategy for the component " << k << " final graph ended.";
    }
  }
  return SUCCESS;
}

// searching the strategy for the final eliminated graph
Status CostGraph::SearchStrategy() {
  MS_LOG(INFO) << "Searching the strategy for the eliminated final graph began.";
  std::vector<OperatorInfoPtr> alive_ops;
  (void)std::for_each(ops_.begin(), ops_.end(), [&alive_ops](const OperatorInfoPtr &op) {
    MS_EXCEPTION_IF_NULL(op);
    if (op->is_alive()) {
      alive_ops.push_back(op);
    }
  });

  if (alive_ops.size() > 2) {
    if (RUN_PHASE == TRAINING_PHASE) {
      // training phase
      return SearchStrategyForMultiNodeFinalGraph(alive_ops);
    } else {
      // inference phase
      MS_LOG(EXCEPTION)
        << "Currently, searching strategy for the multi-node final graph in inference phase is not supported.";
    }
  } else if (alive_ops.size() == 1) {
    MS_LOG(INFO) << "There are 1 single node in the final graph.";
    OperatorInfoPtr u = alive_ops[0];
    auto cost_list = CreateFinalSingleCostList(u);
    CostPtr cost = nullptr;
    if (RUN_PHASE == TRAINING_PHASE) {
      // training phase
      cost = SelectCostWithMinTrainingTime(cost_list, dev_memory_);
    } else {
      // inference phase
      cost = SelectCostWithMinInferenceTime(cost_list, dev_memory_);
    }
    if (cost == nullptr) {
      MS_LOG(ERROR) << "No vaild strategy can be found under the current device memory: " << dev_memory_ << ".";
      return FAILED;
    }
    MS_EXCEPTION_IF_NULL(u);
    MS_EXCEPTION_IF_NULL(cost->decision_ptr_);
    auto decision = cost->decision_ptr_->cast<FinalSingleDecisionPtr>();
    MS_EXCEPTION_IF_NULL(decision);
    u->SetSelectedStrategyAndCost(decision->u_strategy_, decision->u_cost_);
    MS_LOG(INFO) << "Searching the strategy for the eliminated final graph ended.";
    return SUCCESS;
  } else {
    // In this case, the final graph should contains exactly 2 nodes.
    if (alive_ops.empty()) {
      MS_LOG(INFO) << "0 Operator in the final graph.";
      return SUCCESS;
    }
    OperatorInfoPtr u, v;
    MS_EXCEPTION_IF_NULL(alive_ops[0]);
    MS_EXCEPTION_IF_NULL(alive_ops[1]);
    if (!alive_ops[0]->GetAliveSuccEdges().empty() &&
        alive_ops[0]->GetAliveSuccEdges()[0]->next_operator().get() == alive_ops[1].get()) {
      u = alive_ops[0];
      v = alive_ops[1];
    } else if (!alive_ops[1]->GetAliveSuccEdges().empty() &&
               alive_ops[1]->GetAliveSuccEdges()[0]->next_operator().get() == alive_ops[0].get()) {
      u = alive_ops[1];
      v = alive_ops[0];
    } else {
      if (!alive_ops[0]->GetAliveSuccEdges().empty() || !alive_ops[1]->GetAliveSuccEdges().empty()) {
        MS_LOG(EXCEPTION) << "The final graph is not the case of u --> v, " << alive_ops[0]->GetAliveSuccEdges().size()
                          << ", " << alive_ops[1]->GetAliveSuccEdges().size() << ".";
      } else {
        // In this case, the final graph consists of two single nodes
        MS_LOG(INFO) << "There are 2 single nodes in the final graph.";
        std::vector<CostPtrList> all_list;
        auto connected_components = ConstructConnectedComponents(alive_ops);
        MS_LOG(INFO) << "There are " << connected_components.size() << " components in the final graph.";
        for (size_t i = 0; i < connected_components.size(); ++i) {
          MS_LOG(INFO) << "There are 1 operator in a component in the final graph.";
          auto one_component = connected_components[i];
          MS_EXCEPTION_IF_NULL(one_component);
          auto cost_list = one_component->CreateFinalSingleCostList(one_component->GetOperators()[0]);
          all_list.push_back(cost_list);
        }
        CostPtrList selected_cost_list;
        if (RUN_PHASE == TRAINING_PHASE) {
          // training phase
          selected_cost_list = SelectCostListWithMinTrainingTimeMultiple(all_list, dev_memory_);
        } else {
          // inference phase
          MS_LOG(EXCEPTION) << "Currently, searching strategy for the two-separated-node final graph in the inference "
                               "phase is not supported.";
        }
        for (size_t k = 0; k < selected_cost_list.size(); ++k) {
          auto selected_cost = selected_cost_list[k];
          if (selected_cost == nullptr) {
            MS_LOG(ERROR) << "No vaild strategy can be found under the current device memory: " << dev_memory_ << ".";
            return FAILED;
          }
          MS_EXCEPTION_IF_NULL(connected_components[k]);
          auto one_operator = connected_components[k]->GetOperators()[0];
          MS_EXCEPTION_IF_NULL(selected_cost->decision_ptr_);
          auto decision = selected_cost->decision_ptr_->cast<FinalSingleDecisionPtr>();
          MS_EXCEPTION_IF_NULL(decision);
          one_operator->SetSelectedStrategyAndCost(decision->u_strategy_, decision->u_cost_);
          MS_LOG(INFO) << "Searching the strategy for the component " << k << " final graph ended.";
        }

        return SUCCESS;
      }
    }
    MS_LOG(INFO) << "There are 2 nodes in the final graph.";
    // In this case, the finale graph is exactly of the form: u --> v
    MS_EXCEPTION_IF_NULL(u);
    MS_EXCEPTION_IF_NULL(v);
    auto e = u->GetAliveSuccEdges()[0];
    MS_EXCEPTION_IF_NULL(e);
    auto cost_list = CreateFinalCostList(u, e, v);
    CostPtr cost = nullptr;
    if (RUN_PHASE == TRAINING_PHASE) {
      // training phase
      cost = SelectCostWithMinTrainingTime(cost_list, dev_memory_);
    } else {
      MS_LOG(EXCEPTION) << "Currently, searching strategy for the two-connected-node final graph in the inference "
                           "phase is not supported.";
    }
    if (cost == nullptr) {
      MS_LOG(ERROR) << "No vaild strategy can be found under the current device memory: " << dev_memory_ << ".";
      return FAILED;
    }
    MS_EXCEPTION_IF_NULL(cost->decision_ptr_);
    auto decision = cost->decision_ptr_->cast<FinalDecisionPtr>();
    MS_EXCEPTION_IF_NULL(decision);
    u->SetSelectedStrategyAndCost(decision->u_strategy_, decision->left_cost_);
    v->SetSelectedStrategyAndCost(decision->v_strategy_, decision->right_cost_);
    e->set_selected_cost(decision->middle_cost_);
    MS_LOG(INFO) << "Searching the strategy for the eliminated final graph ended.";
    return SUCCESS;
  }
}

// Given a graph which contains the following subgraph: u --> v --> w, the node v can be eliminated
// return the v and the edge u --> v
OperatorInfoPtr CostGraph::CheckOpElimination() const {
  for (auto &op : ops_) {
    bool bool_test = op->is_alive() && op->GetAliveSuccEdges().size() == 1 && op->GetAlivePrevEdges().size() == 1;
    if (bool_test) {
      if ((op->GetAliveSuccEdges()[0]->next_operator() != op) && (op->GetAlivePrevEdges()[0]->prev_operator() != op)) {
        return op;
      }
    }
  }
  return nullptr;
}

// Check the graph whether an EdgeElimination can be performed
std::vector<std::shared_ptr<Edge>> CostGraph::CheckEdgeElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    if (!op->is_alive()) continue;
    std::map<void *, int> count;
    for (auto &edge : op->GetAliveSuccEdges()) {
      MS_EXCEPTION_IF_NULL(edge);
      auto v = edge->next_operator();
      count[v.get()]++;
    }
    for (auto &pair : count) {
      auto *op_ptr = pair.first;
      int op_count = pair.second;
      if (op_count > 1) {
        std::vector<std::shared_ptr<Edge>> ret;
        for (auto &edge : op->GetAliveSuccEdges()) {
          MS_EXCEPTION_IF_NULL(edge);
          if (edge->next_operator().get() == op_ptr) {
            ret.push_back(edge);
          }
        }
        return ret;
      }
    }
  }
  return {};
}

// Check the graph whether a MergeElimination can be performed
OperatorInfoPtr CostGraph::CheckMergeElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    bool bool_test = op->is_alive() && op->GetAlivePrevEdges().empty() && op->GetAliveSuccEdges().size() == 1;
    if (bool_test) {
      auto next_op = op->GetAliveSuccEdges()[0]->next_operator();
      MS_EXCEPTION_IF_NULL(next_op);
      if (!next_op->GetAlivePrevEdges().empty()) {
        return op;
      }
    }
  }
  return nullptr;
}

// Check the graph whether a ContractElimination can be performed
OperatorInfoPtr CostGraph::CheckContractElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    bool bool_test = op->is_alive() && op->GetAlivePrevEdges().size() == 1 && op->GetAliveSuccEdges().empty();
    if (bool_test) {
      auto edge = op->GetAlivePrevEdges()[0];
      MS_EXCEPTION_IF_NULL(edge);
      auto prev_op = edge->prev_operator();
      MS_EXCEPTION_IF_NULL(prev_op);
      if (!prev_op->GetAliveSuccEdges().empty()) {
        return op;
      }
    }
  }
  return nullptr;
}

// Check the graph whether a TriangleElimination can be performed
std::pair<OperatorInfoPtr, std::shared_ptr<Edge>> CostGraph::CheckTriangleElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    bool bool_test = (op->is_alive()) && (op->GetAlivePrevEdges().empty()) && (op->GetAliveSuccEdges().size() == 2);
    if (bool_test) {
      auto edge1 = op->GetAliveSuccEdges()[0];
      auto edge2 = op->GetAliveSuccEdges()[1];
      MS_EXCEPTION_IF_NULL(edge1);
      MS_EXCEPTION_IF_NULL(edge2);
      auto first_op = edge1->next_operator();
      auto second_op = edge2->next_operator();
      MS_EXCEPTION_IF_NULL(first_op);
      for (auto &first_op_succ_edge : first_op->GetAliveSuccEdges()) {
        if (first_op_succ_edge->next_operator() == second_op) {
          return {op, first_op_succ_edge};
        }
      }
      MS_EXCEPTION_IF_NULL(second_op);
      for (auto &second_op_succ_edge : second_op->GetAliveSuccEdges()) {
        if (second_op_succ_edge->next_operator() == first_op) {
          return {op, second_op_succ_edge};
        }
      }
    }
  }
  return {nullptr, nullptr};
}

// Check the graph whether a StarElimination can be performed.
// NOTE: this elimination MUST be performed only when the above 5 operation cannot be applied.
OperatorInfoPtr CostGraph::CheckStarElimination() const {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    bool bool_test = (op->is_alive()) && (op->GetAlivePrevEdges().empty()) && (op->GetAliveSuccEdges().size() > 1);
    if (bool_test) {
      return op;
    }
  }
  return nullptr;
}

// This method is for 'eliminating operator' operation in the DP algorithm. It creates a new edge to replace
// 'lefe_edge', 'op' and 'right_edge'. As a consequence, it creates new costlist for the new edge.
std::shared_ptr<Edge> CostGraph::EliminationOp(const OperatorInfoPtr &op) {
  // in this case, the operators are organised in the form of u-->op-->v, and the goal
  // is to eliminate 'op'.
  MS_EXCEPTION_IF_NULL(op);
  MS_LOG(INFO) << "Now eliminating node: " << op->name() << ".";
  auto edge_u_op = op->GetAlivePrevEdges()[0];
  auto edge_op_v = op->GetAliveSuccEdges()[0];
  MS_EXCEPTION_IF_NULL(edge_u_op);
  MS_EXCEPTION_IF_NULL(edge_op_v);
  auto u = edge_u_op->prev_operator();
  auto v = edge_op_v->next_operator();
  std::vector<size_t> output_indexs, input_indexs;
  size_t output_index, input_index;
  MS_EXCEPTION_IF_NULL(u);
  MS_EXCEPTION_IF_NULL(v);
  std::string new_edge_name = u->name() + OPERATOR_TO_OPERATOR_CONNECTOR + v->name();
  std::shared_ptr<Edge> new_edge;
  if (edge_u_op->is_combined()) {
    output_indexs = edge_u_op->prev_op_output_indexs();
  } else {
    output_index = edge_u_op->prev_op_output_index();
    output_indexs.push_back(output_index);
  }
  if (edge_op_v->is_combined()) {
    input_indexs = edge_op_v->next_op_input_indexs();
  } else {
    input_index = edge_op_v->next_op_input_index();
    input_indexs.push_back(input_index);
  }

  if (!edge_u_op->is_combined() && !edge_op_v->is_combined()) {
    new_edge = std::make_shared<Edge>(new_edge_name, u, v, output_index, input_index, false);
  } else {
    new_edge = std::make_shared<Edge>(new_edge_name, u, v, output_indexs, input_indexs, true);
  }
  MS_EXCEPTION_IF_NULL(new_edge);
  new_edge->set_pre_op_output(edge_u_op->prev_op_output());
  new_edge->set_next_op_input(edge_op_v->next_op_input());
  new_edge->OpEliminationSetNewCost(edge_u_op, op, edge_op_v);
  u->ReplaceSuccEdge(op, new_edge);
  v->ReplacePreEdge(op, new_edge);
  op->SetNotAlive();
  MS_LOG(INFO) << "Eliminating node: " << op->name() << " succeeded.";
  return new_edge;
}

// This method is for 'eliminating edges' operation in the DP algorithm. It creates a new edge to replace the 'edges',
// and sets new costlist for the new edge.
std::shared_ptr<Edge> CostGraph::EliminationEdges(const std::vector<std::shared_ptr<Edge>> &edges) {
  MS_LOG(INFO) << "Now eliminating " << edges.size() << " edges.";
  MS_EXCEPTION_IF_NULL(edges[0]);
  auto u = edges[0]->prev_operator();
  auto v = edges[0]->next_operator();
  MS_EXCEPTION_IF_NULL(u);
  MS_EXCEPTION_IF_NULL(v);
  std::string new_edge_name = u->name() + OPERATOR_TO_OPERATOR_CONNECTOR + v->name();
  std::vector<size_t> output_indexs, input_indexs;

  for (auto &edge : edges) {
    MS_EXCEPTION_IF_NULL(edge);
    if (edge->is_combined()) {
      auto from_output_indexs = edge->prev_op_output_indexs();
      auto from_input_indexs = edge->next_op_input_indexs();
      (void)std::copy(from_output_indexs.begin(), from_output_indexs.end(), std::back_inserter(output_indexs));
      (void)std::copy(from_input_indexs.begin(), from_input_indexs.end(), std::back_inserter(input_indexs));
    } else {
      output_indexs.push_back(edge->prev_op_output_index());
      input_indexs.push_back(edge->next_op_input_index());
    }
  }

  std::shared_ptr<Edge> new_edge = std::make_shared<Edge>(new_edge_name, u, v, output_indexs, input_indexs, true);
  MS_EXCEPTION_IF_NULL(new_edge);
  new_edge->set_pre_op_output(edges[0]->prev_op_output());
  new_edge->set_next_op_input(edges[0]->next_op_input());

  new_edge->EdgeEliminationSetNewCost(u, edges, v);

  u->ReplaceSuccEdges(v, new_edge);
  v->ReplacePreEdges(u, new_edge);
  MS_LOG(INFO) << "Eliminating " << edges.size() << " edges succeeded.";
  return new_edge;
}

// Given 'op_cost_list', 'edge_cost_list', and 'tar_cost_list', this method is to create 'tar_cost_list_new'
// for this contract under the strategy 'op_strategy'
void CostGraph::CreateMergeEliminationSubCostList(StrategyPtr op_strategy, const CostPtrList &op_cost_list,
                                                  const CostPtrList &edge_cost_list, StrategyPtr tar_op_strategy,
                                                  const CostPtrList &tar_cost_list,
                                                  CostPtrList *const tar_cost_list_new) {
  for (size_t i = 0; i < op_cost_list.size(); ++i) {
    auto &op_cost = op_cost_list[i];
    MS_EXCEPTION_IF_NULL(op_cost);
    for (size_t j = 0; j < edge_cost_list.size(); ++j) {
      auto &edge_cost = edge_cost_list[j];
      MS_EXCEPTION_IF_NULL(edge_cost);
      for (size_t k = 0; k < tar_cost_list.size(); ++k) {
        auto &tar_cost = tar_cost_list[k];
        MS_EXCEPTION_IF_NULL(tar_cost);
        double computation = op_cost->computation_cost_ + edge_cost->computation_cost_ + tar_cost->computation_cost_;
        double memory = op_cost->memory_with_reuse_ + edge_cost->memory_with_reuse_ + tar_cost->memory_with_reuse_;
        double communication =
          op_cost->communication_cost_ + edge_cost->communication_cost_ + tar_cost->communication_cost_;
        double communication_forward =
          op_cost->communication_forward_ + edge_cost->communication_forward_ + tar_cost->communication_forward_;
        double communication_without_para = op_cost->communication_without_parameter_ +
                                            edge_cost->communication_without_parameter_ +
                                            tar_cost->communication_without_parameter_;

        auto decision =
          std::make_shared<MergeEliminationDecision>(op_strategy, op_cost, edge_cost, tar_op_strategy, tar_cost);
        auto new_cost = std::make_shared<Cost>(computation, communication, decision);
        MS_EXCEPTION_IF_NULL(new_cost);
        new_cost->communication_without_parameter_ = communication_without_para;
        new_cost->communication_with_partial_para_ =
          communication_without_para + COST_MODEL_GAMMA * (communication - communication_without_para);
        new_cost->memory_with_reuse_ = memory;
        new_cost->communication_forward_ = communication_forward;
        MS_EXCEPTION_IF_NULL(tar_cost_list_new);
        tar_cost_list_new->emplace_back(std::move(new_cost));
      }
    }
  }
}

// This method is for the 'Merge' operation in DP algorithm. It creates new costlist for each strategy in the
// target_op
OperatorInfoPtr CostGraph::EliminationMerge(const OperatorInfoPtr &op) {
  MS_EXCEPTION_IF_NULL(op);
  auto target_op = op->GetAliveSuccEdges()[0]->next_operator();
  auto edge_ptr = op->GetAliveSuccEdges()[0];
  MS_EXCEPTION_IF_NULL(target_op);
  MS_EXCEPTION_IF_NULL(edge_ptr);
  MS_LOG(INFO) << "Now merging " << op->name() << " into " << target_op->name() << ".";
  bool valid = false;

  for (auto &tar_stra_cost : target_op->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(tar_stra_cost);
    auto tar_stra = tar_stra_cost->strategy_ptr;
    auto tar_clist_origin = tar_stra_cost->cost_list;
    CostPtrList tar_clist_new;

    for (auto &op_stra_cost : op->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(op_stra_cost);
      auto op_stra = op_stra_cost->strategy_ptr;
      auto op_clist = op_stra_cost->cost_list;
      auto edge_clist = edge_ptr->GetCostList(op_stra, tar_stra);

      CreateMergeEliminationSubCostList(op_stra, op_clist, edge_clist, tar_stra, tar_clist_origin, &tar_clist_new);
    }
    Simplify(&tar_clist_new);
    // Set the new costlist w.r.t the strategy
    tar_stra_cost->cost_list = tar_clist_new;
    if ((!valid) && (!tar_clist_new.empty())) {
      valid = true;
    }
  }

  if (!valid) {
    MS_LOG(EXCEPTION) << "Merging " << op->name() << " into " << target_op->name() << " failed.";
  }
  op->SetNotAlive();
  MS_LOG(INFO) << "Merging " << op->name() << " into " << target_op->name() << " succeeded.";
  return target_op;
}

// Given 'contract_op_cost_list', 'edge_cost_list', and 'tar_cost_list', this method is to create 'tar_cost_list_new'
// for this contract under the strategy 'contract_op_stra'
void CostGraph::CreateContractEliminationSubCostList(StrategyPtr contract_op_stra,
                                                     const CostPtrList &contract_op_cost_list,
                                                     const CostPtrList &edge_cost_list, StrategyPtr target_op_stra,
                                                     const CostPtrList &tar_cost_list, CostPtrList *tar_cost_list_new) {
  for (size_t i = 0; i < contract_op_cost_list.size(); ++i) {
    auto &contract_op_cost = contract_op_cost_list[i];
    MS_EXCEPTION_IF_NULL(contract_op_cost);
    for (size_t j = 0; j < edge_cost_list.size(); ++j) {
      auto &edge_cost = edge_cost_list[j];
      MS_EXCEPTION_IF_NULL(edge_cost);
      for (size_t k = 0; k < tar_cost_list.size(); ++k) {
        auto &tar_cost = tar_cost_list[k];
        MS_EXCEPTION_IF_NULL(tar_cost);
        double computation =
          contract_op_cost->computation_cost_ + edge_cost->computation_cost_ + tar_cost->computation_cost_;
        double memory =
          contract_op_cost->memory_with_reuse_ + edge_cost->memory_with_reuse_ + tar_cost->memory_with_reuse_;
        double communication =
          contract_op_cost->communication_cost_ + edge_cost->communication_cost_ + tar_cost->communication_cost_;
        double communication_forward = contract_op_cost->communication_forward_ + edge_cost->communication_forward_ +
                                       tar_cost->communication_forward_;
        double communication_without_para = contract_op_cost->communication_without_parameter_ +
                                            edge_cost->communication_without_parameter_ +
                                            tar_cost->communication_without_parameter_;

        auto decision = std::make_shared<ContractEliminationDecision>(contract_op_stra, contract_op_cost, edge_cost,
                                                                      target_op_stra, tar_cost);
        auto new_cost = std::make_shared<Cost>(computation, communication, decision);
        new_cost->communication_without_parameter_ = communication_without_para;
        new_cost->communication_with_partial_para_ =
          communication_without_para + COST_MODEL_GAMMA * (communication - communication_without_para);
        new_cost->memory_with_reuse_ = memory;
        new_cost->communication_forward_ = communication_forward;
        tar_cost_list_new->emplace_back(std::move(new_cost));
      }
    }
  }
}

// This method is for the 'Contract' operation in DP algorithm. It creates new costlist for each strategy in the
// target_op
OperatorInfoPtr CostGraph::EliminationContract(const OperatorInfoPtr &op) {
  MS_EXCEPTION_IF_NULL(op);
  auto target_op = op->GetAlivePrevEdges()[0]->prev_operator();
  auto edge_ptr = op->GetAlivePrevEdges()[0];
  MS_LOG(INFO) << "Now contracting " << op->name() << " into " << target_op->name() << ".";
  bool valid = false;

  for (auto &tar_stra_cost : target_op->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(tar_stra_cost);
    auto tar_stra = tar_stra_cost->strategy_ptr;
    auto tar_clist_origin = tar_stra_cost->cost_list;
    CostPtrList tar_clist_new;

    for (auto &op_stra_cost : op->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(op_stra_cost);
      auto op_stra = op_stra_cost->strategy_ptr;
      auto op_clist = op_stra_cost->cost_list;
      auto edge_clist = edge_ptr->GetCostList(tar_stra, op_stra);

      CreateContractEliminationSubCostList(op_stra, op_clist, edge_clist, tar_stra, tar_clist_origin, &tar_clist_new);
    }
    Simplify(&tar_clist_new);
    // Set the new costlist w.r.t the strategy
    tar_stra_cost->cost_list = tar_clist_new;
    if ((!valid) && (!tar_clist_new.empty())) {
      valid = true;
    }
  }
  if (!valid) {
    MS_LOG(EXCEPTION) << "Contracting " << op->name() << " into " << target_op->name() << " failed.";
  }
  op->SetNotAlive();
  MS_LOG(INFO) << "Contracting " << op->name() << " into " << target_op->name() << " succeeded.";
  return target_op;
}

void CostGraph::CreateTriangleEliminationSubCostList(StrategyPtr elimi_op_stra, StrategyPtr left_op_stra,
                                                     StrategyPtr right_op_stra, const CostPtr &right_op_cost,
                                                     const CostPtrList &elimi_op_clist,
                                                     const CostPtrList &left_edge_clist, const CostPtr &right_edge_cost,
                                                     const CostPtrList &left_node_clist_origin,
                                                     CostPtrList *left_node_clist_new) {
  MS_EXCEPTION_IF_NULL(right_edge_cost);
  MS_EXCEPTION_IF_NULL(right_op_cost);
  MS_EXCEPTION_IF_NULL(left_node_clist_new);
  for (auto &elimi_op_cost : elimi_op_clist) {
    MS_EXCEPTION_IF_NULL(elimi_op_cost);
    for (auto &left_edge_cost : left_edge_clist) {
      MS_EXCEPTION_IF_NULL(left_edge_cost);
      for (auto &left_node_cost : left_node_clist_origin) {
        MS_EXCEPTION_IF_NULL(left_node_cost);
        double new_computation = elimi_op_cost->computation_cost_ + left_edge_cost->computation_cost_ +
                                 left_node_cost->computation_cost_ + right_edge_cost->computation_cost_;
        double new_memory = elimi_op_cost->memory_with_reuse_ + left_edge_cost->memory_with_reuse_ +
                            left_node_cost->memory_with_reuse_ + right_edge_cost->memory_with_reuse_;
        double new_commu_cost = elimi_op_cost->communication_cost_ + left_edge_cost->communication_cost_ +
                                left_node_cost->communication_cost_ + right_edge_cost->communication_cost_;
        double new_commu_forward = elimi_op_cost->communication_forward_ + left_edge_cost->communication_forward_ +
                                   left_node_cost->communication_forward_ + right_edge_cost->communication_forward_;
        double new_commu_without =
          elimi_op_cost->communication_without_parameter_ + left_edge_cost->communication_without_parameter_ +
          left_node_cost->communication_without_parameter_ + right_edge_cost->communication_without_parameter_;

        auto decision = std::make_shared<TriangleEliminationDecision>(elimi_op_stra, elimi_op_cost, left_edge_cost,
                                                                      right_edge_cost, left_op_stra, left_node_cost);
        auto new_cost = std::make_shared<Cost>(new_computation, new_commu_cost, decision);
        new_cost->communication_without_parameter_ = new_commu_without;
        new_cost->communication_with_partial_para_ =
          new_commu_without + COST_MODEL_GAMMA * (new_commu_cost - new_commu_without);
        new_cost->memory_with_reuse_ = new_memory;
        new_cost->communication_forward_ = new_commu_forward;
        left_node_clist_new->emplace_back(std::move(new_cost));
      }
    }
  }
}

void CostGraph::CreateTriangleEliminationCostList(const OperatorInfoPtr &elimi_op, const CostPtrList &right_node_clist,
                                                  const CostPtrList &right_edge_clist, const StrategyPtr &elimi_op_stra,
                                                  const StrategyPtr &left_node_stra, const StrategyPtr &right_node_stra,
                                                  const CostPtrList &elimi_op_clist, const CostPtrList &left_edge_clist,
                                                  const CostPtrList &left_node_clist_origin,
                                                  CostPtrList *left_node_clist_new) {
  MS_EXCEPTION_IF_NULL(elimi_op);
  for (auto &right_node_cost : right_node_clist) {
    MS_EXCEPTION_IF_NULL(right_node_cost);
    for (auto &right_edge_cost : right_edge_clist) {
      MS_EXCEPTION_IF_NULL(right_edge_cost);
      CreateTriangleEliminationSubCostList(elimi_op_stra, left_node_stra, right_node_stra, right_node_cost,
                                           elimi_op_clist, left_edge_clist, right_edge_cost, left_node_clist_origin,
                                           left_node_clist_new);
    }
  }
}

OperatorInfoPtr CostGraph::EliminationTriangle(const OperatorInfoPtr &elimi_op,
                                               const std::shared_ptr<Edge> &edge_left_right) {
  MS_EXCEPTION_IF_NULL(edge_left_right);
  MS_EXCEPTION_IF_NULL(elimi_op);
  MS_LOG(INFO) << "Now eliminating triangle: " << elimi_op->name() << ".";
  auto left_node = edge_left_right->prev_operator();
  auto right_node = edge_left_right->next_operator();
  auto left_edge = elimi_op->GetAliveSuccEdges()[0];
  auto right_edge = elimi_op->GetAliveSuccEdges()[1];
  MS_EXCEPTION_IF_NULL(left_node);
  MS_EXCEPTION_IF_NULL(right_node);
  MS_EXCEPTION_IF_NULL(left_edge);
  MS_EXCEPTION_IF_NULL(right_edge);
  MS_LOG(INFO) << "The left operator is: " << left_node->name() << ".";
  MS_LOG(INFO) << "The right operator is: " << right_node->name() << ".";

  if (left_edge->next_operator() != left_node) {
    auto tmp = left_edge;
    left_edge = right_edge;
    right_edge = tmp;
  }
  bool valid = false;

  for (auto &left_node_stra_cost : left_node->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(left_node_stra_cost);
    auto left_node_stra = left_node_stra_cost->strategy_ptr;
    auto left_node_clist_origin = left_node_stra_cost->cost_list;
    CostPtrList left_node_clist_new;

    for (auto &elimi_op_stra_cost : elimi_op->GetStrategyCost()) {
      MS_EXCEPTION_IF_NULL(elimi_op_stra_cost);
      auto elimi_op_stra = elimi_op_stra_cost->strategy_ptr;
      auto elimi_op_clist = elimi_op_stra_cost->cost_list;
      auto left_edge_clist = left_edge->GetCostList(elimi_op_stra, left_node_stra);

      for (auto &right_node_stra_cost : right_node->GetStrategyCost()) {
        MS_EXCEPTION_IF_NULL(right_node_stra_cost);
        auto right_node_stra = right_node_stra_cost->strategy_ptr;
        auto right_node_clist = right_node_stra_cost->cost_list;
        auto right_edge_clist = right_edge->GetCostList(elimi_op_stra, right_node_stra);

        CreateTriangleEliminationCostList(elimi_op, right_node_clist, right_edge_clist, elimi_op_stra, left_node_stra,
                                          right_node_stra, elimi_op_clist, left_edge_clist, left_node_clist_origin,
                                          &left_node_clist_new);
      }
    }
    Simplify(&left_node_clist_new);
    // Set the new costlist w.r.t the strategy
    left_node_stra_cost->cost_list = left_node_clist_new;
    if ((!valid) && (!left_node_clist_new.empty())) {
      valid = true;
    }
  }

  if (!valid) {
    MS_LOG(EXCEPTION) << "Eliminating triangle: " << elimi_op->name() << " failed.";
  }
  elimi_op->SetNotAlive();
  MS_LOG(INFO) << "Eliminating triangle: " << elimi_op->name() << " succeeded.";
  return left_node;
}

void CostGraph::CreateStarEliminationSubCostList(const StrategyPtr &first_succ_node_stra,
                                                 const CostPtrList &first_succ_node_clist,
                                                 const CostPtrList &first_succ_edge_clist,
                                                 const StrategyPtr &merged_op_stra, const CostPtrList &merged_op_clist,
                                                 std::vector<StrategyPtr> succ_nodes_stras,
                                                 CostPtrList &succ_edges_costs, CostPtrList &succ_nodes_costs,
                                                 CostPtrList *first_succ_node_clist_new) {
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
          }
        }

        auto decision = std::make_shared<StarEliminationDecision>(merged_op_stra, merged_node_cost, succ_edges_costs,
                                                                  succ_nodes_stras, succ_nodes_costs);
        auto new_cost = std::make_shared<Cost>(computation_cost, commu_cost, decision);
        new_cost->communication_without_parameter_ = commu_without;
        new_cost->communication_with_partial_para_ = commu_without + COST_MODEL_GAMMA * (commu_cost - commu_without);
        new_cost->memory_with_reuse_ = memory_cost;
        new_cost->communication_forward_ = commu_forward;
        first_succ_node_clist_new->emplace_back(std::move(new_cost));
      }
    }
  }
}

void CostGraph::CreateStarEliminationCostList(std::vector<std::shared_ptr<Edge>> &succ_edges,
                                              const StrategyPtr &first_succ_node_stra,
                                              const CostPtrList &first_succ_node_clist,
                                              const CostPtrList &first_succ_edge_clist,
                                              const StrategyPtr &merged_op_stra, const CostPtrList &merged_op_clist,
                                              CostPtrList *first_succ_node_clist_new) {
  std::vector<StrategyPtr> succ_nodes_stras(succ_edges.size(), nullptr);
  CostPtrList succ_edges_costs(succ_edges.size(), nullptr), succ_nodes_costs(succ_edges.size(), nullptr);
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

std::vector<std::shared_ptr<Edge>> CostGraph::EliminationStar(const OperatorInfoPtr &merged_op) {
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
    MS_LOG(EXCEPTION) << "Eliminating star centered at: " << merged_op->name() << " failed.";
  }

  merged_op->SetNotAlive();
  MS_LOG(INFO) << "Eliminating star centered at: " << merged_op->name() << " succeeded.";
  return succ_edges;
}

Status CostGraph::InitSelectedStrategy() {
  for (auto &op : ops_) {
    MS_EXCEPTION_IF_NULL(op);
    auto result = op->InitSelectedStrategy(op->selected_strategy());
    if (result != SUCCESS) {
      return result;
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

Status CostGraph::CalculateEdgesMemoryCost() {
  for (auto &edge_pair : edges_) {
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

OperatorInfoPtr CostGraph::FindTmpIdentityByParameterName(std::string &p_name) const {
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
        std::map<size_t, int> output_count;
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
}  // namespace parallel
}  // namespace mindspore
