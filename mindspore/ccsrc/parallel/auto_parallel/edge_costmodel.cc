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

#include "parallel/auto_parallel/edge_costmodel.h"

#include <algorithm>
#include <functional>
#include <utility>
#include <iterator>
#include "parallel/auto_parallel/costmodel.h"
#include "parallel/tensor_layout/tensor_redistribution.h"
#include "parallel/auto_parallel/graph_costmodel.h"

namespace mindspore {
namespace parallel {
Status Edge::InitEdgeCost() {
  bool has_available_cost = false;
  for (auto& swc : prev_op_->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(swc);
    pre_op_output_.emplace_back(std::make_pair(swc->strategy_ptr, swc->outputs_ptr));
  }
  for (auto& swc : next_op_->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(swc);
    next_op_input_.emplace_back(std::make_pair(swc->strategy_ptr, swc->inputs_ptr));
  }
  if (is_identity_edge) {
    for (auto& target_output : pre_op_output_) {
      auto target_output_lyt = target_output.second[prev_op_output_index_].tensor_layout();
      auto target_output_str = target_output.first;
      for (auto& target_input : next_op_input_) {
        auto target_input_lyt = target_input.second[next_op_input_index_].tensor_layout();
        auto target_input_str = target_input.first;
        if (target_output_lyt == target_input_lyt) {
          CostPtrKey ck = {target_output_str, target_input_str};
          CostPtr cost = std::make_shared<Cost>(0.0, 0.0);
          MS_EXCEPTION_IF_NULL(cost);
          cost->communication_without_parameter_ = 0.0;
          cost->communication_with_partial_para_ = 0.0;
          CostPtrList cl;
          cl.push_back(cost);
          (void)cost_map_.emplace(std::make_pair(ck, cl));
          has_available_cost = true;
        }
      }
    }
  } else {
    for (auto& target_output : pre_op_output_) {
      auto target_output_lyt = target_output.second[prev_op_output_index_].tensor_layout();
      auto target_output_str = target_output.first;
      auto type_length = prev_op_->GetOutputTypeLengths()[prev_op_output_index_];
      for (auto& target_input : next_op_input_) {
        auto target_input_lyt = target_input.second[next_op_input_index_].tensor_layout();
        auto target_input_str = target_input.first;
        CostPtr cost;
        if (GetRedistributionCost(target_output_lyt, target_input_lyt, type_length, &cost) != SUCCESS) {
          MS_LOG(EXCEPTION) << "Failure: redistribution cost calculation failed";
        }
        MS_EXCEPTION_IF_NULL(cost);
        MS_LOG(DEBUG) << "The redistribution cost: memory_cost: " << cost->memory_cost_
                      << ", communication_cost: " << cost->communication_cost_
                      << ", communication_without_parameter_: " << cost->communication_without_parameter_
                      << ", communication_with_partial_para_: " << cost->communication_with_partial_para_ << ".";
        // refine communication cost calculation for practice
        RefineForPracticalCost(cost, true);
        CostPtrKey ck = {target_output_str, target_input_str};
        CostPtrList cl;
        cl.push_back(cost);
        (void)cost_map_.emplace(std::make_pair(ck, cl));
        has_available_cost = true;
      }
    }
  }
  if (!has_available_cost) {
    if (!NOT_FULLY_USE_DEVICES) {
      MS_LOG(EXCEPTION) << "Generating cost for edge: " << edge_name_
                        << " failed, it may be caused by setting 'not_fully_use_devices' false. Try to set "
                           "'not_fully_use_devices' true.";
    } else if (ELEMENTWISE_OP_STRA_FOLLOW) {
      MS_LOG(EXCEPTION) << "Generating cost for edge: " << edge_name_
                        << " failed, it may be caused by setting 'elementwise_op_strategy_follow' true. "
                           "Try to set 'elementwise_op_strategy_follow' false.";
    }
    MS_LOG(EXCEPTION) << "Generating cost for edge: " << edge_name_ << " failed.";
  }
  return Status::SUCCESS;
}

Status Edge::GetRedistributionCost(const TensorLayout& prev_op_output_layout, const TensorLayout& next_op_input_layout,
                                   size_t type_length, CostPtr* cost) {
  MS_EXCEPTION_IF_NULL(prev_op_);
  MS_EXCEPTION_IF_NULL(cost);
  RankList dev_list = prev_op_->global_device_list();
  TensorRedistribution tensor_redistribution(false);

  // Init TensorRedistribution
  if (tensor_redistribution.Init(prev_op_output_layout, next_op_input_layout, dev_list) == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution init failed.";
  }

  if (tensor_redistribution.ComputeCost() == FAILED) {
    MS_LOG(EXCEPTION) << "Failure: tensor_redistribution ComputeCost failed.";
  }

  double comm_cost = tensor_redistribution.comm_cost();
  double forward_comm_cost = tensor_redistribution.forward_comm_cost();
  double backward_comm_cost = tensor_redistribution.backward_comm_cost();
  double mem_cost = tensor_redistribution.mem_cost();

  *cost = std::make_shared<Cost>(type_length * mem_cost, type_length * comm_cost);
  (*cost)->communication_without_parameter_ = type_length * comm_cost;
  (*cost)->communication_with_partial_para_ =
    (*cost)->communication_without_parameter_ +
    COST_MODEL_GAMMA * ((*cost)->communication_cost_ - (*cost)->communication_without_parameter_);
  (*cost)->communication_redis_forward_ = type_length * forward_comm_cost;
  (*cost)->communication_redis_backward_ = type_length * backward_comm_cost;
  return Status::SUCCESS;
}

CostPtrList Edge::GetCostList(StrategyPtr output_str, StrategyPtr input_str) {
  CostPtrKey ck = {output_str, input_str};
  CostPtrList result;
  if (cost_map_.find(ck) != cost_map_.end()) {
    return cost_map_.at(ck);
  }
  return result;
}

CostPtrList Edge::CreateEdgeEliminationCostList(const StrategyPtr& output_st_ptr, const std::vector<EdgePtr>& edges,
                                                const StrategyPtr& input_st_ptr) {
  std::function<CostPtrList(EdgePtr)> LocalGetCostList = [&](const EdgePtr& edge) {
    MS_EXCEPTION_IF_NULL(edge);
    return edge->GetCostList(output_st_ptr, input_st_ptr);
  };
  CostPtrList result;
  std::vector<CostPtrList> all_cost_list;
  all_cost_list.resize(edges.size());
  (void)std::transform(edges.begin(), edges.end(), all_cost_list.begin(), LocalGetCostList);

  CostPtrList selected_cost_list(all_cost_list.size(), nullptr);
  std::function<void(size_t, double, double, double)> recursive = [&](size_t k, double memory, double communication,
                                                                      double communication_without_para) {
    if (k == edges.size()) {
      auto decision = std::make_shared<EdgeEliminationDecision>(selected_cost_list);
      CostPtr new_cost = std::make_shared<Cost>(memory, communication);
      MS_EXCEPTION_IF_NULL(new_cost);
      new_cost->communication_without_parameter_ = communication_without_para;
      new_cost->communication_with_partial_para_ =
        communication_without_para + COST_MODEL_GAMMA * (communication - communication_without_para);
      new_cost->decision_ptr_ = decision;
      result.push_back(new_cost);
      return;
    }
    for (auto& c : all_cost_list[k]) {
      MS_EXCEPTION_IF_NULL(c);
      selected_cost_list[k] = c;
      recursive(k + 1, memory + c->memory_cost_, communication + c->communication_cost_,
                communication_without_para + c->communication_without_parameter_);
    }
  };
  recursive(0, 0, 0, 0);
  SimplifyForDreasingCommunicationWithPartialPara(&result);
  return result;
}

void Edge::EdgeEliminationSetNewCost(OperatorInfoPtr, const std::vector<EdgePtr>& edges, OperatorInfoPtr) {
  bool valid = false;
  for (const auto& output_pair : pre_op_output_) {
    StrategyPtr output_st_ptr = output_pair.first;
    for (const auto& input_pair : next_op_input_) {
      StrategyPtr input_st_ptr = input_pair.first;
      CostPtrList clist = CreateEdgeEliminationCostList(output_st_ptr, edges, input_st_ptr);
      CostPtrKey key = {output_st_ptr, input_st_ptr};
      cost_map_[key] = clist;
      if ((!valid) && (!clist.empty())) {
        valid = true;
      }
    }
  }
  if (!valid) {
    MS_LOG(EXCEPTION) << "Creating edge: " << edge_name_ << " failed.";
  }
}

void Edge::CreateOpEliminationSubCostList(StrategyPtr op_strategy, const CostPtrList& left_cost_list,
                                          const CostPtrList& middle_cost_list, const CostPtrList& right_cost_list,
                                          CostPtrList* ret_cost_list) {
  for (auto& left_cost : left_cost_list) {
    MS_EXCEPTION_IF_NULL(left_cost);
    for (auto& middle_cost : middle_cost_list) {
      MS_EXCEPTION_IF_NULL(middle_cost);
      for (auto& right_cost : right_cost_list) {
        MS_EXCEPTION_IF_NULL(right_cost);
        double memory = left_cost->memory_cost_ + middle_cost->memory_cost_ + right_cost->memory_cost_;
        double communication =
          left_cost->communication_cost_ + middle_cost->communication_cost_ + right_cost->communication_cost_;
        double communication_without_para = left_cost->communication_without_parameter_ +
                                            middle_cost->communication_without_parameter_ +
                                            right_cost->communication_without_parameter_;

        auto decision = std::make_shared<OpEliminationDecision>(op_strategy, left_cost, middle_cost, right_cost);
        auto cost = std::make_shared<Cost>(memory, communication, decision);
        MS_EXCEPTION_IF_NULL(cost);
        cost->communication_without_parameter_ = communication_without_para;
        cost->communication_with_partial_para_ =
          communication_without_para + COST_MODEL_GAMMA * (communication - communication_without_para);
        ret_cost_list->emplace_back(std::move(cost));
      }
    }
  }
}

CostPtrList Edge::CreateOpEliminationCostList(const EdgePtr& e1, const StrategyPtr& output_st_ptr,
                                              const OperatorInfoPtr& op, const EdgePtr& e2,
                                              const StrategyPtr& input_st_ptr) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(e1);
  MS_EXCEPTION_IF_NULL(e2);
  CostPtrList result;
  for (const auto& op_strategy : op->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(op_strategy);
    auto middle_strategy = op_strategy->strategy_ptr;
    CreateOpEliminationSubCostList(middle_strategy, e1->GetCostList(output_st_ptr, middle_strategy),
                                   op_strategy->cost_list, e2->GetCostList(middle_strategy, input_st_ptr), &result);
  }
  SimplifyForDreasingCommunicationWithPartialPara(&result);
  return result;
}

void Edge::OpEliminationSetNewCost(const EdgePtr& e1, const OperatorInfoPtr& op, const EdgePtr& e2) {
  bool valid = false;
  for (const auto& output_pair : pre_op_output_) {
    StrategyPtr output_st_ptr = output_pair.first;
    for (const auto& input_pair : next_op_input_) {
      StrategyPtr input_st_ptr = input_pair.first;

      CostPtrList clist = CreateOpEliminationCostList(e1, output_st_ptr, op, e2, input_st_ptr);
      CostPtrKey key = {output_st_ptr, input_st_ptr};
      cost_map_[key] = clist;
      if ((!valid) && (!clist.empty())) {
        valid = true;
      }
    }
  }
  if (!valid) {
    MS_LOG(EXCEPTION) << "Creating edge: " << edge_name_ << " failed.";
  }
}
}  // namespace parallel
}  // namespace mindspore
