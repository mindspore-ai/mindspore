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

#include "frontend/parallel/auto_parallel/edge_costmodel.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <utility>
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/ops_info/reshape_info.h"

namespace mindspore {
namespace parallel {
Status Edge::InitEdgeCost() {
  bool has_available_cost = false;
  pre_op_output_.clear();
  next_op_input_.clear();
  cost_map_.clear();

  for (auto &swc : prev_op_->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(swc);
    (void)pre_op_output_.emplace_back(std::make_pair(swc->strategy_ptr, swc->outputs_ptr));
  }
  for (auto &swc : next_op_->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(swc);
    (void)next_op_input_.emplace_back(std::make_pair(swc->strategy_ptr, swc->inputs_ptr));
  }
  if (is_identity_edge) {
    for (auto &target_output : pre_op_output_) {
      auto target_output_lyt = target_output.second[prev_op_output_index_].tensor_layout();
      auto target_output_str = target_output.first;
      for (auto &target_input : next_op_input_) {
        auto target_input_lyt = target_input.second[next_op_input_index_].tensor_layout();
        auto target_input_str = target_input.first;
        // for identity_info ops, no need to compare device_matrix
        if ((target_output_lyt == target_input_lyt) || (target_output_lyt.IsSameWithoutSplit(target_input_lyt) &&
                                                        edge_name().find(IDENTITY_INFO) != std::string::npos)) {
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
    for (auto &target_output : pre_op_output_) {
      auto target_output_lyt = target_output.second[prev_op_output_index_].tensor_layout();
      auto target_output_str = target_output.first;
      auto type_length = prev_op_->GetOutputTypeLengths()[prev_op_output_index_];
      auto type = prev_op_->outputs_type()[prev_op_output_index_];
      for (auto &target_input : next_op_input_) {
        auto target_input_lyt = target_input.second[next_op_input_index_].tensor_layout();
        auto target_input_str = target_input.first;
        CostPtr cost;
        if (GetRedistributionCost(target_output_lyt, target_input_lyt, type_length, type, &cost) != SUCCESS) {
          MS_LOG(EXCEPTION) << "Failure: redistribution cost calculation failed";
        }
        MS_EXCEPTION_IF_NULL(cost);
        MS_LOG(DEBUG) << "The redistribution cost: computation_cost: " << cost->computation_cost_
                      << ", communication_cost: " << cost->communication_cost_
                      << ", communication_without_parameter_: " << cost->communication_without_parameter_
                      << ", communication_with_partial_para_: " << cost->communication_with_partial_para_ << ".";
        // refine communication cost calculation for practice
        RefineForPracticalCost(cost, true);
        cost->communication_forward_ = cost->communication_redis_forward_;
        CostPtrKey ck = {target_output_str, target_input_str};
        CostPtrList cl;
        cl.push_back(cost);
        (void)cost_map_.emplace(std::make_pair(ck, cl));
        has_available_cost = true;
      }
    }
  }
  if (!has_available_cost) {
    const auto fully_use = CostModelContext::GetInstance()->fully_use_device();
    const auto stra_follow = CostModelContext::GetInstance()->elementwise_stra_follow();
    if (fully_use) {
      MS_LOG(EXCEPTION) << "Generating cost for edge: " << edge_name_
                        << " failed, it may be caused by setting 'fully_use_devices' true. Try to set "
                           "'fully_use_devices' false.";
    } else if (stra_follow) {
      MS_LOG(EXCEPTION) << "Generating cost for edge: " << edge_name_
                        << " failed, it may be caused by setting 'elementwise_op_strategy_follow' true. "
                           "Try to set 'elementwise_op_strategy_follow' false.";
    }
    if (edge_name_.find(RESHAPE) != std::string::npos) {
      MS_LOG(EXCEPTION) << "Generating cost for edge: " << edge_name_
                        << " failed, it may be caused by setting different strategies for operators following Reshape. "
                           "Try to fix that.";
    }
    MS_LOG(EXCEPTION) << "Generating cost for edge: " << edge_name_ << " failed.";
  }
  return Status::SUCCESS;
}

Status Edge::GetRedistributionCost(const TensorLayout &prev_op_output_layout, const TensorLayout &next_op_input_layout,
                                   size_t type_length, const TypePtr &type, CostPtr *cost) {
  MS_EXCEPTION_IF_NULL(prev_op_);
  MS_EXCEPTION_IF_NULL(cost);
  RankList dev_list = prev_op_->stage_device_list();
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
  double computation_cost = tensor_redistribution.computation_cost();
  double mem_cost = tensor_redistribution.memory_cost();
  const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();

  // Now AllGather, ReduceScatter, AlltoAll don't support bool type
  MS_EXCEPTION_IF_NULL(type);
  if ((type->type_id() == kNumberTypeBool) && (comm_cost > 0)) {
    computation_cost = INF;
    comm_cost = INF;
    MS_LOG(WARNING) << "Communication Operators don't support bool dtype!";
  }
  *cost = std::make_shared<Cost>(type_length * computation_cost, type_length * comm_cost);
  (*cost)->communication_without_parameter_ = type_length * comm_cost;
  (*cost)->communication_with_partial_para_ =
    (*cost)->communication_without_parameter_ +
    gamma * ((*cost)->communication_cost_ - (*cost)->communication_without_parameter_);
  (*cost)->communication_redis_forward_ = type_length * forward_comm_cost;
  (*cost)->communication_redis_backward_ = type_length * backward_comm_cost;
  (*cost)->memory_with_reuse_ = mem_cost;
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

CostPtrList Edge::CreateEdgeEliminationCostList(const StrategyPtr &output_st_ptr, const std::vector<EdgePtr> &edges,
                                                const StrategyPtr &input_st_ptr) const {
  std::function<CostPtrList(EdgePtr)> LocalGetCostList = [&](const EdgePtr &edge) {
    MS_EXCEPTION_IF_NULL(edge);
    return edge->GetCostList(output_st_ptr, input_st_ptr);
  };
  CostPtrList result;
  std::vector<CostPtrList> all_cost_list;
  all_cost_list.resize(edges.size());
  (void)std::transform(edges.begin(), edges.end(), all_cost_list.begin(), LocalGetCostList);

  CostPtrList selected_cost_list(all_cost_list.size(), nullptr);
  std::function<void(size_t, double, double, double, double, double)> recursive =
    [&](size_t k, double computation, double memory, double communication, double communication_without_para,
        double communication_forward) {
      const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
      if (k == edges.size()) {
        auto decision = std::make_shared<EdgeEliminationDecision>(selected_cost_list);
        CostPtr new_cost = std::make_shared<Cost>(computation, communication);
        MS_EXCEPTION_IF_NULL(new_cost);
        new_cost->communication_without_parameter_ = communication_without_para;
        new_cost->communication_with_partial_para_ =
          communication_without_para + gamma * (communication - communication_without_para);
        new_cost->memory_with_reuse_ = memory;
        new_cost->communication_forward_ = communication_forward;
        new_cost->decision_ptr_ = decision;
        result.push_back(new_cost);
        return;
      }
      for (auto &c : all_cost_list[k]) {
        MS_EXCEPTION_IF_NULL(c);
        selected_cost_list[k] = c;
        recursive(k + 1, computation + c->computation_cost_, memory + c->memory_with_reuse_,
                  communication + c->communication_cost_,
                  communication_without_para + c->communication_without_parameter_,
                  communication_forward + c->communication_forward_);
      }
    };
  recursive(0, 0.0, 0.0, 0.0, 0.0, 0.0);
  Simplify(&result);
  return result;
}

void Edge::EdgeEliminationSetNewCost(OperatorInfoPtr, const std::vector<EdgePtr> &edges, OperatorInfoPtr) {
  bool valid = false;
  for (const auto &output_pair : pre_op_output_) {
    StrategyPtr output_st_ptr = output_pair.first;
    for (const auto &input_pair : next_op_input_) {
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

void Edge::CreateOpEliminationSubCostList(StrategyPtr op_strategy, const CostPtrList &left_cost_list,
                                          const CostPtrList &middle_cost_list, const CostPtrList &right_cost_list,
                                          CostPtrList *ret_cost_list) const {
  for (auto &left_cost : left_cost_list) {
    MS_EXCEPTION_IF_NULL(left_cost);
    for (auto &middle_cost : middle_cost_list) {
      MS_EXCEPTION_IF_NULL(middle_cost);
      for (auto &right_cost : right_cost_list) {
        MS_EXCEPTION_IF_NULL(right_cost);
        double computation =
          left_cost->computation_cost_ + middle_cost->computation_cost_ + right_cost->computation_cost_;
        double communication =
          left_cost->communication_cost_ + middle_cost->communication_cost_ + right_cost->communication_cost_;
        double communication_forward =
          left_cost->communication_forward_ + middle_cost->communication_forward_ + right_cost->communication_forward_;
        double communication_without_para = left_cost->communication_without_parameter_ +
                                            middle_cost->communication_without_parameter_ +
                                            right_cost->communication_without_parameter_;
        double memory_cost =
          left_cost->memory_with_reuse_ + middle_cost->memory_with_reuse_ + right_cost->memory_with_reuse_;

        auto decision = std::make_shared<OpEliminationDecision>(op_strategy, left_cost, middle_cost, right_cost);
        auto cost = std::make_shared<Cost>(computation, communication, decision);
        const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
        MS_EXCEPTION_IF_NULL(cost);
        cost->communication_without_parameter_ = communication_without_para;
        cost->communication_with_partial_para_ =
          communication_without_para + gamma * (communication - communication_without_para);
        cost->memory_with_reuse_ = memory_cost;
        cost->communication_forward_ = communication_forward;
        (void)ret_cost_list->emplace_back(std::move(cost));
      }
    }
  }
}

CostPtrList Edge::CreateOpEliminationCostList(const EdgePtr &e1, const StrategyPtr &output_st_ptr,
                                              const OperatorInfoPtr &op, const EdgePtr &e2,
                                              const StrategyPtr &input_st_ptr) const {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(e1);
  MS_EXCEPTION_IF_NULL(e2);
  CostPtrList result;
  for (const auto &op_strategy : op->GetStrategyCost()) {
    MS_EXCEPTION_IF_NULL(op_strategy);
    auto middle_strategy = op_strategy->strategy_ptr;
    CreateOpEliminationSubCostList(middle_strategy, e1->GetCostList(output_st_ptr, middle_strategy),
                                   op_strategy->cost_list, e2->GetCostList(middle_strategy, input_st_ptr), &result);
  }
  Simplify(&result);
  return result;
}

void Edge::OpEliminationSetNewCost(const EdgePtr &e1, const OperatorInfoPtr &op, const EdgePtr &e2) {
  bool valid = false;
  for (const auto &output_pair : pre_op_output_) {
    StrategyPtr output_st_ptr = output_pair.first;
    for (const auto &input_pair : next_op_input_) {
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

Status Edge::CalculateMemoryCost() {
  if (is_output_parameter_involve_ == -1) {
    MS_LOG(ERROR) << "is_output_parameter_involve_ is unset.";
    return FAILED;
  }
  if (is_output_parameter_involve_ == 0) {
    // In this case, it is sure that the tensor redistribution along this edge is NOT parameter-involved, thus it is
    // unnecessary to keep them in memory.
    for (auto &cost_kv : cost_map_) {
      auto &cost_v = cost_kv.second;
      if (!cost_v.empty()) {
        cost_v[0]->memory_with_reuse_ = 0;
      }
    }
  }

  return SUCCESS;
}

Status Edge::CalculateMemoryCostForInference() {
  // Currently, memory cost is NOT calculated for redistribution
  if ((is_output_critical_ != 0) && (is_output_critical_ != 1)) {
    MS_LOG(ERROR) << "Failure: unexpected output critical flag value: " << is_output_critical_;
    return FAILED;
  }
  for (const auto &cost_kv : cost_map_) {
    auto &cost_v = cost_kv.second;
    if (!cost_v.empty()) {
      cost_v[0]->memory_with_reuse_ = 0;
    }
  }
  return SUCCESS;
}

CostPtr Edge::GetCostByStrategyPair(const CostPtrKey &stra_pair) {
  if (cost_map_.find(stra_pair) == cost_map_.end()) {
    return nullptr;
  }
  auto cost_vec = cost_map_[stra_pair];
  if (cost_vec.empty()) {
    PrintStrategy(stra_pair.first);
    PrintStrategy(stra_pair.second);
    MS_LOG(EXCEPTION) << "No available cost under current strategy pair of the edge: " << edge_name_;
  }
  if (cost_vec.size() > 1) {
    PrintStrategy(stra_pair.first);
    PrintStrategy(stra_pair.second);
    MS_LOG(INFO) << "Multiple costs available under the stratey pair of the edge: " << edge_name_;
  }
  return cost_vec[0];
}

StrategyPtr Edge::GetNextOpStrategyByPrevOpStrategyWithMiniComm(const StrategyPtr &prev_op_stra) {
  std::vector<std::pair<StrategyPtr, double>> next_op_stras;
  // First, try to find the strategy with zero communication cost.
  for (const auto &key_value : cost_map_) {
    const auto &candidate_prev_op_stra = key_value.first.first;
    if (prev_op_stra->IsEqual(candidate_prev_op_stra) && (key_value.second[0]->communication_cost_ < EPS)) {
      (void)next_op_stras.emplace_back(key_value.first.second, key_value.second[0]->computation_cost_);
    }
  }
  if (next_op_stras.empty()) {
    // Second, if there is not strategy with zero communication cost, find the one with minimum communication cost.
    std::vector<std::pair<StrategyPtr, double>> next_stras;
    for (auto &key_value : cost_map_) {
      const auto &candidate_prev_op_stra = key_value.first.first;
      if (prev_op_stra->IsEqual(candidate_prev_op_stra)) {
        (void)next_stras.emplace_back(key_value.first.second, key_value.second[0]->communication_cost_);
      }
    }
    if (next_stras.empty()) {
      MS_LOG(ERROR) << "There are no available strategy for zero communication cost for edge: " << edge_name_;
      return nullptr;
    }
    MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge_name();
    std::sort(next_stras.begin(), next_stras.end(),
              [this](const std::pair<StrategyPtr, double> &a, const std::pair<StrategyPtr, double> &b) {
                return !IsDoubleEqual(a.second, b.second) ? a.second < b.second : a.first->Compare(b.first);
              });
    return next_stras[0].first;
  }
  if (next_op_stras.size() > 1) {
    MS_LOG(INFO) << "There are multiple strategies for edge: " << edge_name_
                 << " with zero communication cost, choose the one with minimum computation costs.";
  }
  auto next_op = next_op_;
  std::sort(next_op_stras.begin(), next_op_stras.end(),
            [this, &next_op](const std::pair<StrategyPtr, double> &a, const std::pair<StrategyPtr, double> &b) {
              if (!IsDoubleEqual(a.second, b.second)) {
                return a.second < b.second;
              }
              auto cost_a = next_op->GetCostByStrategyPtr(a.first)[0]->communication_without_parameter_;
              auto cost_b = next_op->GetCostByStrategyPtr(b.first)[0]->communication_without_parameter_;
              if (!IsDoubleEqual(cost_a, cost_b)) {
                return cost_a < cost_b;
              }
              return a.first->Compare(b.first);
            });
  return next_op_stras[0].first;
}

StrategyPtr Edge::GetPrevOpStrategyByNextOpStrategyWithMiniComm(const StrategyPtr &next_op_stra) {
  std::vector<std::pair<StrategyPtr, double>> prev_op_stras;
  // First, try to find the strategy with zero communication cost.
  for (const auto &key_value : cost_map_) {
    const auto &candidate_next_op_stra = key_value.first.second;
    if (next_op_stra->IsEqual(candidate_next_op_stra) && (key_value.second[0]->communication_cost_ < EPS)) {
      (void)prev_op_stras.emplace_back(key_value.first.first, key_value.second[0]->computation_cost_);
    }
  }
  if (prev_op_stras.empty()) {
    // Second, if there is no strategy with zero communication cost, find the one with minimum communication cost.
    std::vector<std::pair<StrategyPtr, double>> prev_stras;
    for (auto &key_value : cost_map_) {
      const auto &candidate_next_op_stra = key_value.first.second;
      if (next_op_stra->IsEqual(candidate_next_op_stra)) {
        (void)prev_stras.emplace_back(key_value.first.first, key_value.second[0]->communication_cost_);
      }
    }
    if (prev_stras.empty()) {
      MS_LOG(ERROR) << "There are no available strategy for zero communication cost for edge: " << edge_name_;
      return nullptr;
    }
    MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge_name();
    std::sort(prev_stras.begin(), prev_stras.end(),
              [this](const std::pair<StrategyPtr, double> &a, const std::pair<StrategyPtr, double> &b) {
                return !IsDoubleEqual(a.second, b.second) ? a.second < b.second : a.first->Compare(b.first);
              });
    return prev_stras[0].first;
  }
  if (prev_op_stras.size() > 1) {
    MS_LOG(INFO) << "There are multiple strategies for edge: " << edge_name_
                 << " with zero communication costs, choose the one with minimum computation costs.";
  }
  auto prev_op = prev_op_;
  std::sort(prev_op_stras.begin(), prev_op_stras.end(),
            [this, &prev_op](const std::pair<StrategyPtr, double> &a, const std::pair<StrategyPtr, double> &b) {
              if (!IsDoubleEqual(a.second, b.second)) {
                return a.second < b.second;
              }
              auto cost_a = prev_op->GetCostByStrategyPtr(a.first)[0]->communication_without_parameter_;
              auto cost_b = prev_op->GetCostByStrategyPtr(b.first)[0]->communication_without_parameter_;
              if (!IsDoubleEqual(cost_a, cost_b)) {
                return cost_a < cost_b;
              }
              return a.first->Compare(b.first);
            });
  return prev_op_stras[0].first;
}

int64_t Edge::GetReshapeSWCIndexByNextOpStrategy(const StrategyPtr &next_op_stra) {
  if (!prev_op_->IsReshape()) {
    MS_LOG(EXCEPTION) << "The edge: " << edge_name_ << "'s prev_op is not a Reshape.";
  }
  if (next_op_->IsReshape()) {
    MS_LOG(EXCEPTION) << "The edge: " << edge_name_ << " has two Reshapes, which is not supported currently.";
  }
  const auto &reshape_output_layout = next_op_->GetInputLayoutFromSWCByStrategy(next_op_stra, next_op_input_index_);
  MS_LOG(INFO) << prev_op_->name() << "'s output layout: " << reshape_output_layout.ToString();
  auto reshape_ptr = std::dynamic_pointer_cast<ReshapeInfo>(prev_op_);
  // First, try to find the zero communication strategy.
  auto swc_index = reshape_ptr->GetSWCIndexByOutputLayoutWithZeroComm(reshape_output_layout);
  if (swc_index == -1) {
    // Second, if there is no strategy with zero communication cost, find the strategy with minimum cost.
    swc_index = reshape_ptr->GetSWCIndexByOutputLayoutWithMiniComm(reshape_output_layout);
    if (swc_index != -1) {
      MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge_name();
    }
  }
  if (swc_index == -1) {
    MS_LOG(EXCEPTION) << "No available strategy found at edge: " << edge_name_ << " for: " << prev_op_->name();
  }
  return swc_index;
}

int64_t Edge::GetReshapeSWCIndexByPrevOpStrategy(const StrategyPtr &prev_op_stra) {
  if (!next_op_->IsReshape()) {
    MS_LOG(EXCEPTION) << "The edge: " << edge_name_ << "'s next_op is not a Reshape.";
  }
  if (prev_op_->IsReshape()) {
    MS_LOG(EXCEPTION) << "The edge: " << edge_name_ << " has two Reshapes, which is not supported currently.";
  }
  const auto &reshape_input_lyt = prev_op_->GetOutputLayoutFromSWCByStrategy(prev_op_stra, prev_op_output_index_);
  MS_LOG(INFO) << next_op_->name() << "'s input layout: " << reshape_input_lyt.ToString();
  auto reshape_ptr = std::dynamic_pointer_cast<ReshapeInfo>(next_op_);
  // First, try to find the zero communication strategy.
  auto swc_index = reshape_ptr->GetSWCIndexByInputLayoutWithZeroComm(reshape_input_lyt);
  if (swc_index == -1) {
    // Second, if there is no zero communication strategy, find the strategy with minimum cost.
    swc_index = reshape_ptr->GetSWCIndexByInputLayoutWithMiniComm(reshape_input_lyt);
    if (swc_index != -1) {
      MS_LOG(WARNING) << "Inconsistency occurred at edge: " << edge_name();
    }
  }
  if (swc_index == -1) {
    MS_LOG(EXCEPTION) << "No available strategy found at edge: " << edge_name_ << " for: " << next_op_->name();
  }
  return swc_index;
}

StrategyPtr Edge::GetPrevOpStrategyByReshapeSWCIndex(int64_t swc_index) {
  if (!next_op_->IsReshape()) {
    MS_LOG(EXCEPTION) << "The edge: " << edge_name_ << "'s next_op is not a Reshape.";
  }
  if (prev_op_->IsReshape()) {
    MS_LOG(EXCEPTION) << "The edge: " << edge_name_ << " has two Reshapes, which is not supported currently.";
  }
  auto reshape_ptr = std::dynamic_pointer_cast<ReshapeInfo>(next_op_);
  const auto &reshape_input_lyt = reshape_ptr->GetInputLayoutBySWCIndex(swc_index);
  auto stra = prev_op_->GetStrategyFromSWCByOutputLayout(reshape_input_lyt, prev_op_output_index_);
  if (stra == nullptr) {
    MS_LOG(EXCEPTION) << "No available strategy found at edge: " << edge_name_ << " for: " << prev_op_->name();
  }
  return stra;
}

StrategyPtr Edge::GetNextOpStrategyByReshapeSWCIndex(int64_t swc_index) {
  if (!prev_op_->IsReshape()) {
    MS_LOG(EXCEPTION) << "The edge: " << edge_name_ << "'s next_op is not a Reshape.";
  }
  if (next_op_->IsReshape()) {
    MS_LOG(EXCEPTION) << "The edge: " << edge_name_ << " has two Reshapes, which is not supported currently.";
  }
  auto reshape_ptr = std::dynamic_pointer_cast<ReshapeInfo>(prev_op_);
  const auto &reshape_output_lyt = reshape_ptr->GetOutputLayoutBySWCIndex(swc_index);
  auto stra = next_op_->GetStrategyFromSWCByInputLayout(reshape_output_lyt, next_op_input_index_);
  if (stra == nullptr) {
    MS_LOG(EXCEPTION) << "No available strategy found at edge: " << edge_name_ << " for: " << prev_op_->name();
  }
  return stra;
}

bool Edge::CheckStrategyConsistency(StrategyPtr prev_stra, StrategyPtr next_stra) {
  if (prev_stra == nullptr) {
    MS_LOG(EXCEPTION) << prev_op_->name() << "'s selected strategy is null!";
  }
  if (next_stra == nullptr) {
    MS_LOG(EXCEPTION) << next_op_->name() << "'s selected strategy is null!";
  }
  auto cost = GetCostByStrategyPair({prev_stra, next_stra});
  if (cost == nullptr || cost->communication_cost_ > 0.0) {
    MS_LOG(INFO) << "The edge " << edge_name_ << "'s strategy: ";
    PrintStrategy(prev_stra);
    PrintStrategy(next_stra);
    if (prev_op_->IsTmpIdentity()) {
      MS_LOG(ERROR) << "The parameter: " << prev_op_->refkey_parameter_name()
                    << " has been used by operators with "
                       "different sharding strategies. These operators are: ";
      auto const &succ_edges = prev_op_->succ_edges();
      for (auto const &succ_edge : succ_edges) {
        if (succ_edge->next_operator()->cnodes().empty()) {
          MS_LOG(EXCEPTION) << "No CNODE info has been set in operator: " << succ_edge->next_operator()->name();
        }
        MS_LOG(ERROR) << succ_edge->next_operator()->name() << ", the corresponding fullname is: "
                      << succ_edge->next_operator()->cnodes()[0]->fullname_with_scope();
      }
      MS_LOG(EXCEPTION) << "Configure these operators with consistent sharding strategies.";
    }
    MS_LOG(WARNING) << "There are redistribution cost occurs at edge: " << edge_name() << ".";
    return false;
  }
  return true;
}

void Edge::SetCostMapAndInputOutput(const std::map<CostPtrKey, CostPtrList> &cost_map) {
  cost_map_ = cost_map;
  pre_op_output_.clear();
  next_op_input_.clear();

  for (const auto &key_value : cost_map_) {
    auto &key_pair = key_value.first;
    (void)pre_op_output_.emplace_back(std::pair<StrategyPtr, std::vector<TensorInfo>>(key_pair.first, {}));
    (void)next_op_input_.emplace_back(std::pair<StrategyPtr, std::vector<TensorInfo>>(key_pair.second, {}));
  }
}

// Return true if there are available strategies in this edge.
bool Edge::CheckStrategyCostPossibility() const { return !cost_map_.empty(); }
}  // namespace parallel
}  // namespace mindspore
