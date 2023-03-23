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

#include "frontend/parallel/ops_info/kldiv_loss_info.h"

#include <utility>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
Status KLDivLossInfo::GetAttrs() {
  reduction_ = GetStringAttr(REDUCTION);
  return SUCCESS;
}

Status KLDivLossInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    return FAILED;
  }

  auto strategies = strategy->GetInputDim();
  if (strategies[0] != strategies[1]) {
    MS_LOG(ERROR) << name_ << ": The strategy of 'logits' and 'labels' must be the same, but got strategy "
                  << StrategyToString(strategies);
    return FAILED;
  }
  batch_split_num_ = LongToSize(strategies[0][0]);
  return SUCCESS;
}

Status KLDivLossInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();
  auto strategies = strategy_->GetInputDim();
  auto logits_strategy = strategies.at(0);
  dev_matrix_shape_ = logits_strategy;

  MS_LOG(INFO) << name_ << ": dev matrix is " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status KLDivLossInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  TensorMap sub_tensor_map;
  auto strategies = strategy_->GetInputDim();
  auto logits_strategy = strategies.at(0);
  size_t size = logits_strategy.size();
  for (size_t i = 0; i < size; ++i) {
    sub_tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(sub_tensor_map);
  inputs_tensor_map_.push_back(sub_tensor_map);

  if (reduction_ == ATTR_NONE) {
    (void)outputs_tensor_map_.emplace_back(sub_tensor_map);
  } else {
    (void)outputs_tensor_map_.emplace_back(TensorMap());
  }
  return SUCCESS;
}

std::vector<StrategyPtr> KLDivLossInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape splittable_input(inputs_shape_[0].size());
  std::iota(splittable_input.begin(), splittable_input.end(), 1);
  Shapes splittable_inputs(inputs_shape_.size(), splittable_input);
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for dependent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy.";
  }
  return sp_vector;
}

Status KLDivLossInfo::InferGroup() {
  Shape group_create_map;
  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      group_create_map.push_back(0);
    } else {
      group_create_map.push_back(SizeToLong(dev_matrix_shape_.size()) - 1);
    }
  }

  if (CreateGroupByTensorMap(group_create_map, &group_list_) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status KLDivLossInfo::InferForwardCommunication() {
  if (reduction_ == ATTR_NONE) {
    MS_LOG(DEBUG) << name_ << ": reduction is " << reduction_ << ", there is no need to append reduce op.";
    return SUCCESS;
  }
  if (InferGroup() != SUCCESS) {
    MS_LOG(ERROR) << "Infer group failed";
    return FAILED;
  }
  if (group_list_.empty()) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required";
    return SUCCESS;
  }

  forward_op_.clear();
  if (reduction_ == MEAN) {
    auto element_type = outputs_dtype_->cast<mindspore::TensorTypePtr>()->element();
    forward_op_ = CreateReduceMeanForwardOp(group_list_, element_type);
  } else {
    (void)forward_op_.emplace_back(CreateAllReduceOp(REDUCE_OP_SUM, group_list_[0].name()));
    if (reduction_ == BATCH_MEAN) {
      // Divided by the number of devices in the Batch dimension
      auto element_type = outputs_dtype_->cast<mindspore::TensorTypePtr>()->element();
      (void)forward_op_.emplace_back(CreateDivOpWithType(SizeToFloat(batch_split_num_), element_type));
    }
  }
  MS_LOG(INFO) << name_ << ": The group name of forward all reduce is " << group_list_[0].name();

  return SUCCESS;
}

void KLDivLossInfo::ReComputeBatchSplitFlagList() { split_flag_list_.assign(inputs_shape_.size(), true); }

REGISTER(KLDivLossInfo);
}  // namespace parallel
}  // namespace mindspore
