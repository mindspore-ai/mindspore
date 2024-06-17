/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/group_norm_info.h"

#include <memory>
#include <vector>
#include "utils/ms_context.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
constexpr char NUM_GROUPS[] = "num_groups";
constexpr size_t INPUT_INDEX = 0;
constexpr size_t GAMMA_INDEX = 1;
constexpr size_t BETA_INDEX = 2;
constexpr size_t BATCH_DIM_INDEX = 0;
constexpr size_t NUM_GROUPS_OFFSET = 1;
constexpr size_t EPS_OFFSET = 4;
constexpr size_t INPUT_CNT = 3;
constexpr size_t OUTPUT_CNT = 3;
Status GroupNormInfo::GetAttrs() {
  // input: tensor, num_groups: int, weight_opt: tensor, bias_opt: tensor, eps: float
  std::string op_name = GetPrimNameFromInfoName(this->name_);
  std::optional<int64_t> num_groups_opt_v = GetScalarValueFromInputs<int64_t>(input_value_, op_name, NUM_GROUPS);
  if (!num_groups_opt_v.has_value()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attr of num_groups.";
    return FAILED;
  }
  this->num_groups_ = num_groups_opt_v.value();
  return SUCCESS;
}

Status GroupNormInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  Strategies strategies = strategy->GetInputDim();
  if (strategies.size() != INPUT_CNT) {
    MS_LOG(ERROR) << name_ << ": strategies count must be 3.";
    return FAILED;
  }
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }
  // only support batch dim parallel.
  Dimensions input_strategy = strategies[INPUT_INDEX];
  auto no_split_validator = [](int64_t v) { return v == 1; };
  if (!std::all_of(input_strategy.begin() + 1, input_strategy.end(), no_split_validator)) {
    MS_LOG(ERROR) << name_ << ": only support batch dim parallel for now.";
    return FAILED;
  }
  for (size_t i = 1; i < strategies.size(); ++i) {
    if (!std::all_of(strategies[i].begin(), strategies[i].end(), no_split_validator)) {
      MS_LOG(ERROR) << name_ << ": only input dim can be split for now.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GroupNormInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  Strategies strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy can not be empty";
    return FAILED;
  }
  dev_matrix_shape_ = strategies[INPUT_INDEX];
  return SUCCESS;
}

Status GroupNormInfo::InferTensorMap() {
  // inputs: input(tensor), num_groups(int), weight_opt(tensor), bias_opt(tensor), eps(float)
  // input_strategy: ((n, c, h, w), scalar, (n), (n), scalar)
  // output_strategy: ((n, c, h, w), (n), (n))
  // dev_matrix: (n, c, h, w)
  TensorMap input_tensor_map(this->inputs_shape_[INPUT_INDEX].size(), MAP_NONE);
  input_tensor_map[BATCH_DIM_INDEX] = SizeToLong(this->inputs_shape_[INPUT_INDEX].size()) - 1;
  // input_tensor_map should be: batch_split_index, MAP_NONE, MAP_NONE, MAP_NONE
  TensorMap in_other_tensor_map = {MAP_NONE};

  // Has 3 tensor input: input, weight_opt(gamma), bias_opt(beta)
  inputs_tensor_map_.push_back(input_tensor_map);     // input: tensor
  inputs_tensor_map_.push_back(in_other_tensor_map);  // weight_opt: tensor
  inputs_tensor_map_.push_back(in_other_tensor_map);  // bias_opt: tensor

  TensorMap other_out_tensor_map = {input_tensor_map[BATCH_DIM_INDEX], MAP_NONE};
  // Has 3 output: out(n, c, h, w), meanOut(n, c), rstdOut(n, c)
  outputs_tensor_map_.emplace_back(input_tensor_map);
  outputs_tensor_map_.emplace_back(other_out_tensor_map);
  outputs_tensor_map_.emplace_back(other_out_tensor_map);
  return SUCCESS;
}

Status GroupNormInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }

  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    TensorLayout input_layout;
    if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[i], inputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo input_tensor_info(input_layout);
    inputs_tensor_info_.push_back(input_tensor_info);
  }

  TensorInfo scalar_input_info;
  (void)inputs_tensor_info_.insert(inputs_tensor_info_.cbegin() + NUM_GROUPS_OFFSET, scalar_input_info);
  (void)inputs_tensor_info_.insert(inputs_tensor_info_.cbegin() + EPS_OFFSET, scalar_input_info);

  // (N, C, H, W), (N, C/group_nums), (N, C/group_nums)
  for (size_t i = 0; i < outputs_tensor_map_.size(); ++i) {
    TensorLayout output_layout;
    if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[i], outputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo output_tensor_info(output_layout);
    outputs_tensor_info_.push_back(output_tensor_info);
  }
  return SUCCESS;
}

Status GroupNormInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != OUTPUT_CNT) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map must be 3, but got " << outputs_tensor_map_.size();
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

Status GroupNormInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> GroupNormInfo::GenerateOpStrategies(int64_t stage_id) {
  // Only support data parallel mode for now.
  // If batch_size % dev_count == 0, then batch_size dim is dev_count, reset of it are 1.
  // If batch_size % dev_count != 0, all dims are 1.
  Shape input_strategy(this->inputs_shape_[INPUT_INDEX].size(), 1);
  Shape gamma_strategy(this->inputs_shape_[GAMMA_INDEX].size(), 1);
  Shape beta_strategy(this->inputs_shape_[BETA_INDEX].size(), 1);

  int64_t batch_size = this->inputs_shape_[INPUT_INDEX][BATCH_DIM_INDEX];
  if (batch_size % this->stage_device_size_ == 0) {
    input_strategy[BATCH_DIM_INDEX] = this->stage_device_size_;
  }

  Strategies strategies = {input_strategy, gamma_strategy, beta_strategy};
  StrategyPtr strategy_ptr = std::make_shared<Strategy>(stage_id, strategies);
  std::vector<StrategyPtr> strategy_vec{strategy_ptr};
  return strategy_vec;
}

Status GroupNormInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferMirrorOps failed.";
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  // inputs: input(tensor), num_groups(int), weight_opt(tensor), bias_opt(tensor), eps(float)
  OperatorVector mirror_op_for_num_groups;  // num_groups
  OperatorVector mirror_op_for_eps;         // eps
  (void)mirror_ops_.insert(mirror_ops_.cbegin() + NUM_GROUPS_OFFSET, mirror_op_for_num_groups);
  (void)mirror_ops_.insert(mirror_ops_.cbegin() + EPS_OFFSET, mirror_op_for_eps);
  return SUCCESS;
}

REGISTER(GroupNormInfo);
}  // namespace parallel
}  // namespace mindspore
