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

#include "parallel/ops_info/layer_norm_info.h"
#include <algorithm>
#include <vector>
#include "parallel/device_matrix.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
Status LayerNormInfo::GetAttrs() {
  auto iter = attrs_.find(BEGIN_NORM_AXIS);
  if (iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attr of begin norm axis";
    return FAILED;
  }
  if ((iter->second == nullptr) || !iter->second->isa<Int32Imm>()) {
    MS_LOG(ERROR) << name_ << ": The axis type is not int";
    return FAILED;
  }

  int32_t dim = SizeToInt(input_shape_.size());
  auto axis = GetValue<int32_t>(iter->second);
  if ((axis >= dim) || (axis < -dim)) {
    MS_LOG(ERROR) << name_ << ": The axis(" << axis << ") is out of range[" << -dim << ", " << dim - 1 << "]";
    return FAILED;
  }

  if (axis < 0) {
    axis = axis + dim;
  }
  begin_norm_axis_ = IntToSize(axis);
  return SUCCESS;
}

Status LayerNormInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != LAYER_NORM_INPUT_SIZE) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy size " << stra.size();
    return FAILED;
  }

  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy value";
    return FAILED;
  }

  Dimensions input_strategy = stra[LAYER_NORM_INPUT_INDEX];
  Dimensions gamma_strategy = stra[LAYER_NORM_GAMMA_INDEX];
  Dimensions beta_strategy = stra[LAYER_NORM_BETA_INDEX];
  if (begin_norm_axis_ >= input_strategy.size()) {
    MS_LOG(ERROR) << name_ << ": Invalid begin norm axis " << begin_norm_axis_;
    return FAILED;
  }
  // check input strategy
  for (size_t i = begin_norm_axis_; i < input_strategy.size(); ++i) {
    if (input_strategy[begin_norm_axis_] != NO_SPLIT_STRATEGY) {
      MS_LOG(ERROR) << name_ << ": Invalid input strategy " << ShapeToString(input_strategy);
      return FAILED;
    }
  }

  // check gamma and beta strategy
  if ((gamma_strategy.size() > input_strategy.size()) || (beta_strategy.size() > input_strategy.size())) {
    MS_LOG(ERROR) << name_ << " : The strategy size of gamma or beta is lager than input strategy";
    return FAILED;
  }

  size_t gamma_diff = input_strategy.size() - gamma_strategy.size();
  for (size_t j = 0; j < gamma_strategy.size(); ++j) {
    if (gamma_strategy[j] != input_strategy[gamma_diff + j]) {
      MS_LOG(ERROR) << name_ << ": Invalid gamma strategy " << ShapeToString(gamma_strategy);
      return FAILED;
    }
  }

  size_t beta_diff = input_strategy.size() - beta_strategy.size();
  for (size_t k = 0; k < beta_strategy.size(); ++k) {
    if (beta_strategy[k] != input_strategy[beta_diff + k]) {
      MS_LOG(ERROR) << name_ << ": Invalid beta strategy " << ShapeToString(beta_strategy);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status LayerNormInfo::InferDevMatrixShape() {
  if (strategy_ == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null";
    return FAILED;
  }
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }
  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status LayerNormInfo::CreateTensorMap(size_t input_index) {
  if (inputs_shape_.size() <= input_index) {
    MS_LOG(ERROR) << name_ << ": Invalid index" << input_index;
    return FAILED;
  }
  Shape shape = inputs_shape_[input_index];
  Shape tensor_map;
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor_map.push_back(SizeToInt(shape.size() - i - 1));
  }
  inputs_tensor_map_.push_back(tensor_map);
  outputs_tensor_map_.push_back(tensor_map);
  return SUCCESS;
}

Status LayerNormInfo::InferTensorMap() {
  if ((CreateTensorMap(LAYER_NORM_INPUT_INDEX) != SUCCESS) || (CreateTensorMap(LAYER_NORM_GAMMA_INDEX) != SUCCESS) ||
      (CreateTensorMap(LAYER_NORM_BETA_INDEX) != SUCCESS)) {
    MS_LOG(ERROR) << name_ << ": Create tensor map failed";
    return FAILED;
  }
  return SUCCESS;
}

Status LayerNormInfo::CreateMirrorOp(size_t input_index) {
  if (inputs_tensor_map_.size() <= input_index) {
    MS_LOG(ERROR) << name_ << ": Invalid index " << input_index;
    return FAILED;
  }
  Shape tensor_map = inputs_tensor_map_[input_index];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(tensor_map, &group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create group for input " << input_index << " failed";
    return FAILED;
  }
  OperatorVector mirror_op;
  if (!group.empty()) {
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    MS_LOG(INFO) << name_ << " : Create the mirror ops for input " << input_index << " success, group is "
                 << group[0].name();
  }
  mirror_ops_.push_back(mirror_op);
  return SUCCESS;
}

Status LayerNormInfo::InferMirrorOps() {
  if ((CreateMirrorOp(LAYER_NORM_INPUT_INDEX) != SUCCESS) || (CreateMirrorOp(LAYER_NORM_GAMMA_INDEX) != SUCCESS) ||
      (CreateMirrorOp(LAYER_NORM_BETA_INDEX) != SUCCESS)) {
    MS_LOG(ERROR) << name_ << ": Create mirror op failed";
    return FAILED;
  }
  return SUCCESS;
}

Status LayerNormInfo::CreateTensorInfo(size_t input_index) {
  if ((inputs_shape_.size() <= input_index) || (inputs_tensor_map_.size() <= input_index)) {
    MS_LOG(ERROR) << name_ << ": Invalid input index" << input_index;
    return FAILED;
  }
  Shape tensor_map = inputs_tensor_map_[input_index];
  Shape shape = inputs_shape_[input_index];
  TensorLayout tensor_layout;
  if (tensor_layout.InitFromVector(dev_matrix_shape_, tensor_map, shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init tensor layout for input " << input_index << " failed";
    return FAILED;
  }

  TensorInfo tensor_info(tensor_layout);
  inputs_tensor_info_.push_back(tensor_info);
  outputs_tensor_info_.push_back(tensor_info);
  return SUCCESS;
}

Status LayerNormInfo::InferTensorInfo() {
  if ((CreateTensorInfo(LAYER_NORM_INPUT_INDEX) != SUCCESS) || (CreateTensorInfo(LAYER_NORM_GAMMA_INDEX) != SUCCESS) ||
      (CreateTensorInfo(LAYER_NORM_BETA_INDEX) != SUCCESS)) {
    MS_LOG(ERROR) << name_ << ": Create tensor info failed";
    return FAILED;
  }
  return SUCCESS;
}

Status LayerNormInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != LAYER_NORM_INPUT_SIZE) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map " << outputs_tensor_map_.size() << " is error";
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

Status LayerNormInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Set cost failed";
    return FAILED;
  }
  return SUCCESS;
}

Status LayerNormInfo::GenerateGammaAndBetaStrategies(const std::vector<StrategyPtr> &sp_vector) {
  if ((gamma_shape_.size() > input_shape_.size()) || (beta_shape_.size() > input_shape_.size())) {
    MS_LOG(ERROR) << name_ << ": The dimension of gamma or beta is lager than input";
    return FAILED;
  }

  size_t gamma_diff = input_shape_.size() - gamma_shape_.size();
  size_t beta_diff = input_shape_.size() - beta_shape_.size();
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(ERROR) << name_ << ": Invalid strategy";
      return FAILED;
    }
    std::vector<Dimensions> tmp_strategy;
    Dimensions input_strategy = sp->GetInputDim()[0];
    Dimensions gamma_strategy = input_strategy;
    (void)gamma_strategy.erase(gamma_strategy.begin(),
                               gamma_strategy.begin() + static_cast<different_type>(gamma_diff));
    Dimensions beta_strategy = input_strategy;
    (void)beta_strategy.erase(beta_strategy.begin(), beta_strategy.begin() + static_cast<different_type>(beta_diff));

    // reset the strategy
    tmp_strategy.push_back(input_strategy);
    tmp_strategy.push_back(gamma_strategy);
    tmp_strategy.push_back(beta_strategy);
    sp->ResetInputs(tmp_strategy);
  }
  return SUCCESS;
}

Status LayerNormInfo::GenerateStrategies(int32_t stage_id) {
  if (InitShapes() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init shapes failed";
    return FAILED;
  }
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Get attrs failed";
    return FAILED;
  }
  Shape input_split(input_shape_.size(), SPLIT_FLAG);
  if (begin_norm_axis_ >= input_split.size()) {
    MS_LOG(ERROR) << name_ << ": Invalid begin norm axis " << begin_norm_axis_;
    return FAILED;
  }

  // Can not split the dimensions from begin norm axis
  for (size_t i = begin_norm_axis_; i < input_split.size(); ++i) {
    input_split[i] = NO_SPLIT_FLAG;
  }

  // Generate strategy for input
  Shapes splittable_inputs = {input_split};
  Shapes tmp_inputs_shape = {input_shape_};
  std::vector<StrategyPtr> sp_vector;
  is_auto_parallel_ = true;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Generate input strategy failed";
    return FAILED;
  }

  // Generate the strategies for gamma and beta
  if (GenerateGammaAndBetaStrategies(sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Generate gamma and beta strategies failed";
    return FAILED;
  }

  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(DEBUG) << name_ << ": Successfully generated " << success << " strategy";
    }
  }
  return SUCCESS;
}

Status LayerNormInfo::InitShapes() {
  if (inputs_shape_.size() != LAYER_NORM_INPUT_SIZE) {
    MS_LOG(ERROR) << name_ << ": Invalid inputs size";
    return FAILED;
  }
  input_shape_ = inputs_shape_[LAYER_NORM_INPUT_INDEX];
  gamma_shape_ = inputs_shape_[LAYER_NORM_GAMMA_INDEX];
  beta_shape_ = inputs_shape_[LAYER_NORM_BETA_INDEX];
  return SUCCESS;
}

Status LayerNormInfo::Init(const StrategyPtr &strategy) {
  if ((InitShapes() != SUCCESS) || (InitWithAutoRepeatCalc(strategy)) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success";
  return SUCCESS;
}

Status LayerNormInfo::InitForCostModel(const StrategyPtr &strategy) {
  if ((InitShapes() != SUCCESS) || (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS)) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
