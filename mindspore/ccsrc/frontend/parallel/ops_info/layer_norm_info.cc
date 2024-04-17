/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/layer_norm_info.h"

#include <algorithm>
#include <vector>
#include <utility>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
// the layernorm has three outputs
// if the shape of input is [A, B, C, D], the shape of first output is [A, B, C, D]
// if the begin-norm-axis is 0, the shape of second output is: [1, 1, 1, 1]
// if the begin-norm-axis is 1, the shape of second output is: [A, 1, 1, 1]
// if the begin-norm-axis is 2, the shape of second output is: [A, B, 1, 1]
// if the begin-norm-axis is 3, the shape of second output is: [A, B, C, 1]
// the shape of third output is the same as the shape of second output
Status LayerNormInfo::GetAttrs() {
  if (InitShapes() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init Shape failed";
    return FAILED;
  }
  std::string op_name = GetPrimNameFromInfoName(this->name_);

  std::optional<int64_t> axis_opt = GetScalarValueFromInputs<int64_t>(input_value_, op_name, BEGIN_NORM_AXIS);
  if (!axis_opt.has_value()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attr of begin_norm_axis.";
    return FAILED;
  }

  int64_t dim = SizeToLong(inputs_shape_[0].size());
  auto axis = axis_opt.value();
  if ((axis >= dim) || (axis < -dim)) {
    MS_LOG(ERROR) << name_ << ": The axis(" << axis << ") is out of range[" << (-dim) << ", " << (dim - 1) << "]";
    return FAILED;
  }

  if (axis < 0) {
    axis = axis + dim;
  }
  begin_norm_axis_ = LongToSize(axis);
  return SUCCESS;
}

Status LayerNormInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  Strategies stra = strategy->GetInputDim();
  if (stra.size() != LAYER_NORM_INPUT_SIZE) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy size " << stra.size();
    return FAILED;
  }

  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
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
    if (input_strategy[i] != NO_SPLIT_STRATEGY) {
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
  Strategies stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }
  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status LayerNormInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferMirrorOps failed.";
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  OperatorVector op_for_begin_norm_axis;
  OperatorVector op_for_begin_params_axis;
  OperatorVector op_for_epsilon;
  (void)mirror_ops_.emplace_back(std::move(op_for_begin_norm_axis));
  (void)mirror_ops_.emplace_back(std::move(op_for_begin_params_axis));
  (void)mirror_ops_.emplace_back(std::move(op_for_epsilon));
  return SUCCESS;
}

Status LayerNormInfo::CreateInputTensorMap(size_t input_index) {
  if (inputs_shape_.size() <= input_index) {
    MS_LOG(ERROR) << name_ << ": Invalid index" << input_index;
    return FAILED;
  }
  Shape shape = inputs_shape_[input_index];
  Shape tensor_map;
  for (size_t i = 0; i < shape.size(); ++i) {
    tensor_map.push_back(SizeToLong(shape.size() - i - 1));
  }
  inputs_tensor_map_.push_back(tensor_map);
  return SUCCESS;
}

Status LayerNormInfo::InferTensorMap() {
  if ((CreateInputTensorMap(LAYER_NORM_INPUT_INDEX) != SUCCESS) ||
      (CreateInputTensorMap(LAYER_NORM_GAMMA_INDEX) != SUCCESS) ||
      (CreateInputTensorMap(LAYER_NORM_BETA_INDEX) != SUCCESS)) {
    MS_LOG(ERROR) << name_ << ": Create input tensor map failed";
    return FAILED;
  }

  Shape first_output_tensor_map = inputs_tensor_map_[0];
  Shape second_output_tensor_map = first_output_tensor_map;
  for (size_t i = begin_norm_axis_; i < second_output_tensor_map.size(); ++i) {
    second_output_tensor_map[i] = MAP_NONE;
  }
  Shape third_output_tensor_map = second_output_tensor_map;

  outputs_tensor_map_.push_back(first_output_tensor_map);
  outputs_tensor_map_.push_back(second_output_tensor_map);
  outputs_tensor_map_.push_back(third_output_tensor_map);
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

Status LayerNormInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

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
    Strategies tmp_strategy;
    Dimensions input_strategy = sp->GetInputDim()[0];
    Dimensions gamma_strategy = input_strategy;
    (void)gamma_strategy.erase(gamma_strategy.cbegin(),
                               gamma_strategy.cbegin() + static_cast<different_type>(gamma_diff));
    Dimensions beta_strategy = input_strategy;
    (void)beta_strategy.erase(beta_strategy.cbegin(), beta_strategy.cbegin() + static_cast<different_type>(beta_diff));

    // reset the strategy
    tmp_strategy.push_back(input_strategy);
    tmp_strategy.push_back(gamma_strategy);
    tmp_strategy.push_back(beta_strategy);
    sp->ResetInputs(tmp_strategy);
  }
  return SUCCESS;
}

std::vector<StrategyPtr> LayerNormInfo::GenerateOpStrategies(int64_t stage_id) {
  if (InitShapes() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Init shapes failed";
  }
  Shape input_split(input_shape_.size(), SPLIT_FLAG);
  if (begin_norm_axis_ >= input_split.size()) {
    MS_LOG(EXCEPTION) << name_ << ": Invalid begin norm axis " << begin_norm_axis_;
  }

  // Can not split the dimensions from begin norm axis
  for (size_t i = begin_norm_axis_; i < input_split.size(); ++i) {
    input_split[i] = NO_SPLIT_FLAG;
  }

  // Generate strategy for input
  Shapes splittable_inputs = {input_split};
  Shapes tmp_inputs_shape = {input_shape_};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate input strategy failed";
  }

  // Generate the strategies for gamma and beta
  if (GenerateGammaAndBetaStrategies(sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate gamma and beta strategies failed";
  }

  return sp_vector;
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

Status LayerNormInfo::CheckInputLayout() {
  // Check all device matrix should be the same
  if (inputs_tensor_info_.size() != kSizeThree) {
    MS_LOG(ERROR) << "The size of input_tensor_layout for layernorm is " << inputs_tensor_info_.size()
                  << " rather than 3.";
    return FAILED;
  }
  auto in_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto gamma_layout = inputs_tensor_info_[kIndex1].tensor_layout();
  auto beta_layout = inputs_tensor_info_[kIndex2].tensor_layout();

  // check input layout
  // [begin_norm_axis_, -1] should not shard after begin_norm_axis
  const std::vector<int64_t> np_split_map = {-1};
  for (size_t i = begin_norm_axis_; i < in_layout.tensor_map_before().size(); ++i) {
    if (in_layout.tensor_map_before()[i] != np_split_map) {
      MS_LOG(ERROR) << "Layernorm Invalid input layout " << in_layout.tensor_map_before();
      return FAILED;
    }
  }

  // check gamma and beta layout
  if (gamma_layout.tensor_map_before() != beta_layout.tensor_map_before()) {
    MS_LOG(ERROR) << "The tensor map of gamma " << gamma_layout.tensor_map_before()
                  << " dose not equal to tensor map of beta " << beta_layout.tensor_map_before();
    return FAILED;
  }

  size_t gamma_diff = in_layout.tensor_map_before().size() - gamma_layout.tensor_map_before().size();
  for (size_t j = 0; j < gamma_layout.tensor_map_before().size(); ++j) {
    if (gamma_layout.tensor_map_before()[j] != in_layout.tensor_map_before()[gamma_diff + j]) {
      MS_LOG(ERROR) << "Layernorm Invalid gamma layout " << gamma_layout.tensor_map_before();
      return FAILED;
    }
  }

  size_t beta_diff = in_layout.tensor_map_before().size() - beta_layout.tensor_map_before().size();
  for (size_t j = 0; j < beta_layout.tensor_map_before().size(); ++j) {
    if (beta_layout.tensor_map_before()[j] != in_layout.tensor_map_before()[beta_diff + j]) {
      MS_LOG(ERROR) << "Layernorm Invalid beta layout " << beta_layout.tensor_map_before();
      return FAILED;
    }
  }

  return SUCCESS;
}

Status LayerNormInfo::CheckOutputLayout() {
  // Check all device matrix should be the same
  if (outputs_tensor_info_.size() != kSizeThree) {
    MS_LOG(ERROR) << "The size of output_tensor_layout for layernorm is " << outputs_tensor_info_.size()
                  << " rather than 3.";
    return FAILED;
  }
  if (output_infer_tensor_layout_.tensor_shape_before().array().empty()) {
    MS_LOG(ERROR) << "Parameter of output tensor layout for layernorm is not allowed to be set by users.";
    return FAILED;
  }
  MS_LOG(INFO) << "Using output tensor layout infer by input tensor layout.";
  return SUCCESS;
}

Status LayerNormInfo::InferOutputLayout() {
  auto input_layout = inputs_tensor_info_[kIndex0].tensor_layout();

  TensorLayout output_tensor_layout;
  TensorLayout mean_tensor_layout;
  TensorLayout var_tensor_layout;
  output_tensor_layout = input_layout;
  mean_tensor_layout = output_tensor_layout;
  std::vector<Shape> mean_extended_tensor_map;
  Shape mean_tensor_shape;

  for (size_t i = 0; i < mean_tensor_layout.tensor_shape_before().array().size(); ++i) {
    auto map_dim = input_layout.tensor_map_before()[i];
    auto shp_dim = input_layout.tensor_shape_before().array()[i];
    mean_extended_tensor_map.push_back(map_dim);
    if (i < begin_norm_axis_) {
      mean_tensor_shape.push_back(shp_dim);
    } else {
      mean_tensor_shape.push_back(1);
    }
  }
  mean_tensor_layout.InitFromExtendVector(mean_tensor_layout.device_arrangement_origin().array(),
                                          mean_extended_tensor_map, mean_tensor_shape);
  var_tensor_layout = mean_tensor_layout;

  output_infer_tensor_layout_ = output_tensor_layout;
  mean_infer_tensor_layout_ = mean_tensor_layout;
  var_infer_tensor_layout_ = var_tensor_layout;

  return SUCCESS;
}

Status LayerNormInfo::InferOutputTensorInfo() {
  InferOutputLayout();
  if (output_infer_tensor_layout_.tensor_shape_before().array() != outputs_shape_[kIndex0]) {
    MS_LOG(ERROR) << "The infer output shape " << output_infer_tensor_layout_.tensor_shape_before().array()
                  << " dose not match the output shape " << outputs_shape_[kIndex0];
    return FAILED;
  }
  if (mean_infer_tensor_layout_.tensor_shape_before().array() != outputs_shape_[kIndex1]) {
    MS_LOG(ERROR) << "The infer output mean shape " << mean_infer_tensor_layout_.tensor_shape_before().array()
                  << " dose not match the output shape " << outputs_shape_[kIndex1];
    return FAILED;
  }
  if (var_infer_tensor_layout_.tensor_shape_before().array() != outputs_shape_[kIndex2]) {
    MS_LOG(ERROR) << "The infer output var shape " << var_infer_tensor_layout_.tensor_shape_before().array()
                  << " dose not match the output shape " << outputs_shape_[kIndex2];
    return FAILED;
  }
  TensorInfo output_tensor_info(output_infer_tensor_layout_);
  TensorInfo mean_tensor_info(mean_infer_tensor_layout_);
  TensorInfo var_tensor_info(var_infer_tensor_layout_);
  outputs_tensor_info_.push_back(output_tensor_info);
  outputs_tensor_info_.push_back(mean_tensor_info);
  outputs_tensor_info_.push_back(var_tensor_info);
  return SUCCESS;
}

Status LayerNormInfo::InferForwardCommunicationByLayout() {
  // for layernorm, no ForwardCommunication
  return SUCCESS;
}

REGISTER(LayerNormInfo);
}  // namespace parallel
}  // namespace mindspore
