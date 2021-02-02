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

#include "frontend/parallel/ops_info/prelu_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
/*
 * prelu has 2 input
 *  A: A float tensor of shape [NCHW] representing the output of the preview layer.
 *  w: Float Tensor, w > 0: there is only two shapes are legitimate: 1, or the number of channels at input.
 *  the strategy of w should equal to the channel dimension of strategy of A, or equal to 1
 */
Status PReLUInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    return FAILED;
  }
  Strategys stra = strategy->GetInputDim();
  if (stra[1].size() != PRELU_SECOND_INPUT_SIZE) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy size.";
    return FAILED;
  }
  if (stra[0][PRELU_CHANNEL_INDEX] != stra[1][0] && inputs_shape_[1][0] != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid channel strategy.";
    return FAILED;
  }
  return SUCCESS;
}

/*
 * device matrix is same with the strategy matrix
 */
Status PReLUInfo::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  input_strategy_ = input_strategy;
  dev_matrix_shape_ = input_strategy;
  return SUCCESS;
}

Status PReLUInfo::InferForwardCommunication() { return SUCCESS; }

/*
 * the output tensor map is the same as the input tensor map
 */
Status PReLUInfo::InferTensorMap() {
  TensorMap input_tensor_map;
  // such as 4: input_tensor_map [3,2,1,0]
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    input_tensor_map.push_back((int64_t)(inputs_shape_[0].size() - i - 1));
  }

  TensorMap param_tensor_map;
  if (inputs_shape_[1][0] == 1) {
    param_tensor_map.push_back(-1);
  } else {
    param_tensor_map.push_back(input_tensor_map.at(1));
  }
  inputs_tensor_map_.push_back(input_tensor_map);
  inputs_tensor_map_.push_back(param_tensor_map);
  outputs_tensor_map_.push_back(input_tensor_map);
  return SUCCESS;
}

Dimensions PReLUInfo::GetOutputStrategy() {
  Dimensions output_strategy = input_strategy_;
  return output_strategy;
}

Status PReLUInfo::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout) {
  if (inputs_layout == nullptr || outputs_layout == nullptr) {
    MS_LOG(ERROR) << name_ << ": InferTensorLayout: the layout is null.";
    return FAILED;
  }
  TensorLayout input_layout, param_layout, output_layout;
  if ((input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], inputs_shape_[0]) != SUCCESS) ||
      (param_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[1], inputs_shape_[1]) != SUCCESS) ||
      (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], outputs_shape_[0]) != SUCCESS)) {
    return FAILED;
  }
  inputs_layout->push_back(input_layout);
  inputs_layout->push_back(param_layout);
  outputs_layout->push_back(output_layout);
  return SUCCESS;
}

Status PReLUInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);
  Shape param_shape = inputs_shape_.at(1);
  Shape output_shape = outputs_shape_.at(0);
  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Dimensions output_strategy = GetOutputStrategy();
  Strategys inputs_strategy = strategy_->GetInputDim();
  Strategys outputs_strategy = {output_strategy};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }
  Shape input_slice_shape = inputs_slice_shape.at(0);
  Shape param_slice_shape = inputs_slice_shape.at(1);
  Shape output_slice_shape = outputs_slice_shape.at(0);

  // infer tensor layout
  TensorLayouts inputs_layout, outputs_layout;
  if (InferTensorLayout(&inputs_layout, &outputs_layout) != SUCCESS) {
    return FAILED;
  }

  TensorLayout input_layout = inputs_layout.at(0);
  TensorLayout param_layout = inputs_layout.at(1);
  TensorLayout output_layout = outputs_layout.at(0);
  TensorInfo input_tensor_info(input_layout, input_shape, input_slice_shape);
  TensorInfo param_tensor_info(param_layout, param_shape, param_slice_shape);
  TensorInfo output_tensor_info(output_layout, output_shape, output_slice_shape);

  inputs_tensor_info_.push_back(input_tensor_info);
  inputs_tensor_info_.push_back(param_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status PReLUInfo::GetAttrs() {
  if ((inputs_shape_.size() != PRELU_INPUTS_SIZE) || (outputs_shape_.size() != PRELU_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size " << inputs_shape_.size() << " or outputs shape size "
                  << outputs_shape_.size() << " is wrong.";
    return FAILED;
  }
  return SUCCESS;
}

Status PReLUInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status PReLUInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status PReLUInfo::GenerateStrategies(int64_t stage_id) {
  if (inputs_shape_.size() != PRELU_INPUTS_SIZE) {
    return FAILED;
  }
  if (inputs_shape_[1].size() != PRELU_SECOND_INPUT_SIZE) {
    return FAILED;
  }
  Shape input0_split;
  input0_split.emplace_back(1);
  input0_split.emplace_back(0);
  (void)input0_split.insert(input0_split.end(), inputs_shape_[0].size() - 2, 1);
  Shape input1_split(inputs_shape_[1].size(), 0);
  Shapes splittable_inputs = {input0_split, input1_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateStrategiesForIndependentInputs failed";
    return FAILED;
  }
  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

Status PReLUInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }
}  // namespace parallel
}  // namespace mindspore
