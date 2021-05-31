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

std::vector<StrategyPtr> PReLUInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split;
  input0_split.emplace_back(1);
  input0_split.emplace_back(0);
  (void)input0_split.insert(input0_split.end(), inputs_shape_[0].size() - 2, 1);
  Shape input1_split(inputs_shape_[1].size(), 0);
  Shapes splittable_inputs = {input0_split, input1_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": GenerateStrategies For Independent Inputs failed";
  }
  return sp_vector;
}

Status PReLUInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }
}  // namespace parallel
}  // namespace mindspore
