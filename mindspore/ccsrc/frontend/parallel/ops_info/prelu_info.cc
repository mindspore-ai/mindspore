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
#include "frontend/parallel/dynamic_creator.h"
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
    return FAILED;
  }
  Strategies stra = strategy->GetInputDim();
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
  Strategies stra = strategy_->GetInputDim();
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
    input_tensor_map.push_back(SizeToLong(inputs_shape_[0].size() - i - 1));
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
  infer_strategy_mode_ = INDIVIDUAL_MODE;
  return SUCCESS;
}

std::vector<StrategyPtr> PReLUInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split;
  input0_split.emplace_back(1);
  input0_split.emplace_back(0);
  (void)input0_split.insert(input0_split.cend(), inputs_shape_[0].size() - 2, 1);
  Shape input1_split(inputs_shape_[1].size(), 0);
  Shapes splittable_inputs = {input0_split, input1_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": GenerateStrategies For Independent Inputs failed";
  }
  return sp_vector;
}

Status PReLUInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

// in_strategy: ((A, B, C), ()),  Shapes: ((a, b, c), (b)), return: ((A, B, C), (B))
// in_strategy: ((A, B, C), ()),  Shapes: ((a, b, c), (1)), return: ((A, B, C), (1))
// in_strategy: ((), (B)),  Shapes: ((a, b, c), (b)), return: ((1, B, 1), (B))
Shapes PReLUInfo::InferStrategyIndividualMode(const Shapes &in_strategy) {
  if (in_strategy.size() != 2) {
    MS_LOG(EXCEPTION) << name_ << ": The size of in_strategy must be 3, but got " << in_strategy.size();
  }

  if (!in_strategy[0].empty()) {
    if (in_strategy[0].size() < 2) {
      MS_LOG(EXCEPTION) << name_ << ": The size of in_strategy[0] must be larger than 1, but got "
                        << in_strategy[0].size();
    }
    if (inputs_shape_[1][0] > 1) {
      return Shapes({in_strategy[0], {in_strategy[0][1]}});
    } else {
      return Shapes({in_strategy[0], {1}});
    }
  }

  if (!in_strategy[1].empty()) {
    Shape tmp(inputs_shape_[0].size(), 1);
    tmp[1] = in_strategy[1][0];
    return Shapes({tmp, in_strategy[1]});
  }
  MS_LOG(EXCEPTION) << name_ << ": The in_strategy[0] and in_strategy[1] are empty";
}

REGISTER(PReLUInfo);
}  // namespace parallel
}  // namespace mindspore
