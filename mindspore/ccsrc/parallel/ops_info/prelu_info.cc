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

#include "parallel/ops_info/prelu_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "parallel/device_manager.h"
#include "parallel/device_matrix.h"
#include "parallel/step_parallel.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
/*
 * prelu has 2 input
 *  A: A float tensor of shape [NCHW] representing the output of the preview layer.
 *  w: Float Tensor, w > 0: there is only two shapes are legitimate: 1, or the number of channels at input.
 *  the strategy of w should equal to the channel dimension of strategy of A
 */
Status PReLUInfo::CheckStrategy(const StrategyPtr& strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    }
    return FAILED;
  }
  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra[1].size() != PRELU_SECOND_INPUT_SIZE) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Invalid strategy size.";
    } else {
      MS_LOG(ERROR) << name_ << ": Invalid strategy size.";
    }
    return FAILED;
  }
  if (stra[0][PRELU_CHANNEL_INDEX] != stra[1][0]) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Invalid channel strategy.";
    } else {
      MS_LOG(ERROR) << name_ << ": Invalid channel strategy.";
    }
    return FAILED;
  }
  return SUCCESS;
}

/*
 * device matrix is same with the strategy matrix
 */
Status PReLUInfo::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  input_strategy_ = input_strategy;
  dev_matrix_shape_ = input_strategy;
  return SUCCESS;
}

Status PReLUInfo::InferMirrorOps() {
  Shape param_tensor_map = inputs_tensor_map_[1];
  std::vector<Group> param_group;
  if (CreateGroupByTensorMap(param_tensor_map, &param_group) != SUCCESS) {
    return FAILED;
  } else if (param_group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror ops is empty.";
    return SUCCESS;
  }
  OperatorVector op_for_param;
  op_for_param = CreateMirrorOps(param_group[0].name(), param_group[0].GetDevNum());
  // op_for_inputs is empty
  OperatorVector op_for_inputs;
  mirror_ops_.push_back(op_for_inputs);
  mirror_ops_.push_back(op_for_param);
  std::string group_name = param_group[0].name();
  MS_LOG(INFO) << name_ << ": The mirror ops group is " << group_name;
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
    input_tensor_map.push_back((int32_t)(inputs_shape_[0].size() - i - 1));
  }

  TensorMap param_tensor_map;
  param_tensor_map.push_back(input_tensor_map.at(1));
  inputs_tensor_map_.push_back(input_tensor_map);
  inputs_tensor_map_.push_back(param_tensor_map);
  outputs_tensor_map_.push_back(input_tensor_map);
  return SUCCESS;
}

Dimensions PReLUInfo::GetOutputStrategy() {
  Dimensions output_strategy = input_strategy_;
  return output_strategy;
}

Status PReLUInfo::InferTensorLayout(TensorLayouts* inputs_layout, TensorLayouts* outputs_layout) {
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

Status PReLUInfo::Init(const StrategyPtr& strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status PReLUInfo::InitForCostModel(const StrategyPtr& strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    }
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status PReLUInfo::GenerateStrategies(int32_t stage_id) {
  if (inputs_shape_.size() != PRELU_INPUTS_SIZE) {
    return FAILED;
  }
  if (inputs_shape_[1].size() != PRELU_SECOND_INPUT_SIZE) {
    return FAILED;
  }
  is_auto_parallel_ = true;
  Shape input0_split(inputs_shape_[0].size(), 1);
  input0_split[1] = 0;
  Shape input1_split(inputs_shape_[1].size(), 0);
  Shapes splittable_inputs = {input0_split, input1_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateStrategiesForIndependentInputs failed";
    return FAILED;
  }
  size_t success = 0;
  for (auto& sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

Status PReLUInfo::SetCostUnderStrategy(const StrategyPtr& strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Set cost under strategy failed.";
    }
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
