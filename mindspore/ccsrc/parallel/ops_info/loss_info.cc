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

#include "parallel/ops_info/loss_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "parallel/device_matrix.h"
#include "parallel/strategy.h"
#include "parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status SoftmaxCrossEntropyWithLogitsInfo::CheckStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    }
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  Dimensions label_strategy = stra.at(1);
  if (input_strategy != label_strategy) {
    MS_LOG(ERROR) << name_ << " : Strategies of relevant dimensions are not equal.";
    return FAILED;
  }

  int32_t axis_index = axis_;
  if (axis_ < 0) {
    size_t input_dim = inputs_shape_.at(0).size();
    axis_index = static_cast<int32_t>(input_dim) + axis_;
  }

  int32_t input_axis_strategy = input_strategy.at(IntToSize(axis_index));
  int32_t label_axis_strategy = label_strategy.at(IntToSize(axis_index));
  // Dimension corresponding to axis is un-splittable
  if ((input_axis_strategy != MIN_SLICE_NUM) && (label_axis_strategy != MIN_SLICE_NUM)) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_
                    << " : The strategy corresponding to axis dimension is not 1, input: " << input_axis_strategy
                    << ", label: " << label_axis_strategy;
    } else {
      MS_LOG(ERROR) << name_
                    << " : The strategy corresponding to axis dimension is not 1, input: " << input_axis_strategy
                    << ", label: " << label_axis_strategy;
    }
    return FAILED;
  }

  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::GetAttrs() {
  if ((inputs_shape_.size() != SoftmaxCrossEntropyWithLogitsInputsSize) ||
      (outputs_shape_.size() != SoftmaxCrossEntropyWithLogitsOutputsSize)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }

  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  dev_matrix_shape_ = input_strategy;
  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::InferTensorMap() {
  std::vector<int32_t> tensor_map_index;
  size_t size = inputs_shape_[0].size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int32_t)(size - i - 1));
  }

  std::vector<int32_t> first_output_tensor_map = {tensor_map_index[0]};
  inputs_tensor_map_.push_back(tensor_map_index);          // input
  inputs_tensor_map_.push_back(tensor_map_index);          // label
  outputs_tensor_map_.push_back(first_output_tensor_map);  // output-0
  outputs_tensor_map_.push_back(tensor_map_index);         // output-1
  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);
  Shape first_output_shape = outputs_shape_.at(0);

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Strategys outputs_strategy = {{inputs_strategy[0][0]}, inputs_strategy.at(0)};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }
  Shape input_slice_shape = inputs_slice_shape.at(0);
  Shape first_output_slice_shape = outputs_slice_shape.at(0);

  TensorMap input_tensor_map = inputs_tensor_map_.at(0);
  TensorMap first_output_tensor_map = outputs_tensor_map_.at(0);

  TensorLayout input_tensor_layout, first_output_tensor_layout;
  if ((input_tensor_layout.InitFromVector(dev_matrix_shape_, input_tensor_map, input_shape) != SUCCESS) ||
      (first_output_tensor_layout.InitFromVector(dev_matrix_shape_, first_output_tensor_map, first_output_shape) !=
       SUCCESS)) {
    return FAILED;
  }
  TensorInfo input_tensor_info(input_tensor_layout, input_shape, input_slice_shape);
  TensorInfo first_output_tensor_info(first_output_tensor_layout, first_output_shape, first_output_slice_shape);

  inputs_tensor_info_.push_back(input_tensor_info);          // input
  inputs_tensor_info_.push_back(input_tensor_info);          // label
  outputs_tensor_info_.push_back(first_output_tensor_info);  // output-0
  outputs_tensor_info_.push_back(input_tensor_info);         // output-1

  return SUCCESS;
}

// There are two outputs for SoftmaxCrossEntropyWithLogits, and outputs[1] is used for grad and overload the function.
Status SoftmaxCrossEntropyWithLogitsInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != 2) {
    MS_LOG(ERROR) << name_ << " : The size of outputs tensor map " << outputs_tensor_map_.size() << " is error.";
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[1]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[1]) << ", as_loss_divisor_ is "
               << as_loss_divisor_;
  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << " : Init for cost model failed.";
    }
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}

void SoftmaxCrossEntropyWithLogitsInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    split_flag_list_[i] = true;
  }
}

Status SoftmaxCrossEntropyWithLogitsInfo::GenerateStrategies(int32_t stage_id) {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : GetAttrs failed.";
    return FAILED;
  }
  int32_t axis_index = axis_;
  if (axis_ < 0) {
    size_t input_dim = inputs_shape_[0].size();
    axis_index = static_cast<int32_t>(input_dim) + axis_;
  }
  is_auto_parallel_ = true;

  Shape input0_split;
  (void)input0_split.insert(input0_split.begin(), inputs_shape_[0].size(), 1);
  input0_split[IntToSize(axis_index)] = 0;
  Shapes splittable_inputs = {input0_split, input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesWithBroadcast(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Generate strategies failed.";
    return FAILED;
  }

  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << " : Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }

  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  PrintStrategy(strategy);
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << " : Set cost under strategy failed.";
    }
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
