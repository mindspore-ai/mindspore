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

#include "frontend/parallel/ops_info/loss_info.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status SoftmaxCrossEntropyWithLogitsInfo::CheckStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    return FAILED;
  }

  Strategys stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  Dimensions label_strategy = stra.at(1);
  if (input_strategy != label_strategy) {
    MS_LOG(ERROR) << name_ << " : Strategies of relevant dimensions are not equal.";
    return FAILED;
  }

  int64_t axis_index = axis_;
  if (axis_ < 0) {
    size_t input_dim = inputs_shape_.at(0).size();
    axis_index = static_cast<int64_t>(input_dim) + axis_;
  }

  int64_t input_axis_strategy = input_strategy.at(LongToSize(axis_index));
  int64_t label_axis_strategy = label_strategy.at(LongToSize(axis_index));
  // Dimension corresponding to axis is un-splittable
  if ((input_axis_strategy != MIN_SLICE_NUM) && (label_axis_strategy != MIN_SLICE_NUM)) {
    MS_LOG(ERROR) << name_ << " : The strategy corresponding to axis dimension is not 1, input: " << input_axis_strategy
                  << ", label: " << label_axis_strategy;
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
  Strategys stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  dev_matrix_shape_ = input_strategy;
  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::InferTensorMap() {
  Shape tensor_map_index;
  size_t size = inputs_shape_[0].size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int64_t)(size - i - 1));
  }

  Shape first_output_tensor_map = {tensor_map_index[0]};
  inputs_tensor_map_.push_back(tensor_map_index);          // input
  inputs_tensor_map_.push_back(tensor_map_index);          // label
  outputs_tensor_map_.push_back(first_output_tensor_map);  // output-0
  outputs_tensor_map_.push_back(tensor_map_index);         // output-1
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
    MS_LOG(ERROR) << name_ << " : Init for cost model failed.";
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

std::vector<StrategyPtr> SoftmaxCrossEntropyWithLogitsInfo::GenerateOpStrategies(int64_t stage_id) {
  int64_t axis_index = axis_;
  if (axis_ < 0) {
    size_t input_dim = inputs_shape_[0].size();
    axis_index = static_cast<int64_t>(input_dim) + axis_;
  }

  Shape input0_split;
  (void)input0_split.insert(input0_split.begin(), inputs_shape_[0].size(), 1);
  input0_split[LongToSize(axis_index)] = 0;
  Shapes splittable_inputs = {input0_split, input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesWithBroadcast(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies failed.";
  }

  return sp_vector;
}

Status SoftmaxCrossEntropyWithLogitsInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}
}  // namespace parallel
}  // namespace mindspore
