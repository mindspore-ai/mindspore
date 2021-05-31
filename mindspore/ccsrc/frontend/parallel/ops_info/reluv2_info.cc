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

#include "frontend/parallel/ops_info/reluv2_info.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include <numeric>

#include "frontend/parallel/device_matrix.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
Status ReLUV2Info::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Status ReLUV2Info::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategys stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  if (input_strategy[1] != 1) {
    MS_LOG(ERROR) << name_ << "The second dimension is not splitable.";
    return FAILED;
  }

  return SUCCESS;
}

Status ReLUV2Info::GetAttrs() { return SUCCESS; }

std::vector<StrategyPtr> ReLUV2Info::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  // the second dimension is not splitable
  input0_split[1] = 0;
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }

  return sp_vector;
}

Status ReLUV2Info::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  dev_matrix_shape_ = input_strategy;

  return SUCCESS;
}

Status ReLUV2Info::InferForwardCommunication() {
  // do nothing
  return SUCCESS;
}

Status ReLUV2Info::InferTensorMap() {
  Shape tensor_map_index;
  Shape tensor_map_mask;
  size_t size = inputs_shape_.at(0).size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int64_t)(size - i - 1));
  }

  inputs_tensor_map_.push_back(tensor_map_index);
  // output
  outputs_tensor_map_.push_back(tensor_map_index);
  tensor_map_mask = tensor_map_index;
  // mask format NC1HWC0
  tensor_map_mask.push_back(MAP_NONE);
  outputs_tensor_map_.push_back(tensor_map_mask);
  return SUCCESS;
}

Status ReLUV2Info::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor map is empty.";
    return FAILED;
  }

  if (outputs_tensor_map_[0].empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[0]) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

Status ReLUV2Info::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status ReLUV2Info::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
