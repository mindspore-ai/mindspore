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

#include "frontend/parallel/ops_info/split_info.h"

#include <string>
#include <memory>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "include/common/utils/parallel_context.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status SplitInfo::GetAttrs() {
  int64_t axis = 0;

  std::string attr_axis_name = GetSplitAxisAttrName();
  auto axis_iter = attrs_.find(attr_axis_name);
  if (axis_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(axis_iter->second);
    if (axis_iter->second->isa<Int64Imm>()) {
      axis = axis_iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis is not int";
      return FAILED;
    }
  } else {
    MS_LOG(ERROR) << name_ << ": Can not find the axis attr";
    return FAILED;
  }

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  int dim = SizeToInt(inputs_shape_[0].size());
  if (axis < 0) {
    axis = axis + dim;
  }
  axis_ = LongToSize(axis);

  return SUCCESS;
}

Status SplitInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  if (axis_ >= stra[0].size()) {
    MS_LOG(ERROR) << name_ << ": The axis is out of range, the axis is " << axis_;
    return FAILED;
  }

  if (stra[0][axis_] != 1) {
    MS_LOG(ERROR) << name_ << ": The axis can not be split";
    return FAILED;
  }

  return SUCCESS;
}

Status SplitInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status SplitInfo::InferTensorMap() {
  TensorMap tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << "The inputs shape is empty";
    return FAILED;
  }

  int32_t size = SizeToInt(inputs_shape_[0].size());
  for (int i = 0; i < size; ++i) {
    tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(tensor_map);

  for (size_t i = 0; i < outputs_shape_.size(); ++i) {
    outputs_tensor_map_.push_back(tensor_map);
  }

  return SUCCESS;
}

Status SplitInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> SplitInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape split_flag;
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (i == axis_) {
      split_flag.push_back(0);
    } else {
      split_flag.push_back(1);
    }
  }

  Shapes splittable_input = {split_flag};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }
  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy";
  }

  return sp_vector;
}

std::shared_ptr<Strategies> SplitInfo::GenerateBatchStrategies() {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Get attr failed";
  }
  Dimensions input_strategy(inputs_shape_[0].size(), 1);
  // axis can't split
  if (inputs_shape_[0].size() > 1) {
    if (axis_ != 0) {
      input_strategy[0] = stage_device_size_;
    }
  }
  Strategies strategy_v = {input_strategy};
  return std::make_shared<Strategies>(strategy_v);
}

Status SplitInfo::InferAsLossDivisor() {
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

REGISTER(SplitInfo);
REGISTER(SplitVInfo);
}  // namespace parallel
}  // namespace mindspore
