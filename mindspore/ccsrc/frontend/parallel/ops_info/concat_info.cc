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

#include "frontend/parallel/ops_info/concat_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status ConcatInfo::GetAttrs() {
  int64_t axis = 0;
  auto axis_iter = attrs_.find(AXIS);
  if (axis_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(axis_iter->second);
    if (axis_iter->second->isa<Int64Imm>()) {
      axis = axis_iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis is not int64_t";
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
  int64_t dim = SizeToLong(inputs_shape_[0].size());

  if (axis < 0) {
    axis = axis + dim;
  }

  axis_ = LongToSize(axis);
  return SUCCESS;
}

Status ConcatInfo::CheckStrategy(const StrategyPtr &strategy) {
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

  if (stra.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be equal to the size of inputs shape";
    return FAILED;
  }

  for (size_t i = 0; i < stra.size(); ++i) {
    auto strategy_ele = stra[i];
    auto input_shape_ele = inputs_shape_[i];
    if (strategy_ele.size() != input_shape_ele.size()) {
      MS_LOG(ERROR) << name_ << ": The size of strategy element must be equal to the size of input shape";
      return FAILED;
    }

    if (axis_ >= strategy_ele.size()) {
      MS_LOG(ERROR) << name_ << ": The axis is out of range, the axis is " << axis_;
      return FAILED;
    }

    if (strategy_ele[axis_] != 1) {
      MS_LOG(ERROR) << name_ << ": The axis can not be split";
      return FAILED;
    }

    for (size_t j = 0; j < strategy_ele.size(); ++j) {
      if (strategy_ele[j] != stra[0][j]) {
        MS_LOG(ERROR) << name_ << ": The strategy of each input tensor must be equal";
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status ConcatInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status ConcatInfo::InferTensorMap() {
  TensorMap tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << "The inputs shape is empty";
    return FAILED;
  }

  // cannot use dev_matrix_shape_ replace inputs_shape_[0], because it may not be fully split in all devices.
  int64_t size = SizeToLong(inputs_shape_[0].size());
  for (int64_t i = 0; i < size; ++i) {
    tensor_map.push_back(size - i - 1);
  }

  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    inputs_tensor_map_.push_back(tensor_map);
  }
  outputs_tensor_map_.push_back(tensor_map);
  return SUCCESS;
}

void ConcatInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    split_flag_list_[i] = true;
    if (axis_ == 0) {
      split_flag_list_[i] = false;
    }
  }
}

Status ConcatInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> ConcatInfo::GenerateOpStrategies(int64_t stage_id) {
  if (inputs_shape_.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": The inputs shape is empty";
  }
  Shape input_split;
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (i == axis_) {
      input_split.push_back(0);
    } else {
      input_split.push_back(1);
    }
  }

  // to generate the first input's strategy
  Shapes splittable_input = {input_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  // the others strategies are equal to the first input's strategy
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }
    Strategies tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    for (size_t i = 0; i < inputs_shape_.size(); ++i) {
      tmp_strategy.push_back(first_input_strategy);
    }
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

REGISTER(ConcatInfo);
}  // namespace parallel
}  // namespace mindspore
