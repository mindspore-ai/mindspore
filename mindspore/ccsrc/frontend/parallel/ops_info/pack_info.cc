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

#include "frontend/parallel/ops_info/pack_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status StackInfo::GetAttrs() {
  int64_t axis = 0;
  auto axis_iter = attrs_.find(AXIS);
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
  int64_t dim = SizeToLong(inputs_shape_[0].size());

  if (axis < 0) {
    axis = axis + dim;
  }
  axis_ = LongToSize(axis);
  return SUCCESS;
}

Status StackInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  for (size_t i = 0; i < stra.size(); ++i) {
    auto strategy_ele = stra[i];
    if (axis_ > strategy_ele.size()) {
      MS_LOG(ERROR) << name_ << ": The axis is out of range, the axis is " << axis_;
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

Status StackInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status StackInfo::InferTensorMap() {
  TensorMap in_tensor_map;
  TensorMap out_tensor_map;

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << "The inputs shape is empty";
    return FAILED;
  }

  int32_t size = SizeToInt(inputs_shape_[0].size());
  for (int i = 0; i < size; ++i) {
    in_tensor_map.push_back(size - i - 1);
    out_tensor_map.push_back(size - i - 1);
  }

  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    inputs_tensor_map_.push_back(in_tensor_map);
  }

  (void)out_tensor_map.insert(out_tensor_map.begin() + SizeToLong(axis_), MAP_NONE);
  outputs_tensor_map_.push_back(out_tensor_map);
  return SUCCESS;
}

void StackInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    split_flag_list_[i] = true;
  }
}

Status StackInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> StackInfo::GenerateOpStrategies(int64_t stage_id) {
  if (inputs_shape_.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": The inputs shape is empty";
  }
  Shape input_split;
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    input_split.push_back(1);
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
    Strategys tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    for (size_t i = 0; i < inputs_shape_.size(); ++i) {
      tmp_strategy.push_back(first_input_strategy);
    }
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

Status StackInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status StackInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
