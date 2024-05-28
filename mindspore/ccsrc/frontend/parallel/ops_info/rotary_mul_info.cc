/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/ops_info/rotary_mul_info.h"

namespace mindspore {
namespace parallel {
Status RotaryMulInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies strategies = strategy->GetInputDim();
  if (strategies.size() != ROTARY_MUL_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << " : The strategy size must be " << ROTARY_MUL_INPUTS_SIZE << ", but got "
                  << strategies.size();
    return FAILED;
  }

  Dimensions first_input_stra = strategies[0];
  // except first stra first two dimension, other dimension should not be split
  for (size_t i = 0; i < ROTARY_MUL_INPUTS_SIZE; ++i) {
    Dimensions input_stra = strategies[i];
    for (size_t j = 0; j < input_stra.size(); ++j) {
      if (i == ROTARY_MUL_INPUT_SPLITTABLE_ROW_INDEX && j < ROTARY_MUL_INPUT_SPLITTABLE_COL_INDEX) {
        continue;
      }
      if (input_stra[j] > 1) {
        MS_LOG(ERROR) << name_ << " : The strategy of last 2 dim must be 1, but got " << input_stra[j] << " at index "
                      << i;
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status RotaryMulInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();
  Strategies strategies = strategy_->GetInputDim();

  dev_matrix_shape_.assign(strategies[0].begin(), strategies[0].end());
  MS_LOG(INFO) << name_ << ": dev matrix: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status RotaryMulInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Strategies strategies = strategy_->GetInputDim();
  Shape first_tensor_map;
  size_t dim = strategies[0].size();
  for (size_t i = 0; i < dim; ++i) {
    first_tensor_map.push_back(dim - i - 1);
  }
  (void)inputs_tensor_map_.push_back(first_tensor_map);

  TensorMap non_split_tensor_map = {-1, -1, 1, 0};
  (void)inputs_tensor_map_.push_back(non_split_tensor_map);
  (void)inputs_tensor_map_.push_back(non_split_tensor_map);

  (void)outputs_tensor_map_.emplace_back(std::move(first_tensor_map));
  return SUCCESS;
}

void RotaryMulInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < split_flag_list_.size(); ++i) {
    if (i == ROTARY_MUL_INPUT_SPLITTABLE_ROW_INDEX) {
      split_flag_list_[i] = true;
    } else {
      split_flag_list_[i] = false;
    }
  }
}

std::vector<StrategyPtr> RotaryMulInfo::GenerateOpStrategies(int64_t stage_id) {
  Shapes splittable_inputs;
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    (void)splittable_inputs.emplace_back(inputs_shape_[i].size());
    for (size_t j = 0; j < inputs_shape_[i].size(); ++j) {
      if (i == ROTARY_MUL_INPUT_SPLITTABLE_ROW_INDEX && j < ROTARY_MUL_INPUT_SPLITTABLE_COL_INDEX) {
        splittable_inputs[i][j] = SizeToLong(j) + 1;
      } else {
        splittable_inputs[i][j] = 0;
      }
    }
  }

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for dependent inputs() failed.";
  }

  return sp_vector;
}

REGISTER(RotaryMulInfo);
}  // namespace parallel
}  // namespace mindspore
