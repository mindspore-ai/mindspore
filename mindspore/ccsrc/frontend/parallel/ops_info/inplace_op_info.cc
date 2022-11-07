/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <utility>
#include <algorithm>
#include <functional>

#include "frontend/parallel/ops_info/inplace_op_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status InplaceOpBase::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }
  Strategies strategies = strategy->GetInputDim();
  auto x_strategy = strategies.at(0);
  auto input_v_strategy = strategies.at(1);
  if (x_strategy[0] != 1 || input_v_strategy[0] != 1) {
    MS_LOG(ERROR) << name_ << ": The 1st dimension of x and input_v is not supported sharding, "
                  << "but got strategy " << StrategyToString(strategies);
    return FAILED;
  }
  if (x_strategy != input_v_strategy) {
    MS_LOG(ERROR) << name_ << ": The strategy of x and input_v must be the same, "
                  << "but got strategy " << StrategyToString(strategies);
    return FAILED;
  }
  return SUCCESS;
}

Status InplaceOpBase::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  auto strategies = strategy_->GetInputDim();
  auto x_strategy = strategies.at(0);
  dev_matrix_shape_.assign(x_strategy.begin() + 1, x_strategy.end());

  MS_LOG(INFO) << name_ << ": dev matrix: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status InplaceOpBase::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Shape tensor_map = {-1};
  size_t dev_size = dev_matrix_shape_.size();
  if (repeated_calc_num_ > 1 && repeated_num_in_dev_matrix_right_) {
    --dev_size;
  }
  size_t start = 0;
  if (repeated_calc_num_ > 1 && !repeated_num_in_dev_matrix_right_) {
    ++start;
  }
  for (size_t i = start; i < dev_size; ++i) {
    tensor_map.push_back(dev_size - i - 1);
  }
  inputs_tensor_map_.push_back(tensor_map);
  inputs_tensor_map_.push_back(tensor_map);
  (void)outputs_tensor_map_.emplace_back(std::move(tensor_map));
  return SUCCESS;
}

std::vector<StrategyPtr> InplaceOpBase::GenerateOpStrategies(int64_t stage_id) {
  Shapes splittable_inputs = inputs_shape_;
  for (size_t i = 0; i < splittable_inputs.size(); ++i) {
    for (size_t j = 0; j < splittable_inputs[i].size(); ++j) {
      splittable_inputs[i][j] = SizeToLong(j);
    }
  }
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for dependent inputs() failed.";
  }
  return sp_vector;
}

void InplaceOpBase::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = false;
  split_flag_list_[1] = false;
}

REGISTER(InplaceAddInfo);
REGISTER(InplaceSubInfo);
REGISTER(InplaceUpdateInfo);
}  // namespace parallel
}  // namespace mindspore
