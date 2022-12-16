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

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/ops_info/addn_info.h"

namespace mindspore {
namespace parallel {
Status AddNInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  // The strategy for each input tensor must be equal
  Strategies strategies = strategy->GetInputDim();
  for (size_t i = 1; i < strategies.size(); ++i) {
    if (strategies[i] != strategies[0]) {
      MS_LOG(ERROR) << name_ << ": The strategy for each input must be equal to strategies[0]: " << strategies[0]
                    << ", but got strategies[" << i << "]: " << strategies[i];
      return FAILED;
    }
  }
  return SUCCESS;
}

Status AddNInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  Strategies strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    return SUCCESS;
  }
  dev_matrix_shape_.assign(strategies[0].begin(), strategies[0].end());

  MS_LOG(INFO) << name_ << ": dev matrix: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status AddNInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Shape sub_tensor_map;
  Strategies strategies = strategy_->GetInputDim();
  size_t dim = strategies.at(0).size();
  for (size_t i = 0; i < dim; ++i) {
    sub_tensor_map.push_back(dim - i - 1);
  }

  for (size_t i = 0; i < strategies.size(); ++i) {
    inputs_tensor_map_.push_back(sub_tensor_map);
  }
  (void)outputs_tensor_map_.emplace_back(std::move(sub_tensor_map));
  return SUCCESS;
}

std::vector<StrategyPtr> AddNInfo::GenerateOpStrategies(int64_t stage_id) {
  Shapes splittable_inputs;
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    (void)splittable_inputs.emplace_back(inputs_shape_[i].size());
    for (size_t j = 0; j < inputs_shape_[i].size(); ++j) {
      splittable_inputs[i][j] = SizeToLong(j) + 1;
    }
  }

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for dependent inputs() failed.";
  }

  return sp_vector;
}

void AddNInfo::ReComputeBatchSplitFlagList() {
  bool flag = false;
  if (!inputs_shape_[0].empty()) {
    flag = true;
  }

  // Batch dim of each input can be split
  for (size_t i = 0; i < split_flag_list_.size(); ++i) {
    split_flag_list_[i] = flag;
  }
}

REGISTER(AddNInfo);
}  // namespace parallel
}  // namespace mindspore
