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

#include "frontend/parallel/ops_info/cdist_info.h"

#include <unordered_map>

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status CdistInfo::GetAttrs() {
  input_dims_ = inputs_shape_.at(0).size();
  if (input_dims_ != 2 && input_dims_ != 3) {
    MS_LOG(ERROR) << "Dimension of each input must be 2 or 3, but got dimension is " << input_dims_ << ".";
    return FAILED;
  }
  return SUCCESS;
}

Status CdistInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  auto strategies = strategy->GetInputDim();
  auto input_x_strategy = strategies.at(0);
  auto input_y_strategy = strategies.at(1);
  // input_x shape: (B, P, M), input_y shape: (B, R, M), shard num of B-dim must be equal
  if (input_dims_ == 3 && input_x_strategy[0] != input_y_strategy[0]) {
    MS_LOG(ERROR) << name_ << ": Sharding num of batch-dimension must be equal, "
                  << "but got strategy " << StrategyToString(strategies);
    return FAILED;
  }

  if (input_x_strategy.back() != 1 || input_y_strategy.back() != 1) {
    MS_LOG(ERROR) << name_ << ": The last dimension of each input cannot be shard, "
                  << "but got strategy " << StrategyToString(strategies);
    return FAILED;
  }
  return SUCCESS;
}

Status CdistInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  auto strategies = strategy_->GetInputDim();
  auto input_x_strategy = strategies.at(0);
  auto input_y_strategy = strategies.at(1);
  if (input_dims_ == 2) {
    dev_matrix_shape_ = {input_x_strategy[0], input_y_strategy[0]};
  } else {
    dev_matrix_shape_ = {input_x_strategy[0], input_x_strategy[1], input_y_strategy[1]};
  }

  MS_LOG(INFO) << name_ << ": dev matrix: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status CdistInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  if (input_dims_ == 2) {
    inputs_tensor_map_ = {{1, -1}, {0, -1}};
    outputs_tensor_map_ = {{1, 0}};
  } else {
    inputs_tensor_map_ = {{2, 1, -1}, {2, 0, -1}};
    outputs_tensor_map_ = {{2, 1, 0}};
  }
  return SUCCESS;
}

std::vector<StrategyPtr> CdistInfo::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp;
  Shapes inputs_splittable;
  if (input_dims_ == 2) {
    inputs_splittable = {{1, 0}, {2, 0}};
  } else {
    inputs_splittable = {{1, 2, 0}, {1, 3, 0}};
  }
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, inputs_splittable, &sp) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for dependent inputs() failed.";
  }
  return sp;
}

void CdistInfo::ReComputeBatchSplitFlagList() {
  size_t input_dims = LongToSize(inputs_shape_.at(0).at(0));
  if (input_dims == 3) {
    if (inputs_shape_[0][0] != 1) {
      split_flag_list_[0] = true;
      split_flag_list_[1] = true;
    }
    return;
  }

  // if input_dims is 2, only one of them can be split
  if (inputs_shape_[0][0] != 1) {
    split_flag_list_[0] = true;
  } else if (inputs_shape_[1][0] != 1) {
    split_flag_list_[1] = true;
  }
}

REGISTER(CdistInfo);
}  // namespace parallel
}  // namespace mindspore
