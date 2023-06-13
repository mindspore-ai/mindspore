/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/pad_info.h"
#include <utility>
#include <algorithm>

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status PadV3Info::GetAttrs() {
  mode_ = GetStringAttr(MODE);
  if (mode_ != CONSTANT) {
    MS_LOG(ERROR) << name_ << ": only support the constant mode, but the mode is " << mode_;
    return FAILED;
  }

  if (inputs_shape_.size() != 3) {
    MS_LOG(ERROR) << name_ << ": the size of inputs shape must be 3, but got " << inputs_shape_.size();
    return FAILED;
  }

  if (inputs_shape_[1].size() != 1) {
    MS_LOG(ERROR) << name_ << ": the dim of paddings must be 1, but the shape of paddings is " << inputs_shape_[1];
    return FAILED;
  }

  if (inputs_shape_[1][0] % 2 != 0) {
    MS_LOG(ERROR) << name_ << ": the shape of paddings must be the multiples of 2, but got " << inputs_shape_[1][0];
    return FAILED;
  }

  padding_dim_num_ = inputs_shape_[1][0] / 2;  // the paddings appear in pairs
  MS_LOG(INFO) << name_ << ": the padding dim num is " << padding_dim_num_;

  if (padding_dim_num_ > inputs_shape_[0].size()) {
    MS_LOG(ERROR) << name_
                  << ": the dim of input must be larger than or equal to padding dim num, but the shape of input is "
                  << inputs_shape_[0] << ", and the padding dim num is " << padding_dim_num_;
    return FAILED;
  }
  return SUCCESS;
}

Status PadV3Info::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  // The paddings can not be split
  Strategies strategies = strategy->GetInputDim();
  auto paddings_strategy = strategies[1];
  if (paddings_strategy[0] != NO_SPLIT_STRATEGY) {
    MS_LOG(ERROR) << name_ << ": the paddings can not be split, but its strategy is " << paddings_strategy;
    return FAILED;
  }

  // the last padding_dim_num_ dims of input can not be split
  auto input_strategy = strategies[0];
  size_t input_dim = input_strategy.size();
  for (size_t i = input_dim - padding_dim_num_; i < input_dim; ++i) {
    if (input_strategy[i] != NO_SPLIT_STRATEGY) {
      MS_LOG(ERROR) << name_
                    << ": the last padding_dim_num dims of input can not be split, but the strategy of input is  "
                    << input_strategy;
      return FAILED;
    }
  }
  return SUCCESS;
}

Status PadV3Info::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  Strategies strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << name_ << ": the strategy is empty";
    return FAILED;
  }
  dev_matrix_shape_ = strategies[0];

  MS_LOG(INFO) << name_ << ": dev matrix: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status PadV3Info::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Shape input_tensor_map;
  Strategies strategies = strategy_->GetInputDim();
  size_t dim = strategies[0].size();
  for (size_t i = 0; i < dim; ++i) {
    input_tensor_map.push_back(dim - i - 1);
  }

  inputs_tensor_map_.push_back(input_tensor_map);   // input
  inputs_tensor_map_.push_back({MAP_NONE});         // paddings, can not be split
  inputs_tensor_map_.push_back({});                 // value
  outputs_tensor_map_.push_back(input_tensor_map);  // output
  return SUCCESS;
}

std::vector<StrategyPtr> PadV3Info::GenerateOpStrategies(int64_t stage_id) {
  Shape split_flag;
  size_t input_dim = inputs_shape_[0].size();
  for (size_t i = 0; i < input_dim; ++i) {
    if (i < input_dim - padding_dim_num_) {
      split_flag.push_back(1);
    } else {
      split_flag.push_back(0);  // the last padding_dim_num_ dims of input can not be split
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

  for (auto &sp : sp_vector) {
    Strategies tmp_strategy;
    Dimensions input0_strategy = sp->GetInputDim()[0];
    tmp_strategy.push_back(input0_strategy);      // input
    tmp_strategy.push_back({NO_SPLIT_STRATEGY});  // paddings
    tmp_strategy.push_back({});                   // value
    sp->ResetInputs(tmp_strategy);
  }
  return sp_vector;
}

void PadV3Info::ReComputeBatchSplitFlagList() {
  if (inputs_shape_[0].size() == padding_dim_num_) {
    MS_LOG(EXCEPTION) << name_ << ": all dims are padding, can not use batch parallel";
  }

  split_flag_list_[0] = true;

  for (size_t i = 1; i < split_flag_list_.size(); ++i) {
    split_flag_list_[i] = false;
  }
}

REGISTER(PadV3Info);
}  // namespace parallel
}  // namespace mindspore
