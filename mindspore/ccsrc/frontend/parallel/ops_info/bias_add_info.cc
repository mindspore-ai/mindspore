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

#include "frontend/parallel/ops_info/bias_add_info.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status BiasAddInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  Strategies stra = strategy->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  int64_t channel_a_strategy = sub_a_strategy.at(1);
  int64_t channel_b_strategy = sub_b_strategy.at(0);
  if (channel_a_strategy != channel_b_strategy) {
    MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    return FAILED;
  }
  return SUCCESS;
}

Status BiasAddInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  dev_matrix_shape_ = sub_a_strategy;
  return SUCCESS;
}

void BiasAddInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = true;
  split_flag_list_[1] = false;
}

Status BiasAddInfo::InferTensorMap() {
  TensorMap sub_a_tensor_map;
  TensorMap sub_b_tensor_map;
  Strategies stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  size_t sub_a_strategy_size = sub_a_strategy.size();
  for (size_t i = 0; i < sub_a_strategy_size; ++i) {
    sub_a_tensor_map.push_back(static_cast<int64_t>(LAST_INDEX(sub_a_strategy_size) - i));
  }
  sub_b_tensor_map.push_back(static_cast<int64_t>(LAST_INDEX(sub_a_strategy_size)) - static_cast<int64_t>(1));

  inputs_tensor_map_.push_back(sub_a_tensor_map);
  inputs_tensor_map_.push_back(sub_b_tensor_map);
  outputs_tensor_map_.push_back(sub_a_tensor_map);

  return SUCCESS;
}

Status BiasAddInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> BiasAddInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split, input0_split};

  std::vector<StrategyPtr> sp_vector;
  Shapes tmp_inputs_shape = {inputs_shape_[0], inputs_shape_[0]};
  Shapes tmp_splittable_inputs = {splittable_inputs[0], splittable_inputs[0]};
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, &sp_vector) !=
      SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies failed.";
  }
  MS_LOG(INFO) << name_ << " : Generate strategies success.";

  for (auto &sp : sp_vector) {
    Strategies tmp_strategy;
    Dimensions input0_strategy = sp->GetInputDim()[0];
    tmp_strategy.push_back(input0_strategy);  // input0

    Dimensions input1_strategy = {input0_strategy.at(1)};

    // reset the strategy
    tmp_strategy.push_back(input1_strategy);  // input1
    sp->ResetInputs(tmp_strategy);
  }
  return sp_vector;
}

// in_strategy: ((N, C, H, W), ()), inputs shapes: ((n, c, h, w), (c)), return: ((N, C, H, W), (C))
// in_strategy: ((), (C)), inputs shapes: ((n, c, h, w), (c)), return: ((1, C, 1, 1), (C))
// in_strategy: ((N, C), ()), inputs shapes: ((n, c), (c)), return: ((N, C), (C))
// in_strategy: ((), (C)), inputs shapes: ((n, c), (c)), return: ((1, C), (C))
Shapes BiasAddInfo::InferStrategyIndividualMode(const Shapes &in_strategy) {
  if (in_strategy.size() != 2) {
    MS_LOG(EXCEPTION) << name_ << ": The size of in_strategy must be 2, but got " << in_strategy.size();
  }

  if (!in_strategy[0].empty()) {
    if (in_strategy[0].size() != 2 && in_strategy[0].size() != 4) {
      MS_LOG(EXCEPTION) << name_ << ": The size of in_strategy[0] must be 2 or 4, but got " << in_strategy[0].size();
    }
    return Shapes({in_strategy[0], {in_strategy[0][1]}});
  }

  if (in_strategy[1].size() != 1) {
    MS_LOG(EXCEPTION) << name_ << ": The size of in_strategy[1] must be 1, but got " << in_strategy[1].size();
  }
  Shape tmp(inputs_shape_[0].size(), 1);
  tmp[1] = in_strategy[1][0];
  return Shapes({tmp, in_strategy[1]});
}

REGISTER(BiasAddInfo);
}  // namespace parallel
}  // namespace mindspore
