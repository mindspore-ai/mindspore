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

#include "frontend/parallel/ops_info/iou_info.h"

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status IOUInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies strategies = strategy->GetInputDim();
  if (strategies[0][1] != 1 || strategies[1][1] != 1) {
    MS_LOG(ERROR) << name_ << ": Only supports shard the 0th dimension of each input tensor, but got strategy "
                  << StrategyToString(strategies);
    return FAILED;
  }
  return SUCCESS;
}

Status IOUInfo::InferDevMatrixShape() {
  Strategies strategise = strategy_->GetInputDim();
  int64_t dev1 = strategise[0][0];
  int64_t dev0 = strategise[1][0];

  dev_matrix_shape_.clear();
  dev_matrix_shape_ = {dev1, dev0};
  MS_LOG(INFO) << name_ << ": The dev matrix is " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status IOUInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  (void)inputs_tensor_map_.emplace_back(TensorMap({1, -1}));
  (void)inputs_tensor_map_.emplace_back(TensorMap({0, -1}));
  (void)outputs_tensor_map_.emplace_back(TensorMap({0, 1}));
  return SUCCESS;
}

std::vector<StrategyPtr> IOUInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split({1, 0});
  Shape input1_split({1, 0});
  Shapes splittable_inputs = {input0_split, input1_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs failed.";
  }
  return sp_vector;
}

REGISTER(IOUInfo);
}  // namespace parallel
}  // namespace mindspore
