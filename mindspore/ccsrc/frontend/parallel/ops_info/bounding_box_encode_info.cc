/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/bounding_box_encode_info.h"

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status BoundingBoxEncodeInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies strategies = strategy->GetInputDim();
  Dimensions input_a_strategy = strategies[0];
  Dimensions input_b_strategy = strategies[1];
  if (input_a_strategy != input_b_strategy) {
    MS_LOG(ERROR) << name_ << ": Strategies of relevant dimensions must be equal, but the strategy is "
                  << StrategyToString(strategies);
    return FAILED;
  }
  if (input_a_strategy[1] != 1 || input_b_strategy[1] != 1) {
    MS_LOG(ERROR) << name_ << ": Cannot do this operator in the strategy: " << StrategyToString(strategies)
                  << ", only support shard the first dimension for each input tensor.";
    return FAILED;
  }
  if (input_a_strategy[0] > stage_device_size_) {
    MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(strategies) << ", it requires "
                  << input_a_strategy[0] << " devices, but the device number of this stage is " << stage_device_size_;
    return FAILED;
  }
  return SUCCESS;
}

Status BoundingBoxEncodeInfo::InferDevMatrixShape() {
  Strategies strategies = strategy_->GetInputDim();
  Dimensions input_a_strategy = strategies.at(0);

  dev_matrix_shape_.clear();
  dev_matrix_shape_.push_back(input_a_strategy[0]);
  MS_LOG(INFO) << name_ << ": The dev matrix is " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status BoundingBoxEncodeInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  (void)inputs_tensor_map_.emplace_back(TensorMap({0, -1}));
  (void)inputs_tensor_map_.emplace_back(TensorMap({0, -1}));
  (void)outputs_tensor_map_.emplace_back(TensorMap({0, -1}));
  return SUCCESS;
}

std::vector<StrategyPtr> BoundingBoxEncodeInfo::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp_vector;
  Shape input0_shape = inputs_shape_[0];
  Shape input1_shape = inputs_shape_[1];
  if (input0_shape != input1_shape) {
    MS_LOG(EXCEPTION) << "The shape of inputs must be equal.";
  }
  int64_t input0_length = input0_shape[0];
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  for (int64_t i = 1; i <= SizeToLong(dev_num) && i * i <= input0_length; ++i) {
    if (input0_length % i != 0) {
      continue;
    }
    StrategyPtr sp;
    if (PrepareStrategy(stage_id, i, dev_num, &sp) == SUCCESS) {
      sp_vector.push_back(sp);
    }
    if (PrepareStrategy(stage_id, input0_length / i, dev_num, &sp) == SUCCESS) {
      sp_vector.push_back(sp);
    }
  }

  if (sp_vector.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": No available strategy.";
  }
  return sp_vector;
}

Status BoundingBoxEncodeInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

Status BoundingBoxEncodeInfo::PrepareStrategy(int64_t stage_id, int64_t split_num, size_t dev_num,
                                              StrategyPtr *sp) const {
  const bool fully_use_device = CostModelContext::GetInstance()->fully_use_device();
  if (split_num == 0 || SizeToLong(dev_num) % split_num != 0 ||
      (fully_use_device && split_num != SizeToLong(dev_num))) {
    return FAILED;
  }

  Dimensions input0_partitions = {split_num, 1};
  Dimensions input1_partitions = {split_num, 1};
  Strategies strategies = {input0_partitions, input1_partitions};
  (*sp) = std::make_shared<Strategy>(stage_id, strategies);
  return SUCCESS;
}

void BoundingBoxEncodeInfo::ReComputeBatchSplitFlagList() {
  auto anchor_box_shape = inputs_shape_.at(0);
  auto gt_box_shape = inputs_shape_.at(1);
  anchor_box_shape[0] == 1 ? split_flag_list_[0] = false : split_flag_list_[0] = true;
  gt_box_shape[0] == 1 ? split_flag_list_[1] = false : split_flag_list_[1] = true;
}

REGISTER(BoundingBoxEncodeInfo);
}  // namespace parallel
}  // namespace mindspore
