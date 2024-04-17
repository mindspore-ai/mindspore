/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/random_choice_with_mask_info.h"
#include <algorithm>
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
int64_t RandomChoiceWithMaskInfo::SEED_NUM = 1;

Status RandomChoiceWithMaskInfo::GetAttrs() {
  if (attrs_.find(SEED) != attrs_.end()) {
    seed_ = GetValue<int64_t>(attrs_[SEED]);
  }
  if (attrs_.find(SEED2) != attrs_.end()) {
    seed2_ = GetValue<int64_t>(attrs_[SEED2]);
  }
  return SUCCESS;
}

Status RandomChoiceWithMaskInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies strategies = strategy->GetInputDim();
  Dimensions input_strategy = strategies[0];
  auto is_shard = [](int64_t val) -> bool { return val != 1; };
  if (std::any_of(input_strategy.begin(), input_strategy.end(), is_shard)) {
    MS_LOG(ERROR) << name_ << ": Each dimension of input tensor is not splittable, but the strategy is "
                  << StrategyToString(strategies);
    return FAILED;
  }
  return SUCCESS;
}

Status RandomChoiceWithMaskInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();
  return SUCCESS;
}

Status RandomChoiceWithMaskInfo::InferTensorMap() {
  Shape input0_shape = inputs_shape_.at(0);
  Shape output0_shape = outputs_shape_.at(0);
  Shape output1_shape = outputs_shape_.at(1);

  inputs_tensor_map_.clear();
  (void)inputs_tensor_map_.emplace_back(Shape(input0_shape.size(), -1));
  (void)outputs_tensor_map_.emplace_back(Shape(output0_shape.size(), -1));
  (void)outputs_tensor_map_.emplace_back(Shape(output1_shape.size(), -1));
  return SUCCESS;
}

std::vector<StrategyPtr> RandomChoiceWithMaskInfo::GenerateOpStrategies(int64_t stage_id) {
  Dimensions input_partitions(inputs_shape_[0].size(), 1);
  Strategies strategies = {input_partitions};
  std::vector<StrategyPtr> sp_vector;
  (void)sp_vector.emplace_back(std::make_shared<Strategy>(stage_id, strategies));
  return sp_vector;
}

Status RandomChoiceWithMaskInfo::InferAsLossDivisor() {
  if (out_dev_matrix_shape_.empty()) {
    out_dev_matrix_shape_ = dev_matrix_shape_;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[0]) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

void RandomChoiceWithMaskInfo::ReplaceNodeInputOrAttrs() {
  // seed and seed2 cannot both be 0.
  if (seed_ != 0 || seed2_ != 0) {
    return;
  }
  cnode_->AddAttr(SEED, MakeValue(SEED_NUM));
  cnode_->AddAttr(SEED2, MakeValue(SEED_NUM));
  ++SEED_NUM;
}

void RandomChoiceWithMaskInfo::ReComputeBatchSplitFlagList() { split_flag_list_[0] = false; }

REGISTER(RandomChoiceWithMaskInfo);
}  // namespace parallel
}  // namespace mindspore
