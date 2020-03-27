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

#include "parallel/ops_info/l2_normalize_info.h"

#include <algorithm>
#include <vector>
#include <utility>
#include <memory>

#include "parallel/device_matrix.h"
#include "parallel/strategy.h"
#include "parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status L2NormalizeInfo::CheckStrategy(const StrategyPtr& strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
    } else {
      MS_LOG(INFO) << name_ << " : Init success.";
    }
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  int32_t axis_index = axis_;
  if (axis_ < 0) {
    size_t input_dim = inputs_shape_.at(0).size();
    axis_index = static_cast<int32_t>(input_dim) + axis_;
  }

  if (input_strategy[IntToSize(axis_index)] != 1) {
    MS_LOG(ERROR) << name_ << " : The dim " << axis_index << " of input strategy must be 1.";
    return FAILED;
  }

  return SUCCESS;
}

Status L2NormalizeInfo::GetAttrs() {
  auto iter = attrs_.find(AXIS);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int32Imm>()) {
      axis_ = iter->second->cast<Int32ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << " : The value of axis is not int.";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status L2NormalizeInfo::InferMirrorOps() {
  mirror_ops_.clear();
  Shape input_tensor_map = inputs_tensor_map_.at(0);
  std::vector<Group> input_group;
  if (CreateGroupByTensorMap(input_tensor_map, &input_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create group failed.";
    return FAILED;
  }

  OperatorVector op_for_weight;
  if (input_group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror ops is empty.";
    return SUCCESS;
  } else {
    op_for_weight = CreateMirrorOps(input_group[0].name(), input_group[0].GetDevNum());
    mirror_ops_.push_back(op_for_weight);
    MS_LOG(INFO) << name_ << " : Create the mirror ops success, the group is " << input_group[0].name();
  }

  return SUCCESS;
}

Status L2NormalizeInfo::GenerateStrategies(int32_t stage_id) {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : GetAttrs failed.";
    return FAILED;
  }
  is_auto_parallel_ = true;
  Shape input0_split(inputs_shape_[0].size() - 1, 1);
  int32_t axis_index = axis_;
  if (axis_ < 0) {
    size_t input_dim = inputs_shape_.at(0).size();
    axis_index = static_cast<int32_t>(input_dim) + axis_;
  }
  (void)input0_split.insert(input0_split.begin() + axis_index, 0);
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Generate strategies failed.";
    return FAILED;
  }
  size_t success = 0;
  for (auto& sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << " : Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
