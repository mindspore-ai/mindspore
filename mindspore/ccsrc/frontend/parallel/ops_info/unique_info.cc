/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/unique_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/context.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
/*
 * unique has one input, two outputs. Currently, unique cannot be split.
 */
Status UniqueInfo::InferTensorMap() {
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  for (auto shp : inputs_shape_) {
    TensorMap out_tensor_map;
    TensorMap in_tensor_map;
    for (size_t i = 0; i < shp.size(); ++i) {
      in_tensor_map.push_back(MAP_NONE);
      out_tensor_map.push_back(MAP_NONE);
    }
    inputs_tensor_map_.push_back(in_tensor_map);
    outputs_tensor_map_.push_back(out_tensor_map);
    outputs_tensor_map_.push_back(out_tensor_map);
  }
  return SUCCESS;
}

Status UniqueInfo::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout) {
  if (inputs_layout == nullptr || outputs_layout == nullptr) {
    MS_LOG(ERROR) << name_ << " : The layout is null.";
    return FAILED;
  }
  TensorLayout input_layout;
  TensorLayout output_layout;
  TensorLayout index_layout;
  if ((input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], inputs_shape_[0]) != SUCCESS) ||
      (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], outputs_shape_[0]) != SUCCESS) ||
      (index_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[1], outputs_shape_[1]) != SUCCESS)) {
    return FAILED;
  }
  inputs_layout->push_back(input_layout);
  outputs_layout->push_back(output_layout);
  outputs_layout->push_back(index_layout);
  return SUCCESS;
}

Status UniqueInfo::InferTensorInfo() {
  TensorLayouts inputs_layout;
  TensorLayouts outputs_layout;
  if (InferTensorLayout(&inputs_layout, &outputs_layout) != SUCCESS) {
    return FAILED;
  }
  for (size_t i = 0; i < inputs_layout.size(); ++i) {
    TensorInfo input_tensor_info(inputs_layout[i]);
    inputs_tensor_info_.push_back(input_tensor_info);
  }
  for (size_t i = 0; i < outputs_layout.size(); ++i) {
    TensorInfo output_tensor_info(outputs_layout[i]);
    outputs_tensor_info_.push_back(output_tensor_info);
  }
  return SUCCESS;
}

Status UniqueInfo::InferDevMatrixShape() {
  dev_matrix_shape_.push_back(stage_device_size_);
  return SUCCESS;
}

Status UniqueInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << " : Init success";
  return SUCCESS;
}

Status UniqueInfo::CheckStrategy(const StrategyPtr &strategy) {
  Strategys stras = strategy->GetInputDim();
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    return FAILED;
  }
  for (Dimensions stra : stras) {
    if (stra.size() != UNIQUE_INPUT_SIZE) {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
      return FAILED;
    }
  }

  if (stras[0][0] != 1) {
    MS_LOG(ERROR) << "Currently, unique only support repeat calculate in all devices";
    return FAILED;
  }
  return SUCCESS;
}

Status UniqueInfo::GetAttrs() {
  if ((inputs_shape_.size() != UNIQUE_INPUTS_SIZE) || (outputs_shape_.size() != UNIQUE_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size " << inputs_shape_.size() << " or outputs shape size "
                  << outputs_shape_.size() << " is wrong.";
    return FAILED;
  }
  return SUCCESS;
}

Status UniqueInfo::InferMirrorOps() {
  mirror_ops_.clear();

  Shape tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(tensor_map, &group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create group failed.";
    return FAILED;
  }
  OperatorVector mirror_op;
  if (group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror ops is empty.";
    return SUCCESS;
  } else {
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
    std::string group_name = group[0].name();
    MS_LOG(INFO) << name_ << " : Create the mirror ops success, the group name is " << group_name;
  }

  return SUCCESS;
}

Status UniqueInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init for cost model failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}

Status UniqueInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Status UniqueInfo::GenerateStrategies(int64_t stage_id) {
  if (inputs_shape_.size() != UNIQUE_INPUTS_SIZE) {
    return FAILED;
  }
  if (inputs_shape_[0].size() != UNIQUE_INPUT_SIZE) {
    return FAILED;
  }
  Shape input0_split;
  input0_split.emplace_back(0);
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateStrategiesForIndependentInputs failed";
    return FAILED;
  }
  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
