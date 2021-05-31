/**
#include "utils/log_adapter.h"
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

#include "frontend/parallel/ops_info/tmp_identity_info.h"

#include <memory>
#include <vector>

#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status TmpIdentityInfo::CheckStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  return CheckStrategyValue(strategy, inputs_shape_);
}

Status TmpIdentityInfo::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  dev_matrix_shape_ = input_strategy;
  return SUCCESS;
}

Status TmpIdentityInfo::InferTensorMap() {
  Shape tensor_map_index;
  size_t size = inputs_shape_[0].size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int64_t)(size - 1 - i));
  }

  inputs_tensor_map_.push_back(tensor_map_index);
  outputs_tensor_map_.push_back(tensor_map_index);
  return SUCCESS;
}

Status TmpIdentityInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status TmpIdentityInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status TmpIdentityInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> TmpIdentityInfo::GenerateOpStrategies(int64_t stage_id) {
  if ((inputs_shape_.size() != 1) || (outputs_shape_.size() != 1)) {
    MS_LOG(EXCEPTION) << name_ << ": Inputs shape size or outputs shape size is wrong, " << inputs_shape_.size() << ", "
                      << outputs_shape_.size();
  }
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": GenerateStrategiesForIndependentInputs failed.";
  }

  return sp_vector;
}
}  // namespace parallel
}  // namespace mindspore
