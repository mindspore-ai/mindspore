/**
#include "utils/log_adapter.h"
#include "utils/log_adapter.h"
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

#include "parallel/ops_info/tmp_identity_info.h"

#include <memory>
#include <vector>

#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status TmpIdentityInfo::CheckStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << ": invalid strategy.";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status TmpIdentityInfo::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  dev_matrix_shape_ = input_strategy;
  return SUCCESS;
}

Status TmpIdentityInfo::InferTensorMap() {
  std::vector<int32_t> tensor_map_index;
  size_t size = inputs_shape_[0].size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int32_t)(size - 1 - i));
  }

  inputs_tensor_map_.push_back(tensor_map_index);
  outputs_tensor_map_.push_back(tensor_map_index);
  return SUCCESS;
}

Status TmpIdentityInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Strategys outputs_strategy = {inputs_strategy.at(0)};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }
  Shape input_slice_shape = inputs_slice_shape.at(0);

  TensorLayout input_tensor_layout;
  if (input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], input_shape) != SUCCESS) {
    return FAILED;
  }

  TensorInfo input_tensor_info(input_tensor_layout, input_shape, input_slice_shape);

  inputs_tensor_info_.push_back(input_tensor_info);
  outputs_tensor_info_.push_back(input_tensor_info);  // the same as input

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
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    }
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status TmpIdentityInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Set cost under strategy failed.";
    }
    return FAILED;
  }

  return SUCCESS;
}

Status TmpIdentityInfo::GenerateStrategies(int32_t stage_id) {
  if ((inputs_shape_.size() != 1) || (outputs_shape_.size() != 1)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size or outputs shape size is wrong, " << inputs_shape_.size() << ", "
                  << outputs_shape_.size();
    return FAILED;
  }
  is_auto_parallel_ = true;
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateStrategiesForIndependentInputs failed.";
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
