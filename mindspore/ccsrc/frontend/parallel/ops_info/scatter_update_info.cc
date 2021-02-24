/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/scatter_update_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
// The indices can not be split.
// The strategy of input and the strategy of updates must be equal.
// The first dimension of input or updates can not be split.
Status ScatterUpdateInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 3) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 3";
    return FAILED;
  }

  if (stra[0] != stra[2]) {
    MS_LOG(ERROR) << name_ << ": The strategy[0] and strategy[2] must be equal";
    return FAILED;
  }

  if (stra[0].empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy[0] is empty";
    return FAILED;
  }

  if (stra[0][0] != 1) {
    MS_LOG(ERROR) << name_ << ": The first dimension of input can not be split";
    return FAILED;
  }

  if (!stra[1].empty() && std::accumulate(stra[1].begin(), stra[1].end(), 1, std::multiplies<int64_t>()) != 1) {
    MS_LOG(ERROR) << name_ << ": The indices can not be split";
    return FAILED;
  }

  return SUCCESS;
}

Status ScatterUpdateInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status ScatterUpdateInfo::InferTensorMap() {
  TensorMap input_tensor_map;
  TensorMap indices_tensor_map(inputs_shape_[1].size(), MAP_NONE);
  if (inputs_shape_.size() != 3) {
    MS_LOG(ERROR) << name_ << "The size of inputs shape must be 3";
    return FAILED;
  }

  // cannot use dev_matrix_shape_ replace inputs_shape_[0], because it may not be fully split in all devices.
  int64_t size = SizeToLong(inputs_shape_[0].size());
  for (int64_t i = 0; i < size; ++i) {
    input_tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(input_tensor_map);    // input
  inputs_tensor_map_.push_back(indices_tensor_map);  // indices
  inputs_tensor_map_.push_back(input_tensor_map);    // updates

  outputs_tensor_map_.push_back(input_tensor_map);
  return SUCCESS;
}

Status ScatterUpdateInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }

  TensorLayout input_layout, output_layout;
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    // infer tensor layout
    if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[i], inputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed.";
      return FAILED;
    }
    TensorInfo input_tensor_info(input_layout);
    inputs_tensor_info_.push_back(input_tensor_info);
  }

  if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], outputs_shape_[0]) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed.";
    return FAILED;
  }
  TensorInfo output_tensor_info(output_layout);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

void ScatterUpdateInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    split_flag_list_[i] = false;  // the first dimension can not be split
  }
}

Status ScatterUpdateInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

Status ScatterUpdateInfo::GenerateStrategies(int64_t stage_id) {
  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer attrs failed";
    return FAILED;
  }
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  // to generate the first input's strategy
  Shape input_split(inputs_shape_[0].size(), 1);
  input_split[0] = 0;
  Shapes splittable_input = {input_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Generate strategies failed";
    return FAILED;
  }

  // the others strategies are equal to the first input's strategy
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(ERROR) << name_ << ": The strategy is null or empty";
      return FAILED;
    }
    Strategys tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    Dimensions indices_strategy(inputs_shape_[1].size(), 1);
    tmp_strategy.push_back(first_input_strategy);  // input
    tmp_strategy.push_back(indices_strategy);      // indices
    tmp_strategy.push_back(first_input_strategy);  // updates

    sp->ResetInputs(tmp_strategy);
  }

  size_t success = 0;
  for (auto &sp : sp_vector) {
    PrintStrategy(sp);
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

Status ScatterUpdateInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status ScatterUpdateInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
