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
// The first dimension of input can not be split.
// The indices can not be split.
// The first n dimensions(n is indices' dimension size) of updates can not be split.
// The shape of input:   [A, B, ..., M], the strategy of input: (1, b, ..., m)
// The shape of indices: [N, O, ..., Z], the strategy of indices: (1, 1, ..., 1)
// The shape of updates: [N, O, ..., Z, B, ..., M], the strategy of updates: (1, 1, ..., 1, b, ..., m)
// The shape of output:  [A, B, ..., M], the strategy of output: (1, b, ..., m)
// The dev matrix: (1, b, ..., m)
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

  if (stra[2].empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy[2] is empty";
    return FAILED;
  }

  if (std::accumulate(stra[2].begin(), stra[2].begin() + static_cast<different_type>(stra[1].size()), 1,
                      std::multiplies<int64_t>()) != 1) {
    MS_LOG(ERROR) << name_ << ": The first " << stra[1].size() << " dimensions of updates can not be split";
    return FAILED;
  }

  if (stra[0].size() - 1 != stra[2].size() - stra[1].size()) {
    MS_LOG(ERROR) << name_ << ": updates.strategy must be equal to indices.strategy + input.strategy[1:]";
    return FAILED;
  }

  for (size_t i = 1; i < stra[0].size(); ++i) {
    if (stra[0][i] != stra[2][stra[1].size() + i - 1]) {
      MS_LOG(ERROR) << name_ << ": updates.strategy must be equal to indices.strategy + input.strategy[1:]";
      return FAILED;
    }
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
  if (inputs_shape_.size() != 3) {
    MS_LOG(ERROR) << name_ << "The size of inputs shape must be 3";
    return FAILED;
  }

  TensorMap input_tensor_map, updates_tensor_map;
  TensorMap indices_tensor_map(inputs_shape_[1].size(), MAP_NONE);

  // cannot use dev_matrix_shape_ replace inputs_shape_[0], because it may not be fully split in all devices.
  int64_t size = SizeToLong(inputs_shape_[0].size());
  for (int64_t i = 0; i < size; ++i) {
    input_tensor_map.push_back(size - i - 1);
  }

  // updates_tensor_map = indices_tensor_map + input_tensor_map[1:]
  updates_tensor_map = indices_tensor_map;
  for (size_t i = 1; i < input_tensor_map.size(); ++i) {
    updates_tensor_map.push_back(input_tensor_map[i]);
  }
  inputs_tensor_map_.push_back(input_tensor_map);    // input
  inputs_tensor_map_.push_back(indices_tensor_map);  // indices
  inputs_tensor_map_.push_back(updates_tensor_map);  // updates

  outputs_tensor_map_.push_back(input_tensor_map);
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

std::vector<StrategyPtr> ScatterUpdateInfo::GenerateOpStrategies(int64_t stage_id) {
  // to generate the first input's strategy
  Shape input_split(inputs_shape_[0].size(), 1);
  input_split[0] = 0;
  Shapes splittable_input = {input_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies failed";
  }

  // the others strategies are equal to the first input's strategy
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }
    Strategys tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    Dimensions indices_strategy(inputs_shape_[1].size(), 1);
    // updates_strategy = indices_strategy + input_strategy[1:]
    Dimensions updates_strategy = indices_strategy;
    for (size_t i = 1; i < first_input_strategy.size(); ++i) {
      updates_strategy.push_back(first_input_strategy[i]);
    }

    tmp_strategy.push_back(first_input_strategy);  // input
    tmp_strategy.push_back(indices_strategy);      // indices
    tmp_strategy.push_back(updates_strategy);      // updates

    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
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
