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

#include "frontend/parallel/ops_info/gathernd_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include <functional>
#include <string>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
// the input can not be split, and the last dimension of indices can not be split
Status GatherNdInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of strategies must be 2";
    return FAILED;
  }

  int64_t input_split_size = std::accumulate(stra[0].begin(), stra[0].end(), 1, std::multiplies<int64_t>());
  if (input_split_size != 1) {
    MS_LOG(ERROR) << name_ << ": The input can not be split";
    return FAILED;
  }

  if (stra[1].empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy of indices can not be empty";
    return FAILED;
  }

  if (stra[1].back() != 1) {
    MS_LOG(ERROR) << name_ << ": The last dimension of indices can not be split";
    return FAILED;
  }

  return SUCCESS;
}

// the dev matrix is indices_strategy
Status GatherNdInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << "The size of strategies must be 2";
    return FAILED;
  }

  dev_matrix_shape_ = stra[1];
  return SUCCESS;
}

// input shape: [x, y, z], indices shape: [a, b, c, 2], output shape: [a, b, c, z]
// strategy: ((1, 1, 1), (m, n, o, 1))
// dev-matrix: [m, n, o, 1]
// input map: [-1, -1, -1], indices map: [3, 2, 1, 0], output map: [3, 2, 1, -1]
Status GatherNdInfo::InferTensorMap() {
  if (inputs_shape_.size() != 2) {
    MS_LOG(ERROR) << name_ << "The size of input shapes must be 2";
    return FAILED;
  }

  if (outputs_shape_.empty() || outputs_shape_[0].size() < (inputs_shape_[1].size() - 1)) {
    MS_LOG(ERROR) << name_ << "invalid shapes";
    return FAILED;
  }

  TensorMap input_tensor_map(inputs_shape_[0].size(), MAP_NONE);  // the input can not split

  // cannot use dev_matrix_shape_ replace inputs_shape_[0], because it may not be fully split in all devices.
  TensorMap indices_tensor_map;
  int64_t size = SizeToLong(inputs_shape_[1].size());
  for (int64_t i = 0; i < size; ++i) {
    indices_tensor_map.push_back(size - i - 1);
  }

  TensorMap output_tensor_map(outputs_shape_[0].size(), MAP_NONE);
  for (size_t i = 0; i < (inputs_shape_[1].size() - 1); ++i) {
    output_tensor_map[i] = indices_tensor_map[i];
  }

  inputs_tensor_map_.push_back(input_tensor_map);
  inputs_tensor_map_.push_back(indices_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
  return SUCCESS;
}

void GatherNdInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = false;
  split_flag_list_[1] = true;
}

Status GatherNdInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> GatherNdInfo::GenerateOpStrategies(int64_t stage_id) {
  if (inputs_shape_.empty()) {
    MS_LOG(EXCEPTION) << name_ << ": The inputs shape is empty";
  }

  // to generate the indices' strategy
  Shape input_split(inputs_shape_[1].size(), 1);
  input_split.back() = 0;
  Shapes splittable_input = {input_split};
  Shapes tmp_inputs_shape = {inputs_shape_[1]};

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
    Dimensions indices_strategy = sp->GetInputDim()[0];
    Dimensions input_strategy(inputs_shape_[0].size(), 1);
    tmp_strategy.push_back(input_strategy);
    tmp_strategy.push_back(indices_strategy);
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

Status GatherNdInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status GatherNdInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
