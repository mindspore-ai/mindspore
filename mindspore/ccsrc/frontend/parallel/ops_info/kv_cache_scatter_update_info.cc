/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/kv_cache_scatter_update_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace parallel {
// The first dimension of input can not be split.
// The indices can not be split.
// The first n dimensions(n is indices' dimension size) of updates can not be split.
// The shape of input:   [A, B, Z, M], the strategy of input: (1, b, z, m)
// The shape of indices: [N], the strategy of indices: (1)
// The shape of updates: [A, B, Z, M], the strategy of updates: (1, b, z, m)
// The shape of output:  [A, B, Z, M], the strategy of output: (1, b, z, m)
// The dev matrix: (1, b, z, m)
Status KVCacheScatterUpdateInfo::CheckStrategy(const StrategyPtr &strategy) {
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

  return SUCCESS;
}

Status KVCacheScatterUpdateInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status KVCacheScatterUpdateInfo::InferTensorMap() {
  if (inputs_shape_.size() != 3) {
    MS_LOG(ERROR) << name_ << "The size of inputs shape must be 3";
    return FAILED;
  }

  inputs_tensor_map_.push_back({3, 2, 1, 0});  // input
  inputs_tensor_map_.push_back({-1});          // indices
  inputs_tensor_map_.push_back({3, 2, 1, 0});  // updates

  outputs_tensor_map_.push_back({3, 2, 1, 0});
  return SUCCESS;
}

void KVCacheScatterUpdateInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    split_flag_list_[i] = false;  // the first dimension can not be split
  }
}

Status KVCacheScatterUpdateInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> KVCacheScatterUpdateInfo::GenerateOpStrategies(int64_t stage_id) {
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

    Strategies tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    Dimensions second_input_strategy = sp->GetInputDim()[1];
    Dimensions third_input_strategy = sp->GetInputDim()[2];

    tmp_strategy.push_back(first_input_strategy);   // input
    tmp_strategy.push_back(second_input_strategy);  // indices, can not be split
    tmp_strategy.push_back(third_input_strategy);   // updates

    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

REGISTER(KVCacheScatterUpdateInfo);
}  // namespace parallel
}  // namespace mindspore
