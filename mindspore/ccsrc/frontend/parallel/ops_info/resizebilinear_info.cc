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

#include "frontend/parallel/ops_info/resizebilinear_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status ResizeBilinearInfo::GetAttrs() {
  size_ = GetTupleIntAttr(SIZE);
  if (size_.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of input size must be 2, but got " << size_.size();
    return FAILED;
  }

  if (size_[0] != size_[1]) {
    MS_LOG(ERROR) << name_ << ": The second two elements of size must be the same, but got (" << size_[0] << ", "
                  << size_[1] << ")";
    return FAILED;
  }

  align_corners_ = GetBoolAttr(ALIGN_CORNERS);

  MS_LOG(INFO) << name_ << ": The input size is " << size_ << ", align_corners is " << align_corners_;

  return SUCCESS;
}

Status ResizeBilinearInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 1) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 1, but got " << stra.size();
    return FAILED;
  }

  Dimensions input_strategy = stra[0];
  if (input_strategy.size() != 4) {
    MS_LOG(ERROR) << name_ << ": The size of input strategy must be 4, but got" << input_strategy.size();
    return FAILED;
  }

  if (input_strategy[2] != 1 || input_strategy[3] != 1) {
    MS_LOG(ERROR) << name_ << ": Do not support split from H or W";
    return FAILED;
  }

  return SUCCESS;
}

Status ResizeBilinearInfo::InferDevMatrixShape() {
  // the strategy is (n, c, h, w)
  // the dev matrix is (n, c, h, w)
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status ResizeBilinearInfo::InferTensorMap() {
  // input_strategy: (n, c, h, w)
  // output_strategy: (n, c, h, w)
  // dev_matrix: (n, c, h, w)
  TensorMap input_tensor_map = {3, 2, 1, 0};
  TensorMap output_tensor_map = {3, 2, 1, 0};

  (void)inputs_tensor_map_.emplace_back(std::move(input_tensor_map));
  (void)outputs_tensor_map_.emplace_back(std::move(output_tensor_map));
  return SUCCESS;
}

Status ResizeBilinearInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> ResizeBilinearInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 0);
  input0_split[0] = 1;
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }
  return sp_vector;
}

Status ResizeBilinearInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status ResizeBilinearInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
