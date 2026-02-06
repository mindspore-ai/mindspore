/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/dynamic_quant_ext_info.h"

#include <cstdint>
#include <string>
#include <memory>
#include <vector>
#include <utility>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/status.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "include/utils/parallel_context.h"
#include "frontend/jit/ps/resource.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr size_t kOutStrategySize = 2;
}  // namespace

Status DynamicQuantExtInfo::GetAttrs() { return SUCCESS; }

Status DynamicQuantExtInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  return SUCCESS;
}

Status DynamicQuantExtInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status DynamicQuantExtInfo::InferTensorMap() {
  TensorMap tensor_map_x;
  TensorMap tensor_map_scale;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << "The inputs shape is empty";
    return FAILED;
  }

  int32_t size = SizeToInt(inputs_shape_[0].size());
  for (int i = 0; i < size; ++i) {
    tensor_map_x.push_back(size - i - 1);
  }

  for (int i = 0; i < size - 1; ++i) {
    tensor_map_scale.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(tensor_map_x);

  constexpr size_t smooth_scales_input_size = 2;
  if (strategy()->GetInputDim().size() == smooth_scales_input_size) {
    TensorMap tensor_map_smooth_scales;
    tensor_map_smooth_scales.push_back(0);
    inputs_tensor_map_.push_back(tensor_map_smooth_scales);
  }

  outputs_tensor_map_.push_back(tensor_map_x);
  outputs_tensor_map_.push_back(tensor_map_scale);

  return SUCCESS;
}

Status DynamicQuantExtInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> DynamicQuantExtInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape split_flag_x;
  Shape split_flag_smooth_scales;
  size_t size = inputs_shape_[0].size();
  for (size_t i = 0; i < size; ++i) {
    if (i == size - 1) {
      split_flag_x.push_back(0);
    } else {
      split_flag_x.push_back(1);
    }
  }

  split_flag_smooth_scales.push_back(0);

  Shapes splittable_input = {split_flag_x, split_flag_smooth_scales};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies failed";
  }
  if (sp_vector.empty()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": No available strategy";
  }

  return sp_vector;
}

std::shared_ptr<Strategies> DynamicQuantExtInfo::GenerateBatchStrategies() {
  if (GetAttrs() != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Get attr failed";
  }
  Dimensions input_strategy_x(inputs_shape_[0].size(), 1);
  Dimensions input_strategy_smooth_scales(inputs_shape_[1].size(), 1);
  input_strategy_x[0] = stage_device_size_;
  Strategies strategy_v = {input_strategy_x, input_strategy_smooth_scales};
  return std::make_shared<Strategies>(strategy_v);
}

Status DynamicQuantExtInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor map is empty.";
    return FAILED;
  }

  if (outputs_tensor_map_[0].empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[0]) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

Status DynamicQuantExtInfo::CheckOutputStrategy(const StrategyPtr &out_strategy) {
  if (out_strategy == nullptr) {
    MS_LOG(INFO) << name_ << ": The output strategy is null";
    return SUCCESS;
  }
  std::vector<Dimensions> stra = out_strategy->GetInputDim();
  if (stra.size() != kOutStrategySize) {
    MS_LOG(ERROR) << name_ << ": The output strategy's size must be 2, now is " << stra.size();
    return FAILED;
  }
  Dimensions out1_stra = stra[0];
  Dimensions out2_stra = stra[1];
  if (out1_stra.size() != out2_stra.size() + 1) {
    MS_LOG(ERROR) << name_ << ": The first output strategy's size must be equal to that of the second plus 1, now is "
                  << out1_stra.size() << " and " << out2_stra.size();
    return FAILED;
  }
  for (size_t i = 0; i < out2_stra.size(); ++i) {
    if (out1_stra[i] != out2_stra[i]) {
      MS_LOG(ERROR) << name_ << ": The second output strategy must be equal to the first output strategy[0:-1]";
      return FAILED;
    }
  }
  return SUCCESS;
}

REGISTER(DynamicQuantExtInfo);
}  // namespace parallel
}  // namespace mindspore
