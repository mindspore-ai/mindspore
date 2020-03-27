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

#include "optimizer/parallel/ops_info/dropout_do_mask_info.h"

#include <algorithm>
#include <vector>
#include <utility>
#include <memory>

#include "ir/value.h"
#include "optimizer/parallel/device_matrix.h"
#include "optimizer/parallel/strategy.h"
#include "optimizer/parallel/tensor_layout/tensor_redistribution.h"
#include "optimizer/parallel/auto_parallel/costmodel.h"

namespace mindspore {
namespace parallel {
Status DropoutDoMaskInfo::CheckStrategy(const StrategyPtr& strategy) {
  Shapes input_shape = {inputs_shape_.at(0)};
  if (CheckStrategyValue(strategy, input_shape, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status DropoutDoMaskInfo::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  dev_matrix_shape_ = input_strategy;

  return SUCCESS;
}

Status DropoutDoMaskInfo::InferTensorMap() {
  std::vector<int32_t> tensor_map_index;
  size_t size = inputs_shape_.at(0).size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int32_t)(LAST_INDEX(size) - i));
  }

  TensorMap input_b_tensor_map = {MAP_NONE};
  inputs_tensor_map_.push_back(tensor_map_index);
  inputs_tensor_map_.push_back(input_b_tensor_map);
  outputs_tensor_map_.push_back(tensor_map_index);
  return SUCCESS;
}

Status DropoutDoMaskInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_a_shape = inputs_shape_.at(0);
  Shape input_b_shape = inputs_shape_.at(1);
  Shape output_shape = outputs_shape_.at(0);

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Dimensions input_b_strategy = {1}, input_x_strategy = {};
  inputs_strategy.emplace_back(input_b_strategy);
  inputs_strategy.emplace_back(input_x_strategy);
  Strategys outputs_strategy = {inputs_strategy.at(0)};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }
  Shape input_a_slice_shape = inputs_slice_shape.at(0);
  Shape input_b_slice_shape = inputs_slice_shape.at(1);
  Shape output_slice_shape = outputs_slice_shape.at(0);

  TensorLayout input_a_tensor_layout, input_b_tensor_layout;
  TensorLayout output_tensor_layout;
  if (input_a_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], input_a_shape) != SUCCESS) {
    return FAILED;
  }
  if (input_b_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[1], input_b_shape) != SUCCESS) {
    return FAILED;
  }
  if (output_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], output_shape) != SUCCESS) {
    return FAILED;
  }
  TensorInfo input_a_tensor_info(input_a_tensor_layout, input_a_shape, input_a_slice_shape);
  TensorInfo input_b_tensor_info(input_b_tensor_layout, input_b_shape, input_b_slice_shape);
  TensorInfo output_tensor_info(output_tensor_layout, output_shape, output_slice_shape);

  inputs_tensor_info_.push_back(input_a_tensor_info);
  inputs_tensor_info_.push_back(input_b_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);

  return SUCCESS;
}

Status DropoutDoMaskInfo::SetCostUnderStrategy(const StrategyPtr& strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << " : Set cost under strategy failed.";
    }
    return FAILED;
  }

  return SUCCESS;
}

Status DropoutDoMaskInfo::GenerateStrategies(int32_t stage_id) {
  CheckGlobalDeviceManager();
  is_auto_parallel_ = true;
  size_t dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  Dimensions strategy(inputs_shape_[0].size() - 1, 1);
  (void)strategy.insert(strategy.begin(), SizeToInt(dev_num));
  std::vector<Dimensions> stra = {strategy};
  StrategyPtr sp = std::make_shared<Strategy>(stage_id, stra);
  if (SetCostUnderStrategy(sp) == SUCCESS) {
    MS_LOG(INFO) << name_ << " : Successfully generated batch-parallel-strategy.";
    PrintStrategy(sp);
  } else {
    MS_LOG(ERROR) << name_ << " : Generating batch-parallel-strategy failed.";
    return FAILED;
  }
  return SUCCESS;
}

std::shared_ptr<std::vector<std::vector<int32_t>>> DropoutDoMaskInfo::GenerateBatchStrategies() {
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(0).size();
  Dimensions strategy(inputs_shape_[0].size() - 1, 1);
  (void)strategy.insert(strategy.begin(), SizeToInt(dev_num));
  std::vector<Dimensions> strategy_v = {strategy};
  return std::make_shared<std::vector<std::vector<int32_t>>>(strategy_v);
}

Status DropoutDoMaskInfo::Init(const StrategyPtr& strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status DropoutDoMaskInfo::InitForCostModel(const StrategyPtr& strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << " : Init for cost model failed.";
    }
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
