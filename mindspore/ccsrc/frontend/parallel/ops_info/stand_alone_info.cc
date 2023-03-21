/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/stand_alone_info.h"

#include <memory>
#include <utility>

#include "ir/value.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
Status StandAloneInfo::CheckStrategy(const StrategyPtr &strategy) { return SUCCESS; }

Status StandAloneInfo::InferDevMatrixShape() {
  dev_matrix_shape_.push_back(stage_device_size_);
  return SUCCESS;
}

Status StandAloneInfo::InferForwardCommunication() { return SUCCESS; }

Status StandAloneInfo::GetAttrs() { return SUCCESS; }

Status StandAloneInfo::InferTensorMap() {
  // input tensor map, all -1
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    Shape tensor_map_index;
    for (size_t j = 0; j < inputs_shape_[i].size(); ++j) {
      tensor_map_index.push_back(MAP_NONE);
    }
    inputs_tensor_map_.push_back(tensor_map_index);
  }
  // output tensor map, all -1
  for (size_t i = 0; i < outputs_shape_.size(); i++) {
    Shape tensor_map_index;
    for (size_t j = 0; j < outputs_shape_[i].size(); ++j) {
      tensor_map_index.push_back(MAP_NONE);
    }
    outputs_tensor_map_.push_back(tensor_map_index);
  }
  return SUCCESS;
}

Status StandAloneInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }
  // infer input TensorInfo
  size_t temp = 0;
  for (size_t i = 0; i < input_value_.size(); ++i) {
    if (!input_value_[i]) {
      TensorLayout input_layout;
      if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[temp], inputs_shape_[temp]) != SUCCESS) {
        MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed, the index is " << i;
        return FAILED;
      }
      temp += 1;
      TensorInfo input_tensor_info(input_layout);
      inputs_tensor_info_.push_back(input_tensor_info);
    } else {
      TensorInfo empty_tensor_info;
      inputs_tensor_info_.push_back(empty_tensor_info);
    }
  }
  // infer output TensorInfo
  for (size_t j = 0; j < outputs_shape_.size(); ++j) {
    TensorLayout output_layout;
    if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[j], outputs_shape_[j]) != SUCCESS) {
      return FAILED;
    }
    TensorInfo out_tensor_info(output_layout);
    outputs_tensor_info_.push_back(out_tensor_info);
  }

  return SUCCESS;
}

Status StandAloneInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> StandAloneInfo::GenerateOpStrategies(int64_t stage_id) {
  StrategyPtr sp;
  Strategies strategy;
  ComputeBatchSplitFlagList();
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    Shape temp(inputs_shape_[i].size(), 1);
    strategy.push_back(temp);
  }
  sp = std::make_shared<Strategy>(stage_id, strategy);
  std::vector<StrategyPtr> sp_vector;
  sp_vector.push_back(sp);

  return sp_vector;
}

Status StandAloneInfo::InferAsLossDivisor() {
  as_loss_divisor_ = stage_device_size_;
  return SUCCESS;
}

REGISTER(StandAloneInfo);
}  // namespace parallel
}  // namespace mindspore
