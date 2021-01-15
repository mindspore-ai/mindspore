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

#include "frontend/parallel/ops_info/batch_parallel_info.h"

#include <memory>
#include <utility>

#include "ir/value.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
Status BatchParallelInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    return FAILED;
  }

  size_t strategy_size = strategy->GetInputNumber();
  Strategys stra = strategy->GetInputDim();
  for (size_t i = 0; i < strategy_size; ++i) {
    Shape sub_strategy = stra.at(i);
    size_t strategy_len = sub_strategy.size();
    bool flag = false;
    for (size_t j = 0; j < strategy_len; ++j) {
      int64_t strategy_value = sub_strategy.at(j);
      if (strategy_value > 1) {
        if (flag || strategy_value != stage_device_size_) {
          MS_LOG(ERROR) << name_ << " : It is not a valid data parallel strategy.";
          return FAILED;
        }
        flag = true;
      }
    }
  }
  return SUCCESS;
}

Status BatchParallelInfo::InferDevMatrixShape() {
  dev_matrix_shape_.push_back(stage_device_size_);
  return SUCCESS;
}

Status BatchParallelInfo::InferForwardCommunication() { return SUCCESS; }

Status BatchParallelInfo::InferTensorMap() {
  if (strategy_->GetInputDim()[0][0] != stage_device_size_) {
    MS_LOG(ERROR) << name_ << " : It is not a valid data parallel strategy.";
    return FAILED;
  }
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    Shape tensor_map_index;
    for (size_t j = 0; j < inputs_shape_[i].size(); ++j) {
      if (strategy_->GetInputDim()[i][j] == stage_device_size_ && j == 0) {
        tensor_map_index.push_back(0);
      } else {
        tensor_map_index.push_back(MAP_NONE);
      }
    }
    inputs_tensor_map_.push_back(tensor_map_index);
  }
  for (size_t i = 0; i < outputs_shape_.size(); i++) {
    Shape tensor_map_index;
    for (size_t j = 0; j < outputs_shape_[i].size(); ++j) {
      if (i == 0 && j == 0) {
        tensor_map_index.push_back(0);
      } else {
        tensor_map_index.push_back(MAP_NONE);
      }
    }
    outputs_tensor_map_.push_back(tensor_map_index);
  }
  return SUCCESS;
}

Strategys BatchParallelInfo::GetOutputsStrategy() {
  Strategys outputs_strategy;

  for (size_t i = 0; i < outputs_shape_.size(); ++i) {
    Dimensions strategy;
    for (size_t j = 0; j < outputs_shape_[i].size(); ++j) {
      if (i == 0 && j == 0) {
        strategy.push_back(stage_device_size_);
      } else {
        strategy.push_back(1);
      }
    }
    outputs_strategy.push_back(strategy);
  }

  return outputs_strategy;
}

Status BatchParallelInfo::InferTensorInfo() {
  for (size_t i = 0; i < strategy_->GetInputNumber(); i++) {
    MS_LOG(INFO) << name_ << " : The input size is " << strategy_->GetInputNumber();
    TensorLayout tensor_layout_in;
    if (tensor_layout_in.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(i), inputs_shape_.at(i)) != SUCCESS) {
      return FAILED;
    }
    TensorInfo tensor_info_in(tensor_layout_in);
    inputs_tensor_info_.push_back(tensor_info_in);
  }
  for (size_t i = 0; i < outputs_shape_.size(); i++) {
    TensorLayout tensor_layout_out;
    if (tensor_layout_out.InitFromVector(dev_matrix_shape_, outputs_tensor_map_.at(i), outputs_shape_.at(i)) !=
        SUCCESS) {
      return FAILED;
    }
    TensorInfo tensor_info_out(tensor_layout_out);
    outputs_tensor_info_.push_back(tensor_info_out);
  }
  return SUCCESS;
}

Status BatchParallelInfo::GetAttrs() { return SUCCESS; }

Status BatchParallelInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status BatchParallelInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}

Status BatchParallelInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

Status BatchParallelInfo::GenerateStrategies(int64_t stage_id) {
  StrategyPtr sp;
  Strategys strategy;
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    Shape temp(inputs_shape_[i].size(), 1);
    if (split_flag_list_[i]) {
      temp[0] = stage_device_size_;
    }
    strategy.push_back(temp);
  }
  sp = std::make_shared<Strategy>(stage_id, strategy);

  if (SetCostUnderStrategy(sp) == SUCCESS) {
    MS_LOG(INFO) << name_ << " : Successfully generated batch-parallel-strategy.";
    PrintStrategy(sp);
  } else {
    MS_LOG(ERROR) << name_ << " : Generating batch-parallel-strategy failed.";
    return FAILED;
  }
  return SUCCESS;
}

void SparseSoftmaxCrossEntropyWithLogitsInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    split_flag_list_[i] = true;
  }
}

Status BatchParallelInfo::InferAsLossDivisor() {
  as_loss_divisor_ = 1;
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
