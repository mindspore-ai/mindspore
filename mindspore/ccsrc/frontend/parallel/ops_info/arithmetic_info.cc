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

#include "frontend/parallel/ops_info/arithmetic_info.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Shape ExpendShape(const Shape &bigger_size_shape, Shape smaller_size_shape) {
  size_t insert_num = bigger_size_shape.size() - smaller_size_shape.size();
  for (size_t num = 0; num < insert_num; ++num) {
    (void)smaller_size_shape.insert(smaller_size_shape.begin(), 1);
  }
  return smaller_size_shape;
}

Shapes ArithmeticBase::InferExpendShape() {
  Shape input_a_shape = inputs_shape_.at(0);
  Shape input_b_shape = inputs_shape_.at(1);
  Shapes input_shapes;
  size_t input_a_size = input_a_shape.size();
  size_t input_b_size = input_b_shape.size();
  if (input_a_size > input_b_size) {
    input_shapes.push_back(input_a_shape);
    input_shapes.push_back(ExpendShape(input_a_shape, input_b_shape));
  } else if (input_a_size < input_b_size) {
    input_shapes.push_back(ExpendShape(input_b_shape, input_a_shape));
    input_shapes.push_back(input_b_shape);
  } else {
    input_shapes.push_back(input_a_shape);
    input_shapes.push_back(input_b_shape);
  }
  return input_shapes;
}

Strategys ExpendStrategy(const StrategyPtr &strategy) {
  Strategys expend_strategy;
  Strategys stra = strategy->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  size_t input_a_size = sub_a_strategy.size();
  size_t input_b_size = sub_b_strategy.size();
  if (input_a_size > input_b_size) {
    expend_strategy.push_back(sub_a_strategy);
    expend_strategy.push_back(ExpendShape(sub_a_strategy, sub_b_strategy));
  } else if (input_a_size < input_b_size) {
    expend_strategy.push_back(ExpendShape(sub_b_strategy, sub_a_strategy));
    expend_strategy.push_back(sub_b_strategy);
  } else {
    expend_strategy = stra;
  }
  return expend_strategy;
}

Status ArithmeticBase::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    return FAILED;
  }
  Shapes input_shapes = InferExpendShape();
  Strategys expend_strategy = ExpendStrategy(strategy);
  Dimensions sub_a_strategy = expend_strategy.at(0);
  Dimensions sub_b_strategy = expend_strategy.at(1);
  Shape input_a_shape = input_shapes.at(0);
  Shape input_b_shape = input_shapes.at(1);

  for (size_t i = 0; i < input_a_shape.size(); ++i) {
    if ((sub_a_strategy[i] != sub_b_strategy[i]) && (input_a_shape[i] != 1) && (input_b_shape[i] != 1)) {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ArithmeticBase::InferDevMatrixShape() {
  Strategys expend_strategy = ExpendStrategy(strategy_);
  Dimensions sub_a_strategy = expend_strategy.at(0);
  Dimensions sub_b_strategy = expend_strategy.at(1);
  Shape dev_shape;
  for (size_t i = 0; i < sub_a_strategy.size(); ++i) {
    if (sub_a_strategy[i] != sub_b_strategy[i]) {
      dev_shape.push_back(sub_a_strategy[i] * sub_b_strategy[i]);
    } else {
      dev_shape.push_back(sub_a_strategy[i]);
    }
  }
  dev_matrix_shape_ = dev_shape;

  return SUCCESS;
}

TensorMap SetExpendTensorMap(const Shape &strategy, const Shape &dev_matrix_shape) {
  TensorMap tensor_map_index;
  for (size_t i = 0; i < strategy.size(); ++i) {
    if (strategy[i] == dev_matrix_shape[i]) {
      tensor_map_index.push_back((int64_t)(LAST_INDEX(strategy.size()) - i));
    } else {
      tensor_map_index.push_back(-1);
    }
  }
  return tensor_map_index;
}

TensorMap SetTensorMap(const Shape &strategy_expend, const Shape &dev_matrix_shape, const Shape &strategy) {
  TensorMap expend_map = SetExpendTensorMap(strategy_expend, dev_matrix_shape);
  size_t dev_matrix_size = dev_matrix_shape.size();
  size_t strategy_size = strategy.size();
  if (dev_matrix_size != strategy_size) {
    (void)expend_map.erase(expend_map.begin(),
                           expend_map.begin() + static_cast<different_type>(dev_matrix_size - strategy_size));
  }
  return expend_map;
}

void ArithmeticBase::ReComputeBatchSplitFlagList() {
  Shapes expend_shapes = InferExpendShape();
  Shape expend_a_shape = expend_shapes.at(0);
  Shape expend_b_shape = expend_shapes.at(1);
  if (expend_a_shape.size() != expend_b_shape.size()) {
    MS_LOG(EXCEPTION) << name_ << " : Recompute batch split flag list is wrong.";
  }
  if (expend_a_shape.empty()) {
    split_flag_list_[0] = false;
    split_flag_list_[1] = false;
    return;
  }
  (expend_a_shape.at(0) != 1) ? (split_flag_list_[0] = true) : (split_flag_list_[0] = false);
  (expend_b_shape.at(0) != 1) ? (split_flag_list_[1] = true) : (split_flag_list_[1] = false);
}

Status ArithmeticBase::InferTensorMap() {
  Shape tensor_map_index;
  Strategys expend_strategy = ExpendStrategy(strategy_);
  Dimensions sub_a_expend_strategy = expend_strategy.at(0);
  Dimensions sub_b_expend_strategy = expend_strategy.at(1);
  Strategys stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  for (size_t i = 0; i < sub_a_expend_strategy.size(); ++i) {
    tensor_map_index.push_back((int64_t)(LAST_INDEX(sub_a_expend_strategy.size()) - i));
  }

  Shape dev_shape;
  for (size_t i = 0; i < sub_a_expend_strategy.size(); ++i) {
    if (sub_a_expend_strategy[i] != sub_b_expend_strategy[i]) {
      dev_shape.push_back(sub_a_expend_strategy[i] * sub_b_expend_strategy[i]);
    } else {
      dev_shape.push_back(sub_a_expend_strategy[i]);
    }
  }
  inputs_tensor_map_.push_back(SetTensorMap(sub_a_expend_strategy, dev_shape, sub_a_strategy));
  inputs_tensor_map_.push_back(SetTensorMap(sub_b_expend_strategy, dev_shape, sub_b_strategy));
  outputs_tensor_map_.push_back(tensor_map_index);

  return SUCCESS;
}

Status ArithmeticBase::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> ArithmeticBase::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shape input1_split(inputs_shape_[1].size(), 1);
  Shapes splittable_inputs = {input0_split, input1_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesWithBroadcast(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies with broadcast failed.";
  }
  MS_LOG(INFO) << name_ << " : Generate strategies with broadcast success.";

  return sp_vector;
}

Status ArithmeticBase::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status ArithmeticBase::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
