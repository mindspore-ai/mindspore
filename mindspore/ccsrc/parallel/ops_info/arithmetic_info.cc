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

#include "parallel/ops_info/arithmetic_info.h"

#include <algorithm>
#include <vector>
#include <utility>
#include <memory>

#include "parallel/device_matrix.h"
#include "parallel/strategy.h"
#include "parallel/tensor_layout/tensor_redistribution.h"

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

std::vector<Dimensions> ExpendStrategy(const StrategyPtr &strategy) {
  std::vector<Dimensions> expend_strategy;
  std::vector<Dimensions> stra = strategy->GetInputDim();
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
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    }
    return FAILED;
  }
  Shapes input_shapes = InferExpendShape();
  std::vector<Dimensions> expend_strategy = ExpendStrategy(strategy);
  Dimensions sub_a_strategy = expend_strategy.at(0);
  Dimensions sub_b_strategy = expend_strategy.at(1);
  Shape input_a_shape = input_shapes.at(0);
  Shape input_b_shape = input_shapes.at(1);

  for (size_t i = 0; i < input_a_shape.size(); ++i) {
    if ((sub_a_strategy[i] != sub_b_strategy[i]) && (input_a_shape[i] != 1) && (input_b_shape[i] != 1)) {
      if (is_auto_parallel_) {
        MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
      } else {
        MS_LOG(ERROR) << name_ << " : Invalid strategy.";
      }
      return FAILED;
    }
  }
  return SUCCESS;
}

Status ArithmeticBase::InferDevMatrixShape() {
  std::vector<Dimensions> expend_strategy = ExpendStrategy(strategy_);
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
      tensor_map_index.push_back((int32_t)(LAST_INDEX(SizeToUint(strategy.size())) - i));
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
  std::vector<int32_t> tensor_map_index;
  std::vector<Dimensions> expend_strategy = ExpendStrategy(strategy_);
  Dimensions sub_a_expend_strategy = expend_strategy.at(0);
  Dimensions sub_b_expend_strategy = expend_strategy.at(1);
  Strategys stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  for (size_t i = 0; i < sub_a_expend_strategy.size(); ++i) {
    tensor_map_index.push_back((int32_t)(LAST_INDEX(SizeToUint(sub_a_expend_strategy.size())) - i));
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

Status ArithmeticBase::InferMirrorOps() {
  mirror_ops_.clear();
  Shape input_a_tensor_map = inputs_tensor_map_.at(0);
  Shape input_b_tensor_map = inputs_tensor_map_.at(1);
  std::vector<Group> input_a_group, input_b_group;
  if (CreateGroupByTensorMap(input_a_tensor_map, &input_a_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create group for input a failed.";
    return FAILED;
  }
  if (CreateGroupByTensorMap(input_b_tensor_map, &input_b_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create group for input b failed.";
    return FAILED;
  }

  OperatorVector op_for_input_a, op_for_input_b;
  if (input_a_group.empty() && input_b_group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror group is empty.";
    return SUCCESS;
  }
  if (!input_a_group.empty()) {
    op_for_input_a = CreateMirrorOps(input_a_group[0].name(), input_a_group[0].GetDevNum());
    MS_LOG(INFO) << name_ << " : Create the mirror ops for input a success, group is " << input_a_group[0].name();
  }
  if (!input_b_group.empty()) {
    op_for_input_b = CreateMirrorOps(input_b_group[0].name(), input_b_group[0].GetDevNum());
    MS_LOG(INFO) << name_ << " : Create the mirror ops for input b success, group is " << input_b_group[0].name();
  }
  mirror_ops_.push_back(op_for_input_a);
  mirror_ops_.push_back(op_for_input_b);

  return SUCCESS;
}

Status ArithmeticBase::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout,
                                         const Shape &dev_matrix_array) {
  if ((inputs_layout == nullptr) || (outputs_layout == nullptr)) {
    MS_LOG(ERROR) << name_ << " : The layout is null.";
    return FAILED;
  }
  TensorMap input_a_tensor_map_array = inputs_tensor_map_.at(0);
  TensorMap input_b_tensor_map_array = inputs_tensor_map_.at(1);
  TensorMap out_tensor_map_array = outputs_tensor_map_.at(0);
  Shape input_a_shape_array = inputs_shape_.at(0);
  Shape input_b_shape_array = inputs_shape_.at(1);
  Shape out_shape_array = outputs_shape_.at(0);

  TensorLayout input_a_tensor_layout, input_b_tensor_layout, out_tensor_layout;
  if (input_a_tensor_layout.InitFromVector(dev_matrix_array, input_a_tensor_map_array, input_a_shape_array) !=
      SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create tensor layout for input a failed.";
    return FAILED;
  }
  if (input_b_tensor_layout.InitFromVector(dev_matrix_array, input_b_tensor_map_array, input_b_shape_array) !=
      SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create tensor layout for input b failed.";
    return FAILED;
  }
  if (out_tensor_layout.InitFromVector(dev_matrix_array, out_tensor_map_array, out_shape_array) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create tensor layout for output failed.";
    return FAILED;
  }
  inputs_layout->push_back(input_a_tensor_layout);
  inputs_layout->push_back(input_b_tensor_layout);
  outputs_layout->push_back(out_tensor_layout);

  return SUCCESS;
}

Status ArithmeticBase::InferTensorInfo() {
  // infer tensor shape
  Shape input_a_shape = inputs_shape_.at(0);
  Shape input_b_shape = inputs_shape_.at(1);
  Shape output_shape = outputs_shape_.at(0);

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  std::vector<Dimensions> expend_strategy = ExpendStrategy(strategy_);
  Dimensions sub_a_expend_strategy = expend_strategy.at(0);
  Dimensions sub_b_expend_strategy = expend_strategy.at(1);
  Strategys inputs_strategy = strategy_->GetInputDim();
  Shape dev_shape;
  for (size_t i = 0; i < sub_a_expend_strategy.size(); ++i) {
    if (sub_a_expend_strategy[i] != sub_b_expend_strategy[i]) {
      dev_shape.push_back(sub_a_expend_strategy[i] * sub_b_expend_strategy[i]);
    } else {
      dev_shape.push_back(sub_a_expend_strategy[i]);
    }
  }
  Strategys outputs_strategy = {dev_shape};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }
  Shape input_a_slice_shape = inputs_slice_shape.at(0);
  Shape input_b_slice_shape = inputs_slice_shape.at(1);
  Shape output_slice_shape = outputs_slice_shape.at(0);

  // infer tensor layout
  TensorLayouts inputs_layout, outputs_layout;
  if (InferTensorLayout(&inputs_layout, &outputs_layout, dev_matrix_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Infer tensor layout failed.";
    return FAILED;
  }

  TensorInfo input_a_tensor_info(inputs_layout.at(0), input_a_shape, input_a_slice_shape);
  TensorInfo input_b_tensor_info(inputs_layout.at(1), input_b_shape, input_b_slice_shape);
  TensorInfo out_tensor_info(outputs_layout.at(0), output_shape, output_slice_shape);

  inputs_tensor_info_.push_back(input_a_tensor_info);  // inputs_a
  inputs_tensor_info_.push_back(input_b_tensor_info);  // inputs_b
  outputs_tensor_info_.push_back(out_tensor_info);     // output

  return SUCCESS;
}

Status ArithmeticBase::SetCostUnderStrategy(const StrategyPtr &strategy) {
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

Status ArithmeticBase::GenerateStrategies(int32_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shape input1_split(inputs_shape_[1].size(), 1);
  Shapes splittable_inputs = {input0_split, input1_split};

  std::vector<StrategyPtr> sp_vector;
  is_auto_parallel_ = true;
  if (GenerateStrategiesWithBroadcast(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Generate strategies with broadcast failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << " : Generate strategies with broadcast success.";

  size_t success = 0;
  for (auto &sp : sp_vector) {
    PrintStrategy(sp);
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << " : Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
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
