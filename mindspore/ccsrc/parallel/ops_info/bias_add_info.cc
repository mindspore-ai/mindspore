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

#include "parallel/ops_info/bias_add_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "parallel/device_matrix.h"
#include "parallel/strategy.h"
#include "parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status BiasAddInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    }
    return FAILED;
  }
  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  int32_t channel_a_strategy = sub_a_strategy.at(1);
  int32_t channel_b_strategy = sub_b_strategy.at(0);
  if (channel_a_strategy != channel_b_strategy) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status BiasAddInfo::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  dev_matrix_shape_ = sub_a_strategy;
  return SUCCESS;
}

void BiasAddInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = true;
  split_flag_list_[1] = false;
}

Status BiasAddInfo::InferTensorMap() {
  TensorMap sub_a_tensor_map;
  TensorMap sub_b_tensor_map;
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  size_t sub_a_strategy_size = sub_a_strategy.size();
  for (size_t i = 0; i < sub_a_strategy_size; ++i) {
    sub_a_tensor_map.push_back((int32_t)(LAST_INDEX(SizeToUint(sub_a_strategy_size)) - i));
  }
  sub_b_tensor_map.push_back((int32_t)(LAST_INDEX(SizeToUint(sub_a_strategy_size)) - 1));

  inputs_tensor_map_.push_back(sub_a_tensor_map);
  inputs_tensor_map_.push_back(sub_b_tensor_map);
  outputs_tensor_map_.push_back(sub_a_tensor_map);

  return SUCCESS;
}

Status BiasAddInfo::InferMirrorOps() {
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

Status BiasAddInfo::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout,
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

Status BiasAddInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_a_shape = inputs_shape_.at(0);
  Shape input_b_shape = inputs_shape_.at(1);
  Shape output_shape = outputs_shape_.at(0);

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Strategys outputs_strategy = {inputs_strategy.at(0)};
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

Status BiasAddInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
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

Status BiasAddInfo::GenerateStrategies(int32_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split, input0_split};

  std::vector<StrategyPtr> sp_vector;
  is_auto_parallel_ = true;
  Shapes tmp_inputs_shape = {inputs_shape_[0], inputs_shape_[0]};
  Shapes tmp_splittable_inputs = {splittable_inputs[0], splittable_inputs[0]};
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, &sp_vector) !=
      SUCCESS) {
    return FAILED;
  }
  MS_LOG(INFO) << name_ << " : Generate strategies with broadcast success.";

  for (auto &sp : sp_vector) {
    std::vector<Dimensions> tmp_strategy;
    Dimensions input0_strategy = sp->GetInputDim()[0];
    tmp_strategy.push_back(input0_strategy);  // input0

    Dimensions input1_strategy = {input0_strategy.at(1)};

    // reset the strategy
    tmp_strategy.push_back(input1_strategy);  // input1
    sp->ResetInputs(tmp_strategy);
  }
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

Status BiasAddInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status BiasAddInfo::InitForCostModel(const StrategyPtr &strategy) {
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
