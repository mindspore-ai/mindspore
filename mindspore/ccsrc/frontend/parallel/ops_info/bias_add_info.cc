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

#include "frontend/parallel/ops_info/bias_add_info.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status BiasAddInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  int64_t channel_a_strategy = sub_a_strategy.at(1);
  int64_t channel_b_strategy = sub_b_strategy.at(0);
  if (channel_a_strategy != channel_b_strategy) {
    MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    return FAILED;
  }
  return SUCCESS;
}

Status BiasAddInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  dev_matrix_shape_ = sub_a_strategy;
  return SUCCESS;
}

void BiasAddInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = true;
  split_flag_list_[1] = false;
}

Status BiasAddInfo::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  OperatorVector op_for_axis;
  (void)mirror_ops_.emplace_back(std::move(op_for_axis));
  return SUCCESS;
}

Status BiasAddInfo::InferTensorMap() {
  TensorMap sub_a_tensor_map;
  TensorMap sub_b_tensor_map;
  Strategies stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  size_t sub_a_strategy_size = sub_a_strategy.size();
  for (size_t i = 0; i < sub_a_strategy_size; ++i) {
    sub_a_tensor_map.push_back(static_cast<int64_t>(LAST_INDEX(sub_a_strategy_size) - i));
  }
  sub_b_tensor_map.push_back(static_cast<int64_t>(LAST_INDEX(sub_a_strategy_size)) - static_cast<int64_t>(1));

  inputs_tensor_map_.push_back(sub_a_tensor_map);
  inputs_tensor_map_.push_back(sub_b_tensor_map);
  outputs_tensor_map_.push_back(sub_a_tensor_map);

  return SUCCESS;
}

Status BiasAddInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> BiasAddInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split, input0_split};

  std::vector<StrategyPtr> sp_vector;
  Shapes tmp_inputs_shape = {inputs_shape_[0], inputs_shape_[0]};
  Shapes tmp_splittable_inputs = {splittable_inputs[0], splittable_inputs[0]};
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, &sp_vector) !=
      SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies failed.";
  }
  MS_LOG(INFO) << name_ << " : Generate strategies success.";

  for (auto &sp : sp_vector) {
    Strategies tmp_strategy;
    Dimensions input0_strategy = sp->GetInputDim()[0];
    tmp_strategy.push_back(input0_strategy);  // input0

    Dimensions input1_strategy = {input0_strategy.at(1)};

    // reset the strategy
    tmp_strategy.push_back(input1_strategy);  // input1
    sp->ResetInputs(tmp_strategy);
  }
  return sp_vector;
}

Status BiasAddInfo::CheckInputLayout() {
  // Check all device matrix should be the same
  if (inputs_tensor_info_.size() != kSizeTwo) {
    MS_LOG(ERROR) << "The size of input_tensor_layout for bias_add is " << inputs_tensor_info_.size()
                  << " rather than 2.";
    return FAILED;
  }
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  auto in_layout1 = inputs_tensor_info_[kIndex1].tensor_layout();
  if (in_layout0.device_arrangement_origin().array() != in_layout1.device_arrangement_origin().array()) {
    MS_LOG(ERROR) << "The device_matrix of input0 " << in_layout0.device_arrangement_origin().array()
                  << " dose not equal to device_matrix of input1 " << in_layout1.device_arrangement_origin().array();
    return FAILED;
  }

  if (in_layout0.tensor_map_before().back() != in_layout1.tensor_map_before()[0]) {
    MS_LOG(ERROR) << "The shard size of bias_add is not equal for last dim of input0 and input1";
    return FAILED;
  }
  return SUCCESS;
}

Status BiasAddInfo::InferOutputTensorInfo() {
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  // output layout should be the same as input layout 0
  if (in_layout0.tensor_shape_before().array() != outputs_shape_[kIndex0]) {
    MS_LOG(ERROR) << "The infer output shape " << in_layout0.tensor_shape_before().array()
                  << " dose not match the output shape " << outputs_shape_[kIndex0];
    return FAILED;
  }

  TensorLayout output_tensor_layout;
  output_tensor_layout.InitFromExtendVector(in_layout0.device_arrangement_origin().array(),
                                            in_layout0.tensor_map_before(), in_layout0.tensor_shape_before().array());

  TensorInfo output_tensor_info(output_tensor_layout);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status BiasAddInfo::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeOne) {
    MS_LOG(ERROR) << "The size of output_tensor_layout for bias_add is " << outputs_tensor_info_.size()
                  << " rather than 1.";
    return FAILED;
  }
  auto out_layout = outputs_tensor_info_[kIndex0].tensor_layout();
  auto in_layout0 = inputs_tensor_info_[kIndex0].tensor_layout();
  if (out_layout.tensor_map_before() != in_layout0.tensor_map_before()) {
    MS_LOG(ERROR) << "output layout of bias_add does not match the layout of first input";
    return FAILED;
  }
  MS_LOG(INFO) << "Using output tensor layout infer by input tensor layout.";
  return SUCCESS;
}

REGISTER(BiasAddInfo);
}  // namespace parallel
}  // namespace mindspore
