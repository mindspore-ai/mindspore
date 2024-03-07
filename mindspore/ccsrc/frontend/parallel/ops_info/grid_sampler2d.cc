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

#include "frontend/parallel/ops_info/grid_sampler2d.h"

#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/strategy.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {

constexpr size_t GRID_SAMPLER2D_INPUTS_SIZE = 2;
constexpr size_t GRID_SAMPLER2D_OUTPUTS_SIZE = 1;

Status GridSampler2DInfo::CheckInputOutputSize() {
  if (inputs_shape_.size() != GRID_SAMPLER2D_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": inputs shape size must be 2, but is " << inputs_shape_.size();
    return FAILED;
  }
  if (outputs_shape_.size() != GRID_SAMPLER2D_OUTPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": outputs shape size must be 1, but is " << outputs_shape_.size();
    return FAILED;
  }
  return SUCCESS;
}

Status GridSampler2DInfo::GetAttrs() { return SUCCESS; }

Status GridSampler2DInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckInputOutputSize() == FAILED) {
    return FAILED;
  }
  // The strategy of the first and the second input should be set.
  if (CheckStrategyValue(strategy, {inputs_shape_.at(0), inputs_shape_.at(1)}) != SUCCESS) {
    return FAILED;
  }
  Strategies stra = strategy->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  Dimensions sub_b_strategy = stra.at(1);
  Shape input_b_shape = inputs_shape_.at(1);
  if (sub_a_strategy.size() != input_b_shape.size()) {
    MS_LOG(ERROR) << name_
                  << " : Invalid strategy. The length of strategy of the input_x and grid "
                     "should be same, but they are different. The strategy of the input_x is "
                  << sub_a_strategy << ", and the strategy of grid is " << sub_b_strategy;
    return FAILED;
  }
  // The size of the input b must be equal or smaller than input a
  for (size_t i = 0; i < sub_b_strategy.size(); ++i) {
    if (sub_a_strategy[i] != sub_b_strategy[i]) {
      MS_LOG(ERROR) << name_
                    << " : Invalid strategy. The strategy of the input_x and grid "
                       "should be same, but they are different. The strategy of the input_x is "
                    << sub_a_strategy << ", and the strategy of grid is " << sub_b_strategy;

      return FAILED;
    }
  }
  return SUCCESS;
}

Status GridSampler2DInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  dev_matrix_shape_ = stra.at(0);
  return SUCCESS;
}

Status GridSampler2DInfo::InferMirrorOps() {
  mirror_ops_.clear();

  // Only the first input could be parameter.
  Shape tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(tensor_map, &group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }

  OperatorVector mirror_op;
  OperatorVector op_for_value;
  if (group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror ops is empty. no need to create mirror ops";
  } else {
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
    mirror_ops_.push_back(op_for_value);
    std::string group_name = group[0].name();
    MS_LOG(INFO) << name_ << " : Create the mirror ops success, the group name is " << group_name;
  }

  return SUCCESS;
}

Status GridSampler2DInfo::InferTensorMap() {
  Shape tensor_map_in;
  Shape tensor_map_in_index;
  Shape tensor_map_out;
  size_t input0_size = inputs_shape_.at(0).size();
  for (size_t i = 0; i < input0_size; ++i) {
    auto idx = SizeToInt(input0_size - i - 1);
    tensor_map_in.push_back(idx);
    tensor_map_in_index.push_back(idx);
    tensor_map_out.push_back(idx);
  }

  if (tensor_map_out.size() != outputs_shape_.at(0).size()) {
    MS_LOG(ERROR) << "Out tensor map size is not equal to output size! Out tensor map size is " << tensor_map_out.size()
                  << " output size is " << outputs_shape_.at(0).size();
    return FAILED;
  }

  (void)inputs_tensor_map_.emplace_back(std::move(tensor_map_in));
  (void)inputs_tensor_map_.emplace_back(std::move(tensor_map_in_index));
  (void)outputs_tensor_map_.emplace_back(std::move(tensor_map_out));
  return SUCCESS;
}

// Set the default strategy
std::vector<StrategyPtr> GridSampler2DInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shape input1_split(inputs_shape_[1].size(), 1);
  Shapes splittable_inputs = {input0_split, input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, {inputs_shape_.at(0)}, splittable_inputs, &sp_vector) !=
      SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }
  for (auto &sp : sp_vector) {
    Strategies tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    Dimensions second_input_strategy;
    for (size_t i = 0; i < inputs_shape_[1].size(); ++i) {
      second_input_strategy.push_back(first_input_strategy[i]);
    }
    tmp_strategy.push_back(first_input_strategy);
    tmp_strategy.push_back(second_input_strategy);
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

Status GridSampler2DInfo::InferForwardCommunication() {
  forward_op_.clear();
  std::vector<Group> group_list;
  Shape tmp_group_tensor_map = outputs_tensor_map_.at(0);

  if (CreateGroupByTensorMap(tmp_group_tensor_map, &group_list) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  } else if (group_list.empty()) {
    MS_LOG(INFO) << name_ << " : Forward all reduce is not required.";
    return SUCCESS;
  } else {
    MS_LOG(INFO) << name_ << " : The group name of forward communication is " << group_list[0].name()
                 << ". No need to insert the forward communication, skip this step.";
  }
  return SUCCESS;
}

Status GridSampler2DInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::shared_ptr<Strategies> GridSampler2DInfo::GenerateBatchStrategies() {
  if (CheckInputOutputSize() == FAILED) {
    MS_EXCEPTION(ValueError) << name_ << ": The length of inputs and outputs must be the 2 and 1.";
  }

  Dimensions strategy_a, strategy_b;
  strategy_a.push_back(stage_device_size_);
  for (size_t i = 1; i < inputs_shape_[0].size(); i++) {
    strategy_a.push_back(1);
  }

  strategy_b.push_back(stage_device_size_);
  for (size_t i = 1; i < inputs_shape_[1].size(); i++) {
    strategy_b.push_back(1);
  }
  Strategies strategy_v = {strategy_a, strategy_b};
  return std::make_shared<Strategies>(strategy_v);
}

REGISTER(GridSampler2DInfo);
}  // namespace parallel
}  // namespace mindspore
