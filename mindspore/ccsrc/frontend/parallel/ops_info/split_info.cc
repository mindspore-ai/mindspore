/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/split_info.h"

#include <string>
#include <memory>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/context.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status SplitInfo::GetAttrs() {
  int axis = 0;
  int output_num = 0;

  auto axis_iter = attrs_.find(AXIS);
  if (axis_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(axis_iter->second);
    if (axis_iter->second->isa<Int64Imm>()) {
      axis = axis_iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis is not int";
      return FAILED;
    }
  } else {
    MS_LOG(ERROR) << name_ << ": Can not find the axis attr";
    return FAILED;
  }

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  int dim = SizeToInt(inputs_shape_[0].size());
  if (axis < 0) {
    axis = axis + dim;
  }
  axis_ = SizeToInt(axis);

  auto output_num_iter = attrs_.find(OUTPUT_NUM);
  if (output_num_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(output_num_iter->second);
    if (output_num_iter->second->isa<Int64Imm>()) {
      output_num = output_num_iter->second->cast<Int64ImmPtr>()->value();
    } else {
      MS_LOG(ERROR) << name_ << ": The value of output_num is not int";
      return FAILED;
    }
  } else {
    MS_LOG(ERROR) << name_ << ": Can not find the output_num attr";
    return FAILED;
  }
  output_num_ = output_num;

  return SUCCESS;
}

Status SplitInfo::CheckStrategy(const StrategyPtr &strategy) {
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

  if (axis_ >= stra[0].size()) {
    MS_LOG(ERROR) << name_ << ": The axis is out of range, the axis is " << axis_;
    return FAILED;
  }

  if (stra[0][axis_] != 1) {
    MS_LOG(ERROR) << name_ << ": The axis can not be split";
    return FAILED;
  }

  return SUCCESS;
}

Status SplitInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status SplitInfo::InferTensorMap() {
  TensorMap tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << "The inputs shape is empty";
    return FAILED;
  }

  int32_t size = SizeToInt(inputs_shape_[0].size());
  for (int i = 0; i < size; ++i) {
    tensor_map.push_back(size - i - 1);
  }

  inputs_tensor_map_.push_back(tensor_map);

  for (size_t i = 0; i < outputs_shape_.size(); ++i) {
    outputs_tensor_map_.push_back(tensor_map);
  }

  return SUCCESS;
}

Status SplitInfo::InferMirrorOps() {
  mirror_ops_.clear();
  if (inputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs tensor map is empty";
    return FAILED;
  }

  Shape input_tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(input_tensor_map, &group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group for input failed.";
    return FAILED;
  }

  OperatorVector mirror_op;
  if (group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror group is empty.";
    return SUCCESS;
  } else {
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
    std::string group_name = group[0].name();
    MS_LOG(INFO) << name_ << " : Create the mirror ops success, the group name is " << group_name;
  }

  return SUCCESS;
}

Status SplitInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }

  TensorLayout input_layout, output_layout;
  // infer tensor layout
  if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], inputs_shape_[0]) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed.";
    return FAILED;
  }
  TensorInfo input_tensor_info(input_layout);
  inputs_tensor_info_.push_back(input_tensor_info);

  if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], outputs_shape_[0]) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed.";
    return FAILED;
  }
  for (size_t i = 0; i < outputs_shape_.size(); ++i) {
    TensorInfo output_tensor_info(output_layout);
    outputs_tensor_info_.push_back(output_tensor_info);
  }

  return SUCCESS;
}

Status SplitInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Status SplitInfo::GenerateStrategies(int64_t stage_id) {
  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer attrs failed";
    return FAILED;
  }
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  Shape input_split;
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (i == axis_) {
      input_split.push_back(0);
    } else {
      input_split.push_back(1);
    }
  }

  Shapes splittable_input = {input_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Generate strategies failed";
    return FAILED;
  }

  size_t success = 0;
  for (auto &sp : sp_vector) {
    PrintStrategy(sp);
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

std::shared_ptr<Strategys> SplitInfo::GenerateBatchStrategies() {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Get attr failed";
  }
  Dimensions input_strategy(inputs_shape_[0].size(), 1);
  // axis can't split
  if (inputs_shape_[0].size() > 1) {
    if (axis_ == 0) {
      input_strategy[1] = stage_device_size_;
    } else {
      input_strategy[0] = stage_device_size_;
    }
  }
  Strategys strategy_v = {input_strategy};
  return std::make_shared<Strategys>(strategy_v);
}

Status SplitInfo::InferAsLossDivisor() {
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

Status SplitInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status SplitInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
