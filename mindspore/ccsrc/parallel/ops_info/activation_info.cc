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

#include "parallel/ops_info/activation_info.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

#include "ir/value.h"
#include "parallel/auto_parallel/costmodel.h"
#include "parallel/device_matrix.h"
#include "parallel/strategy.h"

namespace mindspore {
namespace parallel {
Status Activation::SetCostUnderStrategy(const StrategyPtr& strategy) {
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

Status Activation::CheckStrategy(const StrategyPtr& strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    }
    return FAILED;
  }

  return SUCCESS;
}

Status ActivationInfo::GetAttrs() {
  if (attrs_.size() < ACTIVATION_ATTR_SIZE) {
    MS_LOG(ERROR) << name_ << " : The size of attrs small than 1.";
    return FAILED;
  }

  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE) || (outputs_shape_.size() != ACTIVATION_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size(" << inputs_shape_.size() << ") or outputs shape size("
                  << outputs_shape_.size() << "is wrong.";
    return FAILED;
  }

  auto iter = attrs_.find(ACTIVATION_TYPE);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<StringImm>()) {
      std::string val = iter->second->cast<StringImmPtr>()->value();
      if ((val != RELU_TYPE) && (val != RELU6_TYPE) && (val != SIGMOID_TYPE)) {
        MS_LOG(ERROR) << name_ << " : Activation type is wrong.";
        return FAILED;
      }
    } else {
      MS_LOG(ERROR) << name_ << " : The value of activation_type is not string.";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status ActivationOther::GetAttrs() {
  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE) || (outputs_shape_.size() != ACTIVATION_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size(" << inputs_shape_.size() << ") or outputs shape size("
                  << outputs_shape_.size() << "is wrong.";
    return FAILED;
  }
  return SUCCESS;
}

Status Activation::GenerateStrategies(int32_t stage_id) {
  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE) || (outputs_shape_.size() != ACTIVATION_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size(" << inputs_shape_.size() << ") or outputs shape size("
                  << outputs_shape_.size() << "is wrong.";
    return FAILED;
  }

  is_auto_parallel_ = true;
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Generate strategies for independent inputs() failed.";
    return FAILED;
  }
  size_t success = 0;
  for (auto& sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << " : Successfully generated " << success << " strategy";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

Status Softmax::CheckStrategy(const StrategyPtr& strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << " : Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << " : Invalid strategy.";
    }
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  for (auto& element : axis_) {
    int32_t axis_index = element;
    if (element < 0) {
      size_t input_dim = inputs_shape_.at(0).size();
      axis_index = static_cast<int32_t>(input_dim) + element;
    }

    int32_t axis_strategy = input_strategy.at(IntToSize(axis_index));
    // Dimension corresponding to axis is un-splittable
    if (axis_strategy != MIN_SLICE_NUM) {
      if (is_auto_parallel_) {
        MS_LOG(DEBUG) << name_ << " : The strategy corresponding to axis dimension(" << axis_strategy << ") is not 1";
      } else {
        MS_LOG(ERROR) << name_ << " : The strategy corresponding to axis dimension(" << axis_strategy << ") is not 1";
      }
      return FAILED;
    }
  }

  return SUCCESS;
}

Status Softmax::GetAttrs() {
  if (attrs_.size() < SOFTMAX_ATTR_SIZE) {
    MS_LOG(ERROR) << name_ << " : The size of attrs small than 1.";
    return FAILED;
  }

  auto iter = attrs_.find(AXIS);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int32Imm>()) {  // the axis is a number
      int32_t axis_element = iter->second->cast<Int32ImmPtr>()->value();
      axis_.push_back(axis_element);
      MS_LOG(INFO) << name_ << " : The axis is int, value is " << axis_element;
    } else if (iter->second->isa<ValueTuple>()) {  // the axis is a tuple
      ValueTuplePtr value_tuple = iter->second->cast<ValueTuplePtr>();
      if (value_tuple == nullptr) {
        MS_LOG(ERROR) << name_ << " : The value_tuple is nullptr.";
        return FAILED;
      }
      std::vector<ValuePtr> value_vector = value_tuple->value();
      (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(axis_),
                           [](const ValuePtr& value) { return static_cast<int32_t>(GetValue<int>(value)); });
      if (axis_.empty()) {
        MS_LOG(ERROR) << name_ << " : The axis tuple is empty.";
        return FAILED;
      }
      MS_LOG(INFO) << name_ << " : The axis is tuple, value is " << ShapeToString(axis_);
    } else {
      MS_LOG(ERROR) << name_ << " : The value of axis is not int or tuple int.";
      return FAILED;
    }
  }

  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE) || (outputs_shape_.size() != ACTIVATION_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }

  // for example: tensor dimension is 4, then axis range [-4, 3]
  int32_t dim = SizeToInt(inputs_shape_.at(0).size());
  auto it = std::find_if(axis_.begin(), axis_.end(),
                         [dim](const int32_t& element) { return ((element >= dim) || (element < -dim)); });
  if (it != axis_.end()) {
    MS_LOG(ERROR) << name_ << " : The axis(" << *it << ") is out of range[" << -dim << ", " << dim - 1 << "].";
    return FAILED;
  }

  return SUCCESS;
}

Status Softmax::SetCostUnderStrategy(const StrategyPtr& strategy) {
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

Status Softmax::GenerateStrategies(int32_t stage_id) {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : GetAttrs failed.";
    return FAILED;
  }
  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE) || (outputs_shape_.size() != ACTIVATION_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }

  is_auto_parallel_ = true;
  Shape input0_split;
  (void)input0_split.insert(input0_split.begin(), inputs_shape_[0].size(), 1);
  for (auto& element : axis_) {
    int32_t axis_index = element;
    if (element < 0) {
      size_t input_dim = inputs_shape_.at(0).size();
      axis_index = static_cast<int32_t>(input_dim) + element;
    }
    input0_split[IntToSize(axis_index)] = 0;
  }
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Generate strategies for independent inputs failed.";
    return FAILED;
  }
  size_t success = 0;
  for (auto& sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << " : Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

Status ActivationBase::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  dev_matrix_shape_ = input_strategy;

  return SUCCESS;
}

Status ActivationBase::InferMirrorOps() {
  mirror_ops_.clear();

  Shape tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(tensor_map, &group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create group failed.";
    return FAILED;
  }

  OperatorVector mirror_op;
  if (group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror ops is empty.";
    return SUCCESS;
  } else {
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
    std::string group_name = group[0].name();
    MS_LOG(INFO) << name_ << " : Create the mirror ops success, the group name is " << group_name;
  }

  return SUCCESS;
}

Status ActivationBase::InferForwardCommunication() {
  // do nothing
  return SUCCESS;
}

Status ActivationBase::InferTensorMap() {
  std::vector<int32_t> tensor_map_index;
  size_t size = inputs_shape_.at(0).size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int32_t)(size - i - 1));
  }

  inputs_tensor_map_.push_back(tensor_map_index);
  outputs_tensor_map_.push_back(tensor_map_index);
  return SUCCESS;
}

Status ActivationBase::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Strategys outputs_strategy = {inputs_strategy.at(0)};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }
  Shape input_slice_shape = inputs_slice_shape.at(0);

  TensorLayout input_tensor_layout;
  if (input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], input_shape) != SUCCESS) {
    return FAILED;
  }

  TensorInfo input_tensor_info(input_tensor_layout, input_shape, input_slice_shape);

  inputs_tensor_info_.push_back(input_tensor_info);
  outputs_tensor_info_.push_back(input_tensor_info);  // the same as input

  return SUCCESS;
}

Status ActivationBase::Init(const StrategyPtr& strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status ActivationBase::InitForCostModel(const StrategyPtr& strategy) {
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

Status CastInfo::InferMirrorOps() {
  mirror_ops_.clear();

  Shape tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(tensor_map, &group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Create group failed.";
    return FAILED;
  }

  OperatorVector mirror_op;
  OperatorVector op_for_value;
  if (group.empty()) {
    MS_LOG(INFO) << name_ << " : The mirror ops is empty.";
    return SUCCESS;
  } else {
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
    mirror_ops_.push_back(op_for_value);
    std::string group_name = group[0].name();
    MS_LOG(INFO) << name_ << " : Create the mirror ops success, the group name is " << group_name;
  }

  return SUCCESS;
}

Status ExpandDimsInfo::GetAttrs() {
  if (input_value_.size() != EXPANDDIMS_INPUT_SIZE) {
    MS_LOG(ERROR) << name_ << ": Invalid inputs size " << input_value_.size();
    return FAILED;
  }

  if (!input_value_.back()->isa<Int32Imm>()) {
    MS_LOG(ERROR) << name_ << ": The type of axis is not int";
    return FAILED;
  }

  int32_t axis = GetValue<int32_t>(input_value_.back());

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  int32_t dim = SizeToInt(inputs_shape_[0].size());
  if ((axis > dim) || (axis < -dim - 1)) {
    MS_LOG(ERROR) << name_ << ": The axis(" << axis << ") is out of range[" << -dim - 1 << ", " << dim << "]";
    return FAILED;
  }

  if (axis < 0) {
    positive_axis_ = dim + axis + 1;
  } else {
    positive_axis_ = axis;
  }
  MS_LOG(INFO) << name_ << ": The axis is " << axis << ", and the positive axis is " << positive_axis_;
  return SUCCESS;
}

Status ExpandDimsInfo::InferTensorMap() {
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  // for example: if the dimension of input is 3, and the axis is 2,
  // then the input_tensor_map is [2, 1, 0], the output_tensor_map is [2, 1, -1, 0]
  std::vector<int32_t> input_tensor_map, output_tensor_map;
  size_t size = inputs_shape_[0].size();
  for (size_t i = 0; i < size; ++i) {
    input_tensor_map.push_back(SizeToInt(size - i - 1));
  }

  inputs_tensor_map_.push_back(input_tensor_map);

  output_tensor_map = input_tensor_map;
  if ((positive_axis_ < 0) || (positive_axis_ > SizeToInt(size))) {
    MS_LOG(ERROR) << name_ << ": Invalid positive axis " << positive_axis_;
    return FAILED;
  }
  (void)output_tensor_map.insert(output_tensor_map.begin() + positive_axis_, NO_SPLIT_MAP);
  outputs_tensor_map_.push_back(output_tensor_map);

  MS_LOG(INFO) << name_ << ": The tensor map of input is " << ShapeToString(input_tensor_map)
               << ", and the tensor map of output is " << ShapeToString(output_tensor_map);
  return SUCCESS;
}

Status ExpandDimsInfo::InferTensorStrategy() {
  if (strategy_ == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null";
    return FAILED;
  }

  inputs_strategy_ = strategy_->GetInputDim();
  if (inputs_strategy_.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  Shape output_strategy = inputs_strategy_[0];
  if ((positive_axis_ < 0) || (positive_axis_ > SizeToInt(output_strategy.size()))) {
    MS_LOG(ERROR) << name_ << ": Invalid positive axis " << positive_axis_;
    return FAILED;
  }
  (void)output_strategy.insert(output_strategy.begin() + positive_axis_, NO_SPLIT_STRATEGY);
  outputs_strategy_ = {output_strategy};
  return SUCCESS;
}

Status ExpandDimsInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The shape of inputs or outputs is empty";
    return FAILED;
  }

  if (inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The tensor map of inputs or outputs is empty";
    return FAILED;
  }

  Shape input_shape = inputs_shape_[0];
  Shape output_shape = outputs_shape_[0];

  // infer slice shape
  if (InferTensorStrategy() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer tensor strategy failed";
    return FAILED;
  }
  Shapes inputs_slice_shape, outputs_slice_shape;
  if (InferSliceShape(inputs_strategy_, outputs_strategy_, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer slice shape failed";
    return FAILED;
  }

  if (inputs_slice_shape.empty() || outputs_slice_shape.empty()) {
    MS_LOG(ERROR) << name_ << ": The slice shape of inputs or outputs is empty";
    return FAILED;
  }

  Shape input_slice_shape = inputs_slice_shape[0];
  Shape output_slice_shape = outputs_slice_shape[0];

  TensorLayout input_tensor_layout, output_tensor_layout;
  if (input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], input_shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init tensor layout for input failed";
    return FAILED;
  }

  if (output_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], output_shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init tensor layout for output failed";
    return FAILED;
  }

  TensorInfo input_tensor_info(input_tensor_layout, input_shape, input_slice_shape);
  TensorInfo output_tensor_info(output_tensor_layout, output_shape, output_slice_shape);

  inputs_tensor_info_.push_back(input_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status ExpandDimsInfo::InferMirrorOps() {
  mirror_ops_.clear();

  if (inputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The tensor map of inputs is empty";
    return FAILED;
  }

  std::vector<Group> group;
  if (CreateGroupByTensorMap(inputs_tensor_map_[0], &group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create group failed";
    return FAILED;
  }

  if (group.empty()) {
    MS_LOG(INFO) << name_ << ": No need to create mirror ops";
    return SUCCESS;
  }

  OperatorVector mirror_op, placeholder_op;
  mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(mirror_op);
  mirror_ops_.push_back(placeholder_op);
  MS_LOG(INFO) << name_ << ": Create mirror ops success, the group name is " << group[0].name();
  return SUCCESS;
}

Status SqueezeInfo::InferAxis(const ValueTuplePtr& value_tuple) {
  std::vector<int32_t> axis;
  auto axis_list = value_tuple->value();
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  Shape input_shape = inputs_shape_.at(0);
  size_t input_size = input_shape.size();
  // if axis tuple is empty, we should exclude the axis that the corresponding slice shape is 1.
  if (axis_list.empty()) {
    for (size_t i = 0; i < input_size; ++i) {
      if (input_shape[i] == 1) {
        axis.push_back(i);
      }
    }
    axis_ = MakeValue(axis)->cast<ValueTuplePtr>();
    return SUCCESS;
  }

  // convert negative axis to positive.
  for (auto& dim : axis_list) {
    if (!dim->isa<Int32Imm>()) {
      MS_LOG(ERROR) << name_ << ": The type of axis is not int";
      return FAILED;
    }
    int32_t dim_value = GetValue<int32_t>(dim);
    int32_t positive_value = (dim_value < 0) ? (dim_value + SizeToInt(input_size)) : dim_value;
    axis.push_back(positive_value);
  }
  axis_ = MakeValue(axis)->cast<ValueTuplePtr>();
  return SUCCESS;
}

Status SqueezeInfo::GetAttrs() {
  auto iter = attrs_.find(AXIS);
  if (iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Can't find axis attribute.";
    return FAILED;
  }
  MS_EXCEPTION_IF_NULL(iter->second);
  auto value_tuple = iter->second->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(value_tuple);
  InferAxis(value_tuple);
  attrs_[AXIS] = axis_;
  return SUCCESS;
}

Status SqueezeInfo::InferReplaceOps(const StrategyPtr& strategy) {
  Attr attr = std::make_pair(AXIS, axis_);
  OperatorAttrs attrs = {attr};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  replace_op_ = {std::make_pair(SQUEEZE, args)};
  return SUCCESS;
}

Status SqueezeInfo::InferTensorMap() {
  // for example: if the shape of input is [32, 32, 1], and the axis is (2, ),
  // then the input_tensor_map is [2, 1, 0], the output_tensor_map is [2, 1]
  std::vector<int32_t> input_tensor_map, output_tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  size_t size = inputs_shape_[0].size();
  std::vector<int32_t> axis = GetValue<const std::vector<int>>(axis_);
  for (size_t i = 0; i < size; ++i) {
    size_t index = size - i - 1;
    auto iter = std::find(axis.begin(), axis.end(), SizeToInt(i));
    if (iter == axis.end()) {
      output_tensor_map.push_back(SizeToInt(index));
    }
    input_tensor_map.push_back(SizeToInt(index));
  }
  inputs_tensor_map_.push_back(input_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
  MS_LOG(INFO) << name_ << ": The tensor map of input is " << ShapeToString(input_tensor_map)
               << ", and the tensor map of output is " << ShapeToString(output_tensor_map);

  return SUCCESS;
}

Status SqueezeInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The shape of inputs or outputs is empty";
    return FAILED;
  }

  if (inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The tensor map of inputs or outputs is empty";
    return FAILED;
  }

  Shape input_shape = inputs_shape_[0];
  Shape output_shape = outputs_shape_[0];

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Dimensions output_strategy;
  std::vector<int32_t> axis = GetValue<const std::vector<int>>(axis_);
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    auto iter = std::find(axis.begin(), axis.end(), SizeToInt(i));
    if (iter == axis.end()) {
      output_strategy.push_back(inputs_strategy[0].at(i));
    }
  }
  Strategys outputs_strategy = {output_strategy};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer slice shape failed";
    return FAILED;
  }

  if (inputs_slice_shape.empty() || outputs_slice_shape.empty()) {
    MS_LOG(ERROR) << name_ << ": The slice shape of inputs or outputs is empty";
    return FAILED;
  }

  Shape input_slice_shape = inputs_slice_shape[0];
  Shape output_slice_shape = outputs_slice_shape[0];

  // infer tensor layout
  TensorLayout input_tensor_layout, output_tensor_layout;
  if (input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], input_shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init tensor layout for input failed";
    return FAILED;
  }

  if (output_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], output_shape) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init tensor layout for output failed";
    return FAILED;
  }

  TensorInfo input_tensor_info(input_tensor_layout, input_shape, input_slice_shape);
  TensorInfo output_tensor_info(output_tensor_layout, output_shape, output_slice_shape);

  inputs_tensor_info_.push_back(input_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status SqueezeInfo::Init(const StrategyPtr& strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
  }

  if (InferReplaceOps(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Infer replace ops failed";
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
