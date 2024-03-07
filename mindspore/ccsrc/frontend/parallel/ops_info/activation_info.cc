/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/activation_info.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include <numeric>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/redistribution_operator_infer.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
Status Activation::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Status Activation::CheckStrategy(const StrategyPtr &strategy) { return CheckStrategyValue(strategy, inputs_shape_); }

Status ActivationInfo::GetAttrs() {
  if (attrs_.size() < ACTIVATION_ATTR_SIZE) {
    MS_LOG(ERROR) << name_ << " : The size of attrs small than 1.";
    return FAILED;
  }

  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE)) {
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
  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size(" << inputs_shape_.size() << ") or outputs shape size("
                  << outputs_shape_.size() << "is wrong.";
    return FAILED;
  }
  return SUCCESS;
}

std::vector<StrategyPtr> Activation::GenerateOpStrategies(int64_t stage_id) {
  std::vector<StrategyPtr> sp_vector;
  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE)) {
    MS_LOG(EXCEPTION) << name_ << " : Inputs shape size(" << inputs_shape_.size() << ") or outputs shape size("
                      << outputs_shape_.size() << "is wrong.";
  }

  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};

  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }

  return sp_vector;
}

std::vector<StrategyPtr> DropoutInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }
  return sp_vector;
}

Status Softmax::CheckLayoutConfig() {
  for (auto &element : axis_) {
    int64_t axis_index = element;
    if (element < 0) {
      size_t input_dim = inputs_shape_[0].size();
      axis_index = SizeToLong(input_dim) + element;
    }

    int64_t tensor_map = inputs_tensor_map_[0][LongToSize(axis_index)];
    if (tensor_map == MAP_NONE) {
      continue;
    }
    int64_t axis_strategy = dev_matrix_shape_[dev_matrix_shape_.size() - LongToSize(tensor_map) - 1];
    // Dimension corresponding to axis is un-splittable
    if (axis_strategy != MIN_SLICE_NUM) {
      MS_LOG(ERROR) << name_ << " : The strategy corresponding to axis dimension is not 1, the axis is " << axis_
                    << ", dev_matrix is " << dev_matrix_shape_ << ", input tensor map is " << inputs_tensor_map_;
      return FAILED;
    }
  }

  return SUCCESS;
}

Status Softmax::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  for (auto &element : axis_) {
    int64_t axis_index = element;
    if (element < 0) {
      size_t input_dim = inputs_shape_.at(0).size();
      axis_index = static_cast<int64_t>(input_dim) + element;
    }

    int64_t axis_strategy = input_strategy.at(LongToSize(axis_index));
    // Dimension corresponding to axis is un-splittable
    if (axis_strategy != MIN_SLICE_NUM) {
      MS_LOG(ERROR) << name_ << " : The strategy corresponding to axis dimension(" << axis_strategy << ") is not 1";
      return FAILED;
    }
  }

  return SUCCESS;
}

Status Softmax::GetAttrs() {
  std::string op_name = GetPrimNameFromInfoName(this->name_);
  std::optional<std::vector<int64_t>> axis_opt = GetArrayValueFromInputs<int64_t>(input_value_, op_name, AXIS);

  if (!axis_opt.has_value()) {
    MS_LOG(ERROR) << name_ << " : has no axis value.";
    return FAILED;
  }

  std::vector<int64_t> axis_val = axis_opt.value();
  if (axis_val.empty()) {
    MS_LOG(ERROR) << name_ << " axis doesn't have value.";
    return FAILED;
  }
  axis_.swap(axis_val);
  MS_LOG(INFO) << name_ << " : The axis is tuple, value is " << ListToString(axis_);

  if (input_value_.size() != ops::GetOpInputsNum(op_name)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }

  // for example: tensor dimension is 4, then axis range [-4, 3]
  int64_t dim = SizeToLong(inputs_shape_.at(0).size());
  auto it =
    std::find_if(axis_.begin(), axis_.end(), [dim](int64_t element) { return ((element >= dim) || (element < -dim)); });
  if (it != axis_.end()) {
    MS_LOG(ERROR) << name_ << " : The axis(" << *it << ") is out of range[" << (-dim) << ", " << (dim - 1) << "].";
    return FAILED;
  }

  return SUCCESS;
}

Status LogSoftmaxInfo::GetAttrs() {
  std::string op_name = GetPrimNameFromInfoName(this->name_);
  std::optional<int64_t> axis_opt = GetScalarValueFromInputs<int64_t>(input_value_, op_name, AXIS);

  if (!axis_opt.has_value()) {
    MS_LOG(ERROR) << name_ << " : has no axis value.";
    return FAILED;
  }
  int64_t axis_val = axis_opt.value();
  axis_.push_back(axis_val);
  MS_LOG(INFO) << name_ << " : The axis is tuple, value is " << ListToString(axis_);

  if (input_value_.size() != ops::GetOpInputsNum(op_name)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }

  // for example: tensor dimension is 4, then axis range [-4, 3]
  int64_t dim = SizeToLong(inputs_shape_.at(0).size());
  auto it =
    std::find_if(axis_.begin(), axis_.end(), [dim](int64_t element) { return ((element >= dim) || (element < -dim)); });
  if (it != axis_.end()) {
    MS_LOG(ERROR) << name_ << " : The axis(" << *it << ") is out of range[" << (-dim) << ", " << (dim - 1) << "].";
    return FAILED;
  }

  return SUCCESS;
}

Status Softmax::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> Softmax::GenerateOpStrategies(int64_t stage_id) {
  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE)) {
    MS_LOG(EXCEPTION) << name_ << " : Inputs shape size or outputs shape size is wrong.";
  }

  Shape input0_split;
  (void)input0_split.insert(input0_split.cbegin(), inputs_shape_[0].size(), 1);
  for (auto &element : axis_) {
    int64_t axis_index = element;
    if (element < 0) {
      size_t input_dim = inputs_shape_.at(0).size();
      axis_index = static_cast<int64_t>(input_dim) + element;
    }
    input0_split[LongToSize(axis_index)] = 0;
  }
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs failed.";
  }
  return sp_vector;
}

Status CumOpBase::GetAttrs() {
  std::string op_name = GetPrimNameFromInfoName(this->name_);
  if (input_value_.size() != ops::GetOpInputsNum(op_name)) {
    MS_LOG(ERROR) << name_ << ": Invalid inputs size " << input_value_.size()
                  << ", GetOpInputsNum: " << ops::GetOpInputsNum(op_name);
    return FAILED;
  }

  std::optional<int64_t> axis_opt = GetScalarValueFromInputs<int64_t>(input_value_, op_name, AXIS);
  if (!axis_opt.has_value()) {
    MS_LOG(ERROR) << name_ << ": The type of axis has no value.";
    return FAILED;
  }
  int64_t axis = axis_opt.value();

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  int64_t dim = SizeToLong(inputs_shape_[0].size());
  if ((axis > dim - 1) || (axis < -dim)) {
    MS_LOG(ERROR) << name_ << ": The axis(" << axis << ") is out of range [" << -dim << ", " << dim << ")";
    return FAILED;
  }

  if (axis < 0) {
    axis_ = dim + axis;
  } else {
    axis_ = axis;
  }
  MS_LOG(INFO) << name_ << ": The axis is " << axis;
  return SUCCESS;
}

Status CumOpBase::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  if (input_strategy.size() <= LongToSize(axis_)) {
    MS_LOG(ERROR) << "The " << name_ << " input strategy length: " << input_strategy.size() << ", is less ot equal to "
                  << axis_;
    return FAILED;
  }
  auto axis_split = input_strategy[LongToSize(axis_)];
  if (axis_split != NO_SPLIT_STRATEGY) {
    MS_LOG(ERROR) << "Currently, CumSum does not support the sharding strategies which splits axis.";
    return FAILED;
  }

  return SUCCESS;
}

std::vector<StrategyPtr> CumOpBase::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  if (axis_ < 0 || LongToSize(axis_) >= inputs_shape_[0].size()) {
    MS_LOG(EXCEPTION) << "Wrong axis value: " << axis_;
  }
  // Currently, CumSum does not support the sharding strategies which splits axis.
  input0_split[LongToSize(axis_)] = 0;
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }
  return sp_vector;
}

void CumOpBase::ReComputeBatchSplitFlagList() { axis_ == 0 ? split_flag_list_[0] = false : split_flag_list_[0] = true; }

Status CumOpBase::InferMirrorOps() {
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

Status ActivationBase::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  dev_matrix_shape_ = input_strategy;

  return SUCCESS;
}

Status ActivationBase::InferMirrorOps() {
  mirror_ops_.clear();

  Shape tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(tensor_map, &group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
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

  // No need to insert mirror ops
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  int64_t to_be_append = ops::GetOpInputsNum(this->prim_name_) - mirror_ops_.size();
  if (to_be_append <= 0) {
    return SUCCESS;
  }

  std::vector<OperatorVector> op_vec(to_be_append);
  (void)mirror_ops_.insert(mirror_ops_.end(), op_vec.begin(), op_vec.end());

  return SUCCESS;
}

Status ActivationBase::InferForwardCommunication() {
  // do nothing
  return SUCCESS;
}

Status ActivationBase::InferTensorMap() {
  Shape tensor_map_index;
  size_t size = inputs_shape_.at(0).size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back(static_cast<int64_t>(size - i - 1));
  }

  inputs_tensor_map_.push_back(tensor_map_index);
  outputs_tensor_map_.push_back(tensor_map_index);
  return SUCCESS;
}

Status ActivationBase::InferOutputTensorMap() {
  outputs_tensor_map_.push_back(inputs_tensor_map_[0]);
  return SUCCESS;
}

Status DropoutInfo::GetAttrs() {
  auto keep_prob_value = GetScalarValueFromInputsWithCheck<float>(input_value_, name_, KEEP_PROB);
  if (!keep_prob_value.has_value()) {
    return FAILED;
  }
  keep_prob_ = keep_prob_value.value();
  auto seed0_value = GetScalarValueFromInputsWithCheck<int64_t>(input_value_, name_, SEED0);
  if (!seed0_value.has_value()) {
    return FAILED;
  }
  seed0_ = seed0_value.value();
  auto seed1_value = GetScalarValueFromInputsWithCheck<int64_t>(input_value_, name_, SEED1);
  if (!seed1_value.has_value()) {
    return FAILED;
  }
  seed1_ = seed1_value.value();
  return SUCCESS;
}

Status DropoutInfo::InferTensorMap() {
  Shape tensor_map_in;
  size_t size = inputs_shape_.at(0).size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_in.push_back(static_cast<int64_t>(size - i - 1));
  }

  inputs_tensor_map_.push_back(tensor_map_in);
  outputs_tensor_map_.push_back(tensor_map_in);
  outputs_tensor_map_.push_back(tensor_map_in);  // the dropout has two outputs
  return SUCCESS;
}

Status DropoutInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map is empty";
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

void DropoutInfo::InferReplaceOps() {
  if ((seed0_ != 0) || (seed1_ != 0) || (repeated_calc_num_ == 1)) {
    return;
  }
  int64_t seed = get_seed();
  ValuePtr new_seed0 = MakeValue(seed);
  ValuePtr new_seed1 = MakeValue(seed);
  ValuePtr new_keep_prob = MakeValue(keep_prob_);
  Attr attr_seed0 = std::make_pair(SEED0, new_seed0);
  Attr attr_seed1 = std::make_pair(SEED1, new_seed1);
  Attr attr_keep_probs = std::make_pair(KEEP_PROB, new_keep_prob);
  OperatorAttrs attrs = {attr_keep_probs, attr_seed0, attr_seed1};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  replace_op_ = {std::make_pair(DROPOUT, args)};
}

Status CastInfo::InferMirrorOps() {
  mirror_ops_.clear();

  Shape tensor_map = inputs_tensor_map_[0];
  std::vector<Group> group;
  if (CreateGroupByTensorMap(tensor_map, &group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
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

  if (!input_value_.back()->isa<Int64Imm>()) {
    MS_LOG(ERROR) << name_ << ": The type of axis is not int64_t";
    return FAILED;
  }

  int64_t axis = GetValue<int64_t>(input_value_.back());

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }

  int64_t dim = SizeToLong(inputs_shape_[0].size());
  if ((axis > dim) || (axis < -dim - 1)) {
    MS_LOG(ERROR) << name_ << ": The axis(" << axis << ") is out of range[" << (-dim - 1) << ", " << dim << "]";
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
  Shape input_tensor_map, output_tensor_map;
  size_t size = inputs_shape_[0].size();
  for (size_t i = 0; i < size; ++i) {
    input_tensor_map.push_back(SizeToLong(size - i - 1));
  }

  inputs_tensor_map_.push_back(input_tensor_map);

  output_tensor_map = input_tensor_map;
  if ((positive_axis_ < 0) || (positive_axis_ > SizeToLong(size))) {
    MS_LOG(ERROR) << name_ << ": Invalid positive axis " << positive_axis_;
    return FAILED;
  }
  (void)output_tensor_map.insert(output_tensor_map.cbegin() + positive_axis_, NO_SPLIT_MAP);
  outputs_tensor_map_.push_back(output_tensor_map);

  MS_LOG(INFO) << name_ << ": The tensor map of input is " << ShapeToString(input_tensor_map)
               << ", and the tensor map of output is " << ShapeToString(output_tensor_map);
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
    ReportError(name_ + ": Create group failed.");
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

Status SqueezeInfo::InferAxis(const ValueTuplePtr &value_tuple) {
  std::vector<int64_t> axis;
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
  for (auto &dim : axis_list) {
    if (!dim->isa<Int64Imm>()) {
      MS_LOG(ERROR) << name_ << ": The type of axis is not int64_t";
      return FAILED;
    }
    int64_t dim_value = GetValue<int64_t>(dim);
    int64_t positive_value = (dim_value < 0) ? (dim_value + SizeToLong(input_size)) : dim_value;
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
  if (InferAxis(value_tuple) != SUCCESS) {
    return FAILED;
  }
  attrs_[AXIS] = axis_;
  return SUCCESS;
}

void SqueezeInfo::InferReplaceOps() {
  Attr attr = std::make_pair(AXIS, axis_);
  OperatorAttrs attrs = {attr};
  OperatorParams params;
  OperatorArgs args = std::make_pair(attrs, params);
  replace_op_ = {std::make_pair(SQUEEZE, args)};
}

Status SqueezeInfo::InferTensorMap() {
  // for example: if the shape of input is [32, 32, 1], and the axis is (2, ),
  // then the input_tensor_map is [2, 1, 0], the output_tensor_map is [2, 1]
  Shape input_tensor_map, output_tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << name_ << ": The inputs shape is empty";
    return FAILED;
  }
  size_t size = inputs_shape_[0].size();
  std::vector<int64_t> axis = GetValue<const std::vector<int64_t>>(axis_);
  for (size_t i = 0; i < size; ++i) {
    size_t index = size - i - 1;
    auto iter = std::find(axis.begin(), axis.end(), SizeToLong(i));
    if (iter == axis.end()) {
      output_tensor_map.push_back(SizeToLong(index));
    }
    input_tensor_map.push_back(SizeToLong(index));
  }
  inputs_tensor_map_.push_back(input_tensor_map);
  outputs_tensor_map_.push_back(output_tensor_map);
  MS_LOG(INFO) << name_ << ": The tensor map of input is " << ShapeToString(input_tensor_map)
               << ", and the tensor map of output is " << ShapeToString(output_tensor_map);

  return SUCCESS;
}

Status L2LossInfo::InferTensorMap() {
  if (ActivationOther::InferTensorMap() != SUCCESS) {
    return FAILED;
  }
  // outputs_shape is [], so clearing its tensor map.
  outputs_tensor_map_[0].clear();
  return SUCCESS;
}

Status L2LossInfo::InferForwardCommunication() {
  forward_op_.clear();
  Shape group_create_map;
  if (repeated_calc_num_ > 1) {
    if (repeated_num_in_dev_matrix_right_) {
      group_create_map.push_back(0);
    } else {
      group_create_map.push_back(SizeToLong(dev_matrix_shape_.size()) - 1);
    }
  }

  std::vector<Group> group_list;
  if (CreateGroupByTensorMap(group_create_map, &group_list) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }
  if (group_list.empty()) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required";
    return SUCCESS;
  }

  Operator op = CreateAllReduceOp(REDUCE_OP_SUM, group_list[0].name());
  forward_op_.push_back(op);
  MS_LOG(INFO) << name_ << ": The group name of forward all reduce is " << group_list[0].name();

  return SUCCESS;
}

Status CummaxInfo::GetAttrs() {
  std::string op_name = GetPrimNameFromInfoName(this->name_);
  std::optional<int64_t> axis_opt = GetScalarValueFromInputs<int64_t>(input_value_, op_name, AXIS);
  if (!axis_opt.has_value()) {
    MS_LOG(ERROR) << name_ << ": The type of axis has no value.";
    return FAILED;
  }
  axis_ = axis_opt.value();

  if (axis_ < 0) {
    axis_ += SizeToLong(inputs_shape_[0].size());
  }

  MS_LOG(INFO) << name_ << ": The axis is " << axis_;
  return SUCCESS;
}

Status CummaxInfo::InferMirrorOps() { return OperatorInfo::InferMirrorOps(); }

Status CummaxInfo::InferTensorMap() {
  Shape tensor_map_index;
  size_t size = inputs_shape_.at(0).size();
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back(static_cast<int64_t>(size - i - 1));
  }

  inputs_tensor_map_.push_back(tensor_map_index);
  outputs_tensor_map_.push_back(tensor_map_index);
  outputs_tensor_map_.push_back(tensor_map_index);  // cummax has two outputs
  return SUCCESS;
}

Status CummaxInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map is empty";
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

Status SortInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Shape input_tensor_map;
  Strategies strategies = strategy_->GetInputDim();
  size_t dim = strategies[0].size();
  for (size_t i = 0; i < dim; ++i) {
    input_tensor_map.push_back(dim - i - 1);
  }

  inputs_tensor_map_.push_back(input_tensor_map);   // input
  outputs_tensor_map_.push_back(input_tensor_map);  // output
  outputs_tensor_map_.push_back(input_tensor_map);
  return SUCCESS;
}

Status SortInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map is empty";
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

Status SortInfo::GetAttrs() {
  auto iter = attrs_.find(AXIS);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<Int64Imm>()) {  // the axis is a number
      int64_t axis_element = iter->second->cast<Int64ImmPtr>()->value();
      axis_.push_back(axis_element);
      MS_LOG(INFO) << name_ << " : The axis is int64_t, value is " << axis_element;
    } else if (iter->second->isa<ValueTuple>()) {  // the axis is a tuple
      ValueTuplePtr value_tuple = iter->second->cast<ValueTuplePtr>();
      if (value_tuple == nullptr) {
        MS_LOG(ERROR) << name_ << " : The value_tuple is nullptr.";
        return FAILED;
      }
      std::vector<ValuePtr> value_vector = value_tuple->value();
      (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(axis_),
                           [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
      if (axis_.empty()) {
        MS_LOG(ERROR) << name_ << " : The axis tuple is empty.";
        return FAILED;
      }
      MS_LOG(INFO) << name_ << " : The axis is tuple, value is " << ListToString(axis_);
    } else {
      MS_LOG(ERROR) << name_ << " : The value of axis is not int64_t or tuple int64_t.";
      return FAILED;
    }
  }

  if ((inputs_shape_.size() != ACTIVATION_INPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }

  // for example: tensor dimension is 4, then axis range [-4, 3]
  int64_t dim = SizeToLong(inputs_shape_.at(0).size());
  auto it =
    std::find_if(axis_.begin(), axis_.end(), [dim](int64_t element) { return ((element >= dim) || (element < -dim)); });
  if (it != axis_.end()) {
    MS_LOG(ERROR) << name_ << " : The axis(" << *it << ") is out of range[" << (-dim) << ", " << (dim - 1) << "].";
    return FAILED;
  }

  return SUCCESS;
}

Status GeLUInfo::CheckInputLayout() {
  if (inputs_tensor_info_.size() != kSizeOne) {
    MS_LOG(ERROR) << "The size of input_tensor_layout for gelu is " << inputs_tensor_info_.size() << " rather than 1.";
    return FAILED;
  }
  return SUCCESS;
}

Status GeLUInfo::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeOne) {
    MS_LOG(ERROR) << "The size of output_tensor_layout for gelu is " << outputs_tensor_info_.size()
                  << " rather than 1.";
    return FAILED;
  }
  if (output_infer_tensor_layout_.tensor_shape_before().array().empty()) {
    MS_LOG(ERROR) << "Parameter of output tensor layout for gelu is not allowed to be set by users.";
    return FAILED;
  }
  MS_LOG(INFO) << "Using output tensor layout infer by input tensor layout.";
  return SUCCESS;
}

Status GeLUInfo::InferOutputTensorInfo() {
  output_infer_tensor_layout_ = inputs_tensor_info_[kIndex0].tensor_layout();
  TensorInfo output_tensor_info(output_infer_tensor_layout_);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status GeLUInfo::InferForwardCommunicationByLayout() { return SUCCESS; }

REGISTER(ActivationInfo);
REGISTER(GeLUInfo);
REGISTER(FastGeLUInfo);
REGISTER(TanhInfo);
REGISTER(SoftmaxInfo);
REGISTER(SortInfo);
REGISTER(LogSoftmaxInfo);
REGISTER(ReverseV2Info);
REGISTER(CumSumInfo);
REGISTER(CummaxInfo);
REGISTER(CumminInfo);
REGISTER(CumProdInfo);
REGISTER(EluInfo);
REGISTER(ReLUInfo);
REGISTER(SiLUInfo);
REGISTER(identityInfo);
REGISTER(RepeatElementsInfo);
REGISTER(ReLU6Info);
REGISTER(SoftsignInfo);
REGISTER(SoftplusInfo);
REGISTER(CastInfo);
REGISTER(SqrtInfo);
REGISTER(NegInfo);
REGISTER(ExpandDimsInfo);
REGISTER(SqueezeInfo);
REGISTER(SquareInfo);
REGISTER(SigmoidInfo);
REGISTER(DropoutInfo);
REGISTER(HShrinkInfo);
REGISTER(HSigmoidInfo);
REGISTER(IsFiniteInfo);
REGISTER(MishInfo);
REGISTER(RintInfo);
REGISTER(SeLUInfo);
REGISTER(SoftShrinkInfo);
REGISTER(L2LossInfo);
REGISTER(ErfinvInfo);
REGISTER(InvertInfo);           // has not bprop
REGISTER(PopulationCountInfo);  // has not bprop
}  // namespace parallel
}  // namespace mindspore
