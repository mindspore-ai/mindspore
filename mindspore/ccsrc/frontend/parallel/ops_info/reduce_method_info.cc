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

#include "frontend/parallel/ops_info/reduce_method_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status ReduceMethod::CheckStrategy(const StrategyPtr &strategy) { return CheckStrategyValue(strategy, inputs_shape_); }

Status ReduceMethod::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  dev_matrix_shape_ = input_strategy;

  return SUCCESS;
}

std::vector<int64_t> ReduceMethod::reduce_dim() {
  std::vector<int64_t> dim_list;
  if (input_value_.size() < 2) {
    MS_LOG(EXCEPTION) << name_ << ": Input value size is smaller than 2.";
  }
  if (input_value_.back() == nullptr) {
    MS_LOG(EXCEPTION) << name_ << ": Input value is nullptr.";
  }
  MS_ASSERT(inputs_shape_.size() == 1);
  auto input_dim = inputs_shape_.at(0).size();
  if (input_value_.back()->isa<ValueTuple>()) {
    auto attr_axis = GetValue<std::vector<int64_t>>(input_value_.back());
    // axis is (), reduce all dim
    if (attr_axis.empty()) {
      for (size_t i = 0; i < input_dim; ++i) {
        dim_list.push_back(SizeToLong(i));
      }
    } else {
      for (auto &axis : attr_axis) {
        axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
      }
    }
  } else if (input_value_.back()->isa<Int64Imm>()) {
    int64_t axis = GetValue<int64_t>(input_value_.back());
    axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
  } else {
    MS_LOG(EXCEPTION) << "Axis type is invalid.";
  }

  return dim_list;
}

Status ReduceMethod::GetAttrs() {
  // get attr cross_batch and keep_dims
  auto keep_dims_iter = attrs_.find(KEEP_DIMS);
  if (keep_dims_iter == attrs_.end()) {
    MS_LOG(ERROR) << name_ << ": Don't have attr keep_dims.";
    return FAILED;
  }

  if (keep_dims_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(keep_dims_iter->second);
    if (!keep_dims_iter->second->isa<BoolImm>()) {
      MS_LOG(ERROR) << name_ << ": Keep_dims is not a bool.";
      return FAILED;
    }
    keepdims_ = keep_dims_iter->second->cast<BoolImmPtr>()->value();
  }

  auto cross_batch_iter = attrs_.find(CROSS_BATCH);
  if (cross_batch_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(cross_batch_iter->second);
    if (!cross_batch_iter->second->isa<BoolImm>()) {
      MS_LOG(ERROR) << name_ << ": cross_batch is not a bool.";
      return FAILED;
    }
    cross_batch_ = cross_batch_iter->second->cast<BoolImmPtr>()->value();
  }
  auto reducemethodcost = std::dynamic_pointer_cast<ReduceMethodCost>(operator_cost());
  if (reducemethodcost == nullptr) {
    MS_LOG(ERROR) << "Cost cast to ReduceMethodCostPtr failed!";
    return FAILED;
  }
  reducemethodcost->set_cross_batch(cross_batch_);
  return SUCCESS;
}

Status ReduceMethod::InferTensorMap() {
  Shape tensor_map_index, output_tensor_map;
  std::vector<int64_t> dim_list;
  size_t size = inputs_shape_.at(0).size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back((int64_t)(size - 1 - i));
  }
  dim_list = reduce_dim();
  for (size_t i = 0; i < size; ++i) {
    if (find(dim_list.begin(), dim_list.end(), SizeToLong(i)) != dim_list.end()) {
      if (keepdims_) {
        output_tensor_map.push_back(-1);
      } else {
        continue;
      }
    } else {
      output_tensor_map.push_back(tensor_map_index[i]);
    }
  }
  inputs_tensor_map_.push_back(tensor_map_index);
  outputs_tensor_map_.push_back(output_tensor_map);

  return SUCCESS;
}

bool IsDataParallelStrategy(const Dimensions &strategy, int32_t stage_id) {
  CheckGlobalDeviceManager();
  size_t total_dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  if (strategy.empty()) {
    MS_LOG(EXCEPTION) << "IsDataParallelStrategy: strategy is empty";
  }

  return (LongToSize(strategy[0]) == total_dev_num);
}

Status ReduceMethod::InferForwardCommunication() {
  Dimensions stra = strategy_->GetInputDim().at(0);
  if (cross_batch_ && IsDataParallelStrategy(stra, stage_id_)) {
    MS_LOG(INFO) << name_ << ": cross_batch is True, don't need to InferForwardCommunication";
    return SUCCESS;
  }
  if (cross_batch_) {
    MS_LOG(INFO) << name_ << ": cross_batch is True, don't need to InferForwardCommunication";
    return SUCCESS;
  }
  forward_op_.clear();
  std::vector<int64_t> dim_list = reduce_dim();
  size_t size = stra.size();
  // judge if the reduce dim is partitioned.
  Shape group_creat_map;

  // if repeated calculation and the repeated_calc_num_ insert to the first dimension of dev matrix,
  // it need to handle the first dimension of map.
  if ((dev_matrix_shape_.size() > size) && !repeated_num_in_dev_matrix_right_) {
    group_creat_map.push_back(SizeToInt(dev_matrix_shape_.size() - size_t(1)));
  }
  for (size_t index = 0; index < size; ++index) {
    auto pos =
      std::find_if(dim_list.begin(), dim_list.end(), [index](const int64_t &dim) { return SizeToLong(index) == dim; });
    if (pos != dim_list.end() && stra[index] != 1) {
      continue;
    }
    group_creat_map.push_back(SizeToLong(size) - SizeToLong(index) - 1);
  }

  // if repeated calculation and the repeated_calc_num_ insert to the last dimension of dev matrix,
  // it need to handle the group_creat_map and insert the 0 to the last dimension of the group_creat_map.
  if (repeated_num_in_dev_matrix_right_ && (repeated_calc_num_ > 1)) {
    for (auto &ele : group_creat_map) {
      if (ele == MAP_NONE) {
        continue;
      }
      ele += 1;
    }
    group_creat_map.push_back(0);
  }
  std::vector<Group> forward_group;
  if (CreateGroupByTensorMap(group_creat_map, &forward_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferForwardCommunication group failed.";
    return FAILED;
  }
  if (!forward_group.empty()) {
    Operator op = CreateAllReduceOp(reduce_method_, forward_group[0].name());
    forward_op_.push_back(op);
    std::string group_name = forward_group[0].name();
    MS_LOG(INFO) << name_ << ": Forward communication group is " << group_name;
  }

  return SUCCESS;
}

ForwardOp CreateReduceMeanForwardOp(const std::vector<Group> &forward_group, const TypePtr &dtype) {
  // Create AllReduceSum op
  Operator op0 = CreateAllReduceOp(REDUCE_OP_SUM, forward_group[0].name());
  std::string group_name = forward_group[0].name();
  MS_LOG(INFO) << "The group of forward all reduce is " << group_name;

  // Create RealDiv op
  OperatorName operator1_name = REAL_DIV;
  std::vector<Device> device_list = forward_group[0].GetDevicesList();
  auto divisor = static_cast<float>(device_list.size());
  mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(divisor, dtype);
  ValuePtr op1_param_value = MakeValue(tensor_ptr);
  Attr op1_param = std::make_pair("divisor", op1_param_value);
  OperatorParams operator1_params = {std::make_pair(op1_param, 2)};
  OperatorAttrs operator1_attrs;
  OperatorArgs operator1_args = std::make_pair(operator1_attrs, operator1_params);
  Operator op1 = std::make_pair(operator1_name, operator1_args);
  ForwardOp forward_op = {op0, op1};

  std::string dtype_name = dtype->ToString();
  MS_LOG(INFO) << "The divisor of Div op is " << device_list.size() << ", the dtype is " << dtype_name;
  return forward_op;
}

Status ReduceMeanInfo::InferForwardCommunication() {
  Dimensions stra = strategy_->GetInputDim().at(0);
  if (cross_batch_ && IsDataParallelStrategy(stra, stage_id_)) {
    MS_LOG(INFO) << name_ << ": cross_batch is True, don't need to InferForwardCommunication";
    return SUCCESS;
  }
  forward_op_.clear();
  std::vector<int64_t> dim_list = reduce_dim();
  size_t size = stra.size();
  // judge if the reduce dim is partitioned.
  Shape group_creat_map;

  // if repeated calculation and the repeated_calc_num_ insert to the first dimension of dev matrix,
  // it need to handle the first dimension of map.
  if ((dev_matrix_shape_.size() > size) && !repeated_num_in_dev_matrix_right_) {
    group_creat_map.push_back(SizeToInt(dev_matrix_shape_.size() - size_t(1)));
  }

  for (size_t index = 0; index < size; ++index) {
    auto pos =
      std::find_if(dim_list.begin(), dim_list.end(), [index](const int64_t &dim) { return SizeToLong(index) == dim; });
    if (pos != dim_list.end() && stra[index] != 1) {
      continue;
    }
    group_creat_map.push_back(SizeToLong(size) - SizeToLong(index) - 1);
  }

  // if repeated calculation and the repeated_calc_num_ insert to the last dimension of dev matrix,
  // it need to handle the group_creat_map and insert the 0 to the last dimension of the group_creat_map.
  if (repeated_num_in_dev_matrix_right_ && (repeated_calc_num_ > 1)) {
    for (auto &ele : group_creat_map) {
      if (ele == MAP_NONE) {
        continue;
      }
      ele += 1;
    }
    group_creat_map.push_back(0);
  }

  std::vector<Group> forward_group;
  if (CreateGroupByTensorMap(group_creat_map, &forward_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferForwardCommunication group failed.";
    return FAILED;
  }
  if (!forward_group.empty()) {
    if ((outputs_dtype_ == nullptr) || !outputs_dtype_->isa<mindspore::TensorType>()) {
      MS_LOG(ERROR) << name_ << ": The dtype of output is not Array";
      return FAILED;
    }

    auto element_type = outputs_dtype_->cast<mindspore::TensorTypePtr>()->element();
    forward_op_ = CreateReduceMeanForwardOp(forward_group, element_type);
  }

  return SUCCESS;
}

Status ReduceMethod::InferMirrorOps() {
  mirror_ops_.clear();
  Shape input_tensor_map = inputs_tensor_map_.at(0);
  std::vector<Group> input_group;
  if (CreateGroupByTensorMap(input_tensor_map, &input_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " Infer MirrorOps failed.";
    return FAILED;
  }

  OperatorVector op_for_weight;
  OperatorVector op_for_reduce_axis;  // helper node
  if (input_group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror ops is empty.";
    return SUCCESS;
  } else {
    op_for_weight = CreateMirrorOps(input_group[0].name(), input_group[0].GetDevNum());
    mirror_ops_.push_back(op_for_weight);
    mirror_ops_.push_back(op_for_reduce_axis);
    std::string group_name = input_group[0].name();
    MS_LOG(INFO) << name_ << ": Create the mirror ops for weight success, the group is " << group_name;
  }

  return SUCCESS;
}

Status ArgMaxWithValueInfo::InferMirrorOps() {
  mirror_ops_.clear();
  Shape input_tensor_map = inputs_tensor_map_.at(0);
  std::vector<Group> input_group;
  if (CreateGroupByTensorMap(input_tensor_map, &input_group) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer MirrorOps failed.";
    return FAILED;
  }

  OperatorVector op_for_weight;
  if (input_group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror ops is empty.";
    return SUCCESS;
  } else {
    op_for_weight = CreateMirrorOps(input_group[0].name(), input_group[0].GetDevNum());
    mirror_ops_.push_back(op_for_weight);
    MS_LOG(INFO) << name_ << ": Create the mirror ops for weight success.";
  }

  return SUCCESS;
}

Dimensions ReduceMethod::InferOutputStrategy() {
  std::vector<int64_t> dim_list = reduce_dim();
  Dimensions output_strategy;
  Dimensions stra = strategy_->GetInputDim().at(0);
  // if keepdims_ is true,then output strategy is same with input.
  for (size_t i = 0; i < stra.size(); ++i) {
    if (find(dim_list.begin(), dim_list.end(), SizeToLong(i)) != dim_list.end()) {
      if (keepdims_) {
        output_strategy.push_back(1);
      }
    } else {
      output_strategy.push_back(stra[i]);
    }
  }
  return output_strategy;
}

Status ReduceMethod::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);
  Shape output_shape = outputs_shape_.at(0);

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Dimensions output_strategy = InferOutputStrategy();

  Strategys outputs_strategy = {output_strategy};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }
  Shape input_slice_shape = inputs_slice_shape.at(0);
  Shape output_slice_shape = outputs_slice_shape.at(0);

  TensorLayout input_tensor_layout, output_tensor_layout;
  if ((input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], input_shape) != SUCCESS) ||
      (output_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], output_shape) != SUCCESS)) {
    return FAILED;
  }

  std::vector<int64_t> dim_list = reduce_dim();
  TensorInfo input_tensor_info(input_tensor_layout, input_shape, input_slice_shape);
  TensorInfo output_tensor_info(output_tensor_layout, output_shape, output_slice_shape);
  input_tensor_info.set_reduce_dim(dim_list);

  inputs_tensor_info_.push_back(input_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);

  return SUCCESS;
}

Status ReduceMethod::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

Status ReduceMethod::GenerateStrategies(int64_t stage_id) {
  if ((inputs_shape_.size() != 1) || (outputs_shape_.size() != 1)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size or outputs shape size is wrong, " << inputs_shape_.size() << ", "
                  << outputs_shape_.size();
    return FAILED;
  }

  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateStrategiesForIndependentInputs failed.";
    return FAILED;
  }
  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << " strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}

Status ReduceMethod::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }

  return SUCCESS;
}

Status ReduceMethod::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success";
  return SUCCESS;
}

std::vector<int64_t> ArgMaxWithValueInfo::reduce_dim() {
  std::vector<int64_t> dim_list;
  auto iter = attrs_.find(AXIS);
  if (iter == attrs_.end()) {
    MS_LOG(EXCEPTION) << name_ << ": Don't have attr axis.";
  }

  MS_ASSERT(inputs_shape_.size() == 1);
  auto input_dim = inputs_shape_.at(0).size();
  MS_EXCEPTION_IF_NULL(iter->second);
  if (iter->second->isa<ValueTuple>()) {
    auto attr_axis = GetValue<std::vector<int64_t>>(iter->second);
    if (attr_axis.empty()) {
      for (size_t i = 0; i < input_dim; ++i) {
        dim_list.push_back(SizeToLong(i));
      }
    } else {
      for (auto &axis : attr_axis) {
        axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
      }
    }
  } else if (iter->second->isa<Int64Imm>()) {
    int64_t axis = GetValue<int64_t>(iter->second);
    axis < 0 ? dim_list.push_back(axis + SizeToLong(input_dim)) : dim_list.push_back(axis);
  } else {
    MS_LOG(EXCEPTION) << "Axis type is invalid.";
  }

  return dim_list;
}

Status ArgMaxWithValueInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (ReduceMethod::CheckStrategy(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": CheckStrategy for parent class ReduceMethod failed";
    return FAILED;
  }
  std::vector<int64_t> dim_list = reduce_dim();
  MS_ASSERT(dim_list.size() == 1);

  Strategys stra = strategy->GetInputDim();
  MS_ASSERT(stra.size() == 1);
  Shape input_strategy = stra.at(0);
  MS_ASSERT(dim_list.at(0) < input_strategy.size());
  if (input_strategy.at(LongToSize(dim_list.at(0))) != 1) {
    MS_LOG(WARNING)
      << name_
      << " CheckStrategy for ArgMaxWithValueInfo, the strategy corresponding to axis is not one, real strategy "
         "is  "
      << input_strategy.at(LongToSize(dim_list.at(0)))
      << ", the output index may be not compatible with the stand alone Primitive";
  }
  return SUCCESS;
}

Status ArgMaxWithValueInfo::InferTensorMap() {
  if (ReduceMethod::InferTensorMap() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferTensorMap for parent class ReduceMethod failed";
    return FAILED;
  }
  MS_ASSERT(outputs_tensor_map_.size() == 1);
  outputs_tensor_map_.push_back(outputs_tensor_map_[0]);
  return SUCCESS;
}

Status ArgMaxWithValueInfo::InferTensorInfo() {
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);
  Shape output_shape = outputs_shape_.at(0);

  // infer slice shape
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Dimensions output_strategy = InferOutputStrategy();

  Strategys outputs_strategy = {output_strategy, output_strategy};
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }
  Shape input_slice_shape = inputs_slice_shape.at(0);
  Shape output_slice_shape = outputs_slice_shape.at(0);

  TensorLayout input_tensor_layout, output_tensor_layout;
  if ((input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[0], input_shape) != SUCCESS) ||
      (output_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[0], output_shape) != SUCCESS)) {
    return FAILED;
  }

  std::vector<int64_t> dim_list = reduce_dim();
  TensorInfo input_tensor_info(input_tensor_layout, input_shape, input_slice_shape);
  TensorInfo output_tensor_info(output_tensor_layout, output_shape, output_slice_shape);
  input_tensor_info.set_reduce_dim(dim_list);

  inputs_tensor_info_.push_back(input_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

Status ArgMaxWithValueInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor map is empty.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " has two outputs, use output[0] to infer";
  if (outputs_tensor_map_[0].empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size" << as_loss_divisor_ << " as loss divisor.";
    return SUCCESS;
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);

  std::string dev_matrix_shape_str = ShapeToString(dev_matrix_shape_);
  std::string output_tensor_map_str = ShapeToString(outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape, the output tensor map, and loss divisor is " << dev_matrix_shape_str
               << ", " << output_tensor_map_str << ", " << as_loss_divisor_;
  return SUCCESS;
}

Status ArgMaxWithValueInfo::GenerateStrategies(int64_t stage_id) {
  if ((inputs_shape_.size() != 1) || (outputs_shape_.size() != 2)) {
    MS_LOG(ERROR) << name_ << ": Inputs shape size or outputs shape size is wrong, " << inputs_shape_.size() << ", "
                  << outputs_shape_.size();
    return FAILED;
  }
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateStrategiesForIndependentInputs failed.";
    return FAILED;
  }
  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated strategy " << success;
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
