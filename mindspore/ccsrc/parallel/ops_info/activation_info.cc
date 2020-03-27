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
#include <vector>
#include <memory>

#include "ir/value.h"
#include "parallel/device_matrix.h"
#include "parallel/strategy.h"
#include "parallel/auto_parallel/costmodel.h"

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
  Shape input0_split(inputs_shape_[0].size(), 1);
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
}  // namespace parallel
}  // namespace mindspore
