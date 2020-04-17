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

#include "parallel/ops_info/gather_v2_info.h"

#include <memory>
#include <utility>
#include <vector>

#include "ir/meta_tensor.h"
#include "ir/value.h"
#include "parallel/auto_parallel/costmodel.h"
#include "parallel/device_matrix.h"
#include "parallel/graph_util/generate_graph.h"
#include "parallel/strategy.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status GatherV2Info::GetAttrs() {
  if (inputs_shape_.size() != GATHER_V2_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": inputs shape size must be 2, but is " << inputs_shape_.size();
    return FAILED;
  }
  if (outputs_shape_.size() != GATHER_V2_OUTPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": outputs shape size must be 1, but is " << outputs_shape_.size();
    return FAILED;
  }
  if (input_value_.size() != GATHER_V2_INPUTS_VALUE_SIZE) {
    MS_LOG(ERROR) << name_ << ": input value size must be 3, but is " << input_value_.size();
    return FAILED;
  }
  // the second input is the index tensor

  // the third input is the axis, is a ValueNode
  if (input_value_.at(2) == nullptr) {
    MS_LOG(ERROR) << name_ << ": the third input value is nullptr, is not a ValueNode!";
    return FAILED;
  }

  if (inputs_shape_.at(0).size() == 0) {
    MS_LOG(ERROR) << name_ << ": input can not be a scalar!";
    return FAILED;
  }
  int axis = GetValue<int>(input_value_.at(2));
  if (axis >= SizeToInt(inputs_shape_.at(0).size()) || axis < 0 - SizeToInt(inputs_shape_.at(0).size())) {
    MS_LOG(ERROR) << "Axis is " << axis << ", not in [-" << inputs_shape_.at(0).size() << ", "
                  << inputs_shape_.at(0).size() << ").";
  }
  if (axis < 0) {
    axis += SizeToInt(inputs_shape_[0].size());
  }
  axis_ = axis;

  index_size_ = inputs_shape_.at(1).size();

  return SUCCESS;
}

Status GatherV2Info::CheckStrategy(const StrategyPtr& strategy) {
  if (inputs_shape_.size() != GATHER_V2_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": inputs shape size must be " << GATHER_V2_INPUTS_SIZE << ", but is "
                  << inputs_shape_.size();
    return FAILED;
  }
  if (outputs_shape_.size() != GATHER_V2_OUTPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": outputs shape size must be " << GATHER_V2_OUTPUTS_SIZE << ", but is "
                  << outputs_shape_.size();
    return FAILED;
  }
  // Only strategy of the first input should be set.
  if (CheckStrategyValue(strategy, {inputs_shape_.at(0)}, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    }
    return FAILED;
  }
  axis_strategy_ = strategy->GetInputDim().at(0).at(axis_);
  if (index_size_ != 1 && axis_strategy_ != 1) {
    MS_LOG(ERROR) << name_
                  << ": Invalid strategy. If the index is a scalar or a more than 1 dimension vector, the strategy "
                     "corresponding to axis must be 1, but is "
                  << axis_strategy_;
    return FAILED;
  }
  if (index_size_ == 1 && axis_strategy_ != 1 && inputs_shape_.at(1).at(0) % axis_strategy_ != 0) {
    MS_LOG(ERROR) << name_
                  << ": Invalid strategy. The first dimension of index can not be divided by strategy corresponding to "
                     "axis. The first dimension of index is "
                  << inputs_shape_.at(1).at(0) << " strategy corresponding to axis is " << axis_strategy_;
    return FAILED;
  }
  return SUCCESS;
}

Status GatherV2Info::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  dev_matrix_shape_ = stra.at(0);
  return SUCCESS;
}

// If index is a scalar, output dimension is input dimension minus 1;
// If index is a n dimension tensor, output dimension is input dimension plus (n - 1).
// Tensor map dimension is equal to the corresponding input and output dimension.
// If index's dimension is more than 1, we insert -1 for the output tensor map.
Status GatherV2Info::InferTensorMap() {
  if (inputs_shape_.size() != GATHER_V2_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": inputs shape size must be " << GATHER_V2_INPUTS_SIZE << ", but is "
                  << inputs_shape_.size();
    return FAILED;
  }
  if (outputs_shape_.size() != GATHER_V2_OUTPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": outputs shape size must be " << GATHER_V2_OUTPUTS_SIZE << ", but is "
                  << outputs_shape_.size();
    return FAILED;
  }
  std::vector<int32_t> tensor_map_in;
  std::vector<int32_t> tensor_map_out;
  size_t size = inputs_shape_.at(0).size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_in.push_back(SizeToInt(size - i - 1));
    tensor_map_out.push_back(SizeToInt(size - i - 1));
  }

  if (index_size_ == 0) {
    (void)tensor_map_out.erase(tensor_map_out.begin() + axis_);
  } else if (index_size_ > 1) {
    (void)tensor_map_out.insert(tensor_map_out.begin() + axis_, index_size_ - 1, -1);
  }
  if (tensor_map_out.size() != outputs_shape_.at(0).size()) {
    MS_LOG(ERROR) << "Out tensor map size is not equal to output size! Out tensor map size is " << tensor_map_out.size()
                  << " output size is " << outputs_shape_.at(0).size();
    return FAILED;
  }

  std::vector<int32_t> tensor_map_in_index;
  if (index_size_ >= 1) {
    tensor_map_in_index.push_back(SizeToInt(size - axis_ - 1));
  }
  for (size_t i = 1; i < index_size_; ++i) {
    tensor_map_in_index.push_back(-1);
  }
  inputs_tensor_map_.emplace_back(std::move(tensor_map_in));
  inputs_tensor_map_.emplace_back(std::move(tensor_map_in_index));
  outputs_tensor_map_.emplace_back(std::move(tensor_map_out));
  return SUCCESS;
}

Status GatherV2Info::InferTensorInfo() {
  if (inputs_shape_.size() != GATHER_V2_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": inputs shape size must be " << GATHER_V2_INPUTS_SIZE << ", but is "
                  << inputs_shape_.size();
    return FAILED;
  }
  if (outputs_shape_.size() != GATHER_V2_OUTPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": outputs shape size must be " << GATHER_V2_OUTPUTS_SIZE << ", but is "
                  << outputs_shape_.size();
    return FAILED;
  }
  if (inputs_tensor_map_.size() != GATHER_V2_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": inputs tensor map  size must be " << GATHER_V2_INPUTS_SIZE << ", but is "
                  << inputs_tensor_map_.size();
    return FAILED;
  }
  if (outputs_tensor_map_.size() != GATHER_V2_OUTPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": outputs tensor map size must be " << GATHER_V2_OUTPUTS_SIZE << ", but is "
                  << outputs_tensor_map_.size();
    return FAILED;
  }
  // infer tensor shape
  Shape input_shape = inputs_shape_.at(0);
  Shape input_index_shape = inputs_shape_.at(1);
  Shape output_shape = outputs_shape_.at(0);

  TensorLayout input_tensor_layout, input_index_layout, output_tensor_layout;
  if ((input_tensor_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(0), input_shape) != SUCCESS) ||
      (input_index_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(1), input_index_shape) != SUCCESS) ||
      (output_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_.at(0), output_shape) != SUCCESS)) {
    return FAILED;
  }

  TensorInfo input_tensor_info(input_tensor_layout);
  TensorInfo input_index_info(input_index_layout);
  TensorInfo output_tensor_info(output_tensor_layout);

  inputs_tensor_info_.push_back(input_tensor_info);
  inputs_tensor_info_.push_back(input_index_info);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

OperatorVector CreateSubOp(int32_t sub_value) {
  OperatorVector ops;
  OperatorName operator_name = SUB;
  OperatorAttrs operator_attrs;

  py::tuple tuple = py::make_tuple(sub_value);
  mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(tuple, kInt32);
  ValuePtr op_param_value = MakeValue(tensor_ptr);

  Attr op1_param = std::make_pair("", op_param_value);
  OperatorParams operator_param = {std::make_pair(op1_param, 2)};

  OperatorArgs operator_args = std::make_pair(operator_attrs, operator_param);
  Operator op = std::make_pair(operator_name, operator_args);
  ops.push_back(op);
  return ops;
}

Status GatherV2Info::InferTensorSubOps() {
  sub_ops_.clear();
  if ((index_size_ == 0) || (axis_strategy_ == 1)) {
    return SUCCESS;
  }
  int32_t mod_n = 1;
  for (size_t i = IntToSize(axis_) + 1; i < dev_matrix_shape_.size(); i++) {
    mod_n *= dev_matrix_shape_.at(i);
  }
  if ((axis_ >= SizeToInt(dev_matrix_shape_.size())) || axis_ < 0) {
    MS_LOG(ERROR) << "Axis is " << axis_ << ", not in [0, " << dev_matrix_shape_.size() << ").";
  }
  int32_t mod_p = mod_n * dev_matrix_shape_.at(axis_);
  int32_t rank = g_device_manager->global_rank();
  int32_t mod_rank = rank % mod_p;
  mod_rank = static_cast<int32_t>(mod_rank / mod_n);
  if (inputs_shape_.size() != GATHER_V2_INPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << ": inputs shape size must be " << GATHER_V2_INPUTS_SIZE << ", but is "
                  << inputs_shape_.size();
    return FAILED;
  }
  if ((axis_ >= SizeToInt(inputs_shape_.at(0).size())) || axis_ < 0) {
    MS_LOG(ERROR) << "Axis is " << axis_ << ", not in [0, " << inputs_shape_.at(0).size() << ").";
  }
  int32_t sub_value = static_cast<int32_t>(inputs_shape_.at(0).at(axis_) / dev_matrix_shape_.at(axis_)) * mod_rank;

  OperatorVector sub_op;
  sub_ops_.emplace_back(std::move(sub_op));
  sub_op = CreateSubOp(sub_value);
  sub_ops_.emplace_back(std::move(sub_op));
  return SUCCESS;
}

Status GatherV2Info::Init(const StrategyPtr& strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  Status status = InferTensorSubOps();
  if (status != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferTensorSubOps failed.";
    return status;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status GatherV2Info::InitForCostModel(const StrategyPtr& strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Init for cost model failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    }
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status GatherV2Info::GenerateStrategies(int32_t stage_id) {
  if ((inputs_shape_.size() != GATHER_V2_INPUTS_SIZE) || (outputs_shape_.size() != GATHER_V2_OUTPUTS_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size(" << inputs_shape_.size() << ") or outputs shape size("
                  << outputs_shape_.size() << "is wrong.";
    return FAILED;
  }

  is_auto_parallel_ = true;
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, {inputs_shape_.at(0)}, splittable_inputs, &sp_vector) !=
      SUCCESS) {
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

Status GatherV2Info::SetCostUnderStrategy(const StrategyPtr& strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Set cost under strategy failed.";
    }
    return FAILED;
  }
  return SUCCESS;
}

std::shared_ptr<std::vector<std::vector<int32_t>>> GatherV2Info::GenerateBatchStrategies() {
  if (inputs_shape_.size() != GATHER_V2_INPUTS_SIZE) {
    MS_LOG(EXCEPTION) << name_ << ": inputs shape size must be " << GATHER_V2_INPUTS_SIZE << ", but is "
                      << inputs_shape_.size();
  }
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(0).size();
  if (GetAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << "GetAttrs failed!";
  }

  Dimensions strategy;
  if (index_size_ != 1) {
    strategy.push_back(1);
  } else {
    strategy.push_back(SizeToInt(dev_num));
  }
  for (size_t i = 1; i < inputs_shape_[0].size(); i++) {
    strategy.push_back(1);
  }
  std::vector<Dimensions> strategy_v = {strategy};
  return std::make_shared<std::vector<std::vector<int32_t>>>(strategy_v);
}
}  // namespace parallel
}  // namespace mindspore
