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

#include "parallel/ops_info/transpose_info.h"

#include <memory>
#include <vector>

#include "parallel/device_manager.h"
#include "parallel/device_matrix.h"
#include "parallel/step_parallel.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status TransposeInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_, is_auto_parallel_) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(DEBUG) << name_ << ": Invalid strategy.";
    } else {
      MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    }
    return FAILED;
  }

  return SUCCESS;
}

Status TransposeInfo::InferDevMatrixShape() {
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  input_strategy_ = stra.at(0);
  for (auto &iter : input_strategy_) {
    dev_matrix_shape_.push_back(iter);
  }
  return SUCCESS;
}

// there is no Parameter for Transpose Primitive, so no need to do all reduce
Status TransposeInfo::InferMirrorOps() { return SUCCESS; }

// there is no reduction dimension for forward computation of Transpose Primitive, so no need to do all reduce
Status TransposeInfo::InferForwardCommunication() { return SUCCESS; }

/*
 * get perm input of Transpose Primitive
 * perm is a permutation of the dimensions of input
 * the result is saved in axis_v_
 */
Status TransposeInfo::ComputeAxis() {
  if (input_value_[1] == nullptr) {
    MS_LOG(ERROR) << name_ << ": input_value_[1] is nullptr.";
    return FAILED;
  }
  std::vector<ValuePtr> elements;
  ValueTuplePtr dim_tuple = input_value_[1]->cast<ValueTuplePtr>();
  if (dim_tuple == nullptr) {
    MS_LOG(ERROR) << name_ << ": input_value_[1] must be ValueTuplePtr.";
    return FAILED;
  }
  elements = dim_tuple->value();
  if (elements.size() != inputs_shape_[0].size()) {
    MS_LOG(ERROR) << name_ << ": elements size must equal to inputs shape 0 size.";
    return FAILED;
  }
  axis_v_.clear();
  for (auto &element : elements) {
    MS_EXCEPTION_IF_NULL(element);
    if (element->isa<Int32Imm>()) {
      int32_t axis = element->cast<Int32ImmPtr>()->value();
      axis_v_.push_back(axis);
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis must be int32.";
      return FAILED;
    }
  }

  for (int32_t i = 0; i < SizeToInt(axis_v_.size()); i++) {
    auto iter = std::find(axis_v_.begin(), axis_v_.end(), i);
    if (iter == axis_v_.end()) {
      MS_LOG(ERROR) << name_ << ": axis_v_ must be a permutation.";
    }
  }
  return SUCCESS;
}

// the output tensor map is the permutation of input tensor map, the permutation is axis_v
Status TransposeInfo::InferTensorMap() {
  if ((inputs_shape_.size() != 1) || (outputs_shape_.size() != 1)) {
    MS_LOG(ERROR) << name_ << ": inputs_shape_ and outputs_shape_ size must be 1, inputs shape and outputs shape is "
                  << inputs_shape_.size() << ", " << outputs_shape_.size();
    return FAILED;
  }

  std::vector<int32_t> tensor_map_index_input;
  for (size_t j = 0; j < inputs_shape_[0].size(); ++j) {
    tensor_map_index_input.push_back(SizeToInt(inputs_shape_[0].size() - j - 1));
  }
  inputs_tensor_map_.push_back(tensor_map_index_input);

  std::vector<int32_t> tensor_map_index_output = tensor_map_index_input;
  for (uint32_t i = 0; i < tensor_map_index_output.size(); i++) {
    tensor_map_index_output[i] = tensor_map_index_input[IntToUint(axis_v_[i])];
  }
  outputs_tensor_map_.push_back(tensor_map_index_output);
  return SUCCESS;
}

// the output tensor strategy is the permutation of input tensor strategy, the permutation is axis_v
Strategys TransposeInfo::GetOutputsStrategy() {
  Strategys outputs_strategy;
  std::vector<int32_t> strategy = input_strategy_;
  for (uint32_t i = 0; i < strategy.size(); i++) {
    strategy[i] = input_strategy_[IntToUint(axis_v_[i])];
  }
  outputs_strategy.push_back(strategy);
  return outputs_strategy;
}

Status TransposeInfo::InferTensorLayout(TensorLayouts *inputs_layout, TensorLayouts *outputs_layout) {
  if ((inputs_layout == nullptr) || (outputs_layout == nullptr)) {
    MS_LOG(ERROR) << name_ << ": InferTensorLayout: the layout is null.";
    return FAILED;
  }
  Shape shape_in = inputs_shape_.at(0);
  TensorMap tensor_map_in = inputs_tensor_map_.at(0);
  Shape shape_out = outputs_shape_.at(0);
  TensorMap tensor_map_out = outputs_tensor_map_.at(0);

  TensorLayout tensor_layout_in, tensor_layout_out;
  if ((tensor_layout_in.InitFromVector(dev_matrix_shape_, tensor_map_in, shape_in) != SUCCESS) ||
      (tensor_layout_out.InitFromVector(dev_matrix_shape_, tensor_map_out, shape_out) != SUCCESS)) {
    return FAILED;
  }

  inputs_layout->push_back(tensor_layout_in);
  outputs_layout->push_back(tensor_layout_out);
  return SUCCESS;
}

Status TransposeInfo::InferTensorInfo() {
  Shapes inputs_slice_shape, outputs_slice_shape;
  Strategys inputs_strategy = strategy_->GetInputDim();
  Strategys outputs_strategy = GetOutputsStrategy();
  if (InferSliceShape(inputs_strategy, outputs_strategy, &inputs_slice_shape, &outputs_slice_shape) != SUCCESS) {
    return FAILED;
  }

  TensorLayouts inputs_layout, outputs_layout;
  if (InferTensorLayout(&inputs_layout, &outputs_layout) != SUCCESS) {
    return FAILED;
  }
  TensorLayout tensor_layout_in = inputs_layout.at(0);
  TensorLayout tensor_layout_out = outputs_layout.at(0);
  Shape shape_array_in = inputs_shape_.at(0);
  Shape slice_shape_in = inputs_slice_shape.at(0);
  Shape shape_array_out = outputs_shape_.at(0);
  Shape slice_shape_out = outputs_slice_shape.at(0);
  TensorInfo tensor_info_in(tensor_layout_in, shape_array_in, slice_shape_in);
  TensorInfo tensor_info_out(tensor_layout_out, shape_array_out, slice_shape_out);
  inputs_tensor_info_.push_back(tensor_info_in);
  outputs_tensor_info_.push_back(tensor_info_out);
  return SUCCESS;
}

// compute axis_v_ during this method
Status TransposeInfo::GetAttrs() { return ComputeAxis(); }

Status TransposeInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status TransposeInfo::InitForCostModel(const StrategyPtr &strategy) {
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

Status TransposeInfo::SetCostUnderStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  if (SetCostUnderStrategyBase(strategy) != SUCCESS) {
    if (is_auto_parallel_) {
      MS_LOG(ERROR) << name_ << ": Set cost under strategy failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": Set cost under strategy failed.";
    }
    return FAILED;
  }

  return SUCCESS;
}

Status TransposeInfo::GenerateStrategies(int32_t stage_id) {
  if (GetAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GetAttrs failed.";
    return FAILED;
  }
  if ((inputs_shape_.size() != 1) || (outputs_shape_.size() != 1)) {
    MS_LOG(ERROR) << name_ << ": inputs shape size or outputs shape size is wrong, " << inputs_shape_.size() << ", "
                  << outputs_shape_.size();
    return FAILED;
  }
  is_auto_parallel_ = true;
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": GenerateStrategiesForIndependentInputs failed";
    return FAILED;
  }
  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated " << success << "strategy.";
      PrintStrategy(sp);
    }
  }
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
