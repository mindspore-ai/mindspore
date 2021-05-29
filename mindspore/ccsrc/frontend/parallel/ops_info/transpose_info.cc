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

#include "frontend/parallel/ops_info/transpose_info.h"

#include <memory>
#include <vector>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/step_parallel.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
Status TransposeInfo::CheckStrategy(const StrategyPtr &strategy) { return CheckStrategyValue(strategy, inputs_shape_); }

Status TransposeInfo::InferDevMatrixShape() {
  Strategys stra = strategy_->GetInputDim();
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
    if (element->isa<Int64Imm>()) {
      int64_t axis = element->cast<Int64ImmPtr>()->value();
      axis_v_.push_back(axis);
    } else {
      MS_LOG(ERROR) << name_ << ": The value of axis must be int32.";
      return FAILED;
    }
  }

  for (int64_t i = 0; i < SizeToLong(axis_v_.size()); i++) {
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

  Shape tensor_map_index_input;
  for (size_t j = 0; j < inputs_shape_[0].size(); ++j) {
    tensor_map_index_input.push_back(SizeToLong(inputs_shape_[0].size() - j - 1));
  }
  inputs_tensor_map_.push_back(tensor_map_index_input);

  Shape tensor_map_index_output = tensor_map_index_input;
  for (uint64_t i = 0; i < tensor_map_index_output.size(); i++) {
    tensor_map_index_output[i] = tensor_map_index_input[LongToUlong(axis_v_[i])];
  }
  outputs_tensor_map_.push_back(tensor_map_index_output);
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
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}

Status TransposeInfo::SetCostUnderStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

std::vector<StrategyPtr> TransposeInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  Shapes splittable_inputs = {input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": GenerateStrategiesForIndependentInputs failed";
  }

  return sp_vector;
}
}  // namespace parallel
}  // namespace mindspore
