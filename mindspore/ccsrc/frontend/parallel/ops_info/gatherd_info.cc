/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/gatherd_info.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/resource.h"

namespace mindspore {
namespace parallel {
Status GatherDInfo::GetAttrs() {
  if (input_value_.size() != 3) {
    MS_LOG(ERROR) << name_ << ": Invalid input_value's size " << input_value_.size();
    return FAILED;
  }

  if (!input_value_[1]->isa<Int64Imm>()) {
    MS_LOG(ERROR) << name_ << ": The value of dim is not int";
    return FAILED;
  }

  int64_t dim = GetValue<int64_t>(input_value_[1]);
  int64_t input_dim = SizeToLong(inputs_shape_[0].size());
  if ((dim > (input_dim - 1)) || (dim < -input_dim)) {
    MS_LOG(ERROR) << name_ << ": The dim(" << dim << ") is out of range[" << (-input_dim) << ", " << (input_dim - 1)
                  << "]";
    return FAILED;
  }

  if (dim < 0) {
    dim += input_dim;
  }

  dim_ = LongToSize(dim);
  MS_LOG(INFO) << name_ << ": The dim is " << dim_;
  return SUCCESS;
}

Status GatherDInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  if (stra.size() != 2) {
    MS_LOG(ERROR) << name_ << ": The size of strategy must be 2, but got " << stra.size();
    return FAILED;
  }

  Dimensions input_strategy = stra[0];
  Dimensions index_strategy = stra[1];
  if (input_strategy[dim_] != 1 || index_strategy[dim_] != 1) {
    MS_LOG(ERROR) << name_ << ": The dimension of dim can not be split";
    return FAILED;
  }

  if (input_strategy != index_strategy) {
    MS_LOG(ERROR) << name_ << ": The dimension of dim can not be split";
    return FAILED;
  }

  return SUCCESS;
}

Status GatherDInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status GatherDInfo::InferTensorMap() {
  Shape input_tensor_map;
  size_t size = inputs_shape_[0].size();
  for (size_t i = 0; i < size; ++i) {
    input_tensor_map.push_back(SizeToLong(size - i - 1));
  }

  inputs_tensor_map_.push_back(input_tensor_map);   // input
  inputs_tensor_map_.push_back(input_tensor_map);   // index
  outputs_tensor_map_.push_back(input_tensor_map);  // output
  return SUCCESS;
}

Status GatherDInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }

  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    TensorLayout input_layout;
    if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[i], inputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo input_tensor_info(input_layout);
    inputs_tensor_info_.push_back(input_tensor_info);
  }

  TensorInfo dim_tensor_info;
  (void)inputs_tensor_info_.insert(inputs_tensor_info_.begin() + 1, dim_tensor_info);

  for (size_t i = 0; i < outputs_tensor_map_.size(); ++i) {
    TensorLayout output_layout;
    if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[i], outputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo output_tensor_info(output_layout);
    outputs_tensor_info_.push_back(output_tensor_info);
  }
  return SUCCESS;
}

Status GatherDInfo::InferMirrorOps() {
  mirror_ops_.clear();
  if (inputs_shape_.empty()) {
    MS_LOG(INFO) << name_ << ": The inputs size is empty";
    return SUCCESS;
  }

  if (inputs_tensor_map_.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": The size of inputs tensor map is not equal to the size of inputs shape";
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

  OperatorVector mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
  mirror_ops_.push_back(mirror_op);  // input

  OperatorVector tmp_mirror_op;  // dim
  mirror_ops_.push_back(tmp_mirror_op);

  mirror_ops_.push_back(mirror_op);  // index
  return SUCCESS;
}

void GatherDInfo::ReComputeBatchSplitFlagList() {
  if (InferAttrs() != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Infer attrs failed";
  }

  if (dim_ == 0) {
    MS_LOG(EXCEPTION)
      << name_
      << ": Can not generate batch data parallel strategy since the dim is 0, please set others strategy for it";
  }

  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    split_flag_list_[i] = true;
  }
}

Status GatherDInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> GatherDInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
  input0_split[dim_] = 0;
  Shapes splittable_inputs = {input0_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << " : Generate strategies for independent inputs() failed.";
  }

  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG(EXCEPTION) << name_ << ": The strategy is null or empty";
    }
    Strategys tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    for (size_t i = 0; i < inputs_shape_.size(); ++i) {
      tmp_strategy.push_back(first_input_strategy);
    }
    sp->ResetInputs(tmp_strategy);
  }
  return sp_vector;
}

Status GatherDInfo::Init(const StrategyPtr &strategy) {
  if (InitWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init failed.";
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Init success.";
  return SUCCESS;
}

Status GatherDInfo::InitForCostModel(const StrategyPtr &strategy) {
  if (InitForCostModelWithAutoRepeatCalc(strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Init for cost model failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": Init for cost model success.";
  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
