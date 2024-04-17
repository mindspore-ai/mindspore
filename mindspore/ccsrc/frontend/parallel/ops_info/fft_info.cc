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

#include "frontend/parallel/ops_info/fft_info.h"

#include <utility>
#include <algorithm>

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {

void FFTBase::ReComputeBatchSplitFlagList() {
  if (input_split_.empty()) {
    MS_LOG(DEBUG) << "ReComputeBatchSplitFlagList before GetAttrs()?";
    if (GetDimSplit() != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": failed to get dim.";
    }
  }
  split_flag_list_[0] = input_split_[0];
}

std::vector<StrategyPtr> FFTBase::GenerateOpStrategies(int64_t stage_id) {
  if (input_split_.empty()) {
    MS_LOG(DEBUG) << "GenerateOpStrategies before GetAttrs()?";
    if (GetDimSplit() != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": failed to get dim.";
    }
  }

  Shapes splittable_inputs = {input_split_};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  return sp_vector;
}

Status FFTBase::GetAttrs() {
  MS_LOG(DEBUG) << "FFTBase: GetAttrs() start.";
  return GetDimSplit();
}

Status FFTBase::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    return FAILED;
  }

  size_t strategy_size = strategy->GetInputNumber();
  if (strategy_size > 1) {
    MS_LOG(ERROR) << name_ << ": Strategy size cannot be greater than 1.";
    return FAILED;
  }

  Shape input_strategy = strategy->GetInputDim().at(0);
  size_t strategy_len = input_strategy.size();
  if (strategy_len != input_split_.size()) {
    MS_LOG(ERROR) << name_ << ": input strategy size cannot match input size.";
    return FAILED;
  }

  int64_t split_num = 1;
  for (size_t i = 0; i < strategy_len; i++) {
    if (input_strategy[i] > 1 && input_split_[i] == false) {
      MS_LOG(ERROR) << name_ << ": Cannot split tensor on Non-batch dim.";
      return FAILED;
    }
    split_num *= input_strategy[i];
  }

  if (split_num > stage_device_size_) {
    MS_LOG(ERROR) << name_ << " The number of splits cannot be greater than the number of devices.";
    return FAILED;
  }

  return SUCCESS;
}

Status FFTBase::InferTensorMap() {
  size_t size = input_split_.size();
  Shape tensor_map_in(size, -1);
  Shape tensor_map_out(size, -1);
  int64_t tensor_map_index = 0;
  for (size_t i = 0; i < size; i++) {
    size_t index_ = size - i - 1;
    if (input_split_[index_]) {
      tensor_map_in[index_] = tensor_map_index;
      tensor_map_out[index_] = tensor_map_index;
      tensor_map_index++;
    }
  }
  MS_LOG(DEBUG) << name_ << ": TensorMap value = " << ListToString(tensor_map_in);
  (void)inputs_tensor_map_.emplace_back(std::move(tensor_map_in));
  (void)outputs_tensor_map_.emplace_back(std::move(tensor_map_out));
  return SUCCESS;
}

Status FFTBase::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  Strategies stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy can not be empty.";
    return FAILED;
  }
  dev_matrix_shape_ = stra.at(0);
  return SUCCESS;
}

Status FFTBase::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }

  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  size_t index_num = GetIndexNum();
  for (size_t i = 1; i < index_num; i++) {
    // Push empty mirror op for n, dim, norm
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  return SUCCESS;
}

Status FFTIntDim::GetDimSplit() {
  if (input_value_.size() != GetIndexNum()) {
    MS_LOG(ERROR) << name_ << ": Invalid inputs size " << input_value_.size();
    return FAILED;
  }

  if (!input_value_[GetDimIndex()]->isa<Int64Imm>()) {
    MS_LOG(ERROR) << name_ << ": The type of dim is not int64_t";
    return FAILED;
  }

  int64_t dim = GetValue<int64_t>(input_value_[GetDimIndex()]);
  int64_t input_dim = SizeToLong(inputs_shape_[0].size());
  if ((dim > (input_dim - 1)) || (dim < -input_dim)) {
    MS_LOG(ERROR) << name_ << ": The dim(" << dim << ") is out of range[" << (-input_dim) << ", " << (input_dim - 1)
                  << "]";
    return FAILED;
  }

  if (dim < 0) {
    dim += input_dim;
  }

  std::vector<int64_t> splits(inputs_shape_[0].size(), 1);
  splits[dim] = 0;
  input_split_.swap(splits);
  MS_LOG(DEBUG) << name_ << ": The input_split_ size is " << input_split_.size() << ", value is "
                << ListToString(input_split_);
  return SUCCESS;
}

Status FFTTupleDim::GetDimSplit() {
  if (input_value_.size() != GetIndexNum()) {
    MS_LOG(ERROR) << name_ << ": Invalid inputs size " << input_value_.size();
    return FAILED;
  }

  if (!input_value_[GetDimIndex()]->isa<ValueTuple>()) {
    MS_LOG(ERROR) << name_ << ": The type of dim is not tuple";
    return FAILED;
  }

  auto dims = GetValue<std::vector<int64_t>>(input_value_[GetDimIndex()]);
  int64_t input_dim = SizeToLong(inputs_shape_[0].size());

  std::vector<int64_t> splits(inputs_shape_[0].size(), 1);
  for (const auto &dim : dims) {
    if ((dim > (input_dim - 1)) || (dim < -input_dim)) {
      MS_LOG(ERROR) << name_ << ": The dim(" << dim << ") is out of range[" << (-input_dim) << ", " << (input_dim - 1)
                    << "]";
      return FAILED;
    }
    if (dim < 0) {
      splits[dim + input_dim] = 0;
    } else {
      splits[dim] = 0;
    }
  }

  input_split_.swap(splits);
  MS_LOG(DEBUG) << name_ << ": The input_split_ size is " << input_split_.size() << ", value is "
                << ListToString(input_split_);
  return SUCCESS;
}

REGISTER(FFTShiftInfo);
REGISTER(IFFTShiftInfo);
REGISTER(FFTInfo);
REGISTER(IFFTInfo);
REGISTER(FFT2Info);
REGISTER(IFFT2Info);
REGISTER(FFTNInfo);
REGISTER(IFFTNInfo);
REGISTER(RFFTInfo);
REGISTER(IRFFTInfo);
REGISTER(DCTInfo);
REGISTER(IDCTInfo);
REGISTER(DCTNInfo);
REGISTER(IDCTNInfo);
}  // namespace parallel
}  // namespace mindspore
