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

#include "frontend/parallel/ops_info/matmul_ffn_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// MatMulQkv has 3 inputs and 2 outputs
// x:         (batch * seq_len (inc is 1), hidden_size)
// weight_1:         (weight_1_hidden_size, hidden_size)
// weight_2:         (weight_2_hidden_size, hidden_size)
// ------------------------------
// output_1:  (batch * seq_len (inc is 1), weight_1_hidden_size)
// output_2:  (batch * seq_len (inc is 1), weight_2_hidden_size)

constexpr size_t kMatMulFfnOutputSize = 2;

Status MatmulFfnInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  // TODO

  return SUCCESS;
}

Status MatmulFfnInfo::InferDevMatrixShape() {
  auto input_strategies = strategy()->GetInputDim();
  auto x = input_strategies.at(0);  // (batch * seq_len, hidden_size)
  auto weight_1 = input_strategies.at(1);
  // dp   mp
  // 1    0
  dev_matrix_shape_ = {x.at(0), weight_1.at(0)};

  return SUCCESS;
}

Status MatmulFfnInfo::InferTensorMap() {
  Shape x_tensor_map{1, -1};
  Shape weight_1_tensor_map{0, -1};
  Shape weight_2_tensor_map{0, -1};
  inputs_tensor_map_.emplace_back(x_tensor_map);
  inputs_tensor_map_.emplace_back(weight_1_tensor_map);
  inputs_tensor_map_.emplace_back(weight_2_tensor_map);

  Shape output_q_tensor_map{1, 0};
  Shape output_k_tensor_map{1, 0};
  outputs_tensor_map_.emplace_back(output_q_tensor_map);
  outputs_tensor_map_.emplace_back(output_k_tensor_map);

  return SUCCESS;
}

Status MatmulFfnInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != kMatMulFfnOutputSize) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map must be 2, but got " << outputs_tensor_map_.size();
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}
REGISTER(MatmulFfnInfo);
}  // namespace parallel
}  // namespace mindspore