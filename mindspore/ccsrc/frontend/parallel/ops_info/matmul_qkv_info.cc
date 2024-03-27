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

#include "frontend/parallel/ops_info/matmul_qkv_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// MatMulQkv has 3 inputs and 3 outputs
// x:         (batch * seq_len (inc is 1), query_hidden_size)
// q:         (query_hidden_size, query_hidden_size)
// k:         (key_hidden_size, query_hidden_size)
// v:         (value_hidden_size, query_hidden_size)
// ------------------------------
// output_q:  (batch * seq_len (inc is 1), query_hidden_size)
// output_k:  (batch * seq_len (inc is 1), key_hidden_size)
// output_v:  (batch * seq_len (inc is 1), value_hidden_size)

// split strategy
// batch is not able to split
// seq_len is not able to split
// query_hidden_size is able to split
// key_hidden_size is able to split
// value_hidden_size is able to split
constexpr size_t kMatMulQkvOutputSize = 3;

Status MatmulQkvInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  // TODO

  return SUCCESS;
}

Status MatmulQkvInfo::InferDevMatrixShape() {
  auto input_strategies = strategy()->GetInputDim();
  auto x = input_strategies.at(0);  // (batch , seq_len (inc is 1), q_hidden_size)
  auto q = input_strategies.at(1);
  // dp   mp
  // 1    0
  dev_matrix_shape_ = {x.at(0), q.at(0)};

  return SUCCESS;
}

Status MatmulQkvInfo::InferTensorMap() {
  Shape x_tensor_map{1, -1};
  Shape q_tensor_map{0, -1};
  Shape k_tensor_map{0, -1};
  Shape v_tensor_map{0, -1};

  inputs_tensor_map_.emplace_back(x_tensor_map);
  inputs_tensor_map_.emplace_back(q_tensor_map);
  inputs_tensor_map_.emplace_back(k_tensor_map);
  inputs_tensor_map_.emplace_back(v_tensor_map);

  Shape output_q_tensor_map{1, 0};
  Shape output_k_tensor_map{1, 0};
  Shape output_v_tensor_map{1, 0};
  outputs_tensor_map_.emplace_back(output_q_tensor_map);
  outputs_tensor_map_.emplace_back(output_k_tensor_map);
  outputs_tensor_map_.emplace_back(output_v_tensor_map);

  return SUCCESS;
}

Status MatmulQkvInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != kMatMulQkvOutputSize) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map must be 3, but got " << outputs_tensor_map_.size();
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}
REGISTER(MatmulQkvInfo);
}  // namespace parallel
}  // namespace mindspore