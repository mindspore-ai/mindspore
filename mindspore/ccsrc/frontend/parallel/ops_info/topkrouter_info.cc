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

#include "frontend/parallel/ops_info/topkrouter_info.h"

#include <algorithm>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {

// expert_index shape rank is 3, counter shape rank is 2
// expert_index:    (btach, hidden_size, k)
// capacity         (1)
// expert_dim       (1)
//

// ------------------------------
// output:
// dispatch_index:  (batch, expert_dim, capacity)
// combine_index:   (batch, hidden_size, k)

// split strategy
// batch is able to split.
// other is not able to split.

const int64_t OutputSize = 2;

Status TopKRouterInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  auto input_strategys = strategy->GetInputDim();
  auto strategy_input = input_strategys.at(0);  // (4, 1, 1)
  if (strategy_input.at(INDEX_ONE) != 1 || strategy_input.at(INDEX_TWO) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The expert_index last two dimensions can't be shard, but got"
                  << " expert_index's strategy: " << strategy_input;
    return FAILED;
  }
  return SUCCESS;
}
Status TopKRouterInfo::InferMirrorOps() {
  if (inputs_tensor_map_.size() != 1) {
    MS_LOG(ERROR) << "The size of inputs tensor map is not equal 1.";
    return FAILED;
  }
  return SUCCESS;
}

Status TopKRouterInfo::InferDevMatrixShape() {
  // batch, hidden_size, k, <==> 2, 1, 0
  auto input_strategys = strategy()->GetInputDim();
  dev_matrix_shape_ = input_strategys.at(0);
  return SUCCESS;
}

Status TopKRouterInfo::InferTensorMap() {
  // batch, hidden_size, k, expert_dim, capactity <==> 4, 3, 2, 1, 0
  Shape input_tensor_map{2, -1, -1};
  inputs_tensor_map_.emplace_back(input_tensor_map);

  // output shape
  Shape dispatch_tensor_map{2, -1, -1};
  Shape combine_tensor_map{2, -1, -1};
  outputs_tensor_map_.emplace_back(dispatch_tensor_map);
  outputs_tensor_map_.emplace_back(combine_tensor_map);

  return SUCCESS;
}

Status TopKRouterInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != OutputSize) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map " << outputs_tensor_map_.size() << " is error";
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

REGISTER(TopKRouterInfo);
}  // namespace parallel
}  // namespace mindspore
