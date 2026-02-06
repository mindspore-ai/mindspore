/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/wkv_info.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include <numeric>

#include "include/utils/parallel_context.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/costmodel.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/redistribution_operator_infer.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
Status WKVInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto input_strategys = strategy->GetInputDim();
  auto strategy_w = input_strategys.at(0);
  auto strategy_u = input_strategys.at(1);
  auto strategy_k = input_strategys.at(2);
  auto strategy_v = input_strategys.at(3);
  auto strategy_status = input_strategys.at(4);
  auto strategy_status_1 = input_strategys.at(5);
  auto strategy_status_2 = input_strategys.at(6);
  if (strategy_w != strategy_u) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: w's strategy: " << strategy_w
                  << " must be equal to u's strategy: " << strategy_u;
    return FAILED;
  }
  if (strategy_k != strategy_v) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: k's strategy: " << strategy_k
                  << " must be equal to v's strategy: " << strategy_v;
    return FAILED;
  }
  if (strategy_status != strategy_status_1 || strategy_status != strategy_status_2) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: status's strategy must be same, but got: "
                  << "status0: " << strategy_status << "status1: " << strategy_status_1
                  << " status2: " << strategy_status_2;
    return FAILED;
  }
  if (strategy_w.at(0) != strategy_k.at(k_hidden_dim) || strategy_w.at(0) != strategy_status.at(1)) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The hidden dim must be shard at the same time, but got"
                  << " w's strategy: " << strategy_w << " k's strategy: " << strategy_k
                  << " status's strategy: " << strategy_status;
    return FAILED;
  }
  if (strategy_k.at(0) != strategy_status.at(0)) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The batch dim must be shard at the same time, but got"
                  << " k's strategy: " << strategy_k << " status's strategy: " << strategy_status;
    return FAILED;
  }
  if (strategy_k.at(1) != 1) {
    MS_LOG(ERROR) << name_
                  << ": Invalid strategy: The sequence length can't be shard, but got k's strategy: " << strategy_k;
    return FAILED;
  }

  return SUCCESS;
}

Status WKVInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = strategy_->GetInputDim().at(matrix_shape_dim);
  return SUCCESS;
}

Status WKVInfo::InferTensorMap() {
  // w, u: [hidden]
  Shape w_tensor_map{0};
  Shape u_tensor_map{0};
  // Status: [bs, hidden]
  Shape status_tensor_map{2, 0};
  // k, v: [bs, seq_length, hidden]
  Shape k_tensor_map;
  auto input_k_shape = inputs_shape_.at(2).size();
  for (size_t i = 0; i < input_k_shape; ++i) {
    k_tensor_map.emplace_back(SizeToLong(LAST_INDEX(input_k_shape) - i));
  }
  Shape v_tensor_map{k_tensor_map};
  inputs_tensor_map_.emplace_back(w_tensor_map);
  inputs_tensor_map_.emplace_back(u_tensor_map);
  inputs_tensor_map_.emplace_back(k_tensor_map);
  inputs_tensor_map_.emplace_back(v_tensor_map);
  (void)inputs_tensor_map_.insert(inputs_tensor_map_.end(), wkv_insert_num, status_tensor_map);

  Shape out_tensor_map_0{k_tensor_map};
  Shape out_tensor_map_1{status_tensor_map};
  outputs_tensor_map_.emplace_back(out_tensor_map_0);
  (void)outputs_tensor_map_.insert(outputs_tensor_map_.end(), wkv_insert_num, status_tensor_map);
  return SUCCESS;
}

Status WKVInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> WKVInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape splitable_w{3};
  Shape splitable_u{3};
  Shape splitable_status{1, 3};
  Shape splitable_k{1, 0, 3};
  Shape splitable_v{splitable_k};
  Shapes splitable_inputs = {splitable_w, splitable_u, splitable_k, splitable_v};
  (void)splitable_inputs.insert(splitable_inputs.end(), wkv_insert_num, splitable_status);
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, splitable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies for dependent inputs() failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": No valid strategy.";
  }
  return sp_vector;
}

Status WKVInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_map_[0].empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[0]) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

std::shared_ptr<Strategies> WKVInfo::GenerateBatchStrategies() {
  Dimensions stra_w{1};
  Dimensions stra_u{1};
  Dimensions stra_k{stage_device_size_, 1, 1};
  Dimensions stra_v{stage_device_size_, 1, 1};
  Dimensions stra_status{stage_device_size_, 1};
  Strategies batch_strategy = {stra_w, stra_u, stra_k, stra_v};
  (void)batch_strategy.insert(batch_strategy.end(), wkv_insert_num, stra_status);
  return std::make_shared<Strategies>(batch_strategy);
}
REGISTER(WKVInfo);
}  // namespace parallel
}  // namespace mindspore
