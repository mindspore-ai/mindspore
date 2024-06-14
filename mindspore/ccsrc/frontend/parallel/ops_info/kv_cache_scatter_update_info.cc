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

#include "frontend/parallel/ops_info/kv_cache_scatter_update_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace parallel {
Status KVCacheScatterUpdateInfo::CheckStrategy3Dims(const Dimensions &strategy_var, const Dimensions &strategy_update) {
  if (strategy_var.at(1) != 1 || strategy_update.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The seq_len can't be shard, but got"
                  << " strategy_var's seq_len strategy: " << strategy_var.at(1)
                  << "; strategy_update's seq_len strategy: " << strategy_update.at(1);
    return FAILED;
  }
  if (strategy_var.at(2) != strategy_update.at(2)) {
    MS_LOG(ERROR) << name_ << " Invalid strategy: The hidden_size must be shard at the same time, but got"
                  << " strategy_var's strategy: " << strategy_var
                  << ", strategy_update's strategy: " << strategy_update;
    return FAILED;
  }
  return SUCCESS;
}

Status KVCacheScatterUpdateInfo::CheckStrategy4Dims(const Dimensions &strategy_var, const Dimensions &strategy_update) {
  // num_head must be the same strategy.
  if (strategy_var.at(1) != strategy_update.at(1)) {
    MS_LOG(ERROR) << name_ << " Invalid strategy: The num_head must be shard at the same time, but got"
                  << " strategy_var's strategy: " << strategy_var
                  << ", strategy_update's strategy: " << strategy_update;
    return FAILED;
  }
  if (strategy_var.at(2) != 1 || strategy_update.at(2) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The seq_len can't be shard, but got"
                  << " strategy_var's seq_len strategy: " << strategy_var.at(2)
                  << "; update's seq_len strategy: " << strategy_update.at(2);
    return FAILED;
  }
  // hidden_size must be the same strategy.
  if (strategy_var.at(3) != strategy_update.at(3)) {
    MS_LOG(ERROR) << name_ << " Invalid strategy: The hidden_size must be shard at the same time, but got"
                  << " strategy_var's strategy: " << strategy_var
                  << ", strategy_update's strategy: " << strategy_update;
    return FAILED;
  }
  return SUCCESS;
}

Status KVCacheScatterUpdateInfo::SetDims(const StrategyPtr &strategy) {
  auto input_strategys = strategy->GetInputDim();
  auto strategy_var = input_strategys.at(0);

  const size_t input_dims4 = 4;
  const size_t input_dims3 = 3;
  if (strategy_var.size() == input_dims4) {
    is_input_dims_4_ = true;
  } else if (strategy_var.size() == input_dims3) {
    is_input_dims_4_ = false;
  } else {
    return FAILED;
  }

  return SUCCESS;
}

Status KVCacheScatterUpdateInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  if (SetDims(strategy) != SUCCESS) {
    return FAILED;
  }

  auto input_strategys = strategy->GetInputDim();
  auto strategy_var = input_strategys.at(0);      // (1, 8, 1, 1) or (1, 1, 8)
  auto strategy_indices = input_strategys.at(1);  // (1)
  auto strategy_update = input_strategys.at(2);   // (1, 8, 1, 1) or (1, 1, 8)

  if (strategy_indices.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The indices can't be shard, but got"
                  << " indices's strategy: " << strategy_indices;
    return FAILED;
  }

  // batch must be the same strategy.
  if (strategy_var.at(0) != strategy_update.at(0)) {
    MS_LOG(ERROR) << name_ << " Invalid strategy: The batch must be shard at the same time, but got"
                  << " strategy_var's strategy: " << strategy_var
                  << ", strategy_update's strategy: " << strategy_update;
    return FAILED;
  }

  if (is_input_dims_4_) {
    return CheckStrategy4Dims(strategy_var, strategy_update);
  }
  return CheckStrategy3Dims(strategy_var, strategy_update);
}

Status KVCacheScatterUpdateInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << "The strategy is empty";
    return FAILED;
  }

  dev_matrix_shape_ = stra[0];
  return SUCCESS;
}

Status KVCacheScatterUpdateInfo::InferTensorMap() {
  if (is_input_dims_4_) {
    Shape var_tensor_map{3, 2, 1, 0};
    Shape indices_tensor_map{-1};
    Shape update_tensor_map{3, 2, 1, 0};
    inputs_tensor_map_.emplace_back(var_tensor_map);
    inputs_tensor_map_.emplace_back(indices_tensor_map);
    inputs_tensor_map_.emplace_back(update_tensor_map);
  } else {
    Shape cache_tensor_map{2, 1, 0};
    Shape indices_tensor_map{-1};
    Shape update_tensor_map{2, 1, 0};
    inputs_tensor_map_.emplace_back(cache_tensor_map);
    inputs_tensor_map_.emplace_back(indices_tensor_map);
    inputs_tensor_map_.emplace_back(update_tensor_map);
  }

  if (is_input_dims_4_) {
    Shape out_tensor_map{3, 2, 1, 0};
    outputs_tensor_map_.emplace_back(out_tensor_map);
  } else {
    Shape out_tensor_map{2, 1, 0};
    outputs_tensor_map_.emplace_back(out_tensor_map);
  }
  return SUCCESS;
}

REGISTER(KVCacheScatterUpdateInfo);
}  // namespace parallel
}  // namespace mindspore
