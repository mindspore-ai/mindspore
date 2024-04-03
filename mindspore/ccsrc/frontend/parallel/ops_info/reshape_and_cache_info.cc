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

#include "frontend/parallel/ops_info/reshape_and_cache_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// ReshapeAndCache has 5 inputs
// key:           (batch, seq_len, hidden_size)
// value:         (batch, seq_len, hidden_size)
// key_cache:     (block_size, num_blocks, num_head, head_dim)
// value_cache:   (block_size, num_blocks, num_head, head_dim)
// slot_mapping:  (batch * seq_len)
// ------------------------------
// output:        (batch, seq_len, hidden_size)

// split strategy
// batch is able to split
// seq_len is not able to split
// num_blocks is not able to split
// block_size is not able to split
// hidden_size is able to split
// num_head is able to split, same as hidden_size
// head_dim is not able to split
// slot_mapping is not able to split

Status ReshapeAndCacheInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  auto input_strategies = strategy->GetInputDim();
  auto strategy_key_update = input_strategies.at(0);
  auto strategy_value_update = input_strategies.at(1);
  auto strategy_key_cache = input_strategies.at(2);
  auto strategy_value_cache = input_strategies.at(3);
  auto strategy_slot_mapping = input_strategies.at(4);

  if (strategy_slot_mapping.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The slot_mapping can't be shard, but got"
                  << " slot_mapping's strategy: " << strategy_slot_mapping;
    return FAILED;
  }

  if (strategy_key_update.at(1) != 1 || strategy_value_update.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The seq_len can't be shard, but got"
                  << " key's seq_len strategy: " << strategy_key_update.at(0)
                  << ", value's seq_len strategy: " << strategy_value_update.at(0);
    return FAILED;
  }

  if (strategy_key_cache.at(0) != 1 || strategy_value_cache.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The block_size can't be shard, but got"
                  << " key_cache's block_size strategy: " << strategy_key_cache.at(0)
                  << ", value_cache's block_size strategy: " << strategy_value_cache.at(0);
    return FAILED;
  }

  if (strategy_key_cache.at(1) != 1 || strategy_value_cache.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The num_blocks can't be shard, but got"
                  << " key_cache's num_blocks strategy: " << strategy_key_cache.at(1)
                  << ", value_cache's num_blocks strategy: " << strategy_value_cache.at(1);
    return FAILED;
  }

  if ((strategy_key_update.at(2) != strategy_value_update.at(2)) ||
      (strategy_key_cache.at(2) != strategy_value_cache.at(2)) ||
      (strategy_key_update.at(2) != strategy_key_cache.at(2))) {
    MS_LOG(ERROR)
      << name_
      << ": Invalid strategy: The update's hidden_size and cache's num_head must be shard at the same time, but got"
      << " key's strategy: " << strategy_key_update << ", value's strategy: " << strategy_value_update
      << ", key_cache's strategy: " << strategy_key_cache << ", value_cache's strategy: " << strategy_value_cache;
    return FAILED;
  }

  if (strategy_key_cache.at(3) != 1 || strategy_value_cache.at(3) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The num_head can't be shard, but got"
                  << " key_cache's num_blocks strategy: " << strategy_key_cache.at(3)
                  << ", value_cache's num_blocks strategy: " << strategy_value_cache.at(3);
    return FAILED;
  }
  return SUCCESS;
}

Status ReshapeAndCacheInfo::InferDevMatrixShape() {
  auto input_strategies = strategy()->GetInputDim();
  auto kv_update = input_strategies.at(0);  // (batch, seq_len, hidden_size)
  auto cache = input_strategies.at(2);      // (block_size, num_blocks, num_head, head_dim)

  // batch   block_size   num_blocks   seq_len   hidden_size
  // 4         3            2            1          0
  dev_matrix_shape_ = {kv_update.at(0), cache.at(0), cache.at(1), kv_update.at(1), cache.at(2)};

  return SUCCESS;
}

Status ReshapeAndCacheInfo::InferTensorMap() {
  Shape kv_update_tensor_map{4, 1, 0};
  Shape cache_tensor_map{-1, -1, 0, -1};
  Shape slot_tensor_map{-1};
  inputs_tensor_map_.emplace_back(kv_update_tensor_map);
  inputs_tensor_map_.emplace_back(kv_update_tensor_map);
  inputs_tensor_map_.emplace_back(cache_tensor_map);
  inputs_tensor_map_.emplace_back(cache_tensor_map);
  inputs_tensor_map_.emplace_back(slot_tensor_map);

  Shape out_tensor_map{-1, -1, 0};
  outputs_tensor_map_.emplace_back(out_tensor_map);

  return SUCCESS;
}
REGISTER(ReshapeAndCacheInfo);
}  // namespace parallel
}  // namespace mindspore
