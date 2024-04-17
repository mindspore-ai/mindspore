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

#include "frontend/parallel/ops_info/paged_attention_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// PagedAttention has 5 inputs
// query:         (batch * seq_len (inc is 1), num_head, head_dim)
// key_cache:     (block_size, num_blocks, num_head, head_dim)
// value_cache:   (block_size, num_blocks, num_head, head_dim)
// block_tables:  (batch, max_num_block_per_batch)
// context_lens:  (batch * seq_len)
// ------------------------------
// output:        (batch * seq_len (inc is 1), num_head, head_dim)

// split strategy
// num_blocks is not able to split
// block_size is not able to split
// batch is able to split
// num_head is able to split
// head_dim is able to split
// max_num_block_per_batch is not able to split

Status PagedAttentionInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  auto input_strategies = strategy->GetInputDim();
  auto strategy_query = input_strategies.at(0);         // (dp, mp, 1)
  auto strategy_key_cache = input_strategies.at(1);     // (1, 1, mp, 1)
  auto strategy_value_cache = input_strategies.at(2);   // (1, 1, mp, 1)
  auto strategy_block_tables = input_strategies.at(3);  // (dp, 1)

  if (strategy_block_tables.at(1) != 1) {
    MS_LOG(ERROR)
      << name_
      << ": Invalid strategy: The second dim of block_tables \"max_num_block_per_batch\" can't be shard, but got"
      << " block_tables's strategy: " << strategy_block_tables;
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

  if ((strategy_key_cache.at(2) != strategy_value_cache.at(2)) || (strategy_query.at(1) != strategy_key_cache.at(2))) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The num_head must be shard at the same time, but got"
                  << " query's strategy: " << strategy_query << ", key_cache's strategy: " << strategy_key_cache
                  << ", value_cache's strategy: " << strategy_value_cache;
    return FAILED;
  }

  if ((strategy_key_cache.at(3) != strategy_value_cache.at(3)) || (strategy_query.at(2) != strategy_key_cache.at(3))) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The head_dim must be shard at the same time, but got"
                  << " query's strategy: " << strategy_query << ", key_cache's strategy: " << strategy_key_cache
                  << ", value_cache's strategy: " << strategy_value_cache;
    return FAILED;
  }

  return SUCCESS;
}

Status PagedAttentionInfo::InferDevMatrixShape() {
  auto input_strategies = strategy()->GetInputDim();
  auto query = input_strategies.at(0);         // (batch * seq_len (inc is 1), num_head, head_dim)
  auto cache = input_strategies.at(1);         // (block_size, num_blocks, num_head, head_dim)
  auto block_tables = input_strategies.at(3);  // (batch, max_num_block_per_batch)
  auto context_lens = input_strategies.at(4);  // (batch * seq_len (inc is 1))

  // query_batch   context_lens   block_batch   num_head   head_dim
  // 4             3              2             1          0
  dev_matrix_shape_ = {query.at(0), context_lens.at(0), block_tables.at(0), cache.at(2), cache.at(3)};

  return SUCCESS;
}

Status PagedAttentionInfo::InferTensorMap() {
  Shape query_tensor_map{4, 1, 0};
  Shape cache_tensor_map{-1, -1, 1, 0};
  Shape block_tensor_map{2, -1};
  Shape context_tensor_map{3};
  inputs_tensor_map_.emplace_back(query_tensor_map);
  inputs_tensor_map_.emplace_back(cache_tensor_map);
  inputs_tensor_map_.emplace_back(cache_tensor_map);
  inputs_tensor_map_.emplace_back(block_tensor_map);
  inputs_tensor_map_.emplace_back(context_tensor_map);

  Shape out_tensor_map{4, 1, 0};
  outputs_tensor_map_.emplace_back(out_tensor_map);

  return SUCCESS;
}
REGISTER(PagedAttentionInfo);
}  // namespace parallel
}  // namespace mindspore
