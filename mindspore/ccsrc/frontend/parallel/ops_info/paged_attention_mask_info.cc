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

#include "frontend/parallel/ops_info/paged_attention_mask_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// PagedAttentionMask has 6 inputs
// query:         (batch, seq_len, hidden_size)
// key_cache:     (block_size, num_blocks, num_head, head_dim)
// value_cache:   (block_size, num_blocks, num_head, head_dim)
// block_tables:  (batch, max_num_block_per_batch)
// context_lens:  (batch * seq_len)
// alibi_mask:    (batch, num_head, seq_len, seq_len)
// ------------------------------
// output:        (batch, seq_len, hidden_size)

// split strategy
// num_blocks is not able to split
// block_size is not able to split
// batch is able to split
// seq_len is not able to split
// hidden_size is able to split
// num_head is able to split
// head_dim is not able to split
// max_num_block_per_batch is not able to split

Status PagedAttentionMaskInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  auto input_strategies = strategy->GetInputDim();
  auto strategy_query = input_strategies.at(0);         // (dp, 1, mp)
  auto strategy_key_cache = input_strategies.at(1);     // (1, 1, mp, 1)
  auto strategy_value_cache = input_strategies.at(2);   // (1, 1, mp, 1)
  auto strategy_block_tables = input_strategies.at(3);  // (dp, 1)
  auto strategy_alibi_mask = input_strategies.at(5);    // (dp, mp, 1, 1)

  if (strategy_query.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The seq_len can't be shard, but got"
                  << " query's block_size strategy: " << strategy_query.at(1);
    return FAILED;
  }

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

  if ((strategy_key_cache.at(2) != strategy_value_cache.at(2)) || (strategy_query.at(2) != strategy_key_cache.at(2)) ||
      (strategy_key_cache.at(2) != strategy_alibi_mask.at(1))) {
    MS_LOG(ERROR)
      << name_
      << ": Invalid strategy: The cache num_head and update hidden_size must be shard at the same time, but got"
      << " query's strategy: " << strategy_query << ", key_cache's strategy: " << strategy_key_cache
      << ", value_cache's strategy: " << strategy_value_cache << ", alibi_mask's strategy: " << strategy_alibi_mask;
    return FAILED;
  }

  if ((strategy_key_cache.at(3) != strategy_value_cache.at(3)) || (strategy_key_cache.at(3) != 1)) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The head_dim can't be shard, but got"
                  << " query's strategy: " << strategy_query << ", key_cache's strategy: " << strategy_key_cache
                  << ", value_cache's strategy: " << strategy_value_cache;
    return FAILED;
  }

  return SUCCESS;
}

Status PagedAttentionMaskInfo::InferDevMatrixShape() {
  auto input_strategies = strategy()->GetInputDim();
  auto query = input_strategies.at(0);  // (batch, seq_len, hidden_size)

  // dp   mp
  //  1    0
  dev_matrix_shape_ = {query.at(0), query.at(2)};

  return SUCCESS;
}

Status PagedAttentionMaskInfo::InferTensorMap() {
  Shape query_tensor_map{1, -1, 0};
  Shape cache_tensor_map{-1, -1, 0, -1};
  Shape block_tensor_map{1, -1};
  Shape context_tensor_map{1};
  Shape alibi_mask_map{1, 0, -1, -1};
  inputs_tensor_map_.emplace_back(query_tensor_map);
  inputs_tensor_map_.emplace_back(cache_tensor_map);
  inputs_tensor_map_.emplace_back(cache_tensor_map);
  inputs_tensor_map_.emplace_back(block_tensor_map);
  inputs_tensor_map_.emplace_back(context_tensor_map);
  inputs_tensor_map_.emplace_back(alibi_mask_map);

  Shape out_tensor_map{1, -1, 0};
  outputs_tensor_map_.emplace_back(out_tensor_map);

  return SUCCESS;
}
REGISTER(PagedAttentionMaskInfo);
}  // namespace parallel
}  // namespace mindspore
