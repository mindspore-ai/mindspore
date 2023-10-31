/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/prompt_k_v_cache_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// PromptKVCache seven inputs
// cache:           (batch_size, num_head, max_seq_len, hidden_size)
// update:          (batch_size, num_head, update_seq_len, hidden_size)
// valid_seq_len:   (batch)
// batch_index:     (1)
// seq_len_axis:    (1)
// new_max_seq_len: (1)
// cur_max_seq_len: (1)
// ------------------------------
// output:          (batch_size, num_head, max_seq_len, hidden_size)

// split strategy
// batch_size is able to split.
// max_seq_len, update_seq_len is not able to split.
// num_head is able to split.
// hidden_size is able to split.

Status PromptKVCacheInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  auto input_strategys = strategy->GetInputDim();

  auto strategy_cache = input_strategys.at(0);            // (4, 4, 1, 4)
  auto strategy_update = input_strategys.at(1);           // (4, 4, 1, 4)
  auto strategy_valid_seq_len = input_strategys.at(2);    // (4)
  auto strategy_batch_index = input_strategys.at(3);      // (1)
  auto strategy_seq_len_axis = input_strategys.at(4);     // (1)
  auto strategy_new_max_seq_len = input_strategys.at(5);  // (1)
  auto strategy_cur_max_seq_len = input_strategys.at(6);  // (1)

  if (strategy_new_max_seq_len.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The new_max_seq_len can't be shard, but got"
                  << " new_max_seq_len's strategy: " << strategy_new_max_seq_len;
    return FAILED;
  }

  if (strategy_cur_max_seq_len.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The cur_max_seq_len can't be shard, but got"
                  << " cur_max_seq_len's strategy: " << strategy_cur_max_seq_len;
    return FAILED;
  }

  if (strategy_batch_index.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The batch_index can't be shard, but got"
                  << " batch_index's strategy: " << strategy_batch_index;
    return FAILED;
  }

  if (strategy_seq_len_axis.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The seq_len_axis can't be shard, but got"
                  << " seq_len_axis's strategy: " << strategy_seq_len_axis;
    return FAILED;
  }

  if (strategy_cache.at(2) != 1 || strategy_update.at(2) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The seq_len can't be shard, but got"
                  << " cache's seq_len strategy: " << strategy_cache.at(2)
                  << "; update's seq_len strategy: " << strategy_update.at(2);
    return FAILED;
  }

  // batch_size must be the same strategy.
  if (strategy_cache.at(0) != strategy_update.at(0) || strategy_cache.at(0) != strategy_valid_seq_len.at(0)) {
    MS_LOG(ERROR) << name_ << " Invalid strategy: The batch_size must be shard at the same time, but got"
                  << " strategy_cache's strategy: " << strategy_cache
                  << ", strategy_update's strategy: " << strategy_update
                  << ", strategy_valid_seq_len's strategy: " << strategy_valid_seq_len;
    return FAILED;
  }

  // num_head must be the same strategy.
  if (strategy_cache.at(1) != strategy_cache.at(1)) {
    MS_LOG(ERROR) << name_ << " Invalid strategy: The num_head must be shard at the same time, but got"
                  << " strategy_cache's strategy: " << strategy_cache
                  << ", strategy_update's strategy: " << strategy_update;
    return FAILED;
  }

  // hidden_size must be the same strategy.
  if (strategy_cache.at(3) != strategy_update.at(3)) {
    MS_LOG(ERROR) << name_ << " Invalid strategy: The hidden_size must be shard at the same time, but got"
                  << " strategy_cache's strategy: " << strategy_cache
                  << ", strategy_update's strategy: " << strategy_update;
    return FAILED;
  }
  return SUCCESS;
}

Status PromptKVCacheInfo::InferDevMatrixShape() {
  auto input_strategys = strategy()->GetInputDim();
  auto cache = input_strategys.at(0);   // batch_size   num_head    max_seq_len     hidden_size
  auto update = input_strategys.at(1);  // batch_size   num_head   update_seq_len   hidden_size

  // update_seq_len   batch_size   num_head   max_seq_len   hidden_size
  //      4               3           2           1             0
  dev_matrix_shape_ = {update.at(2), cache.at(0), cache.at(1), cache.at(2), cache.at(3)};
  return SUCCESS;
}

Status PromptKVCacheInfo::InferTensorMap() {
  Shape cache_tensor_map{3, 2, 1, 0};
  Shape update_tensor_map{3, 2, 4, 0};
  Shape valid_seq_len_tensor_map{3};
  Shape batch_index_tensor_map{-1};
  Shape seq_lqn_axis_tensor_map{-1};
  Shape new_max_seq_len_tensor_map{-1};
  Shape cur_max_seq_len_tensor_map{-1};
  inputs_tensor_map_.emplace_back(cache_tensor_map);
  inputs_tensor_map_.emplace_back(update_tensor_map);
  inputs_tensor_map_.emplace_back(valid_seq_len_tensor_map);
  inputs_tensor_map_.emplace_back(batch_index_tensor_map);
  inputs_tensor_map_.emplace_back(seq_lqn_axis_tensor_map);
  inputs_tensor_map_.emplace_back(new_max_seq_len_tensor_map);
  inputs_tensor_map_.emplace_back(cur_max_seq_len_tensor_map);

  Shape out_tensor_map{3, 2, 1, 0};
  outputs_tensor_map_.emplace_back(out_tensor_map);
  return SUCCESS;
}
REGISTER(PromptKVCacheInfo);
}  // namespace parallel
}  // namespace mindspore
