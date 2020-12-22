/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "ps/ps_cache/embedding_hash_map.h"

namespace mindspore {
namespace ps {
int EmbeddingHashMap::ParseData(const int id, int *swap_out_index, int *swap_out_ids, const size_t data_step,
                                const size_t graph_running_step, size_t *swap_out_size) {
  MS_EXCEPTION_IF_NULL(swap_out_index);
  MS_EXCEPTION_IF_NULL(swap_out_ids);
  MS_EXCEPTION_IF_NULL(swap_out_size);
  auto hash_index = Hash(id);
  auto need_swap = NeedSwap();
  size_t loop = 0;
  while (true) {
    if (loop++ == hash_capacity_) {
      return INVALID_INDEX_VALUE;
    }
    if (hash_map_elements_[hash_index].IsEmpty()) {
      hash_count_++;
      (void)hash_id_to_index_.emplace(id, hash_index);
      hash_map_elements_[hash_index].set_id(id);
      hash_map_elements_[hash_index].set_step(data_step);
      return hash_index;
    } else if (need_swap && hash_map_elements_[hash_index].IsExpired(graph_running_step)) {
      // Need swap out from the hash table.
      swap_out_index[*swap_out_size] = hash_index;
      swap_out_ids[*swap_out_size] = hash_map_elements_[hash_index].id_;
      (*swap_out_size)++;
      (void)hash_id_to_index_.erase(hash_map_elements_[hash_index].id_);
      (void)hash_id_to_index_.emplace(id, hash_index);
      hash_map_elements_[hash_index].set_id(id);
      hash_map_elements_[hash_index].set_step(data_step);
      return hash_index;
    }
    hash_index = (hash_index + 1) % hash_capacity_;
  }
}

void EmbeddingHashMap::DumpHashMap() {
  MS_LOG(INFO) << "Dump hash map info begin, hash_capacity: " << hash_capacity_ << " hash_count: " << hash_count_;
  MS_LOG(INFO) << "Dump hash_id_to_index: ";
  for (auto iter = hash_id_to_index_.begin(); iter != hash_id_to_index_.end(); ++iter) {
    MS_LOG(INFO) << "  id: " << iter->first << " index: " << iter->second;
  }
  MS_LOG(INFO) << "Dump hash_map_unit: ";
  for (size_t i = 0; i < hash_map_elements_.size(); i++) {
    if (!hash_map_elements_[i].IsEmpty()) {
      MS_LOG(INFO) << "  index: " << i << " id: " << hash_map_elements_[i].id_
                   << " step: " << hash_map_elements_[i].step_;
    }
  }
  MS_LOG(INFO) << "Dump hash map info end.";
}
}  // namespace ps
}  // namespace mindspore
