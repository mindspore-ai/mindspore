/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/backend/distributed/embedding_cache/embedding_hash_map.h"

namespace mindspore {
namespace distributed {
EmbeddingHashMap::EmbeddingHashMap(size_t hash_count, size_t hash_capacity)
    : hash_count_(hash_count),
      hash_capacity_(hash_capacity),
      current_pos_(0),
      current_batch_start_pos_(0),
      graph_running_index_num_(0),
      graph_running_index_pos_(0),
      expired_element_full_(false) {
  hash_map_elements_.resize(hash_capacity);
  // In multi-device mode, embedding table are distributed on different devices by id interval,
  // and ids outside the range of local device will use the front and back positions of the table,
  // the positions are reserved for this.
  hash_map_elements_.front().set_step(SIZE_MAX);
  hash_map_elements_.back().set_step(SIZE_MAX);
  graph_running_index_ = std::make_unique<int[]>(hash_capacity);
}

size_t EmbeddingHashMap::hash_step(const int hash_index) const {
  return hash_map_elements_[IntToSize(hash_index)].step_;
}

void EmbeddingHashMap::set_hash_step(const int hash_index, const size_t step) {
  hash_map_elements_[IntToSize(hash_index)].set_step(step);
}

// Get the id -> index mapping.
const mindspore::HashMap<int, int> &EmbeddingHashMap::hash_id_to_index() const { return hash_id_to_index_; }

// Get capacity of hash map.
size_t EmbeddingHashMap::hash_capacity() const { return hash_capacity_; }

int EmbeddingHashMap::ParseData(const int id, int *const swap_out_index, int *const swap_out_ids,
                                const size_t data_step, const size_t graph_running_step, size_t *const swap_out_size,
                                bool *const need_wait_graph) {
  MS_EXCEPTION_IF_NULL(swap_out_index);
  MS_EXCEPTION_IF_NULL(swap_out_ids);
  MS_EXCEPTION_IF_NULL(swap_out_size);
  bool need_swap = false;
  auto hash_index = FindInsertionPos(data_step, graph_running_step, &need_swap, need_wait_graph);
  if (hash_index == INVALID_INDEX_VALUE) {
    return hash_index;
  }

  if (!need_swap) {
    hash_count_++;
    (void)hash_id_to_index_.emplace(id, hash_index);
    hash_map_elements_[hash_index].set_id(id);
    hash_map_elements_[hash_index].set_step(data_step);
    return hash_index;
  }

  swap_out_index[*swap_out_size] = hash_index;
  swap_out_ids[*swap_out_size] = hash_map_elements_[hash_index].id_;
  (*swap_out_size)++;
  (void)hash_id_to_index_.erase(hash_map_elements_[hash_index].id_);
  (void)hash_id_to_index_.emplace(id, hash_index);
  hash_map_elements_[hash_index].set_id(id);
  hash_map_elements_[hash_index].set_step(data_step);
  return hash_index;
}

int EmbeddingHashMap::FindInsertionPos(const size_t, const size_t graph_running_step, bool *const need_swap,
                                       bool *const need_wait_graph) {
  MS_EXCEPTION_IF_NULL(need_swap);
  MS_EXCEPTION_IF_NULL(need_wait_graph);
  int hash_index = INVALID_INDEX_VALUE;
  while (!expired_element_full_) {
    if (hash_map_elements_[current_pos_].IsEmpty()) {
      hash_index = current_pos_;
    } else if (hash_map_elements_[current_pos_].IsExpired(graph_running_step)) {
      hash_index = current_pos_;
      *need_swap = true;
    } else if (hash_map_elements_[current_pos_].StepEqual(graph_running_step)) {
      graph_running_index_[graph_running_index_num_++] = current_pos_;
    }
    current_pos_ = (current_pos_ + 1) % hash_capacity_;
    if (hash_index != INVALID_INDEX_VALUE) {
      return hash_index;
    }
    if (current_pos_ == current_batch_start_pos_) {
      expired_element_full_ = true;
      MS_LOG(INFO) << "Running step:" << graph_running_step << "(num:" << graph_running_index_num_
                   << ") will be used, index swap will wait until the graph completed.";
    }
  }

  if (graph_running_index_pos_ != graph_running_index_num_) {
    *need_swap = true;
    *need_wait_graph = true;
    return graph_running_index_[graph_running_index_pos_++];
  }
  return INVALID_INDEX_VALUE;
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

void EmbeddingHashMap::Reset() {
  current_batch_start_pos_ = current_pos_;
  graph_running_index_num_ = 0;
  graph_running_index_pos_ = 0;
  expired_element_full_ = false;
}
}  // namespace distributed
}  // namespace mindspore
