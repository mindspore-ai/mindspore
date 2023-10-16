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
#include "distributed/embedding_cache/cache_strategy/lru_cache.h"

namespace mindspore {
namespace distributed {
EmbeddingHashMap::EmbeddingHashMap(size_t hash_capacity) : hash_capacity_(hash_capacity), current_pos_(0) {
  hash_map_elements_.resize(hash_capacity);
  // In multi-device mode, embedding table are distributed on different devices by id interval,
  // and ids outside the range of local device will use the front and back positions of the table(for Ascend platform,
  // out-of-range ids will be Rectified by ReLU and Minimal operaotrs), so these two positions should be reserved for
  // out-of-range ids, otherwise, the in range ids' embedding will be dirtied when optimizer updates the embeddings of
  // out-of-range ids.
  hash_map_elements_.front().set_step(SIZE_MAX);
  hash_map_elements_.back().set_step(SIZE_MAX);
  valid_capacity_ = hash_capacity > kMinimumCapacity ? (hash_capacity - kMinimumCapacity) : 0;
  if (valid_capacity_ == 0) {
    MS_LOG(ERROR) << "The invalid capacity is zero, please enlarge the capacity.";
  }
  ids_to_indices_ = std::make_unique<LRUCache<int, int>>(valid_capacity_);
}

size_t EmbeddingHashMap::hash_step(const int hash_index) const { return hash_map_elements_[hash_index].step_; }

void EmbeddingHashMap::set_hash_step(const int hash_index, const size_t step) {
  hash_map_elements_[hash_index].set_step(step);
}

// Get capacity of hash map.
size_t EmbeddingHashMap::hash_capacity() const { return hash_capacity_; }

bool EmbeddingHashMap::GetIndex(const int id, int *index) const { return ids_to_indices_->Get(id, index); }

const std::list<EmbeddingHashMap::Element> &EmbeddingHashMap::Export() const { return ids_to_indices_->Export(); }

int EmbeddingHashMap::ParseData(const int id, int *const swap_out_index, int *const swap_out_ids,
                                const size_t data_step, const size_t graph_running_step, size_t *const swap_out_size,
                                bool *const need_wait_graph) {
  MS_EXCEPTION_IF_NULL(swap_out_index);
  MS_EXCEPTION_IF_NULL(swap_out_ids);
  MS_EXCEPTION_IF_NULL(swap_out_size);
  bool need_swap = false;
  int swap_out_id;
  auto hash_index = FindInsertionPos(data_step, graph_running_step, &need_swap, need_wait_graph, &swap_out_id);
  if (hash_index == kInvalidIndexValue) {
    return hash_index;
  }

  if (!need_swap) {
    ids_to_indices_->Put(id, hash_index);
    hash_map_elements_[hash_index].set_step(data_step);
    return hash_index;
  }

  swap_out_index[*swap_out_size] = hash_index;
  swap_out_ids[*swap_out_size] = swap_out_id;
  ++(*swap_out_size);
  ids_to_indices_->Put(id, hash_index);
  hash_map_elements_[hash_index].set_step(data_step);
  return hash_index;
}

int EmbeddingHashMap::GetOrInsertDataUnsafe(const int key) {
  int index = kInvalidIndexValue;
  if (GetIndex(key, &index)) {
    return index;
  }

  return InsertDataUnsafe(key);
}

int EmbeddingHashMap::InsertDataUnsafe(const int key) {
  auto hash_index = FindPosUnsafe();
  if (hash_index == kInvalidIndexValue) {
    MS_LOG(WARNING) << "Insert data unsafe failed as map is full.";
    return hash_index;
  }

  ids_to_indices_->Put(key, hash_index);
  hash_map_elements_[hash_index].set_step((size_t)1);
  return hash_index;
}

int EmbeddingHashMap::FindPosUnsafe() {
  if (current_pos_ >= valid_capacity_) {
    return kInvalidIndexValue;
  }
  return static_cast<int>(++current_pos_);
}

int EmbeddingHashMap::FindInsertionPos(const size_t, const size_t graph_running_step, bool *const need_swap,
                                       bool *const need_wait_graph, int *swap_out_id) {
  if (current_pos_ < valid_capacity_) {
    // Start from index 1.
    return ++current_pos_;
  }
  if (valid_capacity_ == 0) {
    return kInvalidIndexValue;
  }

  *need_swap = true;
  int id = ids_to_indices_->Back().first;
  int index = ids_to_indices_->Back().second;
  if (hash_map_elements_[index].IsExpired(graph_running_step)) {
    std::vector<Element> evicted_elements;
    ids_to_indices_->TryEvict(1, &evicted_elements);
    if (evicted_elements.size() != 1) {
      MS_LOG(EXCEPTION) << "Failed to evict tail element in cache, evict element number: " << evicted_elements.size()
                        << ", cache size: " << ids_to_indices_->size()
                        << ", cache capacity: " << ids_to_indices_->capacity();
    }

    *swap_out_id = evicted_elements.front().first;
    if (*swap_out_id != id) {
      MS_LOG(EXCEPTION) << "The evicted id should be: " << id << ", but got: " << *swap_out_id;
    }
    return index;
  }
  return kInvalidIndexValue;
}

void EmbeddingHashMap::DumpHashMap() {
  MS_LOG(INFO) << "Dump hash map info begin, hash_capacity: " << hash_capacity_;
  MS_LOG(INFO) << "Dump hash_id_to_index: ";
  MS_LOG(INFO) << "Dump hash_map_unit: ";
  for (size_t i = 0; i < hash_map_elements_.size(); i++) {
    if (!hash_map_elements_[i].IsEmpty()) {
      MS_LOG(INFO) << "  index: " << i << " step: " << hash_map_elements_[i].step_;
    }
  }
  MS_LOG(INFO) << "Dump hash map info end.";
}
}  // namespace distributed
}  // namespace mindspore
