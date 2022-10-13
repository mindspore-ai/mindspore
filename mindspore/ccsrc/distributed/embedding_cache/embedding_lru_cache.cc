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

#include "distributed/embedding_cache/embedding_lru_cache.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace distributed {
template <typename K, typename V>
bool EmbeddingLRUCache<K, V>::Initialize() {
  keys_lru_cache_ = std::make_unique<LRUCache<K, size_t>>(capacity_);
  return true;
}

template <typename K, typename V>
bool EmbeddingLRUCache<K, V>::Get(const void *input, size_t key_num, const void *keys, void *values, size_t *miss_num,
                                  void *miss_keys, size_t *miss_indices) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(keys);
  MS_EXCEPTION_IF_NULL(values);
  MS_EXCEPTION_IF_NULL(miss_keys);
  MS_EXCEPTION_IF_NULL(miss_indices);

  size_t miss_count = 0;
  auto *miss_keys_list = static_cast<K *>(miss_keys);
  for (size_t i = 0; i < key_num; i++) {
    const K key = static_cast<const K *>(keys)[i];
    if (!keys_lru_cache_->Exists(key)) {
      miss_keys_list[miss_count] = key;
      miss_indices[miss_count] = i;
      miss_count += 1;
      continue;
    }

    size_t index = keys_lru_cache_->Get(key);
    auto ret = memcpy_s(AddressOffset(values, i * value_size_), value_size_,
                        AddressOffset(const_cast<void *>(input), index * value_size_), value_size_);
    if (ret != 0) {
      return false;
    }
  }

  *miss_num = miss_count;
  return true;
}

template <typename K, typename V>
bool EmbeddingLRUCache<K, V>::Put(void *input, size_t key_num, const void *keys, const void *values,
                                  size_t *evicted_num, void *evicted_keys, void *evicted_values) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(keys);
  MS_EXCEPTION_IF_NULL(values);
  MS_EXCEPTION_IF_NULL(evicted_keys);
  MS_EXCEPTION_IF_NULL(evicted_values);

  auto *evicted_keys_list = static_cast<K *>(evicted_keys);
  size_t evicted_count = 0;
  size_t hit_count = 0;
  for (size_t i = 0; i < key_num; i++) {
    hit_count++;
    const K key = static_cast<const K *>(keys)[i];

    if (keys_lru_cache_->Exists(key)) {
      size_t idx = static_cast<size_t>(keys_lru_cache_->Get(key));
      auto ret = memcpy_s(AddressOffset(input, idx * value_size_), value_size_,
                          AddressOffset(const_cast<void *>(values), i * value_size_), value_size_);
      if (ret != 0) {
        MS_LOG(ERROR) << "Failed to update exist key: " << key;
        return false;
      }
      continue;
    }

    if (!keys_lru_cache_->IsFull()) {
      keys_lru_cache_->Put(key, curr_index_);

      auto ret = memcpy_s(AddressOffset(input, curr_index_ * value_size_), value_size_,
                          AddressOffset(const_cast<void *>(values), i * value_size_), value_size_);
      if (ret != 0) {
        return false;
      }
      curr_index_++;
      continue;
    }

    const std::pair<K, V> last = keys_lru_cache_->Back();
    evicted_keys_list[evicted_count] = last.first;
    // Save evicted values
    auto ret = memcpy_s(AddressOffset(evicted_values, evicted_count * value_size_), value_size_,
                        AddressOffset(input, last.second * value_size_), value_size_);
    if (ret != 0) {
      return false;
    }

    // Update input use new values
    ret = memcpy_s(AddressOffset(input, last.second * value_size_), value_size_,
                   AddressOffset(const_cast<void *>(values), i * value_size_), value_size_);
    if (ret != 0) {
      return false;
    }

    // Update key&index cache
    keys_lru_cache_->Put(key, last.second);
    evicted_count++;
  }
  *evicted_num = evicted_count;
  MS_LOG(INFO) << "Embedding lru cache size after put: " << keys_lru_cache_->Size() << ", hit count: " << hit_count;
  return true;
}

template <typename K, typename V>
bool EmbeddingLRUCache<K, V>::IsFull() {
  return curr_index_ >= capacity_;
}

template class EmbeddingLRUCache<int32_t, float>;
template class EmbeddingLRUCache<int32_t, double>;
template class EmbeddingLRUCache<int32_t, int64_t>;
template class EmbeddingLRUCache<int32_t, size_t>;
}  // namespace distributed
}  // namespace mindspore
