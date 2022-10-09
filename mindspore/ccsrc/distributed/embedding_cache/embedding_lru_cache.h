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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_LRU_CHCHE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_LRU_CHCHE_H_

#include <map>
#include <string>
#include <memory>
#include <utility>
#include <list>

#include "utils/hash_map.h"
#include "utils/log_adapter.h"
#include "distributed/embedding_cache/embedding_cache.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
template <typename K, typename V>
class LRUCache {
 public:
  typedef typename std::pair<K, V> Item;
  typedef typename std::list<Item>::iterator ListIter;

  explicit LRUCache(size_t capacity) : capacity_(capacity) {}

  void Put(const K &key, const V &value) {
    auto it = cache_items_map_.find(key);
    cache_items_list_.emplace_front(key, value);
    if (it != cache_items_map_.end()) {
      cache_items_list_.erase(it->second);
    }
    cache_items_map_[key] = cache_items_list_.begin();

    if (IsFull()) {
      auto last = cache_items_list_.end();
      last--;
      cache_items_map_.erase(last->first);
      cache_items_list_.pop_back();
    }
  }

  const V &Get(const K &key) {
    if (!Exists(key)) {
      MS_LOG(EXCEPTION) << "Key not exists: " << key;
    }

    auto it = cache_items_map_.find(key);
    cache_items_list_.splice(cache_items_list_.begin(), cache_items_list_, it->second);
    return it->second->second;
  }

  bool Exists(const K &key) const { return cache_items_map_.find(key) != cache_items_map_.end(); }

  size_t Size() const { return cache_items_map_.size(); }

  bool IsFull() const { return Size() > capacity_; }

 private:
  std::list<Item> cache_items_list_;
  mindspore::HashMap<K, ListIter> cache_items_map_;
  size_t capacity_;
};

// Use LRU algorithm to cache index of saved value of Parameter of embedding.
template <typename K, typename V>
class BACKEND_EXPORT EmbeddingLRUCache : public EmbeddingCache {
 public:
  explicit EmbeddingLRUCache(size_t capacity) : capacity_(capacity) {}
  ~EmbeddingLRUCache() = default;

  bool Initialize();
  bool Finalize() { return true; }

  bool Get(void *input, size_t key_num, const void *keys, void *values) override { return true; }
  bool Put(void *input, size_t key_num, const void *keys, const void *values, size_t evicted_num, void *evicted_keys,
           void *evicted_values) override {
    return true;
  }
  bool IsFull() override { return true; }

 private:
  size_t capacity_;

  // Cache the index of saved value in the Parameter of embedding.
  std::unique_ptr<LRUCache<K, size_t>> keys_lru_cache_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_EMBEDDING_LRU_CHCHE_H_
