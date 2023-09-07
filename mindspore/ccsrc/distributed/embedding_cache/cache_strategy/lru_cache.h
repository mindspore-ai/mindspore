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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_CACHE_STRATEGY_LRU_CHCHE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_CACHE_STRATEGY_LRU_CHCHE_H_

#include <list>
#include <vector>
#include <utility>
#include <functional>

#include "distributed/embedding_cache/cache_strategy/cache.h"
#include "utils/hash_map.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace distributed {
// This class implements a common LRU (least recently used) caching strategy, with the idea that "if data has been
// accessed recently, it is more likely to be accessed in the future."
// The LRUCache implementation uses a linked list to hold elements and a hash table to quickly find the location of an
// element in linked list.
template <typename KeyType, typename ValueType, typename Hash = std::hash<KeyType>,
          typename KeyEqual = std::equal_to<KeyType>>
class LRUCache : public Cache<KeyType, ValueType> {
 public:
  // The elements in cache are stored as key-value pairs.
  using Element = typename Cache<KeyType, ValueType>::Element;
  // The Iter type is the iterator type of the linked list.
  using Iter = typename std::list<Element>::iterator;

  explicit LRUCache(size_t capacity) : Cache<KeyType, ValueType>(capacity) {}

  ~LRUCache() override {
    elements_.clear();
    element_keys_to_iters_.clear();
  }

  // Insert an element (key-value pair) into the lru cache.
  // The newly inserted element is considered hot data and will be placed at the head of the linked list, because this
  // element may have been replaced from a higher level cache.
  void Put(const KeyType &key, const ValueType &value) override {
    const auto &iter = element_keys_to_iters_.find(key);
    // The key exist in lru cache, move this element to the head of list.
    if (iter != element_keys_to_iters_.end()) {
      elements_.splice(elements_.begin(), elements_, iter->second);
      // Update value.
      elements_.begin()->second = value;
      return;
    }

    if (IsFull()) {
      MS_LOG(EXCEPTION) << "There is no space in lru cache.";
    }

    // The key does not exist in lru cache, insert this new element at the head of list.
    elements_.emplace_front(key, value);
    (void)element_keys_to_iters_.emplace(key, elements_.begin());
  }

  // Query the corresponding Value from the cache according to the Key. If the element exists, the corresponding Value
  // is assigned to parameter value and return true. If the element does not exist, return false.
  // The newly accessed element is moved to the head of the list, indicating that it was recently accessed.
  bool Get(const KeyType &key, ValueType *value) override {
    const auto &iter = element_keys_to_iters_.find(key);
    if (iter != element_keys_to_iters_.end()) {
      // For performance, no element was constructed or destroyed.
      elements_.splice(elements_.begin(), elements_, iter->second);
      MS_EXCEPTION_IF_NULL(value);
      *value = iter->second->second;
      return true;
    }
    return false;
  }

  // Get the most recently used element.
  const Element &Front() const override {
    if (elements_.empty()) {
      MS_LOG(EXCEPTION) << "There is no element in lru cache.";
    }
    return elements_.front();
  }

  // Get the least recently used element.
  const Element &Back() const override {
    if (elements_.empty()) {
      MS_LOG(EXCEPTION) << "There is no element in lru cache.";
    }
    return elements_.back();
  }

  // Query whether the element corresponding to a particular key exists in the cache.
  bool Exists(const KeyType &key) const override {
    return element_keys_to_iters_.find(key) != element_keys_to_iters_.end();
  }

  // When the size of the cache is close to capacity, you can use this interface to evict some non-hot data to reserve
  // space for new elements to be inserted into the cache. If the current cache has enough free space, this function
  // does nothing.
  // The input parameter 'reserve_size' indicates the number of element slots that are expected to be reserved. If the
  // reserve_size is less than or equal to the number of slots remaining in the cache, the function does nothing.
  // The output parameter 'evicted_elements' is used to hold the evicted element.
  void TryEvict(size_t reserve_size, std::vector<Element> *evicted_elements) override {
    MS_EXCEPTION_IF_NULL(evicted_elements);
    const auto &capacity = Cache<KeyType, ValueType>::capacity();
    if (reserve_size > capacity) {
      MS_LOG(EXCEPTION) << "The evict number must be less or equal to lru cache capacity: " << capacity
                        << ", but got: " << reserve_size;
    }

    while (size() > capacity - reserve_size) {
      const auto &back_element = elements_.back();
      evicted_elements->emplace_back(back_element.first, back_element.second);
      (void)element_keys_to_iters_.erase(back_element.first);
      elements_.pop_back();
    }
  }

  // Check whether the number of elements in cache reaches capacity.
  bool IsFull() const override { return size() >= Cache<KeyType, ValueType>::capacity(); }

  // Get the current number of elements in the cache.
  size_t size() const override { return element_keys_to_iters_.size(); }

  // Dump all elements in the lru cache.
  const std::list<Element> &Export() const override { return elements_; }

 private:
  // The linked list used to hold elements.
  std::list<Element> elements_;

  // The hash table used to quickly find the location of an element in linked list.
  mindspore::HashMap<KeyType, Iter, Hash, KeyEqual> element_keys_to_iters_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_CACHE_STRATEGY_LRU_CHCHE_H_
