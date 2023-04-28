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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_CACHE_STRATEGY_CHCHE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_CACHE_STRATEGY_CHCHE_H_

#include <utility>
#include <vector>
#include <list>

namespace mindspore {
namespace distributed {
// An abstract class of general cache strategy that provides basic APIs for cache management, such as element access and
// modification APIs: Get, Put, and query whether the cache hits API: Exists, etc.
template <typename KeyType, typename ValueType>
class Cache {
 public:
  // The elements in cache are stored as key-value pairs.
  using Element = std::pair<KeyType, ValueType>;

  explicit Cache(size_t capacity) : capacity_(capacity) {}
  virtual ~Cache() = default;

  // Insert an element (key-value pair) into the cache.
  // Inserting an element into the cache generally affects the location or order of the elements in the cache, depending
  // on different cache strategies.
  virtual void Put(const KeyType &key, const ValueType &value) = 0;

  // Query the corresponding Value from the cache according to the Key. If the element exists, the corresponding Value
  // is assigned to parameter value and return true. If the element does not exist, return false.
  // Access an element of the cache generally affects the location or order of the elements in the cache, depending
  // on different cache strategies.
  virtual bool Get(const KeyType &key, ValueType *value) = 0;

  // Get the most recently used element.
  virtual const Element &Front() const = 0;

  // Get the least recently used element.
  virtual const Element &Back() const = 0;

  // Query whether the element corresponding to a particular key exists in the cache.
  virtual bool Exists(const KeyType &key) const = 0;

  // When the size of the cache is close to capacity, you can use this interface to evict some non-hot data to reserve
  // space for new elements to be inserted into the cache. If the current cache has enough free space, this function
  // does nothing.
  // The input parameter 'reserve_size' indicates the number of element slots that are expected to be reserved. If the
  // reserve_size is less than or equal to the number of slots remaining in the cache, the function does nothing.
  // The output parameter 'evicted_elements' is used to hold the evicted element.
  virtual void TryEvict(size_t reserve_size, std::vector<Element> *evicted_elements) = 0;

  // Check whether the number of elements in cache reaches capacity.
  virtual bool IsFull() const = 0;

  // Get the current number of elements in the cache.
  virtual size_t size() const = 0;

  // Dump all elements in the cache.
  virtual const std::list<Element> &Export() const = 0;

  // Get the maximum number of elements that the cache can hold.
  size_t capacity() const { return capacity_; }

 protected:
  // The maximum number of elements that the cache can hold.
  size_t capacity_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EMBEDDING_CACHE_CACHE_STRATEGY_CHCHE_H_
