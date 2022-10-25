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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_HASH_TABLE_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_HASH_TABLE_H_

#include <string>
#include <utility>

namespace mindspore {
using DataLenPair = std::pair<void *, size_t>;
namespace device {
// The base class of device hash table.
template <typename Key, typename Value>
class HashTable {
 public:
  enum class Status {
    kUnchanged = 0,
    kModified = 1,
    kErased = 2,
  };

  HashTable() = default;
  virtual ~HashTable() = default;

  // Find elements with specific keys, if a key does not exist, initialize the value for the key based on the
  // initialzer and insert the key-value pair into map. The initializer can be 'normal', 'zero' or 'one', and also
  // could be a specific Value type scalar.
  virtual bool Find(const Key *keys, size_t key_num, Value *outputs, void *stream) = 0;

  // Insert elements with specific keys. If key exists, update the value of the key.
  virtual bool Insert(const Key *keys, size_t key_num, const Value *value, void *stream) = 0;

  // Erase elements with specific keys.
  virtual bool Erase(const Key *keys, size_t key_num, void *stream) = 0;

  // Reserves space for at least the specified number of elements.
  virtual bool Reserve(size_t count) = 0;

  // Export all keys and values in hash map, the order of each element of keys and values is consistent.
  // Note: Even if the elements of the hash map are unchanged, the order of the key-value pair returned by the function
  // may be different each time it is called, because there may be multi-threaded concurrent exports inside the
  // function.
  virtual bool GetKeysAndValues(Key *keys, Value *values, void *stream) = 0;

  // Import keys, values into the hash map.
  virtual bool Import(const DataLenPair &input_data) = 0;

  // Export all keys, values and status.
  virtual bool Export(const DataLenPair &keys, const DataLenPair &values, const DataLenPair &status) = 0;

  // Get the max number of elements the container could hold.
  virtual size_t capacity() const = 0;

  // Get the number of elements in the map.
  virtual size_t size() const = 0;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_HASH_TABLE_H_
