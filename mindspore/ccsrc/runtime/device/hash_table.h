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

namespace mindspore {
namespace device {
// The base class of device hash table.
template <typename Key, typename Value>
class HashTable {
 public:
  HashTable() = default;
  virtual ~HashTable() = default;

  // Find elements with specific keys, if the key does not exist, initialize the value for the key based on the
  // initialzer and insert the key-value pair into map.The initializer can be 'normal', 'zero' or 'one'.
  bool Find(const Key *key, size_t key_num, const std::string &initializer, Value *outputs) = 0;

  // Find elements with specific keys, if the key does not exist, initialize the value for the key by 'default_value'
  // and insert the key-value pair into map.
  bool Find(const Key *key, size_t key_num, const Value &default_value, Value *outputs) = 0;

  // Insert elements with specific keys. If key exists, update the value of the key.
  bool Insert(const Key *key, size_t key_num, const Value *value) = 0;

  // Erase elements with specific keys.
  bool Erase(const Key *key, size_t key_num) = 0;

  // Reserves space for at least the specified number of elements.
  bool Reserve(size_t count) = 0;

  // Get the max number of elements the container could hold.
  size_t capacity() const = 0;

  // Get the number of elements in the map.
  size_t size() const = 0;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_HASH_TABLE_H_
