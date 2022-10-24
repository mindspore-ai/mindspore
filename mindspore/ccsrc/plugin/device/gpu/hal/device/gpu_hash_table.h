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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_H_

#include <string>
#include <vector>
#include "runtime/device/hash_table.h"

namespace mindspore {
namespace device {
namespace gpu {
// A hash table base on GPU.
template <typename Key, typename Value, typename Allocator>
class GPUHashTable : public HashTable<Key, Value> {
 public:
  explicit GPUHashTable(int32_t value_dim, const Allocator &alloc = Allocator())
      : value_dim_(value_dim), alloc_(alloc) {}
  ~GPUHashTable() = default;

  using key_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<key>;
  using value_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<Value>;
  using index_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<int>;
  using char_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<char>;

  // Find elements with specific keys, if a key does not exist, initialize the value for the key based on the
  // initialzer and insert the key-value pair into map. The initializer can be 'normal', 'zero' or 'one', and also
  // could be a specific Value type scalar.
  bool Find(const Key *keys, size_t key_num, Value *outputs, void *stream) override;

  // Insert elements with specific keys. If key exists, update the value of the key.
  bool Insert(const Key *key, size_t key_num, const Value *value) override { return true; }

  // Erase elements with specific keys.
  bool Erase(const Key *key, size_t key_num) override { return true; }

  // Reserves space for at least the specified number of elements.
  bool Reserve(size_t count) override { return true; }

  // Export all keys and values in hash map, the order of each element of keys and values is consistent.
  // Note: Even if the elements of the hash map are unchanged, the order of the key-value pair returned by the function
  // may be different each time it is called, because there may be multi-threaded concurrent exports inside the
  // function.
  bool GetKeysAndValues(Key *keys, Value *values, void *stream) override;

  // Import keys, values into the device hash map.
  bool Import(const DataLenPair &input_data) override;

  // Export all keys, values and status to host side.
  bool Export(const DataLenPair &keys, const DataLenPair &values, const DataLenPair &status) override;

  // Get the number of elements that can be held in currently allocated storage.
  size_t capacity() const override { return capacity_; }

  // Get the number of elements.
  size_t size() const override { return size_; };

 private:
  // Find elements with specific keys, if the key does not exist, initialize the value for the key based on the
  // initialzer and insert the key-value pair into map.The initializer can be 'normal', 'zero' or 'one'.
  bool Find(const Key *key, size_t key_num, const std::string &initializer, Value *outputs) { return true; }

  // Find elements with specific keys, if the key does not exist, initialize the value for the key by 'default_value'
  // and insert the key-value pair into map.
  bool Find(const Key *key, size_t key_num, const Value &default_value, Value *outputs) { return true; }

  // Get all indices in blocks according to the key.
  bool GetIndicesByKeys(const Key *key, size_t key_num, bool insert_miss_key, int32_t *indices) const;

  // Get value in blocks according to the indices.
  bool GetValues(const int32_t *indices, Value *value) const;

  // Record all block memory that contain all values.
  std::vector<Value *> blocks_;
  // Record all first address of blocks, the buffer is on device memory.
  Value **blocks_ptr_{nullptr};

  // Record whether every slot is occupied or not for all block.
  std::vector<bool *> idle_flags_;
  // Record whether every slot is occupied or not for all block, the buffer is on device memory.
  bool **idle_flags_ptr_{nullptr};

  // The value dimension for each key.
  size_t value_dim_;

  // The allocator used to alloacte gpu memory.
  Allocator alloc_;

  // Record the number of elements in the map.
  size_t size_{0};

  // Record the number of elements that can be held in currently allocated storage
  size_t capacity_{0};
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_H_
