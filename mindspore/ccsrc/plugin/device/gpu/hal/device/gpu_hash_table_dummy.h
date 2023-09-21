/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_DUMMY_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_DUMMY_H_

#include <string>

#include "runtime/device/hash_table.h"
#include "plugin/device/gpu/hal/device/gpu_allocator.h"

namespace mindspore {
namespace device {
namespace gpu {
// A dummy gpu hash table for compiling asan version.
template <typename Key, typename Value, typename Allocator = GPUAllocator<char>>
class GPUHashTable : public HashTable<Key, Value> {
 public:
  GPUHashTable(int32_t value_dim, const std::string &initializer, uint64_t permit_threshold = 1,
               uint64_t evict_threshold = INT64_MAX, const Allocator &alloc = Allocator()) {}
  GPUHashTable(int32_t value_dim, const Value &default_value, uint64_t permit_threshold = 1,
               uint64_t evict_threshold = INT64_MAX, const Allocator &alloc = Allocator()) {}
  ~GPUHashTable() {}

  // Find elements with specific keys, if a key does not exist, initialize the value for the key based on the
  // initializer and insert the key-value pair into map. The initializer can be 'normal', 'zero' or 'one', and also
  // could be a specific 'Value' type scalar.
  bool Find(const Key *keys, size_t key_num, bool insert_default_value, Value *outputs, void *stream) override;

  // Insert elements with specific keys. If key exists, update the value of the key.
  // If permission is enable, all keys for `Insert` should be contained in gpu hash table already.
  bool Insert(const Key *keys, size_t key_num, const Value *value, void *stream) override;

  // Erase elements with specific keys.
  bool Erase(const Key *keys, size_t key_num, void *stream) override;

  // Reserves space for at least the specified number of elements.
  bool Reserve(size_t new_capacity, void *stream) override;

  // Export all keys and values in hash map, the order of each element of keys and values is consistent.
  // Note: Even if the elements of the hash map are unchanged, the order of the key-value pair returned by the function
  // may be different each time it is called, because there may be multi-threaded concurrent exports inside the
  // function.
  bool GetKeysAndValues(Key *keys, Value *values, void *stream) override;

  // Import keys, values into the device hash map.
  bool Import(const DataLenPair &input_data) override;

  // Export all keys, values and status to host side.
  // Argument `incremental` mean the flag that determine whether export hash table in incremental or full manner, true
  // for incremental export, false for full export.
  HashTableExportData Export(bool incremental) override;

  // Export a slice from the hash table, the size is specified by the parameter 'slice_size_in_mega_bytes' in MB.
  HashTableExportData ExportSlice(bool incremental, bool *last_slice, size_t slice_size_in_mega_bytes) override;

  // Get the number of elements that can be held in currently allocated storage.
  size_t capacity() const override;

  // Get the number of elements.
  size_t size() const override;

  // Gets whether the elements of the hash table have changed since the last export, true means that there has been a
  // change.
  bool is_dirty() const override;

  // Clear all elements of hash table.
  bool Clear() override;
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_DUMMY_H_
