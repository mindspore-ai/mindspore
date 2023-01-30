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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_DEVICE_CPU_HASH_TABLE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_DEVICE_CPU_HASH_TABLE_H_

#include <shared_mutex>
#include <unordered_map>

#include "runtime/hardware/device_context.h"
#include "runtime/device/hash_table.h"

namespace mindspore {
namespace device {
namespace cpu {
using mindspore::HashTableExportData;

// A hash table base on the host side cpu.
template <typename Key, typename Value>
class CPUHashTable : public HashTable<Key, Value> {
 public:
  explicit CPUHashTable(size_t value_dim);
  ~CPUHashTable() override;

  // Initialize the resources (e.g. device context) needed by this hash table.
  bool Initialize();

  // Release all the resources (e.g. the host side memory) used by this hash table.
  bool Finalize();

  // The last parameter `stream` is meaningless for the cpu hash table version.
  bool Find(const Key *keys, size_t key_num, bool insert_default_value, Value *outputs, void *) override;

  bool Insert(const Key *keys, size_t key_num, const Value *value, void *) override;

  bool Erase(const Key *keys, size_t key_num, void *) override;

  bool Reserve(size_t new_capacity, void *) override;

  bool GetKeysAndValues(Key *keys, Value *values, void *) override;

  bool Import(const DataLenPair &input_data) override;

  HashTableExportData Export(bool incremental) override;

  size_t capacity() const override;

  size_t size() const override;

  bool is_dirty() const override;

  bool Clear() override;

 private:
  // The key-value style elements stored in this hash table.
  std::unordered_map<Key, Value *> values_;

  // This mutex is to guarantee the thread-safe of the `values_` above.
  // The perforcemence of this thread-safe mechanism is poor, so a more efficient way could be used later.
  mutable std::shared_mutex mutex_;

  // The value dimension and byte size for each key.
  size_t value_dim_;
  size_t value_size_;

  // The flag records whether the elements of the hash table have changed since the last export, true means that there
  // has been a change.
  bool is_dirty_{true};

  // The device context is used to allocate host side memory from the pool for values in the hash table.
  DeviceContext *device_ctx_{nullptr};
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_HAL_DEVICE_CPU_HASH_TABLE_H_
