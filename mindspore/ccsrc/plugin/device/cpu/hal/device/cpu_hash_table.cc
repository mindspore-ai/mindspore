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
#include <memory>
#include <vector>

#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/cpu/hal/device/cpu_hash_table.h"

namespace mindspore {
namespace device {
namespace cpu {
template <typename Key, typename Value>
CPUHashTable<Key, Value>::CPUHashTable(size_t value_dim) : value_dim_(value_dim), value_size_(0) {
  Initialize();
}

template <typename Key, typename Value>
CPUHashTable<Key, Value>::~CPUHashTable() {
  Finalize();
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::Initialize() {
  value_size_ = value_dim_ * sizeof(Value);
  return true;
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::Finalize() {
  return Clear();
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::Find(const Key *keys, size_t key_num, bool, Value *outputs, void *) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  // Find and copy values to output buffer if the keys exist.
  for (size_t i = 0; i < key_num; ++i) {
    const auto &key = keys[i];
    if (values_.find(key) != values_.end()) {
      size_t offset = i * value_dim_;
      size_t src_size = value_size_;
      size_t dst_size = value_size_;

      // Copy the value of the key from the hash table to the outputs.
      auto value = values_[key];
      MS_EXCEPTION_IF_NULL(value);
      auto ret = memcpy_s(outputs + offset, dst_size, value, src_size);
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return false;
      }
    } else {
      MS_LOG(ERROR) << "The key: " << key << " does not exist in the hash table.";
      return false;
    }
  }
  return true;
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::Insert(const Key *keys, size_t key_num, const Value *value, void *) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  for (size_t i = 0; i < key_num; ++i) {
    const auto &key = keys[i];

    // The the key does not exist, a new value buffer should be allocated firstly.
    if (values_.find(key) == values_.end()) {
      auto value_addr = AllocateMemory(value_size_);
      MS_EXCEPTION_IF_NULL(value_addr);
      values_[key] = reinterpret_cast<Value *>(value_addr);
    }

    // Do the insertion copy.
    size_t offset = i * value_dim_;
    size_t src_size = value_size_;
    size_t dst_size = value_size_;
    auto ret = memcpy_s(values_[key], dst_size, value + offset, src_size);
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
  }
  is_dirty_ = true;
  return true;
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::Erase(const Key *keys, size_t key_num, void *) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  // Erase all the keys in the hash table.
  for (size_t i = 0; i < key_num; ++i) {
    const auto &key = keys[i];
    if (values_.find(key) != values_.end()) {
      auto value = values_[keys[i]];
      values_.erase(keys[i]);

      // Return the memory of value to the pool.
      MS_EXCEPTION_IF_NULL(value);
      FreeMemory(value);
    } else {
      MS_LOG(ERROR) << "The key: " << key << " does not exist in the hash table.";
      return false;
    }
  }
  return true;
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::Reserve(size_t new_capacity, void *) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  values_.reserve(new_capacity);
  return true;
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::GetKeysAndValues(Key *keys, Value *values, void *) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  size_t index = 0;
  for (auto iter = values_.begin(); iter != values_.end(); iter++) {
    auto key = iter->first;
    auto value = iter->second;
    // Copy the key.
    keys[index] = key;

    // Copy the value.
    size_t offset = index * value_dim_;
    size_t src_size = value_size_;
    size_t dst_size = value_size_;
    auto ret = memcpy_s(values + offset, dst_size, value, src_size);
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
    ++index;
  }
  return true;
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::Import(const DataLenPair &input_data) {
  return true;
}

template <typename Key, typename Value>
HashTableExportData CPUHashTable<Key, Value>::Export(bool) {
  const size_t size = values_.size();
  auto host_keys = std::make_shared<std::vector<char>>(size * sizeof(Key));
  auto host_values = std::make_shared<std::vector<char>>(size * value_size_);
  auto value_data = host_values->data();
  auto host_statuses = std::make_shared<std::vector<char>>(size * sizeof(HashTableElementStatus));

  size_t index = 0;
  for (auto iter = values_.begin(); iter != values_.end(); iter++) {
    auto key = iter->first;
    auto value = iter->second;
    // Export the key and value.
    (*host_keys)[index] = key;

    size_t offset = index * value_dim_;
    size_t src_size = value_size_;
    size_t dst_size = value_size_;
    auto ret = memcpy_s(value_data + offset, dst_size, value, src_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    }
    ++index;
  }
  return {host_keys, host_values, host_statuses};
}

template <typename Key, typename Value>
size_t CPUHashTable<Key, Value>::capacity() const {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  return values_.size();
}

template <typename Key, typename Value>
size_t CPUHashTable<Key, Value>::size() const {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  return values_.size();
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::is_dirty() const {
  return is_dirty_;
}

template <typename Key, typename Value>
bool CPUHashTable<Key, Value>::Clear() {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  // Return all the memory of values in hash table to the memory pool.
  for (auto iter = values_.begin(); iter != values_.end(); iter++) {
    auto key = iter->first;
    auto value = iter->second;
    if (value != nullptr) {
      FreeMemory(value);
    }
    values_.erase(key);
  }
  return true;
}

template <typename Key, typename Value>
void *CPUHashTable<Key, Value>::AllocateMemory(size_t size) const {
  return CPUMemoryPool::GetInstance().AllocTensorMem(size, false);
}

template <typename Key, typename Value>
void CPUHashTable<Key, Value>::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  CPUMemoryPool::GetInstance().FreeTensorMem(ptr);
}

template class CPUHashTable<int32_t, float>;
template class CPUHashTable<int64_t, float>;
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
