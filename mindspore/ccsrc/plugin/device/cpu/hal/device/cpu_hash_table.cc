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
#include "plugin/device/cpu/hal/device/cpu_hash_table.h"

#include <memory>
#include <vector>
#include <algorithm>

#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace device {
namespace cpu {
template <typename Key, typename Value>
CPUHashTable<Key, Value>::CPUHashTable(size_t value_dim) : value_dim_(value_dim), value_size_(0) {
  (void)Initialize();
}

template <typename Key, typename Value>
CPUHashTable<Key, Value>::~CPUHashTable() {
  (void)Finalize();
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
      auto value = values_[key].first;
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
bool CPUHashTable<Key, Value>::Insert(const Key *keys, size_t key_num, const Value *values, void *) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  MS_ERROR_IF_NULL(keys);
  MS_ERROR_IF_NULL(values);

  for (size_t i = 0; i < key_num; ++i) {
    const auto &key = keys[i];

    auto iter = values_.find(key);
    // The the key does not exist, a new value buffer should be allocated firstly.
    if (iter == values_.end()) {
      auto value_addr = AllocateMemory(value_size_);
      MS_EXCEPTION_IF_NULL(value_addr);
      iter = values_.emplace(key, std::make_pair(reinterpret_cast<Value *>(value_addr), Status::kModified)).first;
    }

    // Do the insertion copy.
    size_t offset = i * value_dim_;
    size_t src_size = value_size_;
    size_t dst_size = value_size_;
    auto ret = memcpy_s(iter->second.first, dst_size, values + offset, src_size);
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
    iter->second.second = Status::kModified;
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
    auto iter = values_.find(key);
    if (iter != values_.end()) {
      auto value = iter->second.first;
      (void)values_.erase(key);

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
    auto value = iter->second.first;
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
bool CPUHashTable<Key, Value>::Import(const DataLenPair &) {
  return true;
}

template <typename Key, typename Value>
HashTableExportData CPUHashTable<Key, Value>::ExportIncrementally() {
  // 1. Count export number of all modified elememts.
  size_t update_elements_size = std::count_if(
    values_.begin(), values_.end(), [](typename std::unordered_map<Key, ValueStatusPair>::const_reference item) {
      return item.second.second == Status::kModified;
    });

  auto keys = std::make_shared<std::vector<char>>(update_elements_size * sizeof(Key));
  auto keys_data = reinterpret_cast<Key *>(keys->data());
  auto values = std::make_shared<std::vector<char>>(update_elements_size * value_size_);
  auto value_data = reinterpret_cast<Value *>(values->data());
  auto statuses = std::make_shared<std::vector<char>>(update_elements_size * sizeof(HashTableElementStatus));
  auto statuses_data = reinterpret_cast<Status *>(statuses->data());

  // 2. Export all modified elememts.
  size_t index = 0;
  for (auto iter = values_.begin(); iter != values_.end(); iter++) {
    auto key = iter->first;
    auto value = iter->second.first;
    auto status = iter->second.second;
    if (status != Status::kModified) {
      continue;
    }

    // Export the key.
    keys_data[index] = key;
    // Export the status.
    statuses_data[index] = status;

    // Export the value.
    size_t offset = index * value_dim_;
    size_t src_size = value_size_;
    size_t dst_size = value_size_;
    auto ret = memcpy_s(value_data + offset, dst_size, value, src_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    }
    ++index;
  }
  return {keys, values, statuses};
}

template <typename Key, typename Value>
HashTableExportData CPUHashTable<Key, Value>::ExportFully() {
  const size_t size = values_.size();
  auto keys = std::make_shared<std::vector<char>>(size * sizeof(Key));
  auto keys_data = reinterpret_cast<Key *>(keys->data());
  auto values = std::make_shared<std::vector<char>>(size * value_size_);
  auto value_data = reinterpret_cast<Value *>(values->data());
  auto statuses = std::make_shared<std::vector<char>>(size * sizeof(HashTableElementStatus));
  auto statuses_data = reinterpret_cast<Status *>(statuses->data());

  size_t index = 0;
  for (auto iter = values_.begin(); iter != values_.end(); iter++) {
    auto key = iter->first;
    auto value = iter->second.first;
    auto status = iter->second.second;

    // Export the key.
    keys_data[index] = key;
    // Export the status.
    statuses_data[index] = status;

    // Export the value.
    size_t offset = index * value_dim_;
    size_t src_size = value_size_;
    size_t dst_size = value_size_;
    auto ret = memcpy_s(value_data + offset, dst_size, value, src_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    }
    ++index;
  }
  return {keys, values, statuses};
}

template <typename Key, typename Value>
HashTableExportData CPUHashTable<Key, Value>::Export(bool incremental) {
  // Update is_dirty_ to false because already get latest content after export.
  is_dirty_ = false;

  if (incremental) {
    return ExportIncrementally();
  }
  return ExportFully();
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
    auto value = iter->second.first;
    if (value != nullptr) {
      FreeMemory(value);
    }
  }
  (void)values_.clear();
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
