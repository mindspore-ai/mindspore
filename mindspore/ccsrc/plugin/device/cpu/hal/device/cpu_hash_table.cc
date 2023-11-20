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

#include <string>
#include <algorithm>

#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace device {
namespace cpu {
template <typename Key, typename Value>
CPUHashTable<Key, Value>::CPUHashTable(size_t value_dim, const std::string &initializer)
    : value_dim_(value_dim), value_size_(0), initializer_(initializer), default_value_(0) {
  (void)Initialize();
}

template <typename Key, typename Value>
CPUHashTable<Key, Value>::CPUHashTable(size_t value_dim, const Value &default_value)
    : value_dim_(value_dim), value_size_(0), initializer_(""), default_value_(default_value) {
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
bool CPUHashTable<Key, Value>::Find(const Key *keys, size_t key_num, bool insert_default_value, Value *outputs,
                                    void *) {
  std::unique_lock<std::shared_mutex> lock(mutex_);

  MS_EXCEPTION_IF_NULL(outputs);
  // Find and copy values to output buffer if the keys exist.
  for (size_t i = 0; i < key_num; ++i) {
    const auto &key = keys[i];
    size_t offset = i * value_dim_;
    size_t src_size = value_size_;
    size_t dst_size = value_size_;
    if (values_.find(key) != values_.end()) {
      // Copy the value of the key from the hash table to the outputs.
      auto value = values_[key].first;
      MS_EXCEPTION_IF_NULL(value);
      auto ret = memcpy_s(outputs + offset, dst_size, value, src_size);
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return false;
      }
      continue;
    }

    if (!insert_default_value) {
      MS_LOG(ERROR) << "The key: " << key << " does not exist in the hash table.";
      return false;
    }

    // Insert key-value pair into values_ by default_value or initializer
    if (!initializer_.empty()) {
      // allocate memory for value
      auto value_addr = static_cast<Value *>(AllocateMemory(value_size_));
      MS_EXCEPTION_IF_NULL(value_addr);
      (void)values_.emplace(key, std::make_pair(static_cast<Value *>(value_addr), Status::kModified));
      if (initializer_ == kNormalDistribution) {
        // initialize normal distribution parameter
        const double mean = 0.0;
        const double sigma = 0.01;
        std::random_device rd;
        const std::uint64_t seed = rd();
        size_t skip = 0;
        random::GenerateRandoms<Value, Generator, NormalDistribution>(seed, skip, value_addr, value_dim_, mean, sigma);
      } else if (initializer_ == kOnesDistribution) {
        default_value_ = 1;
        for (size_t k = 0; k < value_dim_; ++k) {
          value_addr[k] = default_value_;
        }
      } else if (initializer_ == kZerosDistribution) {
        default_value_ = 0;
        for (size_t k = 0; k < value_dim_; ++k) {
          value_addr[k] = default_value_;
        }
      } else {
        MS_LOG(ERROR) << "Unsupported initializer: " << initializer_;
        return false;
      }
      auto ret = memcpy_s(outputs + offset, dst_size, value_addr, src_size);
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return false;
      }
    } else {
      // if there's no key in values_
      auto value_addr = static_cast<Value *>(AllocateMemory(value_size_));
      MS_EXCEPTION_IF_NULL(value_addr);
      (void)values_.emplace(key, std::make_pair(static_cast<Value *>(value_addr), Status::kModified));
      for (size_t k = 0; k < value_dim_; ++k) {
        value_addr[k] = default_value_;
      }
      auto ret = memcpy_s(outputs + offset, dst_size, value_addr, src_size);
      if (ret != EOK) {
        MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
        return false;
      }
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
      iter = values_.emplace(key, std::make_pair(static_cast<Value *>(value_addr), Status::kModified)).first;
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
bool CPUHashTable<Key, Value>::Insert(const Key *keys, size_t key_num, const Value *values, Status *statuses, void *) {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  MS_ERROR_IF_NULL(keys);
  MS_ERROR_IF_NULL(values);
  MS_ERROR_IF_NULL(statuses);

  for (size_t i = 0; i < key_num; ++i) {
    const auto &key = keys[i];

    auto iter = values_.find(key);
    // The the key does not exist, a new value buffer should be allocated firstly.
    if (iter == values_.end()) {
      auto value_addr = AllocateMemory(value_size_);
      MS_EXCEPTION_IF_NULL(value_addr);
      iter = values_.emplace(key, std::make_pair(static_cast<Value *>(value_addr), statuses[i])).first;
    }

    auto ret = memcpy_s(iter->second.first, value_size_, values + (i * value_dim_), value_size_);
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy_s error, errorno(" << ret << ")";
      return false;
    }
    iter->second.second = statuses[i];
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
bool CPUHashTable<Key, Value>::Import(const DataLenPair &input_data) {
  // 1. import input tensor data once receiving kImportTensorNum(3) input tensors: {key_tensor, value_tensor,
  // status_tensor}
  static std::vector<DataLenPair> input_data_list;
  if (input_data_list.size() < kImportTensorNum) {
    (void)input_data_list.emplace_back(input_data);
  }
  if (input_data_list.size() != kImportTensorNum) {
    return true;
  }

  const auto &input_keys = input_data_list[0];
  const auto &input_values = input_data_list[1];
  void *host_keys = input_keys.first;
  void *host_values = input_values.first;
  MS_ERROR_IF_NULL(host_keys);
  MS_ERROR_IF_NULL(host_values);

  size_t keys_len = input_keys.second;
  if (keys_len == 0) {
    return true;
  }

  Key *device_keys = static_cast<Key *>(host_keys);
  Value *device_values = static_cast<Value *>(host_values);
  Status *statuses = new Status[keys_len];
  (void)std::fill_n(statuses, keys_len, Status::kUnchanged);
  if (!Insert(device_keys, keys_len / sizeof(Key), device_values, statuses, nullptr)) {
    MS_LOG(ERROR) << "Insert keys and values failed.";
    delete[] statuses;
    return false;
  }

  input_data_list.clear();  // Clear the list of input tensors
  delete[] statuses;

  return true;
}

template <typename Key, typename Value>
HashTableExportData CPUHashTable<Key, Value>::ExportSliceFully(size_t begin, size_t end) {
  if (end < begin) {
    MS_LOG(EXCEPTION) << "Invalid export position parameter, begin: " << begin << ", end: " << end;
  }

  const size_t size = end - begin;
  auto keys = std::make_shared<std::vector<char>>(size * sizeof(Key));
  auto keys_data = reinterpret_cast<Key *>(keys->data());
  auto values = std::make_shared<std::vector<char>>(size * value_size_);
  auto values_data = reinterpret_cast<Value *>(values->data());
  auto statuses = std::make_shared<std::vector<char>>(size * sizeof(HashTableElementStatus));
  auto statuses_data = reinterpret_cast<Status *>(statuses->data());

  size_t index = 0;
  auto begin_iter = values_.begin();
  std::advance(begin_iter, begin);
  auto end_iter = values_.begin();
  std::advance(end_iter, end);
  for (auto iter = begin_iter; iter != end_iter; ++iter) {
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
    auto ret = memcpy_s(values_data + offset, dst_size, value, src_size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s error, errorno(" << ret << ")";
    }
    ++index;
  }
  return {keys, values, statuses};
}

template <typename Key, typename Value>
HashTableExportData CPUHashTable<Key, Value>::ExportSliceIncrementally(size_t begin, size_t end) {
  if (end < begin) {
    MS_LOG(EXCEPTION) << "Invalid export position parameter, begin: " << begin << ", end: " << end;
  }

  auto begin_iter = values_.begin();
  std::advance(begin_iter, begin);
  auto end_iter = values_.begin();
  std::advance(end_iter, end);

  // 1. Count export number of all modified elememts.
  size_t update_elements_size = LongToSize(
    std::count_if(begin_iter, end_iter, [](typename std::unordered_map<Key, ValueStatusPair>::const_reference item) {
      return item.second.second != Status::kUnchanged;
    }));

  auto keys = std::make_shared<std::vector<char>>(update_elements_size * sizeof(Key));
  auto keys_data = reinterpret_cast<Key *>(keys->data());
  auto values = std::make_shared<std::vector<char>>(update_elements_size * value_size_);
  auto values_data = reinterpret_cast<Value *>(values->data());
  auto statuses = std::make_shared<std::vector<char>>(update_elements_size * sizeof(HashTableElementStatus));
  auto statuses_data = reinterpret_cast<Status *>(statuses->data());

  // 2. Export all modified elememts.
  size_t index = 0;
  for (auto iter = begin_iter; iter != end_iter; ++iter) {
    auto key = iter->first;
    auto value = iter->second.first;
    auto status = iter->second.second;
    if (status == Status::kUnchanged) {
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
    auto ret = memcpy_s(values_data + offset, dst_size, value, src_size);
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

  if (size() == 0) {
    return HashTableExportData();
  }

  size_t begin = 0;
  size_t end = size();

  if (incremental) {
    return ExportSliceIncrementally(begin, end);
  }
  return ExportSliceFully(begin, end);
}

template <typename Key, typename Value>
HashTableExportData CPUHashTable<Key, Value>::ExportSlice(bool incremental, bool *last_slice,
                                                          size_t slice_size_in_mega_bytes) {
  MS_EXCEPTION_IF_NULL(last_slice);
  if (size() == 0) {
    *last_slice = true;
    return HashTableExportData();
  }

  constexpr size_t mega_byte_to_byte_rate = static_cast<size_t>(1) << 20;
  size_t slice_size = slice_size_in_mega_bytes * mega_byte_to_byte_rate / value_size_;
  if (slice_size == 0) {
    MS_LOG(EXCEPTION) << "The parameter[slice_size_in_mega_bytes] " << slice_size_in_mega_bytes
                      << " should be greater than the length in meta bytes of one element in hash map: "
                      << value_size_ / mega_byte_to_byte_rate;
  }

  if (end_ == 0) {
    end_ = std::min(begin_ + slice_size, size());
  }

  HashTableExportData ret;
  if (incremental) {
    ret = ExportSliceIncrementally(begin_, end_);
  } else {
    ret = ExportSliceFully(begin_, end_);
  }

  *last_slice = (end_ == size());
  if (*last_slice) {
    begin_ = 0;
    end_ = 0;
  } else {
    begin_ += slice_size;
    end_ = std::min(begin_ + slice_size, size());
  }
  return ret;
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
