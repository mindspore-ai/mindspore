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

#include "plugin/device/gpu/hal/device/gpu_hash_table_dummy.h"

namespace mindspore {
namespace device {
namespace gpu {
template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *keys, size_t key_num, bool insert_default_value,
                                               Value *outputs, void *stream) {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Insert(const Key *keys, size_t key_num, const Value *value, void *stream) {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Erase(const Key *keys, size_t key_num, void *stream) {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Clear() {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
size_t GPUHashTable<Key, Value, Allocator>::capacity() const {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
size_t GPUHashTable<Key, Value, Allocator>::size() const {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::is_dirty() const {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Reserve(size_t new_capacity, void *stream) {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::GetKeysAndValues(Key *keys, Value *values, void *stream) {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Import(const DataLenPair &input_data) {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::Export(bool incremental) {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::ExportSlice(bool incremental, bool *last_slice, size_t) {
  MS_LOG(EXCEPTION) << "Call dummy implement function.";
}

template class GPUHashTable<int32_t, float>;
template class GPUHashTable<int64_t, float>;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
