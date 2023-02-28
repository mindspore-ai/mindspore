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

#include "plugin/device/gpu/hal/device/gpu_hash_table.h"

#if CUDA_VERSION > 11000
#include <cuco/dynamic_map.cuh>
#include <random>
#include <algorithm>

#include "plugin/device/gpu/hal/device/gpu_hash_table_kernel.cuh"
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"

namespace mindspore {
namespace device {
namespace gpu {
template <typename Key, typename Value, typename Allocator>
using CucoDynamicMap = cuco::dynamic_map<Key, Value, cuda::thread_scope_device, Allocator>;

// CudaDynamicMap is a wrapper of cuco::dynamic_map, gpu_hash_table.h needs to be used by other cpp source files, in
// order for g++ to compile properly, the declaration of the cuco::dynamic_map type cannot appear in the header file
// gpu_hash_table.h, through the CudaDynamicMap type gpu_hash_ table.h pre-declaration to solve compilation problems.
template <typename Key, typename Value, typename Allocator>
struct CudaDynamicMap {
  CucoDynamicMap<Key, Value, Allocator> dynamic_map_;

  CudaDynamicMap(const Key &empty_key, const Value &empty_value, const Key &erased_key, const Allocator &alloc,
                 cudaStream_t stream = 0)
      : dynamic_map_(kInitialCapacity, cuco::sentinel::empty_key<Key>{empty_key},
                     cuco::sentinel::empty_value<Value>{empty_value}, cuco::sentinel::erased_key<Key>{erased_key},
                     alloc, stream) {}

  ~CudaDynamicMap() = default;
};

template <typename Key, typename Value, typename Allocator>
std::vector<int8_t> GPUHashTable<Key, Value, Allocator>::idle_flags_initializer_ =
  std::vector<int8_t>(GPUHashTable<Key, Value, Allocator>::elements_per_block_, 1);

template <typename Key, typename Value, typename Allocator>
std::vector<size_t> GPUHashTable<Key, Value, Allocator>::lookup_counter_initializer_ =
  std::vector<size_t>(GPUHashTable<Key, Value, Allocator>::elements_per_block_, 0);

template <typename Key, typename Value, typename Allocator>
GPUHashTable<Key, Value, Allocator>::GPUHashTable(int32_t value_dim, const std::string &initializer,
                                                  uint64_t permit_threshold, uint64_t evict_threshold,
                                                  const Allocator &alloc)
    : value_dim_(value_dim),
      initializer_(initializer),
      default_value_(0),
      char_alloc_(alloc),
      permit_threshold_(permit_threshold),
      evict_threshold_(evict_threshold) {
  Initialize(alloc);
}

template <typename Key, typename Value, typename Allocator>
GPUHashTable<Key, Value, Allocator>::GPUHashTable(int32_t value_dim, const Value &default_value,
                                                  uint64_t permit_threshold, uint64_t evict_threshold,
                                                  const Allocator &alloc)
    : value_dim_(value_dim),
      initializer_(""),
      default_value_(default_value),
      char_alloc_(alloc),
      permit_threshold_(permit_threshold),
      evict_threshold_(evict_threshold) {
  Initialize(alloc);
}

template <typename Key, typename Value, typename Allocator>
GPUHashTable<Key, Value, Allocator>::~GPUHashTable() {
  Finalize();
}

template <typename Key, typename Value, typename Allocator>
void GPUHashTable<Key, Value, Allocator>::Initialize(const Allocator &alloc) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  cuda_dynamic_map_ = std::make_unique<CudaDynamicMap<Key, int32_t, Allocator>>(
    static_cast<Key>(kEmptyKey), kEmptyValue, static_cast<Key>(kErasedKey), alloc, stream);

  CudaAtomicSize host_init_atomic_size_t(0);
  CudaAtomicInt host_init_atomic_int(0);

  AllocateMemory(sizeof(CudaAtomicSize), &current_index_);
  AllocateMemory(sizeof(CudaAtomicInt), &erased_counter_);

  CHECK_CUDA_RET(
    cudaMemcpyAsync(current_index_, &host_init_atomic_size_t, sizeof(CudaAtomicSize), cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync");
  CHECK_CUDA_RET(
    cudaMemcpyAsync(erased_counter_, &host_init_atomic_int, sizeof(CudaAtomicInt), cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync");

  CHECK_CUDA_RET(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  CHECK_CUDA_RET(cudaMallocManaged(&insert_success_number_, sizeof(CudaAtomicSize)), "cudaMallocManaged");
}

template <typename Key, typename Value, typename Allocator>
void GPUHashTable<Key, Value, Allocator>::Finalize() {
  cuda_dynamic_map_ = nullptr;

  FreeMemory(current_index_);
  FreeMemory(erased_counter_);

  if (erased_slot_) {
    FreeMemory(erased_slot_);
  }

  for (size_t i = 0; i < blocks_.size(); i++) {
    FreeMemory(blocks_[i]);
    FreeMemory(idle_flags_[i]);
    FreeMemory(statuses_[i]);
    FreeMemory(lookup_cnts_[i]);
    FreeMemory(update_timestamps_[i]);
  }

  FreeAllBlockRecorders();

  if (random_gen_state_) {
    FreeMemory(random_gen_state_);
  }

  CHECK_CUDA_RET(cudaFree(insert_success_number_), "cudaFree");
}

template <typename Key, typename Value, typename Allocator>
template <typename T>
void GPUHashTable<Key, Value, Allocator>::AllocateMemory(size_t size, T **ptr) {
  MS_EXCEPTION_IF_NULL(ptr);
  *ptr = reinterpret_cast<T *>(std::allocator_traits<CharAllocatorType>::allocate(char_alloc_, size));
}

template <typename Key, typename Value, typename Allocator>
void GPUHashTable<Key, Value, Allocator>::FreeMemory(void *ptr) {
  std::allocator_traits<CharAllocatorType>::deallocate(char_alloc_, reinterpret_cast<char *>(ptr), 0);
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *keys, size_t key_num, bool insert_default_value,
                                               Value *outputs, void *stream) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!initializer_.empty()) {
    return Find(keys, key_num, insert_default_value, initializer_, outputs, stream);
  }
  return Find(keys, key_num, insert_default_value, default_value_, outputs, stream);
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *keys, size_t key_num, bool insert_default_value,
                                               const std::string &initializer, Value *outputs, void *stream) {
  MS_ERROR_IF_NULL(keys);
  MS_ERROR_IF_NULL(outputs);
  MS_ERROR_IF_NULL(stream);
  int *indices = nullptr;
  AllocateMemory(key_num * sizeof(int), &indices);
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  Reserve(dynamic_map.get_size() + key_num, stream);

  // 1. Get all indices in blocks according to the keys.
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  RETURN_IF_FALSE_WITH_LOG(GetIndicesByKeys(keys, key_num, insert_default_value, indices, cuda_stream),
                           "Get indices by keys failed.");

  RETURN_IF_FALSE_WITH_LOG(UpdateSize(key_num, indices, cuda_stream, insert_default_value),
                           "Update hash table size failed.");

  // 2. Insert default value according to initializer, initializer can be 'normal', 'zeros' or 'ones'.
  RETURN_IF_FALSE_WITH_LOG(InsertDefaultValueByInitializer(key_num, initializer, indices, cuda_stream),
                           "Insert default value for miss keys failed.");

  // 3. Get all values by indices in blocks.
  size_t total_size = value_dim_ * key_num;
  GetValues<<<GET_BLOCKS(total_size), GET_THREADS, 0, cuda_stream>>>(value_dim_, total_size, indices,
                                                                     elements_per_block_, blocks_ptr_, outputs);
  FreeMemory(indices);
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *keys, size_t key_num, bool insert_default_value,
                                               const Value &default_value, Value *outputs, void *stream) {
  MS_ERROR_IF_NULL(keys);
  MS_ERROR_IF_NULL(outputs);
  MS_ERROR_IF_NULL(stream);
  int *indices = nullptr;
  AllocateMemory(key_num * sizeof(int), &indices);
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  Reserve(dynamic_map.get_size() + key_num, stream);

  // 1. Get all indices in blocks according to the keys.
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  RETURN_IF_FALSE_WITH_LOG(GetIndicesByKeys(keys, key_num, insert_default_value, indices, cuda_stream),
                           "Get indices by keys failed.");

  RETURN_IF_FALSE_WITH_LOG(UpdateSize(key_num, indices, cuda_stream, insert_default_value),
                           "Update hash table size failed.");

  // 2. Insert default value into map by specific value.
  InsertDefaultValue<<<GET_BLOCKS(key_num), GET_THREADS, 0, cuda_stream>>>(
    value_dim_, key_num, indices, elements_per_block_, lookup_cnts_ptr_, permit_threshold_, default_value,
    idle_flags_ptr_, blocks_ptr_);

  // 3. Get all values by indices in blocks.
  size_t total_size = value_dim_ * key_num;
  GetValues<<<GET_BLOCKS(total_size), GET_THREADS, 0, cuda_stream>>>(value_dim_, total_size, indices,
                                                                     elements_per_block_, blocks_ptr_, outputs);
  FreeMemory(indices);
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Insert(const Key *keys, size_t key_num, const Value *value, void *stream) {
  std::unique_lock<std::mutex> lock(mutex_);
  MS_ERROR_IF_NULL(keys);
  MS_ERROR_IF_NULL(value);
  MS_ERROR_IF_NULL(stream);
  int *indices = nullptr;
  AllocateMemory(key_num * sizeof(int), &indices);
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  Reserve(dynamic_map.get_size() + key_num, stream);

  // 1. Get all indices in blocks according to the keys.
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  RETURN_IF_FALSE_WITH_LOG(GetIndicesByKeys(keys, key_num, true, indices, cuda_stream), "Get indices by keys failed.");

  // Update size but do not update lookup count, the method `Find` should update lookup count.
  RETURN_IF_FALSE_WITH_LOG(UpdateSize(key_num, indices, cuda_stream, false), "Update hash table size failed.");

  global_timestamp_++;

  // 2. Insert values into map by indices in blocks.
  size_t total_insert_size = value_dim_ * key_num;
  auto block_size = GET_THREADS_MAXSIZE(kBlockSize);
  auto grid_size = CUDA_BLOCKS_CAL(GET_CTX_DEVICE_ID, total_insert_size, block_size);
  InsertValues<<<grid_size, block_size, 0, cuda_stream>>>(
    value_dim_, total_insert_size, indices, value, elements_per_block_, lookup_cnts_ptr_, permit_threshold_,
    global_timestamp_, update_timestamps_ptr_, statuses_ptr_, idle_flags_ptr_, blocks_ptr_);

  is_dirty_ = true;
  FreeMemory(indices);

  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Erase(const Key *keys, size_t key_num, void *stream) {
  std::unique_lock<std::mutex> lock(mutex_);
  MS_ERROR_IF_NULL(keys);
  MS_ERROR_IF_NULL(stream);

  int *indices = nullptr;
  AllocateMemory(key_num * sizeof(int), &indices);
  MS_ERROR_IF_NULL(indices);

  // 1. Get all indices in blocks according to the key.
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  RETURN_IF_FALSE_WITH_LOG(GetIndicesByKeys(keys, key_num, false, indices, cuda_stream), "Get indices by keys failed.");

  // 2. Erase elements by indices and update erased statistics recorder.
  RETURN_IF_FALSE_WITH_LOG(EraseElements(keys, key_num, indices, cuda_stream), "Erase elements failed.");

  FreeMemory(indices);
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Clear() {
  std::unique_lock<std::mutex> lock(mutex_);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  // Need wait all task on stream finish.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  size_ = 0;
  // 1. Reset cuda dynamic map.
  cuda_dynamic_map_ = std::make_unique<CudaDynamicMap<Key, int32_t, Allocator>>(
    static_cast<Key>(-1), -1, static_cast<Key>(-2), Allocator(), stream);

  CudaAtomicSize host_init_atomic_size_t(0);
  CudaAtomicInt host_init_atomic_int(0);

  // 2. Reset cuda atomic counter.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(current_index_, &host_init_atomic_size_t, sizeof(CudaAtomicSize), cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(erased_counter_, &host_init_atomic_int, sizeof(CudaAtomicInt), cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync");

  // 3. Reset idle status.
  for (size_t i = 0; i < idle_flags_.size(); i++) {
    CHECK_CUDA_RET_WITH_RETURN_FALSE(
      cudaMemcpyAsync(idle_flags_[i], idle_flags_initializer_.data(), idle_flags_initializer_.size() * sizeof(bool),
                      cudaMemcpyHostToDevice, stream),
      "cudaMemcpyAsync");
  }

  // 4. Reset status.
  for (size_t i = 0; i < statuses_.size(); i++) {
    CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemsetAsync(statuses_[i], 0, elements_per_block_ * sizeof(Status), stream),
                                     "cudaMemsetAsync");
  }

  // 5. Reset lookup counter.
  for (size_t i = 0; i < lookup_cnts_.size(); i++) {
    CHECK_CUDA_RET_WITH_RETURN_FALSE(
      cudaMemcpyAsync(lookup_cnts_[i], lookup_counter_initializer_.data(),
                      lookup_counter_initializer_.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream),
      "cudaMemcpyAsync");
  }

  global_timestamp_ = 0;

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Reserve(size_t new_capacity, void *stream) {
  // There is sufficient space in hash table, need not to reserve.
  if (capacity() >= new_capacity) {
    return true;
  }
  MS_ERROR_IF_NULL(stream);
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

  // Allocate new block and idle flag until the capacity of hash table reaches desired capacity.
  size_t remain_num = new_capacity - capacity();
  while (remain_num > 0) {
    RETURN_IF_FALSE_WITH_LOG(AddNewBlock(cuda_stream), "Add a new block for hash table failed.");

    remain_num -= std::min(remain_num, elements_per_block_);
    capacity_ += elements_per_block_;
  }

  // Wait all task on the stream finish, because the blocks_ptr_ need to reallocate, there may be some kernels are using
  // the blocks_ptr_ and idle_flags_ptr_.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize");

  return ResetAllBlockRecorders(cuda_stream);
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::AddNewBlock(cudaStream_t stream) {
  // Allocate a new block.
  Value *new_block = nullptr;
  AllocateMemory(value_dim_ * elements_per_block_ * sizeof(Value), &new_block);
  MS_ERROR_IF_NULL(new_block);
  blocks_.push_back(new_block);

  // Allocate new idle flag memory for new block.
  bool *new_block_idle_flag = nullptr;
  AllocateMemory(elements_per_block_ * sizeof(bool), &new_block_idle_flag);
  MS_ERROR_IF_NULL(new_block_idle_flag);
  idle_flags_.push_back(new_block_idle_flag);

  // Allocate new status recorder memory for new block.
  Status *new_block_status = nullptr;
  AllocateMemory(elements_per_block_ * sizeof(Status), &new_block_status);
  MS_ERROR_IF_NULL(new_block_status);
  statuses_.push_back(new_block_status);

  // Allocate new lookup counter memory for new block.
  size_t *new_lookup_cnt = nullptr;
  AllocateMemory(elements_per_block_ * sizeof(size_t), &new_lookup_cnt);
  MS_ERROR_IF_NULL(new_lookup_cnt);
  lookup_cnts_.push_back(new_lookup_cnt);

  // Allocate new timestamps counter memory for new block.
  size_t *new_update_timestamp = nullptr;
  AllocateMemory(elements_per_block_ * sizeof(size_t), &new_update_timestamp);
  MS_ERROR_IF_NULL(new_update_timestamp);
  update_timestamps_.push_back(new_update_timestamp);

  // Set initialized value for idle flag.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(new_block_idle_flag, idle_flags_initializer_.data(), idle_flags_initializer_.size() * sizeof(bool),
                    cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync");

  // Set initialized value for status.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemsetAsync(new_block_status, 0, elements_per_block_ * sizeof(Status), stream),
                                   "cudaMemsetAsync");

  // Set initialized value for lookup counter.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(new_lookup_cnt, lookup_counter_initializer_.data(),
                    lookup_counter_initializer_.size() * sizeof(size_t), cudaMemcpyHostToDevice, stream),
    "cudaMemcpyAsync");

  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::ResetAllBlockRecorders(cudaStream_t cuda_stream) {
  // 1. Free the buffers that record pointers for block, idle status, lookup counter and updated timestamps.
  FreeAllBlockRecorders();

  size_t cur_blocks_num = blocks_.size();
  // 2. Allocate new GPU memory for blocks_ptr_.
  Value *new_blocks_ptr = nullptr;
  AllocateMemory(cur_blocks_num * sizeof(Value *), &new_blocks_ptr);
  blocks_ptr_ = reinterpret_cast<Value **>(new_blocks_ptr);
  MS_ERROR_IF_NULL(blocks_ptr_);

  // Update the content for blocks pointer recorder.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(blocks_ptr_, blocks_.data(), cur_blocks_num * sizeof(Value *), cudaMemcpyHostToDevice, cuda_stream),
    "cudaMemcpyAsync");

  // 3. Allocate new GPU memory for idle_flags_ptr_.
  bool *new_idle_flags_ptr = nullptr;
  AllocateMemory(cur_blocks_num * sizeof(bool *), &new_idle_flags_ptr);
  idle_flags_ptr_ = reinterpret_cast<bool **>(new_idle_flags_ptr);
  MS_ERROR_IF_NULL(idle_flags_ptr_);

  // Update the content for idle flags pointer recorder.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(idle_flags_ptr_, idle_flags_.data(), cur_blocks_num * sizeof(bool *),
                                                   cudaMemcpyHostToDevice, cuda_stream),
                                   "cudaMemcpyAsync");

  // 3. Allocate new GPU memory for statuses_ptr_.
  Status *new_status_ptr = nullptr;
  AllocateMemory(cur_blocks_num * sizeof(Status *), &new_status_ptr);
  statuses_ptr_ = reinterpret_cast<Status **>(new_status_ptr);
  MS_ERROR_IF_NULL(statuses_ptr_);

  // Update the content for status pointer recorder.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(statuses_ptr_, statuses_.data(), cur_blocks_num * sizeof(Status *),
                                                   cudaMemcpyHostToDevice, cuda_stream),
                                   "cudaMemcpyAsync");

  // 4. Allocate new GPU memory for lookup_cnts_ptr_.
  bool *new_lookup_cnts_ptr = nullptr;
  AllocateMemory(cur_blocks_num * sizeof(size_t *), &new_lookup_cnts_ptr);
  lookup_cnts_ptr_ = reinterpret_cast<size_t **>(new_lookup_cnts_ptr);
  MS_ERROR_IF_NULL(lookup_cnts_ptr_);

  // Update the content for lookup counter pointer recorder.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(lookup_cnts_ptr_, lookup_cnts_.data(), cur_blocks_num * sizeof(size_t *), cudaMemcpyHostToDevice,
                    cuda_stream),
    "cudaMemcpyAsync");

  // 5. Allocate new GPU memory for update_timestamps_ptr_.
  bool *new_update_timestamps_ptr = nullptr;
  AllocateMemory(cur_blocks_num * sizeof(size_t *), &new_update_timestamps_ptr);
  update_timestamps_ptr_ = reinterpret_cast<size_t **>(new_update_timestamps_ptr);
  MS_ERROR_IF_NULL(update_timestamps_ptr_);

  // Update the content for updated timestamps pointer recorder.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(update_timestamps_ptr_, update_timestamps_.data(), cur_blocks_num * sizeof(size_t *),
                    cudaMemcpyHostToDevice, cuda_stream),
    "cudaMemcpyAsync");
  return true;
}

template <typename Key, typename Value, typename Allocator>
void GPUHashTable<Key, Value, Allocator>::FreeAllBlockRecorders() {
  if (blocks_ptr_) {
    FreeMemory(blocks_ptr_);
  }
  if (idle_flags_ptr_) {
    FreeMemory(idle_flags_ptr_);
  }
  if (statuses_ptr_) {
    FreeMemory(statuses_ptr_);
  }
  if (lookup_cnts_ptr_) {
    FreeMemory(lookup_cnts_ptr_);
  }
  if (update_timestamps_ptr_) {
    FreeMemory(update_timestamps_ptr_);
  }
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::GetKeysAndValues(Key *keys, Value *values, void *stream) {
  MS_ERROR_IF_NULL(keys);
  MS_ERROR_IF_NULL(values);
  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  int *indices = nullptr;
  AllocateMemory(dynamic_map.get_size() * sizeof(int), &indices);
  MS_ERROR_IF_NULL(indices);

  // 1. Export all keys and indices from dynamic map.
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  RETURN_IF_FALSE_WITH_LOG(dynamic_map.get_keys_values(keys, indices, cuda_stream),
                           "Get keys and values from cuda dynamic map failed.");

  // 2. Get all values by indices in blocks.
  size_t total_size = value_dim_ * size_;
  GetValues<<<GET_BLOCKS(total_size), GET_THREADS, 0, cuda_stream>>>(value_dim_, total_size, indices,
                                                                     elements_per_block_, blocks_ptr_, values);
  FreeMemory(indices);
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::EvictExpiredElements(cudaStream_t stream) {
  // If evict_threshold_ is greater than or equal to kMaxEvictThreshold, eviction is disable.
  if (evict_threshold_ >= kMaxEvictThreshold) {
    return true;
  }

  MS_ERROR_IF_NULL(stream);
  // 1. Count all expired elements number in hash table.
  size_t expired_num = 0;
  RETURN_IF_FALSE_WITH_LOG(CountExpiredElements(stream, &expired_num), "Count expired elements failed.");
  if (expired_num == 0) {
    return true;
  }

  Key *expired_keys = nullptr;
  int *expired_indices = nullptr;
  AllocateMemory(expired_num * sizeof(Key), &expired_keys);
  AllocateMemory(expired_num * sizeof(int), &expired_indices);
  MS_ERROR_IF_NULL(expired_keys);
  MS_ERROR_IF_NULL(expired_indices);

  // 2. Find all keys and indices of expired elements.
  RETURN_IF_FALSE_WITH_LOG(FindExpiredElements(expired_keys, expired_indices, stream), "Find expired elements failed.");

  // 3. Erase all expired elements.
  RETURN_IF_FALSE_WITH_LOG(EraseElements(expired_keys, expired_num, expired_indices, stream), "Erase elements failed.");

  FreeMemory(expired_keys);
  FreeMemory(expired_indices);

  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::CountExpiredElements(cudaStream_t stream, size_t *expired_num) {
  MS_ERROR_IF_NULL(stream);
  MS_ERROR_IF_NULL(expired_num);

  // 1. Initialize device expired counter.
  CudaAtomicSize host_expired_counter(0);
  CudaAtomicSize *device_expired_counter = nullptr;
  AllocateMemory(sizeof(CudaAtomicSize), &device_expired_counter);
  MS_ERROR_IF_NULL(device_expired_counter);

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(device_expired_counter, &host_expired_counter,
                                                   sizeof(CudaAtomicSize), cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync");

  const uint32_t block_size = kBlockSize;
  if (IntToUint(GET_THREADS) < block_size) {
    MS_LOG(ERROR) << "The max thread per block is less than: " << block_size << " of this GPU";
    return false;
  }
  uint32_t device_id = GET_CTX_DEVICE_ID;
  const uint32_t grid_size = IntToUint(CUDA_BLOCKS_CAL(device_id, elements_per_block_, block_size));

  // 2. Count the number for expired elements.
  CountExpiredNum<block_size><<<grid_size, block_size, 0, stream>>>(
    blocks_.size(), permit_threshold_, global_timestamp_, evict_threshold_, elements_per_block_, idle_flags_ptr_,
    lookup_cnts_ptr_, update_timestamps_ptr_, device_expired_counter);

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(&host_expired_counter, device_expired_counter,
                                                   sizeof(CudaAtomicSize), cudaMemcpyDeviceToHost, stream),
                                   "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");
  *expired_num = host_expired_counter;

  FreeMemory(device_expired_counter);
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::FindExpiredElements(Key *expired_keys, int *expired_indices,
                                                              cudaStream_t stream) {
  MS_ERROR_IF_NULL(expired_keys);
  MS_ERROR_IF_NULL(expired_indices);
  MS_ERROR_IF_NULL(stream);

  // 1. Initialize device expired counter.
  CudaAtomicSize host_expired_counter(0);
  CudaAtomicSize *device_expired_counter = nullptr;
  AllocateMemory(sizeof(CudaAtomicSize), &device_expired_counter);
  MS_ERROR_IF_NULL(device_expired_counter);

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(device_expired_counter, &host_expired_counter,
                                                   sizeof(CudaAtomicSize), cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync");

  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  // Note: size of dynamic_map maybe greater than size_.
  auto size = dynamic_map.get_size();
  Key *all_keys = nullptr;
  int *all_indices = nullptr;
  AllocateMemory(size * sizeof(Key), &all_keys);
  AllocateMemory(size * sizeof(int), &all_indices);
  MS_ERROR_IF_NULL(all_keys);
  MS_ERROR_IF_NULL(all_indices);

  // 2. Export all keys and indices from dynamic map.
  RETURN_IF_FALSE_WITH_LOG(dynamic_map.get_keys_values(all_keys, all_indices, stream),
                           "Get keys and values from cuda dynamic map failed.");

  // 3. Find all keys and indices for expired elememts in hash table.
  FindExpiredKeysAndIndices<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(
    size, elements_per_block_, permit_threshold_, global_timestamp_, evict_threshold_, idle_flags_ptr_,
    lookup_cnts_ptr_, update_timestamps_ptr_, all_keys, all_indices, device_expired_counter, expired_keys,
    expired_indices);

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");
  FreeMemory(all_keys);
  FreeMemory(all_indices);
  FreeMemory(device_expired_counter);

  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::EraseElements(const Key *keys, size_t key_num, const int *indices,
                                                        cudaStream_t stream) {
  // 1. Store new erased slots.
  CudaAtomicInt host_erased_counter(0);
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(&host_erased_counter, erased_counter_, sizeof(CudaAtomicInt), cudaMemcpyDeviceToHost, stream),
    "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  size_t new_erased_counter = key_num + host_erased_counter;
  int32_t *new_erased_slot = nullptr;
  AllocateMemory(new_erased_counter * sizeof(int32_t), &new_erased_slot);
  MS_ERROR_IF_NULL(new_erased_slot);

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(new_erased_slot, erased_slot_, host_erased_counter * sizeof(int32_t),
                                                   cudaMemcpyDeviceToDevice, stream),
                                   "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");
  if (erased_slot_) {
    FreeMemory(erased_slot_);
  }
  erased_slot_ = new_erased_slot;

  AddErasedSlots<<<GET_BLOCKS(key_num), GET_THREADS, 0, stream>>>(key_num, kEmptyValue, indices, erased_counter_,
                                                                  erased_slot_);

  // 2. Update idle status for erased slot.
  EraseElementsByIndices<<<GET_BLOCKS(key_num), GET_THREADS, 0, stream>>>(key_num, elements_per_block_, kEmptyValue,
                                                                          indices, idle_flags_ptr_, statuses_ptr_);

  // 3. Erase all keys in dynamic map.
  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  size_t size_before_erase = dynamic_map.get_size();
  dynamic_map.erase(keys, keys + key_num, stream);
  size_t size_after_erase = dynamic_map.get_size();

  // 4. Update size.
  // Note: The erased keys should be exist in hash map.
  size_ -= (size_before_erase - size_after_erase);

  // 5. Record erased keys.
  size_t old_erased_keys_num = erased_keys_.size();
  erased_keys_.resize(old_erased_keys_num + key_num);
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(&erased_keys_[old_erased_keys_num], keys, key_num * sizeof(Key), cudaMemcpyDeviceToHost, stream),
    "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Import(const DataLenPair &input_data) {
  // 1. Store input tensor data until receiving kImportTensorNum(3) input tensor.
  // Really import input data to hash table when receive kImportTensorNum(3) input tensor.
  static std::vector<DataLenPair> input_data_list;
  if (input_data_list.size() < kImportTensorNum) {
    input_data_list.emplace_back(input_data);
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
  size_t values_len = input_values.second;

  // 2. Allocate temp buffer to keys and values.
  Key *device_keys = nullptr;
  AllocateMemory(keys_len, &device_keys);
  MS_ERROR_IF_NULL(device_keys);

  Value *device_values = nullptr;
  AllocateMemory(values_len, &device_values);
  MS_ERROR_IF_NULL(device_values);

  // 3. Copy input keys and values to device.
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(device_keys, host_keys, keys_len, cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(device_values, host_values, values_len, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");

  // 4. Insert input keys and values to hash table.
  RETURN_IF_FALSE_WITH_LOG(Insert(device_keys, keys_len / sizeof(Key), device_values, stream),
                           "Insert keys and values failed.");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  // 5. Free temp buffer to keys and values.
  FreeMemory(device_keys);
  FreeMemory(device_values);

  input_data_list.clear();
  return true;
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::Export(bool incremental) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  // Evict expired element before export.
  MS_EXCEPTION_IF_CHECK_FAIL(EvictExpiredElements(stream), "Evict expired elements failed.");

  // Update is_dirty_ to false because host side will get latest content after export.
  is_dirty_ = false;

  if (incremental) {
    return ExportIncrementally(stream);
  }
  return ExportFully(stream);
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::ExportFully(cudaStream_t stream) {
  if (size_ == 0) {
    return {std::make_shared<std::vector<char>>(), std::make_shared<std::vector<char>>(),
            std::make_shared<std::vector<char>>()};
  }
  MS_EXCEPTION_IF_NULL(stream);

  // 1. Allocate temp buffer to keys, values and statuses.
  Key *device_keys = nullptr;
  Value *device_values = nullptr;
  AllocateMemory(size_ * sizeof(Key), &device_keys);
  AllocateMemory(size_ * value_dim_ * sizeof(Value), &device_values);
  MS_EXCEPTION_IF_NULL(device_keys);
  MS_EXCEPTION_IF_NULL(device_values);

  // 2. Get all keys and values and store into temp buffer.
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  if (size_ != dynamic_map.get_size()) {
    MS_LOG(EXCEPTION) << "The gpu hash table's size[" << size_ << "] is not equal to cuda dynamic map size["
                      << dynamic_map.get_size() << "]";
  }
  int *indices = nullptr;
  AllocateMemory(dynamic_map.get_size() * sizeof(int), &indices);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_CHECK_FAIL(dynamic_map.get_keys_values(device_keys, indices, stream),
                             "Get keys and values from cuda dynamic map failed.");
  size_t total_size = value_dim_ * size_;
  GetValues<<<GET_BLOCKS(total_size), GET_THREADS, 0, stream>>>(value_dim_, total_size, indices, elements_per_block_,
                                                                blocks_ptr_, device_values);

  // 3. Export keys, values and statuses.
  auto host_keys = std::make_shared<std::vector<char>>(size_ * sizeof(Key));
  auto host_values = std::make_shared<std::vector<char>>(size_ * value_dim_ * sizeof(Value));
  auto host_statuses = std::make_shared<std::vector<char>>(size_ * sizeof(Status));
  // Copy keys.
  CHECK_CUDA_RET_WITH_EXCEPTION(
    cudaMemcpyAsync(host_keys->data(), device_keys, size_ * sizeof(Key), cudaMemcpyDeviceToHost, stream),
    "cudaMemcpyAsync");
  // Copy values.
  CHECK_CUDA_RET_WITH_EXCEPTION(cudaMemcpyAsync(host_values->data(), device_values, size_ * value_dim_ * sizeof(Value),
                                                cudaMemcpyDeviceToHost, stream),
                                "cudaMemcpyAsync");
  // Copy statuses.
  std::vector<Status> modified_status(size_, Status::kModified);
  auto ret = memcpy_s(host_statuses->data(), host_statuses->size(), modified_status.data(), size_ * sizeof(Status));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Memcpy for gpu hash table status failed, errno[" << ret << "]";
  }

  // 4. Update status to unchanged after export.
  UpdateStatus<<<GET_BLOCKS(size_), GET_THREADS, 0, stream>>>(size_, elements_per_block_, kEmptyValue, indices,
                                                              Status::kUnchanged, statuses_ptr_);

  CHECK_CUDA_RET_WITH_EXCEPTION(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");
  FreeMemory(indices);
  FreeMemory(device_keys);
  FreeMemory(device_values);
  return {host_keys, host_values, host_statuses};
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::ExportIncrementally(cudaStream_t stream) {
  MS_EXCEPTION_IF_NULL(stream);

  // 1. Count the number of element whose statuses are Status::kModified,
  // which means these elements are modified since last export.
  size_t modified_num = 0;
  MS_EXCEPTION_IF_CHECK_FAIL(CountModifiedElements(stream, &modified_num), " Count modified elements failed.");
  if (modified_num == 0) {
    return {std::make_shared<std::vector<char>>(), std::make_shared<std::vector<char>>(),
            std::make_shared<std::vector<char>>()};
  }

  // 2. Allocate device temp buffer to modified keys, indices and values.
  Key *device_modified_keys = nullptr;
  int *device_modified_indices = nullptr;
  Value *device_modified_values = nullptr;
  AllocateMemory(modified_num * sizeof(Key), &device_modified_keys);
  AllocateMemory(modified_num * sizeof(int), &device_modified_indices);
  AllocateMemory(modified_num * value_dim_ * sizeof(Value), &device_modified_values);
  MS_EXCEPTION_IF_NULL(device_modified_keys);
  MS_EXCEPTION_IF_NULL(device_modified_indices);
  MS_EXCEPTION_IF_NULL(device_modified_values);

  // 3. Get the keys and values for elements which are modified.
  MS_EXCEPTION_IF_CHECK_FAIL(FindModifiedElements(device_modified_keys, device_modified_indices, stream),
                             "Find modified elements failed");
  size_t total_size = value_dim_ * modified_num;
  GetValues<<<GET_BLOCKS(total_size), GET_THREADS, 0, stream>>>(
    value_dim_, total_size, device_modified_indices, elements_per_block_, blocks_ptr_, device_modified_values);

  // 4. Export the keys, values and statuses for elements which are modified or erased.
  HashTableExportData export_data =
    ExportModifiedAndErasedElements(modified_num, device_modified_keys, device_modified_values, stream);

  // 5. Update status to unchanged after export.
  UpdateStatus<<<GET_BLOCKS(modified_num), GET_THREADS, 0, stream>>>(
    modified_num, elements_per_block_, kEmptyValue, device_modified_indices, Status::kUnchanged, statuses_ptr_);

  CHECK_CUDA_RET_WITH_EXCEPTION(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  FreeMemory(device_modified_keys);
  FreeMemory(device_modified_indices);
  FreeMemory(device_modified_values);
  return export_data;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::CountModifiedElements(cudaStream_t stream, size_t *modified_num) {
  MS_ERROR_IF_NULL(stream);
  MS_ERROR_IF_NULL(modified_num);

  // 1. Initialize device modified counter.
  CudaAtomicSize host_modified_counter(0);
  CudaAtomicSize *device_modified_counter = nullptr;
  AllocateMemory(sizeof(CudaAtomicSize), &device_modified_counter);
  MS_ERROR_IF_NULL(device_modified_counter);

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(device_modified_counter, &host_modified_counter,
                                                   sizeof(CudaAtomicSize), cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync");

  const uint32_t block_size = kBlockSize;
  if (IntToUint(GET_THREADS) < block_size) {
    MS_LOG(ERROR) << "The max thread per block is less than: " << block_size << " of this GPU";
    return false;
  }
  uint32_t device_id = GET_CTX_DEVICE_ID;
  const uint32_t grid_size = IntToUint(CUDA_BLOCKS_CAL(device_id, elements_per_block_, block_size));

  // 2. Count the number for modified elements.
  CountModifiedNum<block_size>
    <<<grid_size, block_size, 0, stream>>>(blocks_.size(), elements_per_block_, statuses_ptr_, device_modified_counter);

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(&host_modified_counter, device_modified_counter,
                                                   sizeof(CudaAtomicSize), cudaMemcpyDeviceToHost, stream),
                                   "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");
  *modified_num = host_modified_counter;

  FreeMemory(device_modified_counter);
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::FindModifiedElements(Key *modified_keys, int *modified_indices,
                                                               cudaStream_t stream) {
  MS_ERROR_IF_NULL(modified_keys);
  MS_ERROR_IF_NULL(modified_indices);
  MS_ERROR_IF_NULL(stream);

  // 1. Initialize device modified counter.
  CudaAtomicSize host_modified_counter(0);
  CudaAtomicSize *device_modified_counter = nullptr;
  AllocateMemory(sizeof(CudaAtomicSize), &device_modified_counter);
  MS_ERROR_IF_NULL(device_modified_counter);

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(device_modified_counter, &host_modified_counter,
                                                   sizeof(CudaAtomicSize), cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync");

  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  // Note: size of dynamic_map maybe greater than size_.
  auto size = dynamic_map.get_size();
  Key *all_keys = nullptr;
  int *all_indices = nullptr;
  AllocateMemory(size * sizeof(Key), &all_keys);
  AllocateMemory(size * sizeof(int), &all_indices);
  MS_ERROR_IF_NULL(all_keys);
  MS_ERROR_IF_NULL(all_indices);

  // 2. Export all keys and indices from dynamic map.
  RETURN_IF_FALSE_WITH_LOG(dynamic_map.get_keys_values(all_keys, all_indices, stream),
                           "Get keys and values from cuda dynamic map failed.");

  // 3. Find all keys and indices for modified elememts in hash table.
  FindModifiedKeysAndIndices<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(
    size, elements_per_block_, all_keys, all_indices, statuses_ptr_, device_modified_counter, modified_keys,
    modified_indices);

  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");
  FreeMemory(all_keys);
  FreeMemory(all_indices);
  FreeMemory(device_modified_counter);

  return true;
}

template <typename Key, typename Value, typename Allocator>
HashTableExportData GPUHashTable<Key, Value, Allocator>::ExportModifiedAndErasedElements(
  size_t modified_num, const Key *device_modified_keys, const Value *device_modified_values, cudaStream_t stream) {
  MS_EXCEPTION_IF_NULL(stream);

  // 1. Export the keys, values and statuses for elements which are modified.
  // The host buffer for modified keys, values and statuses.
  size_t modified_and_erased_num = modified_num + erased_keys_.size();
  auto host_keys = std::make_shared<std::vector<char>>(modified_and_erased_num * sizeof(Key));
  auto host_values = std::make_shared<std::vector<char>>(modified_num * value_dim_ * sizeof(Value));
  auto host_statuses = std::make_shared<std::vector<char>>(modified_and_erased_num * sizeof(Status));

  // Copy keys.
  CHECK_CUDA_RET_WITH_EXCEPTION(cudaMemcpyAsync(host_keys->data(), device_modified_keys, modified_num * sizeof(Key),
                                                cudaMemcpyDeviceToHost, stream),
                                "cudaMemcpyAsync");
  // Copy values.
  CHECK_CUDA_RET_WITH_EXCEPTION(
    cudaMemcpyAsync(host_values->data(), device_modified_values, modified_num * value_dim_ * sizeof(Value),
                    cudaMemcpyDeviceToHost, stream),
    "cudaMemcpyAsync");
  // Copy statuses.
  std::vector<Status> modified_status(modified_num, Status::kModified);
  auto ret =
    memcpy_s(host_statuses->data(), host_statuses->size(), modified_status.data(), modified_num * sizeof(Status));
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Memcpy for gpu hash table status failed, errno[" << ret << "]";
  }

  // 2. Export the keys and statuses for elements which are erased.
  if (!erased_keys_.empty()) {
    ret = memcpy_s(host_keys->data() + modified_num * sizeof(Key), host_keys->size() - modified_num * sizeof(Key),
                   erased_keys_.data(), erased_keys_.size() * sizeof(Key));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Memcpy for gpu hash table erased keys failed, errno[" << ret << "]";
    }

    std::vector<Status> erased_status(erased_keys_.size(), Status::kErased);
    ret = memcpy_s(host_statuses->data() + modified_num * sizeof(Status),
                   host_statuses->size() - modified_num * sizeof(Status), erased_status.data(),
                   erased_status.size() * sizeof(Status));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Memcpy for gpu hash table erased status failed, errno[" << ret << "]";
    }

    erased_keys_.clear();
  }

  CHECK_CUDA_RET_WITH_EXCEPTION(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");
  return {host_keys, host_values, host_statuses};
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::GetIndicesByKeys(const Key *key, size_t key_num, bool insert_miss_key,
                                                           int32_t *indices, cudaStream_t stream) {
  MS_ERROR_IF_NULL(key);
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(stream);
  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  if (insert_miss_key) {
    dynamic_map.reserve(key_num + dynamic_map.get_size());
  }

  size_t submap_idx = 0;
  uint32_t device_id = GET_CTX_DEVICE_ID;
  size_t remaining_key_num = key_num;
  MS_ERROR_IF_NULL(insert_success_number_);

  while (remaining_key_num > 0) {
    auto &submap_ptr = dynamic_map.get_submaps()[submap_idx];
    MS_ERROR_IF_NULL(submap_ptr);
    // 1. Get reamaining capacity in current submap, max load faltor and min insert size need to be considered.
    size_t submap_remaining_capacity =
      submap_ptr->get_capacity() * dynamic_map.get_max_load_factor() - submap_ptr->get_size();
    if (submap_remaining_capacity < dynamic_map.get_min_insert_size()) {
      submap_idx++;
      continue;
    }

    *(insert_success_number_) = 0;
    CHECK_CUDA_RET_WITH_RETURN_FALSE(
      cudaMemPrefetchAsync(insert_success_number_, sizeof(CudaAtomicSize), device_id, stream), "cudaMemPrefetchAsync");

    // 2. Get the key number could be handled by current submap.
    size_t item_num = std::min(submap_remaining_capacity, remaining_key_num);
    const uint32_t tile_size = kTileSize;
    const uint32_t block_size = kBlockSize;
    if (IntToUint(GET_THREADS) < block_size) {
      MS_LOG(ERROR) << "The max thread per block is less than: " << block_size << " of this GPU";
      return false;
    }
    const uint32_t grid_size = IntToUint(CUDA_BLOCKS_CAL(device_id, tile_size * item_num, block_size));

    // 3. Transform all keys into indices in blocks. If the key exist in map already, just return the index,
    // otherwise find a valid position in block.
    LookupIndices<block_size, tile_size, Key, typename CucoDynamicMap<Key, int32_t, Allocator>::mutable_view_type,
                  typename CucoDynamicMap<Key, int32_t, Allocator>::view_type><<<grid_size, block_size, 0, stream>>>(
      key, item_num, insert_miss_key, dynamic_map.get_submaps().size(), submap_idx, erased_slot_, erased_counter_,
      dynamic_map.get_submap_mutable_views().data().get(), dynamic_map.get_submap_views().data().get(),
      insert_success_number_, current_index_, indices);

    // 4. Update size for dynamic map and static submap.
    CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    size_t insert_success_num = insert_success_number_->load(cuda::std::memory_order_relaxed);
    dynamic_map.update_submap_size(submap_idx, submap_ptr->get_size() + insert_success_num);
    dynamic_map.update_size(dynamic_map.get_size() + insert_success_num);

    indices += item_num;
    key += item_num;
    remaining_key_num -= item_num;
    submap_idx++;
  }
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::UpdateSize(size_t key_num, const int *indices, cudaStream_t stream,
                                                     bool update_lookup_count) {
  size_t old_size = size_;
  // If permit_threshold_ is less than or equal to kMinPermitThreshold, permission is disable.
  if (permit_threshold_ <= kMinPermitThreshold) {
    MS_EXCEPTION_IF_NULL(cuda_dynamic_map_);
    auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
    size_ = dynamic_map.get_size();
    if (size_ != old_size) {
      is_dirty_ = true;
    }
    return true;
  }

  if (!update_lookup_count) {
    // Note: if need not to update lookup count and permission is enable, the size of gpu hash table could not change.
    // For example, all keys for `Insert` should be contained in gpu hash table already.
    return true;
  }

  // Count the number for new permission elements.
  MS_EXCEPTION_IF_NULL(stream);
  MS_EXCEPTION_IF_NULL(insert_success_number_);
  *(insert_success_number_) = 0;
  uint32_t device_id = GET_CTX_DEVICE_ID;
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemPrefetchAsync(insert_success_number_, sizeof(CudaAtomicSize), device_id, stream), "cudaMemPrefetchAsync");

  const uint32_t block_size = kBlockSize;
  if (IntToUint(GET_THREADS) < block_size) {
    MS_LOG(ERROR) << "The max thread per block is less than: " << block_size << " of this GPU";
    return false;
  }
  const uint32_t grid_size = IntToUint(CUDA_BLOCKS_CAL(device_id, key_num, block_size));

  // Launch kernel to count new permitted elements number.
  CountPermissionNum<block_size><<<grid_size, block_size, 0, stream>>>(
    elements_per_block_, key_num, indices, lookup_cnts_ptr_, permit_threshold_, insert_success_number_);
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  // Update hash table size.
  size_t insert_success_num = insert_success_number_->load(cuda::std::memory_order_relaxed);
  size_ += insert_success_num;
  if (size_ != old_size) {
    is_dirty_ = true;
  }

  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::InsertDefaultValueByInitializer(size_t key_num,
                                                                          const std::string &initializer,
                                                                          const int *indices, cudaStream_t stream) {
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(stream);
  if (initializer == kNormalDistribution) {
    // Normal distribution.
    RETURN_IF_FALSE_WITH_LOG(InitNormalDistRandomGenerator(stream),
                             "Initialize normal distribution random generator failed.");
    Value mean = static_cast<Value>(0);
    Value stddev = static_cast<Value>(0.01);

    InsertNormalDistRandomValue<<<random_gen_block_count_, random_gen_threads_per_block_, 0, stream>>>(
      value_dim_, key_num, indices, elements_per_block_, lookup_cnts_ptr_, permit_threshold_, mean, stddev,
      random_gen_state_, idle_flags_ptr_, blocks_ptr_);
  } else if (initializer == kOnesDistribution) {
    // One distribution.
    InsertDefaultValue<<<GET_BLOCKS(key_num), GET_THREADS, 0, stream>>>(
      value_dim_, key_num, indices, elements_per_block_, lookup_cnts_ptr_, permit_threshold_, static_cast<Value>(1.0),
      idle_flags_ptr_, blocks_ptr_);
  } else if (initializer == kZerosDistribution) {
    // Zero distribution.
    InsertDefaultValue<<<GET_BLOCKS(key_num), GET_THREADS, 0, stream>>>(
      value_dim_, key_num, indices, elements_per_block_, lookup_cnts_ptr_, permit_threshold_, static_cast<Value>(0),
      idle_flags_ptr_, blocks_ptr_);
  } else {
    MS_LOG(ERROR) << "Unsupported initializer: " << initializer;
    return false;
  }

  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::InitNormalDistRandomGenerator(cudaStream_t stream) {
  MS_ERROR_IF_NULL(stream);
  if (random_gen_init_.load()) {
    return true;
  }

  // 1. Allocate memory for all random generator states.
  auto total_random_state_num = random_gen_threads_per_block_ * random_gen_block_count_;
  AllocateMemory(IntToSize(total_random_state_num) * sizeof(curandStatePhilox4_32_10_t), &random_gen_state_);
  MS_ERROR_IF_NULL(random_gen_state_);

  // 2. Initialize normal distribution random generator states.
  std::random_device rd;
  uint32_t seed = rd();
  InitNormalDisRandomGen<<<random_gen_block_count_, random_gen_threads_per_block_, 0, stream>>>(seed,
                                                                                                random_gen_state_);

  random_gen_init_ = true;
  return true;
}

template class GPUHashTable<int32_t, float>;
template class GPUHashTable<int64_t, float>;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif
