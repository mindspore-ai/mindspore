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

#if CUDA_VERSION > 11000
#include "plugin/device/gpu/hal/device/gpu_hash_table.h"

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
#define CHECK_CUDA_RET(expression, message)                                                \
  {                                                                                        \
    cudaError_t cuda_ret = (expression);                                                   \
    if (cuda_ret != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << cuda_ret << " " \
                    << cudaGetErrorString(cuda_ret);                                       \
    }                                                                                      \
  }

#define CHECK_CUDA_RET_WITH_RETURN_FALSE(expression, message)                              \
  {                                                                                        \
    cudaError_t cuda_ret = (expression);                                                   \
    if (cuda_ret != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << cuda_ret << " " \
                    << cudaGetErrorString(cuda_ret);                                       \
      return false;                                                                        \
    }                                                                                      \
  }

#define ASSERT_EQUAL(lhs, rhs, message) \
  {                                     \
    if ((lhs) != (rhs)) {               \
      MS_LOG(ERROR) << message;         \
      return false;                     \
    }                                   \
  }

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
GPUHashTable<Key, Value, Allocator>::GPUHashTable(int32_t value_dim, const std::string &initializer,
                                                  const Allocator &alloc)
    : value_dim_(value_dim), initializer_(initializer), default_value_(0), index_alloc_(alloc), char_alloc_(alloc) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  cuda_dynamic_map_ = std::make_unique<CudaDynamicMap<Key, int32_t, Allocator>>(static_cast<Key>(-1), -1,
                                                                                static_cast<Key>(-2), alloc, stream);
  CHECK_CUDA_RET(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  CHECK_CUDA_RET(
    cudaMallocManaged(&insert_success_number_, sizeof(cuda::atomic<std::size_t, cuda::thread_scope_device>)),
    "cudaMallocManaged");
}

template <typename Key, typename Value, typename Allocator>
GPUHashTable<Key, Value, Allocator>::GPUHashTable(int32_t value_dim, const Value &default_value, const Allocator &alloc)
    : value_dim_(value_dim), initializer_(""), default_value_(default_value), index_alloc_(alloc), char_alloc_(alloc) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  cuda_dynamic_map_ = std::make_unique<CudaDynamicMap<Key, int32_t, Allocator>>(static_cast<Key>(-1), -1,
                                                                                static_cast<Key>(-2), alloc, stream);
  CHECK_CUDA_RET(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  CHECK_CUDA_RET(
    cudaMallocManaged(&insert_success_number_, sizeof(cuda::atomic<std::size_t, cuda::thread_scope_device>)),
    "cudaMallocManaged");
}

template <typename Key, typename Value, typename Allocator>
GPUHashTable<Key, Value, Allocator>::~GPUHashTable() {
  CHECK_CUDA_RET(cudaFree(insert_success_number_), "cudaFree");
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *keys, size_t key_num, Value *outputs, void *stream) {
  if (!initializer_.empty()) {
    return Find(keys, key_num, initializer_, outputs, stream);
  }
  return Find(keys, key_num, default_value_, outputs, stream);
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *key, size_t key_num, const std::string &initializer,
                                               Value *outputs, void *stream) {
  MS_ERROR_IF_NULL(key);
  MS_ERROR_IF_NULL(outputs);
  MS_ERROR_IF_NULL(stream);
  int *indices = std::allocator_traits<IndexAllocatorType>::allocate(index_alloc_, key_num);
  MS_ERROR_IF_NULL(indices);

  // 1. Get all indices in blocks according to the key.
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  RETURN_IF_FALSE_WITH_LOG(GetIndicesByKeys(key, key_num, true, indices, cuda_stream), "Get indices by keys failed.");

  Reserve(size_ + key_num);
  // 2. Insert default value according to initializer, initializer can be 'normal', 'zeros' or 'ones'.
  RETURN_IF_FALSE_WITH_LOG(InsertDefaultValueByInitializer(key_num, initializer, indices, cuda_stream),
                           "Insert default value for miss keys failed.");

  // 3. Get all values by indices in blocks.
  size_t total_size = value_dim_ * key_num;
  GetValues<<<GET_BLOCKS(total_size), GET_THREADS, 0, cuda_stream>>>(value_dim_, total_size, indices,
                                                                     elements_per_block_, blocks_ptr_, outputs);
  std::allocator_traits<IndexAllocatorType>::deallocate(index_alloc_, indices, key_num);
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Find(const Key *key, size_t key_num, const Value &default_value,
                                               Value *outputs, void *stream) {
  MS_ERROR_IF_NULL(key);
  MS_ERROR_IF_NULL(outputs);
  MS_ERROR_IF_NULL(stream);
  int *indices = std::allocator_traits<IndexAllocatorType>::allocate(index_alloc_, key_num);
  MS_ERROR_IF_NULL(indices);

  // 1. Get all indices in blocks according to the key.
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  RETURN_IF_FALSE_WITH_LOG(GetIndicesByKeys(key, key_num, true, indices, cuda_stream), "Get indices by keys failed.");

  Reserve(size_ + key_num);
  // 2. Insert default value into map by specific value.
  InsertDefaultValue<<<GET_BLOCKS(key_num), GET_THREADS, 0, cuda_stream>>>(
    value_dim_, key_num, indices, elements_per_block_, default_value, idle_flags_ptr_, blocks_ptr_);

  // 3. Get all values by indices in blocks.
  size_t total_size = value_dim_ * key_num;
  GetValues<<<GET_BLOCKS(total_size), GET_THREADS, 0, cuda_stream>>>(value_dim_, total_size, indices,
                                                                     elements_per_block_, blocks_ptr_, outputs);
  std::allocator_traits<IndexAllocatorType>::deallocate(index_alloc_, indices, key_num);
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Insert(const Key *key, size_t key_num, const Value *value, void *stream) {
  MS_ERROR_IF_NULL(key);
  MS_ERROR_IF_NULL(value);
  MS_ERROR_IF_NULL(stream);
  int *indices = std::allocator_traits<IndexAllocatorType>::allocate(index_alloc_, key_num);
  MS_ERROR_IF_NULL(indices);

  // 1. Get all indices in blocks according to the key.
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  RETURN_IF_FALSE_WITH_LOG(GetIndicesByKeys(key, key_num, true, indices, cuda_stream), "Get indices by keys failed.");

  Reserve(size_ + key_num);
  // 2. Insert values into map by indices in blocks.
  size_t total_insert_size = value_dim_ * key_num;
  InsertValues<<<GET_BLOCKS(total_insert_size), GET_THREADS, 0, cuda_stream>>>(
    value_dim_, total_insert_size, indices, value, elements_per_block_, idle_flags_ptr_, blocks_ptr_);

  std::allocator_traits<IndexAllocatorType>::deallocate(index_alloc_, indices, key_num);
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Erase(const Key *keys, size_t key_num, void *stream) {
  MS_ERROR_IF_NULL(keys);
  MS_ERROR_IF_NULL(stream);
  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  dynamic_map.erase(keys, keys + key_num, reinterpret_cast<cudaStream_t>(stream));
  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::GetKeysAndValues(Key *keys, Value *values, void *stream) {
  MS_ERROR_IF_NULL(keys);
  MS_ERROR_IF_NULL(values);
  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  int *indices = std::allocator_traits<IndexAllocatorType>::allocate(index_alloc_, size_);
  MS_ERROR_IF_NULL(indices);

  // 1. Export all keys and indices from dynamic map.
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  RETURN_IF_FALSE_WITH_LOG(dynamic_map.get_keys_values(keys, indices, cuda_stream),
                           "Get keys and values from cuda dynamic map failed.");

  // 2. Get all values by indices in blocks.
  size_t total_size = value_dim_ * size_;
  GetValues<<<GET_BLOCKS(total_size), GET_THREADS, 0, cuda_stream>>>(value_dim_, total_size, indices,
                                                                     elements_per_block_, blocks_ptr_, values);
  std::allocator_traits<IndexAllocatorType>::deallocate(index_alloc_, indices, size_);
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
  char *device_keys = std::allocator_traits<CharAllocatorType>::allocate(char_alloc_, keys_len);
  char *device_values = std::allocator_traits<CharAllocatorType>::allocate(char_alloc_, values_len);
  MS_ERROR_IF_NULL(device_keys);
  MS_ERROR_IF_NULL(device_values);

  // 3. Copy input keys and values to device.
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(device_keys, host_keys, keys_len, cudaMemcpyHostToDevice, stream),
                                   "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(device_values, host_values, values_len, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync");

  // 4. Insert input keys and values to hash table.
  RETURN_IF_FALSE_WITH_LOG(Insert(reinterpret_cast<Key *>(device_keys), keys_len / sizeof(Key),
                                  reinterpret_cast<Value *>(device_values), stream),
                           "Insert keys and values failed.");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  // 5. Free temp buffer to keys and values.
  std::allocator_traits<CharAllocatorType>::deallocate(char_alloc_, device_keys, keys_len);
  std::allocator_traits<CharAllocatorType>::deallocate(char_alloc_, device_values, values_len);
  input_data_list.clear();

  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::Export(const DataLenPair &keys, const DataLenPair &values,
                                                 const DataLenPair &status) {
  MS_ERROR_IF_NULL(keys.first);
  MS_ERROR_IF_NULL(values.first);
  MS_ERROR_IF_NULL(status.first);

  size_t keys_len = size_ * sizeof(Key);
  size_t values_len = size_ * value_dim_ * sizeof(Value);
  size_t status_len = size_ * sizeof(Status);
  // 1. Check length for output tensor.
  ASSERT_EQUAL(
    keys_len, keys.second,
    std::string("Need keys len[") + std::to_string(keys_len) + "], but got:[" + std::to_string(keys.second) + "].");
  ASSERT_EQUAL(values_len, values.second,
               std::string("Need values len[") + std::to_string(values_len) + "], but got:[" +
                 std::to_string(values.second) + "].");
  ASSERT_EQUAL(status_len, status.second,
               std::string("Need status len[") + std::to_string(status_len) + "], but got:[" +
                 std::to_string(status.second) + "].");

  // 2. Allocate temp buffer to keys, values and status.
  char *device_keys = std::allocator_traits<CharAllocatorType>::allocate(char_alloc_, keys_len);
  char *device_values = std::allocator_traits<CharAllocatorType>::allocate(char_alloc_, values_len);
  char *device_status = std::allocator_traits<CharAllocatorType>::allocate(char_alloc_, status_len);
  MS_ERROR_IF_NULL(device_keys);
  MS_ERROR_IF_NULL(device_values);
  MS_ERROR_IF_NULL(device_status);

  // 3. Export all keys and indices and store into temp buffer.
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(GPUDeviceManager::GetInstance().default_stream());
  RETURN_IF_FALSE_WITH_LOG(
    GetKeysAndValues(reinterpret_cast<Key *>(device_keys), reinterpret_cast<Value *>(device_values), stream),
    "Get keys and values failed.");

  // Note: Get all status.
  // 4. Copy keys, values and status from device temp buffer to host.
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemcpyAsync(keys.first, device_keys, keys_len, cudaMemcpyDeviceToHost, stream),
                                   "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(values.first, device_values, values_len, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(
    cudaMemcpyAsync(status.first, device_status, status_len, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");
  CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize default cuda stream");

  // 5. Free temp buffer to keys, values and status.
  std::allocator_traits<CharAllocatorType>::deallocate(char_alloc_, device_keys, keys_len);
  std::allocator_traits<CharAllocatorType>::deallocate(char_alloc_, device_values, values_len);
  std::allocator_traits<CharAllocatorType>::deallocate(char_alloc_, device_status, status_len);

  return true;
}

template <typename Key, typename Value, typename Allocator>
bool GPUHashTable<Key, Value, Allocator>::GetIndicesByKeys(const Key *key, size_t key_num, bool insert_miss_key,
                                                           int32_t *indices, cudaStream_t stream) {
  MS_ERROR_IF_NULL(key);
  MS_ERROR_IF_NULL(indices);
  MS_ERROR_IF_NULL(stream);
  MS_ERROR_IF_NULL(cuda_dynamic_map_);
  auto &dynamic_map = cuda_dynamic_map_->dynamic_map_;
  dynamic_map.reserve(key_num + dynamic_map.get_size());
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
    CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaMemPrefetchAsync(insert_success_number_, sizeof(CudaAtomicSize), device_id),
                                     "cudaMemPrefetchAsync");

    // 2. Get the key number could be handled by current submap.
    size_t item_num = std::min(submap_remaining_capacity, remaining_key_num);
    const uint32_t tile_size = kTileSize;
    const uint32_t block_size = kBlockSize;
    if (IntToUint(GET_THREADS) < block_size) {
      MS_LOG(ERROR) << "The max thread per block is less than: " << block_size << " of this GPU";
    }
    const uint32_t grid_size = IntToUint(CUDA_BLOCKS_CAL(device_id, tile_size * item_num, block_size));

    // 3. Transform all keys into indices in blocks. If the key exist in map already ,just return the index,
    // otherwise find a valid position in block.
    LookupIndices<block_size, tile_size, Key, typename CucoDynamicMap<Key, int32_t, Allocator>::mutable_view_type,
                  typename CucoDynamicMap<Key, int32_t, Allocator>::view_type><<<grid_size, block_size, 0, stream>>>(
      key, item_num, insert_miss_key, dynamic_map.get_submaps().size(), submap_idx, idel_slot_, idle_index_,
      dynamic_map.get_submap_mutable_views().data().get(), dynamic_map.get_submap_views().data().get(),
      insert_success_number_, current_index_, indices);

    // 4. Update size for dynamic map and static submap.
    CHECK_CUDA_RET_WITH_RETURN_FALSE(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    size_t insert_success_num = insert_success_number_->load(cuda::std::memory_order_relaxed);
    dynamic_map.update_submap_size(submap_idx, submap_ptr->get_size() + insert_success_num);
    dynamic_map.update_size(dynamic_map.get_size() + insert_success_num);
    size_ += insert_success_num;

    indices += item_num;
    key += item_num;
    remaining_key_num -= item_num;
    submap_idx++;
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
      value_dim_, key_num, indices, elements_per_block_, mean, stddev, random_gen_state_, idle_flags_ptr_, blocks_ptr_);
  } else if (initializer == kOnesDistribution) {
    // One distribution.
    InsertDefaultValue<<<GET_BLOCKS(key_num), GET_THREADS, 0, stream>>>(
      value_dim_, key_num, indices, elements_per_block_, static_cast<Value>(1.0), idle_flags_ptr_, blocks_ptr_);
  } else if (initializer == kZerosDistribution) {
    // Zero distribution.
    InsertDefaultValue<<<GET_BLOCKS(key_num), GET_THREADS, 0, stream>>>(
      value_dim_, key_num, indices, elements_per_block_, static_cast<Value>(0), idle_flags_ptr_, blocks_ptr_);
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
  random_gen_state_ = reinterpret_cast<curandStatePhilox4_32_10_t *>(std::allocator_traits<CharAllocatorType>::allocate(
    char_alloc_, IntToSize(total_random_state_num) * sizeof(curandStatePhilox4_32_10_t)));
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
