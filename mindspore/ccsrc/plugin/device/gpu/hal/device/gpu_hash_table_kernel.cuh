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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_KERNEL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_KERNEL_CUH_

#if CUDA_VERSION > 11000
#include <cuco/dynamic_map.cuh>
#include <curand_kernel.h>
#include "plugin/device/gpu/hal/device/gpu_hash_table_common.h"

namespace mindspore {
namespace device {
namespace gpu {
namespace cg = cooperative_groups;

// Check whether the key exist in map already.
template <typename CG, typename Key, typename View>
__device__ __forceinline__ void CheckKeyExist(const CG &g, const Key &key, View *submap_views, size_t submaps_num,
                                              int32_t *index_in_block) {
  for (size_t i = 0; i < submaps_num; ++i) {
    auto &submap_view = submap_views[i];
    auto iter = submap_view.find(g, key);
    if (iter != submap_view.end()) {
      *index_in_block = iter->second;
      break;
    }
  }
}

// Get a valid position in block and return the offset index.
template <typename CG>
__device__ __forceinline__ int32_t GetInsertIndex(const CG &g, const int32_t *erased_slot,
                                                  CudaAtomicInt *erased_counter, CudaAtomicSize *current_index) {
  int32_t candidate_index = 0;
  if (g.thread_rank() == 0) {
    if (erased_counter->load(cuda::std::memory_order_relaxed) > 0) {
      // Idle slot position is preferred.
      int32_t idle_index = erased_counter->fetch_sub(1, cuda::std::memory_order_relaxed) - 1;
      // Idle slot position compete fail.
      if (idle_index < 0) {
        erased_counter->store(0, cuda::std::memory_order_relaxed);
        candidate_index = current_index->fetch_add(1, cuda::std::memory_order_relaxed);
      } else {
        candidate_index = erased_slot[idle_index];
      }
    } else {
      // If idle slot is empty, use new position in blocks.
      candidate_index = current_index->fetch_add(1, cuda::std::memory_order_relaxed);
    }
  }
  // Sync index in block in cooperative group.
  int32_t index_in_block = g.shfl(candidate_index, 0);
  return index_in_block;
}

// Transform all keys into indices in blocks. If the key exist in map already ,just return the index,
// otherwise find a valid position in block.
template <uint32_t block_size, uint32_t tile_size, typename Key, typename MutableView, typename View>
__global__ void LookupIndices(const Key *keys, size_t key_num, bool insert_miss_key, size_t submaps_num,
                              size_t submap_idx, const int32_t *erased_slot, CudaAtomicInt *erased_counter,
                              MutableView *submap_mutable_views, View *submap_views, CudaAtomicSize *insert_success_num,
                              CudaAtomicSize *current_index, int32_t *indices) {
  typedef cub::BlockReduce<size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_t insert_success_num_per_thread = 0;

  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  size_t key_idx = global_thread_index / tile_size;
  int32_t empty_key = submap_views[0].get_empty_key_sentinel();
  int32_t empty_value = submap_views[0].get_empty_value_sentinel();

  while (key_idx < key_num) {
    Key key = keys[key_idx];
    CUDA_KERNEL_ASSERT(key != empty_key);
    int32_t index_in_block = empty_value;
    // 1. Check whether the key exist in map already.
    CheckKeyExist(tile, key, submap_views, submaps_num, &index_in_block);

    // 2. Handle the key doesn't exist in map.
    if (index_in_block == empty_value && insert_miss_key) {
      index_in_block = GetInsertIndex(tile, erased_slot, erased_counter, current_index);
      bool ret = submap_mutable_views[submap_idx].insert(tile, cuco::pair_type<Key, int32_t>{key, index_in_block});
      CUDA_KERNEL_ASSERT(ret);
      if (tile.thread_rank() == 0) {
        insert_success_num_per_thread++;
      }
    }

    CUDA_KERNEL_ASSERT(index_in_block != empty_value);
    CUDA_KERNEL_ASSERT(index_in_block >= 0);
    if (tile.thread_rank() == 0) {
      // 3. Update the final index.
      *(indices + key_idx) = index_in_block;
    }
    key_idx += (gridDim.x * blockDim.x) / tile_size;
  }

  // 4. Count successfully inserting number.
  size_t insert_success_num_per_block = BlockReduce(temp_storage).Sum(insert_success_num_per_thread);
  if (threadIdx.x == 0) {
    *insert_success_num += insert_success_num_per_block;
  }
}

// Initialize normal distribution random generator states.
__global__ void InitNormalDisRandomGen(uint32_t seed, curandStatePhilox4_32_10_t *state) {
  int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, global_thread_index, 0, &state[global_thread_index]);
}

// Insert default normal distribution random value.
template <typename Value>
__global__ void InsertNormalDistRandomValue(size_t value_dim, size_t total_insert_elements_num, const int *indices,
                                            size_t elements_per_block, size_t *const *lookup_cnts_ptr,
                                            size_t permit_threshold, const Value mean, const Value stddev,
                                            curandStatePhilox4_32_10_t *state, bool **idle_flags_ptr,
                                            Value *const *blocks_ptr) {
  size_t total_thread_num = blockDim.x * gridDim.x;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_insert_elements_num; pos += total_thread_num) {
    CUDA_KERNEL_ASSERT(indices[pos] >= 0);

    const size_t block_idx = indices[pos] / elements_per_block;
    const size_t offset_in_block = indices[pos] % elements_per_block;

    bool enable_permission = permit_threshold > kMinPermitThreshold;
    bool meet_permit_cond = lookup_cnts_ptr[block_idx][offset_in_block] == permit_threshold;
    // Need not to initialize if the current slot is not idle.
    if (!idle_flags_ptr[block_idx][offset_in_block]) {
      continue;
    }
    // Need not to initialize if the current slot does not meet the permission condition.
    if (enable_permission && !meet_permit_cond) {
      continue;
    }

    const size_t current_pos_in_block = offset_in_block * value_dim;
    Value *element_ptr = &blocks_ptr[block_idx][current_pos_in_block];

    if (enable_permission && !meet_permit_cond) {
      for (size_t i = 0; i < value_dim; i++) {
        // Initialize by zero for elements which do not meet the permission condition.
        element_ptr[i] = 0;
      }
      continue;
    }

    // 1. Copy state to local memory for performance.
    curandStatePhilox4_32_10_t localState = state[pos % total_thread_num];

    float2 x;
    size_t mod = value_dim % 2;
    // Note: could use tile or block to parallel.
    for (size_t i = 0; i < value_dim - mod; i += 2) {
      // 2. Genetate two random number once.
      x = curand_normal2(&localState);
      element_ptr[i] = (Value)(x.x) * stddev + mean;
      element_ptr[i + 1] = (Value)(x.y) * stddev + mean;
    }

    // 3. Handle the last number.
    if (mod != 0) {
      x = curand_normal2(&localState);
      element_ptr[value_dim - 1] = (Value)x.x * stddev + mean;
    }

    // 4. Update latest random state back to global memory.
    state[pos % total_thread_num] = localState;
    idle_flags_ptr[block_idx][offset_in_block] = false;
  }
}

// Insert default value into map by specific value.
template <typename Value>
__global__ void InsertDefaultValue(size_t value_dim, size_t total_insert_elements_num, const int *indices,
                                   size_t elements_per_block, size_t *const *lookup_cnts_ptr, size_t permit_threshold,
                                   const Value default_value, bool **idle_flags_ptr, Value *const *blocks_ptr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_insert_elements_num;
       pos += blockDim.x * gridDim.x) {
    CUDA_KERNEL_ASSERT(indices[pos] >= 0);
    const size_t block_idx = indices[pos] / elements_per_block;
    const size_t offset_in_block = indices[pos] % elements_per_block;

    bool enable_permission = permit_threshold > kMinPermitThreshold;
    bool meet_permit_cond = lookup_cnts_ptr[block_idx][offset_in_block] == permit_threshold;
    // Need not to initialize if the current slot is not idle.
    if (!idle_flags_ptr[block_idx][offset_in_block]) {
      continue;
    }

    // Need not to initialize if the current slot does not meet the permission condition.
    if (enable_permission && !meet_permit_cond) {
      continue;
    }

    const size_t current_pos_in_block = offset_in_block * value_dim;
    Value *element_ptr = &blocks_ptr[block_idx][current_pos_in_block];

    // Note: could use tile or block to parallel.
    if (enable_permission && !meet_permit_cond) {
      for (size_t i = 0; i < value_dim; i++) {
        // Initialize by zero for elements which do not meet the permission condition.
        element_ptr[i] = 0;
      }
    } else {
      for (size_t i = 0; i < value_dim; i++) {
        element_ptr[i] = default_value;
      }
    }

    idle_flags_ptr[block_idx][offset_in_block] = false;
  }
}

// Get all values by indices in blocks.
template <typename Value>
__global__ void GetValues(size_t value_dim, size_t total_size, const int *indices, const size_t elements_per_block,
                          Value *const *blocks_ptr, Value *outputs) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / value_dim;
    const size_t offset = pos % value_dim;
    CUDA_KERNEL_ASSERT(indices[index] >= 0);

    const size_t block_idx = indices[index] / elements_per_block;
    const size_t offset_in_block = indices[index] % elements_per_block;

    const size_t current_pos_in_block = offset_in_block * value_dim + offset;
    outputs[pos] = blocks_ptr[block_idx][current_pos_in_block];
  }
}

// Insert values into map by indices in blocks.
template <typename Value>
__global__ void InsertValues(size_t value_dim, size_t total_insert_size, const int *indices, const Value *insert_values,
                             const size_t elements_per_block, const size_t *const *lookup_cnts_ptr,
                             size_t permit_threshold, size_t global_timestamp, size_t *const *update_timestamps_ptr,
                             Status *const *statuses_ptr, bool *const *idle_flags_ptr, Value *const *blocks_ptr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_insert_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / value_dim;
    const size_t offset = pos % value_dim;
    CUDA_KERNEL_ASSERT(indices[index] >= 0);

    const size_t block_idx = indices[index] / elements_per_block;
    const size_t offset_in_block = indices[index] % elements_per_block;

    if (permit_threshold > kMinPermitThreshold && lookup_cnts_ptr[block_idx][offset_in_block] < permit_threshold) {
      continue;
    }

    const size_t current_pos_in_block = offset_in_block * value_dim + offset;
    blocks_ptr[block_idx][current_pos_in_block] = insert_values[pos];

    // Update idle status.
    if (pos % value_dim == 0 && idle_flags_ptr[block_idx][offset_in_block]) {
      idle_flags_ptr[block_idx][offset_in_block] = false;
    }

    // Update status to kModified.
    if (pos % value_dim == 0) {
      statuses_ptr[block_idx][offset_in_block] = Status::kModified;
    }

    // Update timestamp.
    if (pos % value_dim == 0) {
      update_timestamps_ptr[block_idx][offset_in_block] = global_timestamp;
    }
  }
}

template <uint32_t block_size>
__global__ void CountPermissionNum(size_t elements_per_block, size_t key_num, const int *indices,
                                   size_t *const *lookup_cnts_ptr, size_t permit_threshold,
                                   CudaAtomicSize *insert_counter) {
  typedef cub::BlockReduce<size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_t local_counter = 0;

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < key_num; pos += blockDim.x * gridDim.x) {
    CUDA_KERNEL_ASSERT(indices[pos] >= 0);
    const size_t block_idx = indices[pos] / elements_per_block;
    const size_t offset_in_block = indices[pos] % elements_per_block;
    lookup_cnts_ptr[block_idx][offset_in_block] += 1;

    if (lookup_cnts_ptr[block_idx][offset_in_block] == permit_threshold) {
      // Insert a new element.
      local_counter++;
    }
  }

  size_t insert_num_per_block = BlockReduce(temp_storage).Sum(local_counter);
  if (threadIdx.x == 0) {
    *insert_counter += insert_num_per_block;
  }
}

template <uint32_t block_size>
__global__ void CountExpiredNum(size_t blocks_num, size_t permit_threshold, size_t global_timestamp,
                                uint64_t evict_threshold, size_t elements_per_block, bool *const *idle_flags_ptr,
                                size_t *const *lookup_cnts_ptr, size_t *const *update_timestamps_ptr,
                                CudaAtomicSize *expired_counter) {
  typedef cub::BlockReduce<size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_t local_expired_counter = 0;

  for (size_t i = 0; i < blocks_num; i++) {
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < elements_per_block; pos += blockDim.x * gridDim.x) {
      // Ignore elements at idle slot or ones that have not been really inserted into hash table yet.
      if (idle_flags_ptr[i][pos] || lookup_cnts_ptr[i][pos] < permit_threshold) {
        continue;
      }

      // Counter expired element.
      size_t timestamp = update_timestamps_ptr[i][pos];
      if (global_timestamp > timestamp && (global_timestamp - timestamp) > evict_threshold) {
        local_expired_counter++;
      }
    }
  }

  size_t expired_num_per_block = BlockReduce(temp_storage).Sum(local_expired_counter);
  if (threadIdx.x == 0) {
    *expired_counter += expired_num_per_block;
  }
}

template <typename Key>
__global__ void FindExpiredKeysAndIndices(size_t key_num, size_t elements_per_block, size_t permit_threshold,
                                          size_t global_timestamp, uint64_t evict_threshold,
                                          bool *const *idle_flags_ptr, size_t *const *lookup_cnts_ptr,
                                          size_t *const *update_timestamps_ptr, const Key *all_keys,
                                          const int *all_indices, CudaAtomicSize *expired_counter, Key *expired_keys,
                                          int *expired_indices) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < key_num; pos += blockDim.x * gridDim.x) {
    Key key = all_keys[pos];
    int index = all_indices[pos];

    const size_t block_idx = index / elements_per_block;
    const size_t offset_in_block = index % elements_per_block;

    // Ignore elements at idle slot or ones that have not been really inserted into hash table yet.
    if (idle_flags_ptr[block_idx][offset_in_block] || lookup_cnts_ptr[block_idx][offset_in_block] < permit_threshold) {
      continue;
    }

    // Record expired keys and indices.
    size_t timestamp = update_timestamps_ptr[block_idx][offset_in_block];
    if (global_timestamp > timestamp && (global_timestamp - timestamp) > evict_threshold) {
      size_t expired_index = expired_counter->fetch_add(1, cuda::std::memory_order_relaxed);
      expired_keys[expired_index] = key;
      expired_indices[expired_index] = index;
    }
  }
}

// Erase elements in hash map, update idle status for erased slots.
__global__ void EraseElementsByIndices(size_t erase_num, size_t elements_per_block, const int *erased_indices,
                                       bool *const *idle_flags_ptr, Status *const *statuses_ptr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < erase_num; pos += blockDim.x * gridDim.x) {
    CUDA_KERNEL_ASSERT(erased_indices[pos] >= 0);

    const size_t block_idx = erased_indices[pos] / elements_per_block;
    const size_t offset_in_block = erased_indices[pos] % elements_per_block;
    idle_flags_ptr[block_idx][offset_in_block] = true;

    statuses_ptr[block_idx][offset_in_block] = Status::kErased;
  }
}

__global__ void AddErasedSlots(size_t erased_num, const int *erased_indices, CudaAtomicInt *erased_counter,
                               int *erased_slot) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < erased_num; pos += blockDim.x * gridDim.x) {
    CUDA_KERNEL_ASSERT(erased_indices[pos] >= 0);

    erased_slot[erased_counter->fetch_add(1, cuda::std::memory_order_relaxed)] = erased_indices[pos];
  }
}

// Update status of element in hash table.
__global__ void UpdateStatus(size_t key_num, size_t elements_per_block, int empty_index, const int *indices,
                             Status new_status, Status *const *statuses_ptr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < key_num; pos += blockDim.x * gridDim.x) {
    // Ignore Elements which do not exist in hash map.
    if (indices[pos] == empty_index) {
      continue;
    }

    const size_t block_idx = indices[pos] / elements_per_block;
    const size_t offset_in_block = indices[pos] % elements_per_block;

    statuses_ptr[block_idx][offset_in_block] = new_status;
  }
}

// Count the number of element whose statuses are modified.
template <uint32_t block_size>
__global__ void CountModifiedNum(size_t blocks_num, size_t elements_per_block, const Status *const *statuses_ptr,
                                 CudaAtomicSize *modified_counter) {
  typedef cub::BlockReduce<size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_t local_modified_counter = 0;

  for (size_t i = 0; i < blocks_num; i++) {
    for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < elements_per_block; pos += blockDim.x * gridDim.x) {
      // Counter modified element.
      if (statuses_ptr[i][pos] == Status::kModified) {
        local_modified_counter++;
      }
    }
  }

  size_t modified_num_per_block = BlockReduce(temp_storage).Sum(local_modified_counter);
  if (threadIdx.x == 0) {
    *modified_counter += modified_num_per_block;
  }
}

// Find all keys and indices for elememts whose statuses are modified.
template <typename Key>
__global__ void FindModifiedKeysAndIndices(size_t key_num, size_t elements_per_block, const Key *all_keys,
                                           const int *all_indices, const Status *const *statuses_ptr,
                                           CudaAtomicSize *modified_counter, Key *modified_keys,
                                           int *modified_indices) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < key_num; pos += blockDim.x * gridDim.x) {
    Key key = all_keys[pos];
    int index = all_indices[pos];

    const size_t block_idx = index / elements_per_block;
    const size_t offset_in_block = index % elements_per_block;

    // Record modified keys and indices.
    if (statuses_ptr[block_idx][offset_in_block] == Status::kModified) {
      size_t modified_index = modified_counter->fetch_add(1, cuda::std::memory_order_relaxed);
      modified_keys[modified_index] = key;
      modified_indices[modified_index] = index;
    }
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_KERNEL_CUH_
