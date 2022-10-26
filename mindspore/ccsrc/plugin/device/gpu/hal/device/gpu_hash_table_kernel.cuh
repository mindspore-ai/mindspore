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

namespace mindspore {
namespace device {
namespace gpu {
namespace cg = cooperative_groups;
using CudaAtomicSize = cuda::atomic<std::size_t, cuda::thread_scope_device>;

// Check whether the key exist in map already.
template <typename CG, typename Key, typename View>
__device__ __forceinline__ void CheckKeyExist(const CG &g, const Key &key, View *submap_views, size_t submaps_num,
                                              int32_t *index_in_block) {
  for (auto i = 0; i < submaps_num; ++i) {
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
__device__ __forceinline__ int32_t GetInsertIndex(const CG &g, const int32_t *idle_slot, CudaAtomicSize *idle_index,
                                                  CudaAtomicSize *current_index) {
  int32_t candidate_index;
  if (g.thread_rank() == 0) {
    if (idle_index->load(cuda::std::memory_order_relaxed) != 0) {
      // Idle slot position is preferred.
      candidate_index = idle_slot[idle_index->fetch_sub(1)];
    } else {
      // If idle slot is empty, use new position in blocks.
      candidate_index = current_index->fetch_add(1);
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
                              size_t submap_idx, const int32_t *idle_slot, CudaAtomicSize *idle_index,
                              MutableView *submap_mutable_views, View *submap_views, CudaAtomicSize *insert_success_num,
                              CudaAtomicSize *current_index, int32_t *indices) {
  typedef cub::BlockReduce<size_t, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  size_t insert_success_num_per_thread = 0;

  auto tile = cg::tiled_partition<tile_size>(cg::this_thread_block());
  auto global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;
  size_t key_idx = global_thread_index / tile_size;
  int32_t empty_value = submap_views[0].get_empty_value_sentinel();

  while (key_idx < key_num) {
    Key key = keys[key_idx];
    int32_t index_in_block = empty_value;
    // 1. Check whether the key exist in map already.
    CheckKeyExist(tile, key, submap_views, submaps_num, &index_in_block);

    // 2. Handle the key doesn't exist in map.
    if (index_in_block == empty_value && insert_miss_key) {
      index_in_block = GetInsertIndex(tile, idle_slot, idle_index, current_index);
      if (submap_mutable_views[submap_idx].insert(tile, cuco::pair_type<Key, int32_t>{key, index_in_block}) &&
          tile.thread_rank() == 0) {
        insert_success_num_per_thread++;
      }
    }

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
                                            size_t elements_per_block, const Value &mean, const Value &stddev,
                                            curandStatePhilox4_32_10_t *state, bool **idle_flags_ptr,
                                            Value *const *blocks_ptr) {
  size_t total_thread_num = blockDim.x * gridDim.x;
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_insert_elements_num; pos += total_thread_num) {
    const size_t block_idx = indices[pos] / elements_per_block;
    const size_t offset_in_block = indices[pos] % elements_per_block;
    if (idle_flags_ptr[block_idx][offset_in_block] != true) {
      return;
    }

    const size_t current_pos_in_block = offset_in_block * value_dim;
    Value *element_ptr = &blocks_ptr[block_idx][current_pos_in_block];

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
    state[pos] = localState;
    idle_flags_ptr[block_idx][offset_in_block] = false;
  }
}

// Insert default value into map by specific value.
template <typename Value>
__global__ void InsertDefaultValue(size_t value_dim, size_t total_insert_elements_num, const int *indices,
                                   size_t elements_per_block, const Value &default_value, bool **idle_flags_ptr,
                                   Value *const *blocks_ptr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_insert_elements_num;
       pos += blockDim.x * gridDim.x) {
    const size_t block_idx = indices[pos] / elements_per_block;
    const size_t offset_in_block = indices[pos] % elements_per_block;
    if (idle_flags_ptr[block_idx][offset_in_block] != true) {
      return;
    }

    const size_t current_pos_in_block = offset_in_block * value_dim;
    Value *element_ptr = &blocks_ptr[block_idx][current_pos_in_block];

    // Note: could use tile or block to parallel.
    for (size_t i = 0; i < value_dim; i++) {
      element_ptr[i] = default_value;
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
    if (indices[index] < 0) {
      continue;
    }

    const size_t block_idx = indices[index] / elements_per_block;
    const size_t offset_in_block = indices[index] % elements_per_block;

    const size_t current_pos_in_block = offset_in_block * value_dim + offset;
    outputs[pos] = blocks_ptr[block_idx][current_pos_in_block];
  }
}

// Insert values into map by indices in blocks.
template <typename Value>
__global__ void InsertValues(size_t value_dim, size_t total_insert_size, const int *indices, const Value *insert_values,
                             const size_t elements_per_block, bool **idle_flags_ptr, Value *const *blocks_ptr) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < total_insert_size; pos += blockDim.x * gridDim.x) {
    const size_t index = pos / value_dim;
    const size_t offset = pos % value_dim;
    if (indices[index] < 0) {
      continue;
    }
    const size_t block_idx = indices[index] / elements_per_block;
    const size_t offset_in_block = indices[index] % elements_per_block;

    const size_t current_pos_in_block = offset_in_block * value_dim + offset;
    blocks_ptr[block_idx][current_pos_in_block] = insert_values[pos];

    if (pos % value_dim == 0 && idle_flags_ptr[block_idx][offset_in_block]) {
      idle_flags_ptr[block_idx][offset_in_block] = false;
    }
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_KERNEL_CUH_
