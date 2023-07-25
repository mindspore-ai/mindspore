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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sort_fixed_size.cuh"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/strided_pointer.cuh"

#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 2048;
#endif

constexpr uint32_t CUDA_THREADS_PRE_BLOCK_FALLBACK = 256;

#define MS_MAX_THREAD_PER_BLOCK(val) (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) : CUDA_THREADS_PRE_BLOCK_FALLBACK)

// Maximum size per grid dimension that we assume (compute capability >= 2.0)
constexpr int64_t MAX_GRID_SIZE = 65535LL;

#define ceil_div(x, y) (((x) + (y)-1) / (y))

static bool GetGridFromTiles(int64_t grid_tiles, dim3 *grid) {
  if (grid_tiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    return false;
  }

  int64_t grid_x = grid_tiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : grid_tiles;
  int64_t grid_y = 1;
  int64_t grid_z = 1;

  if (grid_tiles > MAX_GRID_SIZE) {
    grid_tiles = ceil_div(grid_tiles, MAX_GRID_SIZE);
    grid_y = grid_tiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : grid_tiles;

    if (grid_tiles > MAX_GRID_SIZE) {
      grid_tiles = ceil_div(grid_tiles, MAX_GRID_SIZE);
      grid_z = grid_tiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : grid_tiles;
    }
  }

  *grid = dim3(grid_x, grid_y, grid_z);
  return true;
}

template <typename index_t>
__device__ __forceinline__ index_t GetLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
}

template <typename T, int Dims>
struct IndexToOffset {
  static __device__ int64_t get(int64_t linear_id, const TensorLayoutHelper &info) {
    int64_t offset = 0;

    // Uses static dims
    for (int i = Dims - 1; i > 0; --i) {
      int64_t cur_dim_index = linear_id % info.sizes_[i];
      int64_t cur_dim_offset = cur_dim_index * info.strides_[i];
      offset += cur_dim_offset;
      linear_id /= info.sizes_[i];
    }

    return offset + linear_id * info.strides_[0];
  }
};

template <typename T>
struct IndexToOffset<T, -1> {
  static inline __device__ int64_t get(int64_t linear_id, const TensorLayoutHelper &info) {
    int64_t offset = 0;

    for (int i = info.dim_size_ - 1; i > 0; --i) {
      int64_t cur_dim_index = linear_id % info.sizes_[i];
      int64_t cur_dim_offset = cur_dim_index * info.strides_[i];
      offset += cur_dim_offset;
      linear_id /= info.sizes_[i];
    }

    return offset + linear_id * info.strides_[0];
  }
};

template <int kKeyDims, int kValueDims, int kBlockSize, int kItemsPerThread, typename K, typename V>
__global__ void __launch_bounds__(MS_MAX_THREAD_PER_BLOCK(kBlockSize))
  RadixSortKVInPlace(TensorLayoutHelper keys, K *key_data, int64_t key_slices, int64_t key_slice_size,
                     int64_t key_slice_stride, TensorLayoutHelper values, V *value_data, int64_t value_slice_stride,
                     bool descending) {
  static_assert(kBlockSize > 0, "");

  // Find the slice of the tensor that we are sorting
  const int64_t linearIndex = GetLinearBlockId<int64_t>();
  // Tiling the slices could have us be out of bounds, if there are a
  // lot of slices to sort
  if (linearIndex >= key_slices) {
    return;
  }

  const int64_t keyStartOffset = IndexToOffset<K, kKeyDims>::get(linearIndex, keys);
  const int64_t valueStartOffset = IndexToOffset<V, kValueDims>::get(linearIndex, values);

  K *keys_slice = &(key_data[keyStartOffset]);
  V *values_slice = &(value_data[valueStartOffset]);

  StridedPointer<K, int64_t> keys_iter(keys_slice, key_slice_stride);
  StridedPointer<V, int64_t> values_iter(values_slice, value_slice_stride);

  using LoadKeys = cub::BlockLoad<K, kBlockSize, kItemsPerThread, cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;
  using LoadValues = cub::BlockLoad<V, kBlockSize, kItemsPerThread, cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE>;
  using Sort = cub::BlockRadixSort<K, kBlockSize, kItemsPerThread, V>;
  using StoreKeys = cub::BlockStore<K, kBlockSize, kItemsPerThread, cub::BLOCK_STORE_TRANSPOSE>;
  using StoreValues = cub::BlockStore<V, kBlockSize, kItemsPerThread, cub::BLOCK_STORE_TRANSPOSE>;

  __shared__ union {
    typename LoadKeys::TempStorage load_keys;
    typename LoadValues::TempStorage load_values;
    typename Sort::TempStorage sort;
    typename StoreKeys::TempStorage store_keys;
    typename StoreValues::TempStorage store_values;
  } tmp_storage;

  // cub's Block operations operate on a fixed number of items, but the
  // actual slice we are sorting might be smaller. So, we need to make
  // up the difference with keys that will always sort higher.
  const K invalid_key = [descending] {
    using radix_t = typename cub::Traits<K>::UnsignedBits;
    union {
      K key;
      radix_t radix;
    } tmp;
    tmp.radix = descending ? cub::Traits<K>::LOWEST_KEY : cub::Traits<K>::MAX_KEY;
    return tmp.key;
  }();
  const V invalid_value = static_cast<V>(0);

  // Load inputs
  K local_keys[kItemsPerThread];
  V local_values[kItemsPerThread];

  LoadKeys(tmp_storage.load_keys).Load(keys_iter, local_keys, key_slice_size, invalid_key);
  __syncthreads();
  LoadValues(tmp_storage.load_values).Load(values_iter, local_values, key_slice_size, invalid_value);
  __syncthreads();

  // Sort!
  if (descending) {
    auto sorter = Sort(tmp_storage.sort);
    sorter.SortDescending(reinterpret_cast<K(&)[kItemsPerThread]>(local_keys), local_values);
  } else {
    Sort(tmp_storage.sort).Sort(reinterpret_cast<K(&)[kItemsPerThread]>(local_keys), local_values);
  }
  __syncthreads();

  // Store outputs
  StoreKeys(tmp_storage.store_keys).Store(keys_iter, local_keys, key_slice_size);
  __syncthreads();
  StoreValues(tmp_storage.store_values).Store(values_iter, local_values, key_slice_size);
}

template <int A, int kSortSize, int kItemsPerThread, typename K, typename V>
CUDA_LIB_EXPORT cudaError_t SortFixedSize(const TensorLayoutHelper &key_info, K *key_data, int64_t key_slices,
                                          int64_t key_slice_size, int64_t key_slice_stride,
                                          const TensorLayoutHelper &value_info, V *value_data,
                                          int64_t value_slice_stride, bool descending, cudaStream_t cuda_stream) {
  static_assert(kSortSize % kItemsPerThread == 0, "SortSize mod ItemsPerThread should be equal to zero.");
  constexpr int block = kSortSize / kItemsPerThread;
  dim3 grid;
  if (!GetGridFromTiles(key_slices, &grid)) {
    fprintf(stderr, "GetGridFromTiles failed\n");
    return cudaErrorNotReady;
  }

  RadixSortKVInPlace<A, -1, block, kItemsPerThread>
    <<<grid, block, 0, cuda_stream>>>(key_info, key_data, key_slices, key_slice_size, key_slice_stride, value_info,
                                      value_data, value_slice_stride, descending);
  return GetCudaStatus();
}

#define SortFixedSizeSpec(A, kSortSize, kItemsPerThread, K, V)                                                 \
  template CUDA_LIB_EXPORT cudaError_t SortFixedSize<A, kSortSize, kItemsPerThread, K, V>(                     \
    const TensorLayoutHelper &key_info, K *key_data, int64_t key_slices, int64_t key_slice_size,               \
    int64_t key_slice_stride, const TensorLayoutHelper &value_info, V *value_data, int64_t value_slice_stride, \
    bool descending, cudaStream_t cuda_stream)

#define SortFixedSizeSpecKV(K, V)                                                                            \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLast, kFixedSizeLevel1, kFixedSizeLevel1ItemPreThread, K, V);       \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLast, kFixedSizeLevel2, kFixedSizeLevel2ItemPreThread, K, V);       \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLast, kFixedSizeLevel3, kFixedSizeLevel3ItemPreThread, K, V);       \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLast, kFixedSizeLevel4, kFixedSizeLevel4ItemPreThread, K, V);       \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLast, kFixedSizeLevel5, kFixedSizeLevel5ItemPreThread, K, V);       \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLastSecond, kFixedSizeLevel1, kFixedSizeLevel1ItemPreThread, K, V); \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLastSecond, kFixedSizeLevel2, kFixedSizeLevel2ItemPreThread, K, V); \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLastSecond, kFixedSizeLevel3, kFixedSizeLevel3ItemPreThread, K, V); \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLastSecond, kFixedSizeLevel4, kFixedSizeLevel4ItemPreThread, K, V); \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsLastSecond, kFixedSizeLevel5, kFixedSizeLevel5ItemPreThread, K, V); \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsSecond, kFixedSizeLevel1, kFixedSizeLevel1ItemPreThread, K, V);     \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsSecond, kFixedSizeLevel2, kFixedSizeLevel2ItemPreThread, K, V);     \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsSecond, kFixedSizeLevel3, kFixedSizeLevel3ItemPreThread, K, V);     \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsSecond, kFixedSizeLevel4, kFixedSizeLevel4ItemPreThread, K, V);     \
  SortFixedSizeSpec(kFixedSizeSortKeyDimsSecond, kFixedSizeLevel5, kFixedSizeLevel5ItemPreThread, K, V);

SortFixedSizeSpecKV(bool, int64_t);
SortFixedSizeSpecKV(int8_t, int64_t);
SortFixedSizeSpecKV(int16_t, int64_t);
SortFixedSizeSpecKV(int32_t, int64_t);
SortFixedSizeSpecKV(int64_t, int64_t);
SortFixedSizeSpecKV(uint8_t, int64_t);
SortFixedSizeSpecKV(half, int64_t);
SortFixedSizeSpecKV(float, int64_t);
SortFixedSizeSpecKV(double, int64_t);

SortFixedSizeSpecKV(bool, int32_t);
SortFixedSizeSpecKV(int8_t, int32_t);
SortFixedSizeSpecKV(int16_t, int32_t);
SortFixedSizeSpecKV(int32_t, int32_t);
SortFixedSizeSpecKV(int64_t, int32_t);
SortFixedSizeSpecKV(uint8_t, int32_t);
SortFixedSizeSpecKV(half, int32_t);
SortFixedSizeSpecKV(float, int32_t);
SortFixedSizeSpecKV(double, int32_t);
