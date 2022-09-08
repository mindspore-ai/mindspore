/*copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <limits>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/segment_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

// Basic function

template <typename DataType>
struct MinFunc {
  __device__ __host__ __forceinline__ DataType operator()(const DataType &lhs, const DataType &rhs) {
    return lhs < rhs ? lhs : rhs;
  }
};

template <>
struct MinFunc<Complex<float>> {
  __device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const Complex<float> &rhs) {
    return 0;
  }
};

template <>
struct MinFunc<Complex<double>> {
  __device__ __host__ __forceinline__
    Complex<double> operator()(const Complex<double> &lhs, const Complex<double> &rhs) {
    return 0;
  }
};


template <typename DataType>
struct MaxFunc {
  __device__ __host__ __forceinline__ DataType operator()(const DataType &lhs, const DataType &rhs) {
    return lhs > rhs ? lhs : rhs;
  }
};

template <>
struct MaxFunc<Complex<float>> {
  __device__ __host__ __forceinline__ Complex<float> operator()(const Complex<float> &lhs, const Complex<float> &rhs) {
    return 0;
  }
};

template <>
struct MaxFunc<Complex<double>> {
  __device__ __host__ __forceinline__
    Complex<double> operator()(const Complex<double> &lhs, const Complex<double> &rhs) {
    return 0;
  }
};


template <typename T>
struct AddFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs + rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs + rhs);
  }
};

template <typename T>
struct MulFunc {
  __device__ __host__ __forceinline__ T operator()(const T &lhs, const T &rhs) { return (lhs * rhs); }
  __device__ __host__ __forceinline__ Complex<T> operator()(const Complex<T> &lhs, const Complex<T> &rhs) {
    return (lhs * rhs);
  }
};

template <typename DataType>
DataType max_val_init() {
  return std::numeric_limits<DataType>::max();
}

template <>
half max_val_init() {
  return 65504;  // Max value for Half
}

template <typename DataType>
DataType min_val_init() {
  return std::numeric_limits<DataType>::lowest();
}

template <>
half min_val_init() {
  return -65504;  // Max value for Half
}

template <typename DataType>
DataType get_default_value(size_t op) {
  return static_cast<DataType>(0);
}

template <>
half get_default_value(size_t op) {
  return op == 0 ? -65504 : 65504;
}

template <>
float get_default_value(size_t op) {
  return op == 0 ? std::numeric_limits<double>::lowest() : -std::numeric_limits<float>::lowest();
}

template <>
double get_default_value(size_t op) {
  return op == 0 ? std::numeric_limits<double>::lowest() : -std::numeric_limits<double>::lowest();
}

template <typename IndexType>
__global__ void CalSegmentPos(const IndexType *segment_ids_ptr, size_t *segment_pos_ptr, const size_t segment_size) {
  for (size_t pos = threadIdx.x + blockIdx.x * blockDim.x; pos <= segment_size; pos += blockDim.x * gridDim.x) {
    IndexType temp =
      (segment_size > (segment_ids_ptr[segment_size - 1]) + 1) ? segment_size : (segment_ids_ptr[segment_size - 1] + 1);
    IndexType begin_pos = (pos == 0) ? 0 : (segment_ids_ptr[pos - 1] + 1);
    IndexType end_pos = (pos != segment_size) ? segment_ids_ptr[pos] : temp;
    const IndexType max_size = static_cast<IndexType>(temp);
    const IndexType min_size = IndexType(0);
    begin_pos = max(min_size, min(max_size, begin_pos));
    end_pos = max(min_size, min(max_size, end_pos));
    for (IndexType j = begin_pos; j <= end_pos; ++j) {
      segment_pos_ptr[j] = pos;
    }
  }
}

template <typename DataType, typename Func>
__device__ DataType ReduceWithinBlock(const DataType &value) {
  // Refer to reduce3 from Mark Harris, et al. 'Optimizing Parallel Reduction in CUDA'.
  extern __shared__ __align__(16) char share_data[];
  DataType *share_data_ptr = reinterpret_cast<DataType *>(share_data);
  const unsigned int x = threadIdx.x;
  const unsigned int y = threadIdx.y;
  const unsigned int tid = y * blockDim.x + x;
  share_data_ptr[tid] = value;
  __syncthreads();
  // Reduce over the y dimension of the block.
  for (unsigned k = blockDim.y / 2; k > 0; k /= 2) {
    if (y < k) {
      share_data_ptr[tid] = Func()(share_data_ptr[tid], share_data_ptr[(y + k) * blockDim.x + x]);
    }
    __syncthreads();
  }
  return share_data_ptr[tid];
}

template <typename DataType, typename Func, typename IndexType>
__global__ void SegmentProcess(DataType *inp_ptr, DataType *out_ptr, size_t *seg_pos_ptr, const size_t inner_size,
                               const size_t outer_size, const size_t outer_class, size_t op, DataType init_K,
                               DataType default_value, IndexType *seg_id_ptr) {
  for (size_t thread_x = threadIdx.x + blockIdx.x * blockDim.x; thread_x < inner_size;
       thread_x += blockDim.x * gridDim.x) {
    for (size_t block_idx_y = blockIdx.y; block_idx_y < outer_class; block_idx_y += gridDim.y) {
      size_t begin_pos = seg_pos_ptr[block_idx_y];
      size_t end_pos = seg_pos_ptr[block_idx_y + 1];
      DataType res = init_K;
      DataType cur_data = init_K;
      for (size_t pos = begin_pos; pos < end_pos; pos += blockDim.y) {
        size_t thread_y = pos + threadIdx.y;
        cur_data = (thread_y < end_pos) ? inp_ptr[thread_y * inner_size + thread_x] : static_cast<DataType>(0);
        cur_data = ReduceWithinBlock<DataType, Func>(cur_data);
        if (threadIdx.y == 0) {
          res = Func()(res, cur_data);
        }
      }
      if (threadIdx.y == 0) {
        if (op == 2) {
          DataType segment_len = DataType(static_cast<double>(end_pos - begin_pos));
          out_ptr[block_idx_y * inner_size + thread_x] =
            (begin_pos >= end_pos) ? static_cast<DataType>(0) : res / segment_len;
        } else if (op == 3) {
          out_ptr[block_idx_y * inner_size + thread_x] = (begin_pos >= end_pos) ? static_cast<DataType>(0) : res;
        } else if (op == 4) {
          out_ptr[block_idx_y * inner_size + thread_x] = (begin_pos >= end_pos) ? static_cast<DataType>(1) : res;
        } else {
          out_ptr[block_idx_y * inner_size + thread_x] = (begin_pos >= end_pos) ? default_value : res;
        }
      }
    }
  }
}

inline int Log2Floor(uint32_t n) {
  if (n == 0) return -1;
  int log = 0;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32_t x = n >> shift;
    if (x) {
      n = x;
      log += shift;
    }
  }
  return log;
}

inline int Log2Floor64(uint64_t n) {
  // Scan n first high 32 then low 32 bits.
  const uint32_t high_32_bit = static_cast<uint32_t>(n >> 32);
  if (high_32_bit == 0) {
    return Log2Floor(static_cast<uint32_t>(n));
  } else {
    return 32 + Log2Floor(high_32_bit);
  }
}

inline int Log2Ceil64(uint64_t n) {
  int floor = Log2Floor64(n);
  if (n == (n & ~(n - 1)))
    return floor;
  else
    return floor + 1;
}

template <typename DataType, typename IndexType>
void CalSegmentCombination(DataType *inp_ptr, DataType *out_ptr, IndexType *seg_id_ptr,
                           size_t *seg_pos_ptr, size_t op, const size_t inner_size,
                           const size_t outer_size, const size_t outer_class, uint32_t device_id,
                           cudaStream_t cuda_stream) {
  // Get start position of each segment and set to segment_pos_ptr.
  // The last element of segment_pos_ptr must equal to indices_size.
  const unsigned int segment_size = outer_size + 1;
  // size_t segment_pos_length[1] = {0};
  CalSegmentPos<<<CUDA_BLOCKS(device_id, segment_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    seg_id_ptr, seg_pos_ptr, outer_size);
  const unsigned int max_grid_x = (1u << 31) - 1;
  const unsigned int max_grid_y = (1u << 16) - 1;
  const unsigned int max_block_x = 1024;
  const unsigned int max_block_y = 64;
  unsigned int inner_power2 = 1u << Log2Ceil64(inner_size);
  unsigned int avg_reduce_size = UP_DIV(outer_size, outer_size);
  unsigned int avg_reduce_size_power2 = 1u << Log2Ceil64(avg_reduce_size);
  unsigned int block_x = std::min(inner_power2, max_block_x);
  unsigned int block_y = std::min(avg_reduce_size_power2, UP_DIV(max_block_y, block_x));
  unsigned int grid_x = std::min(static_cast<unsigned int>(UP_DIV(inner_size, block_x)), max_grid_x);
  unsigned int grid_y = std::min(segment_size, max_grid_y);
  dim3 block(block_x, block_y);
  dim3 grid(grid_x, grid_y);
  unsigned int shared_memory_size = block_x * block_y * sizeof(DataType);
  DataType init_K = std::numeric_limits<DataType>::lowest();
  DataType default_value = get_default_value<DataType>(op);
  switch (op) {
    case 0:
      init_K = min_val_init<DataType>();
      return SegmentProcess<DataType, MaxFunc<DataType>><<<grid, block, shared_memory_size, cuda_stream>>>(
        inp_ptr, out_ptr, seg_pos_ptr, inner_size, outer_size, outer_class, op, init_K, default_value, seg_id_ptr);
    case 1:
      init_K = max_val_init<DataType>();
      return SegmentProcess<DataType, MinFunc<DataType>><<<grid, block, shared_memory_size, cuda_stream>>>(
        inp_ptr, out_ptr, seg_pos_ptr, inner_size, outer_size, outer_class, op, init_K, default_value, seg_id_ptr);
    case 2:
      init_K = 0.0;
      return SegmentProcess<DataType, AddFunc<DataType>><<<grid, block, shared_memory_size, cuda_stream>>>(
        inp_ptr, out_ptr, seg_pos_ptr, inner_size, outer_size, outer_class, op, init_K, default_value, seg_id_ptr);
    case 3:
      init_K = 0.0;
      return SegmentProcess<DataType, AddFunc<DataType>><<<grid, block, shared_memory_size, cuda_stream>>>(
        inp_ptr, out_ptr, seg_pos_ptr, inner_size, outer_size, outer_class, op, init_K, default_value, seg_id_ptr);
    case 4:
      init_K = 1.0;
      return SegmentProcess<DataType, MulFunc<DataType>><<<grid, block, shared_memory_size, cuda_stream>>>(
        inp_ptr, out_ptr, seg_pos_ptr, inner_size, outer_size, outer_class, op, init_K, default_value, seg_id_ptr);
    default:
      break;
  }
}

template CUDA_LIB_EXPORT void CalSegmentCombination<float, int32_t>(float *inp_ptr, float *out_ptr,
                                                                    int32_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                    size_t op, const size_t inner_size,
                                                                    const size_t outer_size, const size_t outer_class,
                                                                    uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<double, int32_t>(double *inp_ptr, double *out_ptr,
                                                                     int32_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                     size_t op, const size_t inner_size,
                                                                     const size_t outer_size, const size_t outer_class,
                                                                     uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<half, int32_t>(half *inp_ptr, half *out_ptr, int32_t *seg_id_addr,
                                                                   size_t *seg_pos_ptr, size_t op,
                                                                   const size_t inner_size, const size_t outer_size,
                                                                   const size_t outer_class, uint32_t device_id,
                                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<Complex<float>, int32_t>(
  Complex<float> *inp_ptr, Complex<float> *out_ptr, int32_t *seg_id_addr, size_t *seg_pos_ptr, size_t op,
  const size_t inner_size, const size_t outer_size, const size_t outer_class, uint32_t device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<Complex<double>, int32_t>(
  Complex<double> *inp_ptr, Complex<double> *out_ptr, int32_t *seg_id_addr, size_t *seg_pos_ptr, size_t op,
  const size_t inner_size, const size_t outer_size, const size_t outer_class, uint32_t device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<int8_t, int32_t>(int8_t *inp_ptr, int8_t *out_ptr,
                                                                     int32_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                     size_t op, const size_t inner_size,
                                                                     const size_t outer_size, const size_t outer_class,
                                                                     uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<int16_t, int32_t>(int16_t *inp_ptr, int16_t *out_ptr,
                                                                      int32_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                      size_t op, const size_t inner_size,
                                                                      const size_t outer_size, const size_t outer_class,
                                                                      uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<int32_t, int32_t>(int32_t *inp_ptr, int32_t *out_ptr,
                                                                      int32_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                      size_t op, const size_t inner_size,
                                                                      const size_t outer_size, const size_t outer_class,
                                                                      uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<int64_t, int32_t>(int64_t *inp_ptr, int64_t *out_ptr,
                                                                      int32_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                      size_t op, const size_t inner_size,
                                                                      const size_t outer_size, const size_t outer_class,
                                                                      uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<uint8_t, int32_t>(uint8_t *inp_ptr, uint8_t *out_ptr,
                                                                      int32_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                      size_t op, const size_t inner_size,
                                                                      const size_t outer_size, const size_t outer_class,
                                                                      uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<uint16_t, int32_t>(
  uint16_t *inp_ptr, uint16_t *out_ptr, int32_t *seg_id_addr, size_t *seg_pos_ptr, size_t op, const size_t inner_size,
  const size_t outer_size, const size_t outer_class, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<uint32_t, int32_t>(
  uint32_t *inp_ptr, uint32_t *out_ptr, int32_t *seg_id_addr, size_t *seg_pos_ptr, size_t op, const size_t inner_size,
  const size_t outer_size, const size_t outer_class, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<uint64_t, int32_t>(
  uint64_t *inp_ptr, uint64_t *out_ptr, int32_t *seg_id_addr, size_t *seg_pos_ptr, size_t op, const size_t inner_size,
  const size_t outer_size, const size_t outer_class, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<float, int64_t>(float *inp_ptr, float *out_ptr,
                                                                    int64_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                    size_t op, const size_t inner_size,
                                                                    const size_t outer_size, const size_t outer_class,
                                                                    uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<double, int64_t>(double *inp_ptr, double *out_ptr,
                                                                     int64_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                     size_t op, const size_t inner_size,
                                                                     const size_t outer_size, const size_t outer_class,
                                                                     uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<half, int64_t>(half *inp_ptr, half *out_ptr, int64_t *seg_id_addr,
                                                                   size_t *seg_pos_ptr, size_t op,
                                                                   const size_t inner_size, const size_t outer_size,
                                                                   const size_t outer_class, uint32_t device_id,
                                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<Complex<float>, int64_t>(
  Complex<float> *inp_ptr, Complex<float> *out_ptr, int64_t *seg_id_addr, size_t *seg_pos_ptr, size_t op,
  const size_t inner_size, const size_t outer_size, const size_t outer_class, uint32_t device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<Complex<double>, int64_t>(
  Complex<double> *inp_ptr, Complex<double> *out_ptr, int64_t *seg_id_addr, size_t *seg_pos_ptr, size_t op,
  const size_t inner_size, const size_t outer_size, const size_t outer_class, uint32_t device_id,
  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<int8_t, int64_t>(int8_t *inp_ptr, int8_t *out_ptr,
                                                                     int64_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                     size_t op, const size_t inner_size,
                                                                     const size_t outer_size, const size_t outer_class,
                                                                     uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<int16_t, int64_t>(int16_t *inp_ptr, int16_t *out_ptr,
                                                                      int64_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                      size_t op, const size_t inner_size,
                                                                      const size_t outer_size, const size_t outer_class,
                                                                      uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<int32_t, int64_t>(int32_t *inp_ptr, int32_t *out_ptr,
                                                                      int64_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                      size_t op, const size_t inner_size,
                                                                      const size_t outer_size, const size_t outer_class,
                                                                      uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<int64_t, int64_t>(int64_t *inp_ptr, int64_t *out_ptr,
                                                                      int64_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                      size_t op, const size_t inner_size,
                                                                      const size_t outer_size, const size_t outer_class,
                                                                      uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<uint8_t, int64_t>(uint8_t *inp_ptr, uint8_t *out_ptr,
                                                                      int64_t *seg_id_addr, size_t *seg_pos_ptr,
                                                                      size_t op, const size_t inner_size,
                                                                      const size_t outer_size, const size_t outer_class,
                                                                      uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<uint16_t, int64_t>(
  uint16_t *inp_ptr, uint16_t *out_ptr, int64_t *seg_id_addr, size_t *seg_pos_ptr, size_t op, const size_t inner_size,
  const size_t outer_size, const size_t outer_class, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<uint32_t, int64_t>(
  uint32_t *inp_ptr, uint32_t *out_ptr, int64_t *seg_id_addr, size_t *seg_pos_ptr, size_t op, const size_t inner_size,
  const size_t outer_size, const size_t outer_class, uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalSegmentCombination<uint64_t, int64_t>(
  uint64_t *inp_ptr, uint64_t *out_ptr, int64_t *seg_id_addr, size_t *seg_pos_ptr, size_t op, const size_t inner_size,
  const size_t outer_size, const size_t outer_class, uint32_t device_id, cudaStream_t cuda_stream);
