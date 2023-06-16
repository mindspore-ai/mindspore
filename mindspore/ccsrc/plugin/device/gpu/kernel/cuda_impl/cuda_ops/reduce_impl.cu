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

#include <cub/cub.cuh>
#include <limits>
#include <algorithm>

#include "include/cuda_runtime.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/reduce_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/permutation_in_iterator.cuh"

constexpr int thread_per_warp = 32;
constexpr int kUnroll = 8;

template <typename T>
struct Sum {
  __host__ __device__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct Max {
  __host__ __device__ T operator()(const T &a, const T &b) const { return (a != a ? a : (a > b ? a : b)); }
};

template <typename T>
struct Min {
  __host__ __device__ T operator()(const T &a, const T &b) const { return (a != a ? a : (a < b ? a : b)); }
};

template <typename T>
struct Prod {
  __host__ __device__ T operator()(const T &a, const T &b) const { return a * b; }
};

struct And {
  __host__ __device__ bool operator()(const bool &a, const bool &b) const { return a && b; }
};

struct Or {
  __host__ __device__ bool operator()(const bool &a, const bool &b) const { return a || b; }
};

template <typename T, typename Op>
struct IsSum {
  constexpr static bool flag = std::is_same<Op, Sum<T>>::value;
};

template <typename T, typename Op>
struct IsProd {
  constexpr static bool flag = std::is_same<Op, Prod<T>>::value;
};

template <typename T, typename Op>
struct IsMax {
  constexpr static bool flag = std::is_same<Op, Max<T>>::value;
};

template <typename T, typename Op>
struct IsMin {
  constexpr static bool flag = std::is_same<Op, Min<T>>::value;
};

template <typename Op>
struct IsAll {
  constexpr static bool flag = std::is_same<Op, And>::value;
};

template <typename Op>
struct IsAny {
  constexpr static bool flag = std::is_same<Op, Or>::value;
};

template <typename T, typename Op>
struct GetInit {
  static_assert(IsSum<T, Op>::flag || IsProd<T, Op>::flag || IsMax<T, Op>::flag || IsMin<T, Op>::flag ||
                  IsAll<Op>::flag || IsAny<Op>::flag,
                "Not support this type");

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<IsSum<U, OpCopy>::flag, U>::type init = U(0)) {
    return init;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<IsProd<U, OpCopy>::flag, U>::type init = U(1)) {
    return init;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<IsMax<U, OpCopy>::flag, U>::type init = std::numeric_limits<T>::lowest()) {
    return init;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<IsMin<U, OpCopy>::flag, U>::type init = std::numeric_limits<T>::max()) {
    return init;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<IsAll<OpCopy>::flag, bool>::type init = true) {
    return init;
  }

  template <typename U = T, typename OpCopy = Op>
  U operator()(typename std::enable_if<IsAny<OpCopy>::flag, bool>::type init = false) {
    return init;
  }
};

template <typename T>
__global__ void Average(const size_t size, const size_t divisor, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] /= divisor;
  }
}

template <>
__global__ void Average(const size_t size, const size_t divisor, half *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
#if CUDA_VERSION >= 11000
    output[pos] /= __ull2half_rn(divisor);
#else
    output[pos] /= static_cast<half>(static_cast<float>(divisor));
#endif  // CUDA_VERSION > 11000
  }
}

template <>
__global__ void Average(const size_t size, const size_t divisor, Complex<float> *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    Complex<float> ret;
    ret.real(output[pos].real() / divisor);
    ret.imag(output[pos].imag() / divisor);
    output[pos] = ret;
  }
}

template <>
__global__ void Average(const size_t size, const size_t divisor, Complex<double> *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    Complex<double> ret;
    ret.real(output[pos].real() / divisor);
    ret.imag(output[pos].imag() / divisor);
    output[pos] = ret;
  }
}

struct GatherOp {
  __host__ __device__ GatherOp(const int &extent_x, const int &extent_y, const int &extent_z, bool kOne)
      : extent_x_(extent_x), extent_y_(extent_y), extent_z_(extent_z), kOne_(kOne) {
    if (kOne_)
      group_size_ = extent_y_;
    else
      group_size_ = extent_x_ * extent_z_;
  }

  __host__ __device__ int operator()(const int &ind) const {
    const int group = kOne_ ? ind / group_size_ : ind % group_size_;
    const int offset = kOne_ ? ind % group_size_ : ind / group_size_;

    const int x = group / extent_z_;
    const int z = group % extent_z_;

    return x * extent_y_ * extent_z_ + z + offset * extent_z_;
  }

  int extent_x_;
  int extent_y_;
  int extent_z_;
  bool kOne_;
  int group_size_;
};

struct ComputeOffset {
  __host__ __device__ explicit ComputeOffset(const int &cols) : cols_(cols) {}

  __host__ __device__ int operator()(const int &x) const { return cols_ * x; }

  int cols_;
};

template <typename T, int NUM_THREADS, typename Op>
__global__ __launch_bounds__(1024) void BlockReduceKernel(const T *input, T *output, const size_t size, Op op,
                                                          const T init) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int gid = blockDim.x * bid + tid;
  const int stride = blockDim.x * gridDim.x;
  T sum = init;
  if (gid < size) {
    sum = input[gid];
    for (size_t i = gid + stride; i < size; i += stride) {
      sum = op(sum, input[i]);
    }
  }
  typedef cub::BlockReduce<T, NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int num_elem_need_reduce =
    max(min(static_cast<float>(size - bid * blockDim.x), static_cast<float>(NUM_THREADS)), static_cast<float>(0.0));
  sum = BlockReduce(temp_storage).Reduce(sum, op, num_elem_need_reduce);
  if (tid == 0) output[bid] = sum;
}

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void CleanupSegments(const T *temp, T *output, const size_t num_rows,
                                                        const size_t num_cols, const size_t size, Op op, T init) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  T temp_res = init;
  if (tid < size * num_cols) {
    temp_res = temp[tid];
  }
  typedef cub::WarpReduce<T> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage;
  const bool head_flag = (threadIdx.x % size) == 0;
  T sum = WarpReduce(temp_storage).HeadSegmentedReduce(temp_res, head_flag, op);
  if (head_flag && tid < size * num_cols) {
    output[tid / size] = sum;
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void RowReduceKernel(const T *input, T *output, const size_t num_rows,
                                                        const size_t num_cols, Op op, T init) {
  CUDA_KERNEL_ASSERT(blockDim.x % thread_per_warp == 0);
  int warp_per_block = blockDim.x / thread_per_warp;
  int warp_index = threadIdx.x / thread_per_warp;
  const int row_index = blockIdx.x * warp_per_block + warp_index;
  const int lane_index = threadIdx.x % thread_per_warp;

  if (num_cols == 1) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < num_rows) {
      output[gid] = input[gid];
    }
    return;
  }
  T sum = init;
  int col_index = lane_index;
  if (row_index < num_rows && col_index < num_cols) {
    sum = input[row_index * num_cols + col_index];
    col_index += thread_per_warp;
    for (; col_index < num_cols; col_index += thread_per_warp) {
      sum = op(sum, input[row_index * num_cols + col_index]);
    }
  }

  typedef cub::WarpReduce<T> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage;
  sum = WarpReduce(temp_storage).Reduce(sum, op, min(static_cast<int>(num_cols), thread_per_warp));
  if (row_index < num_rows && lane_index == 0) {
    output[row_index] = sum;
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void ColumnReduce16Kernel(const T *input, T *output, const size_t num_rows,
                                                             const size_t num_cols, Op op, T init) {
  const int rows_per_warp = thread_per_warp / num_cols;

  const int lane_index = threadIdx.x % thread_per_warp;
  const int lane_row = lane_index / num_cols;

  const int start_row_warp = rows_per_warp * (blockIdx.y * blockDim.y + threadIdx.y);
  const int start_row_lane = start_row_warp + lane_row;
  int row_index = start_row_lane;
  int col_index = lane_index % num_cols;

  T sum = init;
  if (row_index * num_cols + col_index < num_rows * num_cols) sum = input[row_index * num_cols + col_index];
  __shared__ __align__(alignof(T)) char partial_sums_raw[thread_per_warp * (thread_per_warp + 1) * sizeof(T)];
  T *partial_sums = reinterpret_cast<T *>(partial_sums_raw);

  row_index += rows_per_warp * gridDim.y * blockDim.y;
  for (; row_index < num_rows; row_index += rows_per_warp * gridDim.y * blockDim.y) {
    int global_pos = row_index * num_cols + col_index;
    if (global_pos < (num_rows * num_cols)) sum = op(sum, input[row_index * num_cols + col_index]);
  }

  const int rows_in_this_warp = min(rows_per_warp, static_cast<int>(num_rows - start_row_warp));
  for (int i = 1; i < rows_in_this_warp; ++i) {
    T tmp = cub::ShuffleIndex<thread_per_warp, T>(sum, static_cast<int>(threadIdx.x + i * num_cols), 0xffffffff);
    if (lane_index < num_cols) sum = op(sum, tmp);
  }

  if (lane_index < num_cols) partial_sums[lane_index * (thread_per_warp + 1) + threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.y == 0 && threadIdx.x < num_cols) {
    T total_sum = partial_sums[threadIdx.x * (thread_per_warp + 1)];

    if (blockDim.y > 1) {
      for (int row_index = 1; row_index < blockDim.y; ++row_index) {
        T block_sum = partial_sums[threadIdx.x * (thread_per_warp + 1) + row_index];
        total_sum = op(total_sum, block_sum);
      }
    }

    output[col_index * gridDim.y + blockIdx.y] = total_sum;
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void ColumnReduceKernel(const T *input, T *output, const size_t num_rows,
                                                           const size_t num_cols, Op op, T init) {
  size_t row_index = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col_index = blockIdx.x * thread_per_warp + threadIdx.x;

  T sum = init;
  if (row_index < num_rows && col_index < num_cols) {
    sum = input[row_index * num_cols + col_index];
  }

  __shared__ __align__(alignof(T)) char partial_sums_raw[thread_per_warp * (thread_per_warp + 1) * sizeof(T)];
  T *partial_sums = reinterpret_cast<T *>(partial_sums_raw);

  row_index += gridDim.y * blockDim.y;

  if (col_index < num_cols) {
    for (; row_index < num_rows; row_index += gridDim.y * blockDim.y) {
      sum = op(sum, input[row_index * num_cols + col_index]);
    }
  }

  partial_sums[threadIdx.x * (thread_per_warp + 1) + threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.y == 0 && col_index < num_cols) {
    T total_sum = partial_sums[threadIdx.x * (thread_per_warp + 1)];
    const int numRowsThisBlock =
      min(static_cast<int>(blockDim.y), static_cast<int>(num_rows - blockIdx.y * blockDim.y));
    for (int row_index = 1; row_index < numRowsThisBlock; ++row_index) {
      T block_sum = partial_sums[threadIdx.x * (thread_per_warp + 1) + row_index];
      total_sum = op(total_sum, block_sum);
    }
    output[col_index * gridDim.y + blockIdx.y] = total_sum;
  }
}

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void ColumnReduceSimpleKernel(const T *input, T *output, const size_t num_matrix,
                                                                 const size_t num_rows, const size_t num_cols, Op op) {
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int matrix_size = num_rows * num_cols;

  const int matrix_index = gid / num_cols;
  const int col_index = gid % num_cols;

  if (matrix_index >= num_matrix) return;

  if (num_rows == 1) {
    output[matrix_index * matrix_size + col_index] = input[matrix_index * matrix_size + col_index];
    return;
  }

  T sum = op(input[matrix_index * matrix_size + col_index], input[matrix_index * matrix_size + num_cols + col_index]);
  for (int row_index = 2; row_index < num_rows; ++row_index) {
    sum = op(sum, input[matrix_index * matrix_size + row_index * num_cols + col_index]);
  }

  output[matrix_index * num_cols + col_index] = sum;
}

template <typename T, typename Op>
__device__ __inline__ T ComputeSum(const T *input, const int plane, const int num_out_rows, int num_rows, int num_cols,
                                   const int col, Op op) {
  const int out_rows = num_rows / (2 * kUnroll);
  const int num_rem_rows = num_rows % (2 * kUnroll);
  const int elems_per_plane = num_rows * num_cols;
  T reg[2 * kUnroll];
  T sum;
  int offset = 0;
  if (out_rows != 0) {
    for (int i = 0; i < 2 * kUnroll; i++) {
      reg[i] = input[plane * elems_per_plane + i * (num_out_rows * num_cols) + col];
    }
    sum = reg[0];
    for (int i = 1; i < 2 * kUnroll; i++) {
      sum = op(sum, reg[i]);
    }
    offset = 2 * kUnroll * (num_out_rows * num_cols);
  }

  if (col < num_cols && num_rem_rows > 0) {
    reg[0] = input[plane * elems_per_plane + offset + 0 * num_cols + col];
    if (out_rows != 0) {
      sum = op(sum, reg[0]);
    } else {
      sum = reg[0];
    }
    for (int i = 1; i < num_rem_rows; i++) {
      reg[0] = input[plane * elems_per_plane + offset + i * num_cols + col];
      sum = op(sum, reg[0]);
    }
  }
  return sum;
}

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void ColumnReduceInToTempKernel(T *temp, int temp_in_offset, int temp_out_offset,
                                                                   const T *input, const size_t num_planes,
                                                                   int num_rows, const size_t num_cols, Op op) {
  T *t = reinterpret_cast<T *>(temp);
  T *out_ = t + temp_out_offset;

  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const int num_out_rows = max(1, num_rows / (2 * kUnroll));
  const int plane = gid / (num_out_rows * num_cols);
  const int col = gid % (num_out_rows * num_cols);

  if (plane >= num_planes) return;

  T sum;
  if (temp_in_offset == -1) {
    auto in_ = input;
    sum = ComputeSum(in_, plane, num_out_rows, num_rows, num_cols, col, op);
  } else {
    auto in_ = t + temp_in_offset;
    sum = ComputeSum(in_, plane, num_out_rows, num_rows, num_cols, col, op);
  }
  out_[plane * num_out_rows * num_cols + col] = sum;
}

template <typename T, typename Op>
__global__ __launch_bounds__(1024) void ColumnReduceTempToOutKernel(T *temp, int temp_in_offset, const T *input,
                                                                    T *output, const size_t num_planes, int num_rows,
                                                                    const size_t num_cols, Op op) {
  T *t = temp;
  const int tid = threadIdx.x;
  const int gid = threadIdx.x + blockIdx.x * blockDim.x;
  int elems_per_plane = num_rows * num_cols;

  if (num_rows == 1) {
    if (gid >= num_planes * num_cols) return;
    if (temp_in_offset == -1) {
      auto in_ = input;
      output[gid] = in_[gid];
    } else {
      auto in_ = t + temp_in_offset;
      output[gid] = in_[gid];
    }
    return;
  }

  const int planes_per_block = 1;
  const int plane = blockIdx.x * planes_per_block + tid / elems_per_plane;
  const int col = tid % elems_per_plane;
  const int local_plane = plane % planes_per_block;

  if (tid >= planes_per_block * elems_per_plane || plane >= num_planes) return;

  extern __shared__ __align__(8) char ss[];
  T *smem = reinterpret_cast<T *>(ss);

  if (temp_in_offset == -1) {
    auto in_ = input;
    smem[local_plane * elems_per_plane + col] = in_[plane * elems_per_plane + col];
  } else {
    auto in_ = t + temp_in_offset;
    smem[local_plane * elems_per_plane + col] = in_[plane * elems_per_plane + col];
  }
  __syncthreads();

  int num_in_rows = num_rows;
  int num_out_rows;
  int num_rem_rows;

  int in_offset = 0;
  int out_offset = blockDim.x;

  int in_elems_per_plane = elems_per_plane;
  int out_elems_per_plane;

  while (num_in_rows > 1) {
    num_out_rows = num_in_rows / 2;
    num_rem_rows = num_in_rows % 2;
    out_elems_per_plane = num_out_rows * num_cols;

    if (col < out_elems_per_plane) {
      T sum;
      sum = op(smem[in_offset + local_plane * in_elems_per_plane + col],
               smem[in_offset + local_plane * in_elems_per_plane + out_elems_per_plane + col]);
      if (num_rem_rows == 1 && col < num_cols) {
        sum = op(sum, smem[in_offset + local_plane * in_elems_per_plane + 2 * out_elems_per_plane + col]);
      }
      smem[out_offset + local_plane * out_elems_per_plane + col] = sum;
    }

    num_in_rows = num_out_rows;
    in_elems_per_plane = out_elems_per_plane;
    int t_offset = in_offset;
    in_offset = out_offset;
    out_offset = t_offset;
    __syncthreads();
  }

  if (col < num_cols) {
    output[plane * num_cols + col] = smem[in_offset + local_plane * out_elems_per_plane + col];
  }
}

template <typename T, typename Op>
void CalReduceColumn16(const T *input, const size_t num_rows, const size_t num_cols, Op op, T init, T *output,
                       cudaStream_t cuda_stream) {
  int rows_per_warp = thread_per_warp / num_cols;
  const int block_y = std::min<int>(((num_rows + rows_per_warp - 1) / rows_per_warp), (1024 / thread_per_warp));

  dim3 num_threads(thread_per_warp, block_y, 1);

  const int grid_y = (num_rows + rows_per_warp * num_threads.y - 1) / (rows_per_warp * num_threads.y);
  dim3 num_blocks(1, grid_y, 1);

  num_blocks.y = std::min<int>(static_cast<int>(num_blocks.y), thread_per_warp);

  if (num_blocks.y > 2 && num_blocks.y < thread_per_warp) {
    int log2 = Log2Floor(num_blocks.y);
    num_blocks.y = 1 << log2;
  }

  if (num_blocks.y == 1) {
    ColumnReduce16Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(input, output, num_rows, num_cols, op, init);
  } else {
    T *temp_storage = nullptr;
    (void)cudaMalloc(&temp_storage, sizeof(T) * num_cols * num_blocks.y);
    ColumnReduce16Kernel<<<num_blocks, num_threads, 0, cuda_stream>>>(input, temp_storage, num_rows, num_cols, op,
                                                                      init);

    const int grid_x = (num_blocks.y * num_cols + thread_per_warp - 1) / thread_per_warp;
    dim3 new_num_blocks(grid_x, 1, 1);
    dim3 new_num_threads(128, 1, 1);
    CleanupSegments<<<new_num_blocks, new_num_threads, 0, cuda_stream>>>(temp_storage, output, num_rows, num_cols,
                                                                         num_blocks.y, op, init);
    (void)cudaFree(temp_storage);
  }
}

template <typename T, typename Op>
void CalReduceColumn4096(const T *input, const size_t num_rows, const size_t num_cols, Op op, T init, T *output,
                         cudaStream_t cuda_stream) {
  dim3 num_threads(thread_per_warp, std::min<int>(num_rows, (1024 / thread_per_warp)), 1);
  dim3 num_blocks((num_cols + thread_per_warp - 1) / thread_per_warp, 1, 1);

  if (num_blocks.x < 16) {
    num_blocks.y = std::min<int>((num_rows + thread_per_warp - 1) / thread_per_warp, thread_per_warp);
  }
  if (num_blocks.y > 2 && num_blocks.y < thread_per_warp) {
    int log2 = Log2Floor(num_blocks.y);
    num_blocks.y = 1 << log2;
  }

  if (num_blocks.y == 1) {
    ColumnReduceKernel<<<num_blocks, num_threads, 0, cuda_stream>>>(input, output, num_rows, num_cols, op, init);
  } else {
    T *temp_storage = nullptr;
    (void)cudaMalloc(&temp_storage, sizeof(T) * num_cols * num_blocks.y);
    ColumnReduceKernel<<<num_blocks, num_threads, 0, cuda_stream>>>(input, temp_storage, num_rows, num_cols, op, init);

    dim3 new_num_blocks((num_blocks.y * num_cols + thread_per_warp - 1) / thread_per_warp, 1, 1);
    CleanupSegments<<<new_num_blocks, num_threads, 0, cuda_stream>>>(temp_storage, output, num_rows, num_cols,
                                                                     num_blocks.y, op, init);
    (void)cudaFree(temp_storage);
  }
}

template <typename T, typename Op>
void CalReduceToScalar(const T *input, const size_t size, T *output, Op op, T init, cudaStream_t cuda_stream) {
  if (size <= 4096) {
    const int num_blocks = 1;
    const int num_threads = 256;
    BlockReduceKernel<T, num_threads, Op><<<num_blocks, num_threads, 0, cuda_stream>>>(input, output, size, op, init);
    return;
  } else if (size <= 1 << 18) {
    const int num_threads = 256;
    const int num_blocks = std::min<int>(thread_per_warp, ((static_cast<int>(size) + num_threads - 1) / num_threads));
    T *temp_storage = nullptr;
    (void)cudaMalloc(&temp_storage, num_blocks * sizeof(T));
    BlockReduceKernel<T, num_threads, Op>
      <<<num_blocks, num_threads, 0, cuda_stream>>>(input, temp_storage, size, op, init);
    const int last_blocks = 1;
    const int num_rows = 1;
    const int num_cols = 1;
    CleanupSegments<<<last_blocks, thread_per_warp, 0, cuda_stream>>>(temp_storage, output, num_rows, num_cols,
                                                                      num_blocks, op, init);
    cudaFree(temp_storage);
    return;
  }
  size_t temp_storage_size = 0;
  auto reduce = [&](void *temp_storage) {
    auto res = cub::DeviceReduce::Reduce(temp_storage, temp_storage_size, input, output, size, op, init, cuda_stream);
    if (res != cudaSuccess) {
      return;
    }
  };
  reduce(nullptr);
  T *temp_storage;
  (void)cudaMalloc(&temp_storage, temp_storage_size);
  reduce(temp_storage);
  cudaFree(temp_storage);
}

template <typename T, typename Op>
void CalReduceRow(const T *input, const size_t num_rows, const size_t num_cols, Op op, T init, T *output,
                  cudaStream_t cuda_stream) {
  if (num_cols < 1024) {
    const int num_threads = 128;
    const int num_warps = num_threads / thread_per_warp;
    const int num_blocks = (num_rows + num_warps - 1) / num_warps;
    RowReduceKernel<<<num_blocks, num_threads, 0, cuda_stream>>>(input, output, num_rows, num_cols, op, init);
    return;
  }
  ComputeOffset computeoffset(num_cols);
  cub::CountingInputIterator<int> counting_iter(0);
  cub::TransformInputIterator<int, ComputeOffset, cub::CountingInputIterator<int>> transform_iter(counting_iter,
                                                                                                  computeoffset);
  size_t temp_storage_size = 0;
  auto reduce = [&](void *temp_storage) {
    auto res = cub::DeviceSegmentedReduce::Reduce(temp_storage, temp_storage_size, input, output, num_rows,
                                                  transform_iter, transform_iter + 1, op, init, cuda_stream);
    if (res != cudaSuccess) {
      return;
    }
  };

  reduce(nullptr);
  T *temp_storage = nullptr;
  (void)cudaMalloc(&temp_storage, temp_storage_size);

  reduce(temp_storage);
  cudaFree(temp_storage);
}

template <typename T, typename Op>
void CalReduceColumn(const T *input, const size_t dim0, const size_t dim1, Op op, T init, T *output,
                     cudaStream_t cuda_stream) {
  if (dim1 <= 16) {
    CalReduceColumn16(input, dim0, dim1, op, init, output, cuda_stream);
  } else if (dim1 <= 4096) {
    CalReduceColumn4096(input, dim0, dim1, op, init, output, cuda_stream);
  } else {
    const int num_threads = 128;
    const int num_blocks = (dim1 + num_threads - 1) / num_threads;
    const size_t num_matrix = 1;
    ColumnReduceSimpleKernel<<<num_blocks, num_threads, 0, cuda_stream>>>(input, output, num_matrix, dim0, dim1, op);
  }
}

template <typename T, typename Op>
void CalReduce3DXZ(const T *input, const size_t dim0, const size_t dim1, const size_t dim2, Op op, T init, T *output,
                   cudaStream_t cuda_stream) {
  ComputeOffset computeoffset(dim0 * dim2);
  cub::CountingInputIterator<int> counting_iter(0);
  cub::TransformInputIterator<int, ComputeOffset, cub::CountingInputIterator<int>> transform_iter(counting_iter,
                                                                                                  computeoffset);
  GatherOp gather_op(dim0, dim1, dim2, false);
  typedef cub::TransformInputIterator<int, GatherOp, cub::CountingInputIterator<int>> gatherIterType;
  gatherIterType gather_iter(counting_iter, gather_op);
  PermutationInputIterator<T, gatherIterType> permute_iter(input, gather_iter);
  std::size_t temp_storage_size = 0;
  auto reduce = [&](void *temp_storage_ptr) {
    auto res = cub::DeviceSegmentedReduce::Reduce(temp_storage_ptr, temp_storage_size, permute_iter, output, dim1,
                                                  transform_iter, transform_iter + 1, op, init, cuda_stream);
    if (res != cudaSuccess) {
      return;
    }
  };

  reduce(nullptr);
  T *temp_storage = nullptr;
  (void)cudaMalloc(&temp_storage, temp_storage_size);
  reduce(temp_storage);
  cudaFree(temp_storage);
}

template <typename T, typename Op>
void CalReduce3DY(const T *input, const size_t dim0, const size_t dim1, const size_t dim2, Op op, T init, T *output,
                  cudaStream_t cuda_stream) {
  int num_threads = 128;
  int n_group_in = dim1;
  int n_size = dim2;

  std::size_t temp_storage_size = 0;
  while (n_group_in >= 2 && n_group_in * n_size > num_threads) {
    int n_group_out = std::max(1, n_group_in / (2 * kUnroll));
    temp_storage_size += n_group_out * n_size;
    n_group_in = n_group_out;
  }
  temp_storage_size *= dim0 * sizeof(T);
  T *temp_storage = nullptr;
  (void)cudaMalloc(&temp_storage, temp_storage_size);

  n_group_in = dim1;
  int temp_in_offset = -1;
  int temp_out_offset = 0;
  int num_blocks;
  while (n_group_in >= 2 && n_group_in * n_size > num_threads) {
    int n_group_out = std::max(1, n_group_in / (2 * kUnroll));
    num_blocks = (static_cast<int>(dim0) * n_group_out * n_size + num_threads - 1) / num_threads;
    ColumnReduceInToTempKernel<<<num_blocks, num_threads, 0, cuda_stream>>>(
      temp_storage, temp_in_offset, temp_out_offset, input, dim0, n_group_in, dim2, op);
    n_group_in = n_group_out;
    temp_in_offset = temp_out_offset;
    temp_out_offset = temp_in_offset + dim0 * n_group_out * n_size;
  }

  if (n_group_in * n_size <= num_threads) {
    num_blocks = dim0;
  } else if (n_group_in != 1) {
    return;
  } else {
    num_blocks = (static_cast<int>(dim0) * n_size + num_threads - 1) / num_threads;
  }
  ColumnReduceTempToOutKernel<<<num_blocks, num_threads, 2 * sizeof(T) * num_threads, cuda_stream>>>(
    temp_storage, temp_in_offset, input, output, dim0, n_group_in, dim2, op);
  cudaFree(temp_storage);
}

template <typename T, typename Op>
void CalReduce3DYLight(const T *input, const size_t dim0, const size_t dim1, const size_t dim2, Op op, T init,
                       T *output, cudaStream_t cuda_stream) {
  int threads_per_block = 128;
  int num_blocks = (dim0 * dim2 + threads_per_block - 1) / threads_per_block;
  ColumnReduceSimpleKernel<<<num_blocks, threads_per_block, 0, cuda_stream>>>(input, output, dim0, dim1, dim2, op);
}

template <typename T, typename Op>
void ReduceImpl(const T *input, const std::vector<size_t> &input_reshape, const bool reduce_first_axis, Op op,
                T *output, cudaStream_t cuda_stream) {
  T init = GetInit<T, Op>()();
  const size_t dim0 = input_reshape[0];
  const size_t dim1 = input_reshape.size() >= 2 ? input_reshape[1] : 1;
  const size_t dim2 = input_reshape.size() >= 3 ? input_reshape[2] : 1;
  if (input_reshape.size() == 1 && reduce_first_axis) {
    CalReduceToScalar(input, dim0, output, op, init, cuda_stream);
  } else if ((input_reshape.size() == 2) && (reduce_first_axis)) {
    CalReduceColumn(input, dim0, dim1, op, init, output, cuda_stream);
  } else if ((input_reshape.size() == 2) && (!reduce_first_axis)) {
    CalReduceRow(input, dim0, dim1, op, init, output, cuda_stream);
  } else if ((input_reshape.size() == 3) && (reduce_first_axis)) {
    CalReduce3DXZ(input, dim0, dim1, dim2, op, init, output, cuda_stream);
  } else if ((input_reshape.size() == 3) && (!reduce_first_axis)) {
    int num_per_thread = dim1 / (dim0 * dim2);
    if (num_per_thread >= 16) {
      CalReduce3DY(input, dim0, dim1, dim2, op, init, output, cuda_stream);
    } else {
      CalReduce3DYLight(input, dim0, dim1, dim2, op, init, output, cuda_stream);
    }
  } else {
    fprintf(stderr, "Invalid shapes and axis to reduce.");
    exit(1);
  }
}

template <typename T>
cudaError_t ArrayReduce(const T *input, const std::vector<size_t> &input_reshape, const bool reduce_first_axis,
                        ReduceType_t type, T *output, cudaStream_t cuda_stream) {
  switch (type) {
    case ReduceSum:
      ReduceImpl<T, Sum<T>>(input, input_reshape, reduce_first_axis, Sum<T>(), output, cuda_stream);
      break;
    case ReduceMax:
      ReduceImpl<T, Max<T>>(input, input_reshape, reduce_first_axis, Max<T>(), output, cuda_stream);
      break;
    case ReduceMin:
      ReduceImpl<T, Min<T>>(input, input_reshape, reduce_first_axis, Min<T>(), output, cuda_stream);
      break;
    case ReduceProd:
      ReduceImpl<T, Prod<T>>(input, input_reshape, reduce_first_axis, Prod<T>(), output, cuda_stream);
      break;
    case ReduceAll:
      ReduceImpl<T, And>(input, input_reshape, reduce_first_axis, And(), output, cuda_stream);
      break;
    case ReduceAny:
      ReduceImpl<T, Or>(input, input_reshape, reduce_first_axis, Or(), output, cuda_stream);
      break;
    case ReduceMean:
      size_t reduce_size = 1;
      size_t unreduce_size = 1;
      if (input_reshape.size() == 1) {
        reduce_size = input_reshape[0];
        unreduce_size = 1;
      } else if ((input_reshape.size() == 2) && (reduce_first_axis)) {
        reduce_size = input_reshape[0];
        unreduce_size = input_reshape[1];
      } else if ((input_reshape.size() == 2) && (!reduce_first_axis)) {
        reduce_size = input_reshape[1];
        unreduce_size = input_reshape[0];
      } else if ((input_reshape.size() == 3) && (reduce_first_axis)) {
        reduce_size = input_reshape[0] * input_reshape[2];
        unreduce_size = input_reshape[1];
      } else if ((input_reshape.size() == 3) && (!reduce_first_axis)) {
        reduce_size = input_reshape[1];
        unreduce_size = input_reshape[0] * input_reshape[2];
      }
      ReduceImpl<T, Sum<T>>(input, input_reshape, reduce_first_axis, Sum<T>(), output, cuda_stream);
      Average<<<(unreduce_size + 256) / 256, 256, 0, cuda_stream>>>(unreduce_size, reduce_size, output);
      break;
  }
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template <typename T>
cudaError_t ArrayReduceComplex(const T *input, const std::vector<size_t> &input_reshape, const bool reduce_first_axis,
                               ReduceType_t type, T *output, cudaStream_t cuda_stream) {
  switch (type) {
    case ReduceMax:
      break;
    case ReduceMin:
      break;
    case ReduceAll:
      break;
    case ReduceAny:
      break;
    case ReduceSum:
      ReduceImpl<T, Sum<T>>(input, input_reshape, reduce_first_axis, Sum<T>(), output, cuda_stream);
      break;
    case ReduceProd:
      ReduceImpl<T, Prod<T>>(input, input_reshape, reduce_first_axis, Prod<T>(), output, cuda_stream);
      break;
    case ReduceMean:
      size_t reduce_size = 1;
      size_t unreduce_size = 1;
      if (input_reshape.size() == 1) {
        reduce_size = input_reshape[0];
        unreduce_size = 1;
      } else if ((input_reshape.size() == 2) && (reduce_first_axis)) {
        reduce_size = input_reshape[0];
        unreduce_size = input_reshape[1];
      } else if ((input_reshape.size() == 2) && (!reduce_first_axis)) {
        reduce_size = input_reshape[1];
        unreduce_size = input_reshape[0];
      } else if ((input_reshape.size() == 3) && (reduce_first_axis)) {
        reduce_size = input_reshape[0] * input_reshape[2];
        unreduce_size = input_reshape[1];
      } else if ((input_reshape.size() == 3) && (!reduce_first_axis)) {
        reduce_size = input_reshape[1];
        unreduce_size = input_reshape[0] * input_reshape[2];
      }
      ReduceImpl<T, Sum<T>>(input, input_reshape, reduce_first_axis, Sum<T>(), output, cuda_stream);
      Average<<<(unreduce_size + 256) / 256, 256, 0, cuda_stream>>>(unreduce_size, reduce_size, output);
      break;
  }
  CHECK_CUDA_LAUNCH_SUCCESS();
}

#define ARRAY_REDUCE_REGISTER(T)                                                                               \
  template CUDA_LIB_EXPORT cudaError_t ArrayReduce(const T *input, const std::vector<size_t> &input_reshape,   \
                                                   const bool reduce_first_axis, ReduceType_t type, T *output, \
                                                   cudaStream_t cuda_stream)

ARRAY_REDUCE_REGISTER(float);
ARRAY_REDUCE_REGISTER(double);
ARRAY_REDUCE_REGISTER(half);
ARRAY_REDUCE_REGISTER(bool);
ARRAY_REDUCE_REGISTER(int8_t);
ARRAY_REDUCE_REGISTER(int16_t);
ARRAY_REDUCE_REGISTER(int32_t);
ARRAY_REDUCE_REGISTER(int64_t);
ARRAY_REDUCE_REGISTER(uint8_t);
ARRAY_REDUCE_REGISTER(uint16_t);
ARRAY_REDUCE_REGISTER(uint32_t);
ARRAY_REDUCE_REGISTER(uint64_t);

#define ARRAY_REDUCE_COMPLEX_REGISTER(T)                                                                              \
  template CUDA_LIB_EXPORT cudaError_t ArrayReduceComplex(const T *input, const std::vector<size_t> &input_reshape,   \
                                                          const bool reduce_first_axis, ReduceType_t type, T *output, \
                                                          cudaStream_t cuda_stream)

ARRAY_REDUCE_COMPLEX_REGISTER(Complex<float>);
ARRAY_REDUCE_COMPLEX_REGISTER(Complex<double>);
