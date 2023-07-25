/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <math.h>
#include <iostream>
#include <limits>
#include <algorithm>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/softmax_impl.cuh"

#define WARPSIZE 32
const int max_threads = 1024;
constexpr int ALIGN_BYTES = 16;

inline dim3 SpatialSoftMaxGetGridSize(dim3 *block, uint32_t activate_block, uint64_t outer_size, uint64_t dim_size,
                                      uint64_t inner_size) {
  uint32_t inner = (inner_size + block->y - 1) / block->y;
  if (inner > activate_block) {
    inner = activate_block;
  }
  uint32_t outer = (activate_block + inner - 1) / inner;
  if (outer > outer_size) {
    outer = outer_size;
  }
  return dim3(outer, inner);
}

inline dim3 SpatialSoftMaxGetBlockSize(uint64_t outer_size, uint64_t dim_size, uint64_t inner_size) {
  uint32_t inner_ths = inner_size;
  inner_ths = std::min(inner_ths, static_cast<uint32_t>(max_threads));
  uint32_t dim_threads = 1;
  if (inner_ths <= 64 && dim_size >= 64) {
    while ((inner_ths * dim_threads <= max_threads) && (dim_threads <= dim_size)) {
      dim_threads *= 2;
    }
    dim_threads /= 2;
  }
  return dim3(dim_threads, inner_ths);
}

template <typename accumulate_t, typename Kernel>
void SpatialSoftMaxGetLaunchSizes(Kernel k, uint64_t outer_size, uint64_t dim_size, uint64_t inner_size, dim3 *grid,
                                  dim3 *block, uint32_t *smem_size, uint32_t device_id) {
  *block = SpatialSoftMaxGetBlockSize(outer_size, dim_size, inner_size);
  uint32_t block_ths = block->x * block->y;
  if (block->x == 1) {
    *smem_size = 0;
  } else {
    *smem_size = block_ths * sizeof(accumulate_t);
  }
  int activate_size;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activate_size, k, block_ths, *smem_size);
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  activate_size *= prop.multiProcessorCount;
  *grid = SpatialSoftMaxGetGridSize(block, activate_size, outer_size, dim_size, inner_size);
}

int log2_ceil(int val) {
  int final_val = 0;
  while ((1 << final_val) < val) ++final_val;
  return final_val;
}

inline dim3 SoftMaxGetBlockSize(int ins, uint64_t dim) {
  uint64_t block_size = 1;
  uint64_t max_block_size = std::min(dim / ins, static_cast<uint64_t>(max_threads));

  if (ins > 1) {
    max_block_size /= 2;
  }

  while (block_size < (max_block_size)) {
    block_size *= 2;
  }
  block_size = std::max(block_size, static_cast<uint64_t>(WARPSIZE));
  return dim3(block_size);
}

template <typename T, typename AccT>
struct GetMaxFloat {
  __device__ __forceinline__ AccT operator()(AccT max, T v) const { return ::max(max, (AccT)v); }
};

template <typename T, typename AccT>
struct GetSumExpFloat {
  __device__ __forceinline__ GetSumExpFloat(AccT v) : max_k(v) {}

  __device__ __forceinline__ AccT operator()(AccT sum, T v) const { return sum + std::exp((AccT)v - max_k); }

  const AccT max_k;
};

template <typename T>
__device__ __forceinline__ T WARPSHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff) {
#ifndef __HIP_PLATFORM_HCC__
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

template <typename T, typename Function>
__forceinline__ __device__ T SpatialBlockReduceX(T *memsha, T val) {
  Function r = Function();
  memsha += threadIdx.y * blockDim.x;
  __syncthreads();
  memsha[threadIdx.x] = val;
  int offset = blockDim.x / 2;
  while (offset > 0) {
    __syncthreads();
    if (threadIdx.x < offset) memsha[threadIdx.x] = r(memsha[threadIdx.x], memsha[threadIdx.x + offset]);
    offset /= 2;
  }
  __syncthreads();
  return memsha[0];
}

template <typename input_t, typename accumulate_t, typename output_t, bool is_log_softmax>
__global__ void SpatialSoftMaxForward(output_t *output, input_t *input, uint32_t outer_size, uint32_t dim_size,
                                      uint32_t inner_size) {
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accumulate_t *>(smem);
  const uint32_t outer_stride = inner_size * dim_size;
  const uint32_t dim_stride = inner_size;
  for (uint32_t outer_index = blockIdx.x; outer_index < outer_size; outer_index += gridDim.x) {
    const uint32_t outer_offset = outer_index * outer_stride;
    for (uint32_t inner_index = blockIdx.y * blockDim.y + threadIdx.y; inner_index < inner_size;
         inner_index += blockDim.y * gridDim.y) {
      const uint32_t offset = outer_offset + inner_index;
      if (blockDim.x > 1) {
        accumulate_t max_data_input = std::numeric_limits<accumulate_t>::lowest();
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          const accumulate_t value = static_cast<accumulate_t>(input[offset + d * dim_stride]);
          max_data_input = atomic::Max()(max_data_input, value);
        }
        max_data_input = SpatialBlockReduceX<accumulate_t, atomic::Max>(sdata, max_data_input);
        accumulate_t sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += std::exp(static_cast<accumulate_t>(input[offset + d * dim_stride]) - max_data_input);
        sum = SpatialBlockReduceX<accumulate_t, atomic::Add>(sdata, sum);
        SoftMaxForwardEpilogue<input_t, accumulate_t, output_t, is_log_softmax> epilogue(max_data_input, sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          output[offset + d * dim_stride] = epilogue(input[offset + d * dim_stride]);
      } else {
        accumulate_t max_data_input = std::numeric_limits<accumulate_t>::lowest();
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x) {
          const accumulate_t value = static_cast<accumulate_t>(input[offset + d * dim_stride]);
          max_data_input = atomic::Max()(max_data_input, value);
        }
        accumulate_t sum = 0;
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          sum += std::exp(static_cast<accumulate_t>(input[offset + d * dim_stride]) - max_data_input);

        SoftMaxForwardEpilogue<input_t, accumulate_t, output_t, is_log_softmax> epilogue(max_data_input, sum);
        for (uint32_t d = threadIdx.x; d < dim_size; d += blockDim.x)
          output[offset + d * dim_stride] = epilogue(input[offset + d * dim_stride]);
      }
    }
  }
}

template <int InsP, typename input_t, typename accum_t, typename output_t, bool is_log_softmax>
__device__ __forceinline__ void WriteResults(int clas, input_t *input, output_t *output, input_t max_k,
                                             input_t sum_all) {
  SoftMaxForwardEpilogue<input_t, accum_t, output_t, is_log_softmax> epilogue(max_k, sum_all);
  int offset = threadIdx.x;
  int last = clas % (InsP * blockDim.x);
  for (; offset < clas - last; offset += blockDim.x * InsP) {
    input_t tmp[InsP];
#pragma unroll
    for (int j = 0; j < InsP; ++j) {
      tmp[j] = input[offset + j * blockDim.x];
    }
#pragma unroll
    for (int j = 0; j < InsP; ++j) {
      output[offset + j * blockDim.x] = epilogue(tmp[j]);
    }
  }
  for (; offset < clas; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

template <int InsP, typename input_t, typename accum_t, typename output_t, bool is_log_softmax>
__device__ __forceinline__ void WriteResultsVectorized(int size, const int deviate, input_t *input, output_t *output,
                                                       input_t max_k, input_t sum_all) {
  SoftMaxForwardEpilogue<input_t, accum_t, output_t, is_log_softmax> epilogue(max_k, sum_all);
  using loadT = aligned_vector<input_t>;
  using storeT = aligned_vector<output_t>;
  int offset = threadIdx.x;
  if (deviate > 0) {
    input -= deviate;
    output -= deviate;
    size += deviate;
    if (threadIdx.x >= deviate) {
      output[offset] = epilogue(input[offset]);
    }
    size -= blockDim.x;
    input += blockDim.x;
    output += blockDim.x;
  }
  const int last = size % (InsP * blockDim.x);
  input_t in_v[InsP];
  loadT *in_value = reinterpret_cast<loadT *>(&in_v);
  output_t out_v[InsP];
  storeT *out_value = reinterpret_cast<storeT *>(&out_v);
  for (; offset * InsP < (size - last); offset += blockDim.x) {
    *in_value = reinterpret_cast<loadT *>(input)[offset];
#pragma unroll
    for (int j = 0; j < InsP; ++j) {
      out_v[j] = epilogue(in_v[j]);
    }
    reinterpret_cast<storeT *>(output)[offset] = *out_value;
  }
  offset = size - last + threadIdx.x;
  for (; offset < size; offset += blockDim.x) {
    output[offset] = epilogue(input[offset]);
  }
}

template <typename Reduction, typename AccT>
__device__ __forceinline__ AccT ReduceBlock(AccT *sharemen, AccT val, AccT initVal) {
  Reduction r = Reduction();
  __syncthreads();
  sharemen[threadIdx.x] = val;
  __syncthreads();
  AccT warpVal = initVal;
  uint32_t mask = (((uint64_t)1) << (blockDim.x / WARPSIZE)) - 1;
  if (threadIdx.x < WARPSIZE) {
    int lane = threadIdx.x % WARPSIZE;
    if (lane < blockDim.x / WARPSIZE) {
#pragma unroll
      for (int i = 0; i < WARPSIZE; ++i) {
        warpVal = r(warpVal, sharemen[lane * WARPSIZE + i]);
      }
#ifndef __HIP_PLATFORM_HCC__
      __syncwarp(mask);
#endif
      sharemen[lane] = warpVal;
    }
  }
  __syncthreads();
  AccT blockVal = initVal;
  if (threadIdx.x == 0) {
    for (int i = 0; i < blockDim.x / WARPSIZE; ++i) {
      blockVal = r(blockVal, sharemen[i]);
    }
    sharemen[0] = blockVal;
  }
  __syncthreads();
  return sharemen[0];
}

template <template <typename, typename> class Reduction, int InsP, typename T, typename AccT>
__device__ __forceinline__ AccT ILPReduce(int shift, T *data, int size, const Reduction<T, AccT> &r, AccT initVal) {
  using loadT = aligned_vector<T>;
  AccT threadVal = initVal;
  int offset = threadIdx.x;
  if (shift > 0) {
    data -= shift;
    size += shift;
    if (threadIdx.x >= shift) {
      threadVal = r(threadVal, data[offset]);
    }
    size -= blockDim.x;
    data += blockDim.x;
  }
  int last = size % (InsP * blockDim.x);
  T v[InsP];
  loadT *value = reinterpret_cast<loadT *>(&v);
  for (; offset * InsP < (size - last); offset += blockDim.x) {
    *value = reinterpret_cast<loadT *>(data)[offset];
#pragma unroll
    for (int j = 0; j < InsP; ++j) {
      threadVal = r(threadVal, v[j]);
    }
  }
  offset = size - last + threadIdx.x;
  for (; offset < size; offset += blockDim.x) threadVal = r(threadVal, data[offset]);
  return threadVal;
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, typename func>
__device__ __forceinline__ void warp_reduce(acc_t *sum) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int i = 0; i < WARP_BATCH; ++i) {
      acc_t b = WARPSHFL_XOR(sum[i], offset, WARP_SIZE);
      sum[i] = func()(sum[i], b);
    }
  }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax, bool is_masked>
__global__ void SoftMaxWarpForward(output_t *dst, const input_t *src, int batch_size, int stride, int element_count,
                                   const bool *mask = nullptr, const int head_chunk_size = -1,
                                   bool is_transformer_mask = false) {
  constexpr int next_power_of_two = 1 << log2_elements;
  constexpr int WARP_SIZE = (next_power_of_two < WARPSIZE) ? next_power_of_two : WARPSIZE;
  constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
  constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

  int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;
  int local_batches = batch_size - first_batch;
  if (local_batches > WARP_BATCH) local_batches = WARP_BATCH;
  int local_idx = threadIdx.x;
  int idx_offset = first_batch * stride + local_idx;

  src += idx_offset;
  dst += idx_offset;

  if (is_transformer_mask) {
    mask += ((first_batch * stride) / head_chunk_size) * stride + local_idx;
  } else {
    mask += idx_offset;
  }
  acc_t elements[WARP_BATCH][WARP_ITERATIONS];
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < batch_element_count) {
        elements[i][it] = src[i * element_count + it * WARP_SIZE];
      } else {
        elements[i][it] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }

  acc_t max_value[WARP_BATCH];
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
    bool is_meaningful_max = false;
    max_value[i] = elements[i][0];
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      if (is_masked) {
        int idx = it * WARP_SIZE;
        if ((idx + local_idx) < batch_element_count) {
          if (!is_transformer_mask) {
            idx += i * element_count;
          }
          if (!mask[idx]) {
            max_value[i] = (is_meaningful_max && max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
            is_meaningful_max = true;
          }
        }
      } else {
        max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
      }
    }
    if (is_masked) {
      if (!is_meaningful_max) {
        max_value[i] = -std::numeric_limits<acc_t>::infinity();
      }
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, atomic::Max>(max_value);

  acc_t sum[WARP_BATCH]{0.0f};
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    int batch_element_count = (i >= local_batches) ? 0 : element_count;
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      if (!is_masked) {
        if (is_log_softmax) {
          sum[i] += std::exp(elements[i][it] - max_value[i]);
        } else {
          elements[i][it] = std::exp(elements[i][it] - max_value[i]);
          sum[i] += elements[i][it];
        }
      } else {
        int idx = it * WARP_SIZE;
        bool valid = (idx + local_idx) < batch_element_count;
        if (!is_transformer_mask) {
          idx += i * element_count;
        }
        if (valid) {
          if (!mask[idx]) {
            if (is_log_softmax) {
              sum[i] += std::exp(elements[i][it] - max_value[i]);
            } else {
              elements[i][it] = std::exp(elements[i][it] - max_value[i]);
              sum[i] += elements[i][it];
            }
          } else {
            if (!is_log_softmax) {
              elements[i][it] = 0;
            }
          }
        } else {
          if (!is_log_softmax) {
            elements[i][it] = 0.;
          }
        }
      }
    }
  }
  warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, atomic::Add>(sum);
#pragma unroll
  for (int i = 0; i < WARP_BATCH; ++i) {
    if (i >= local_batches) break;
    if (is_log_softmax) sum[i] = std::log(sum[i]);
#pragma unroll
    for (int it = 0; it < WARP_ITERATIONS; ++it) {
      int element_index = local_idx + it * WARP_SIZE;
      if (element_index < element_count) {
        if (is_log_softmax) {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
        } else if (sum[i] == 0) {
          dst[i * element_count + it * WARP_SIZE] = std::numeric_limits<acc_t>::quiet_NaN();
        } else {
          dst[i * element_count + it * WARP_SIZE] = elements[i][it] / sum[i];
        }
      } else {
        break;
      }
    }
  }
}

template <int InsP, typename T, typename accumulate_t, bool is_log_softmax>
__global__ void cunn_SoftMaxForward(T *output, T *input, int classes) {
  extern __shared__ unsigned char smem[];
  auto sdata = reinterpret_cast<accumulate_t *>(smem);

  using loadT = aligned_vector<T>;
  using storeT = aligned_vector<T>;
  input += blockIdx.x * classes;
  output += blockIdx.x * classes;
  const int shift = ((uint64_t)input) % ALIGN_BYTES / sizeof(T);
  const int output_shift = ((uint64_t)output) % ALIGN_BYTES / sizeof(T);

  accumulate_t threadMax = ILPReduce<GetMaxFloat, InsP, T, accumulate_t>(
    shift, input, classes, GetMaxFloat<T, accumulate_t>(), -std::numeric_limits<accumulate_t>::max());
  accumulate_t max_k =
    ReduceBlock<atomic::Max, accumulate_t>(sdata, threadMax, -std::numeric_limits<accumulate_t>::max());

  accumulate_t threadExp = ILPReduce<GetSumExpFloat, InsP, T, accumulate_t>(
    shift, input, classes, GetSumExpFloat<T, accumulate_t>(max_k), static_cast<accumulate_t>(0));
  accumulate_t sumAll = ReduceBlock<atomic::Add, accumulate_t>(sdata, threadExp, static_cast<accumulate_t>(0));

  if (shift == output_shift) {
    WriteResultsVectorized<InsP, T, accumulate_t, T, is_log_softmax>(classes, shift, input, output, max_k, sumAll);
  } else {
    WriteResults<InsP, T, accumulate_t, T, is_log_softmax>(classes, input, output, max_k, sumAll);
  }
}

// end of kernel function
template <typename input_t, typename output_t, typename acc_t, bool is_log_softmax, bool is_masked>
void dispatch_softmax_forward(output_t *dst, const input_t *src, int softmax_elements, int softmax_elements_stride,
                              int batch_count, cudaStream_t stream, const bool *mask = nullptr, int chunk_size = -1,
                              bool is_transformer_mask = false) {
  if (softmax_elements == 0) {
    return;
  } else {
    int log2_elements = log2_ceil(softmax_elements);
    const int next_power_of_two = 1 << log2_elements;

    int warp_size = (next_power_of_two < WARPSIZE) ? next_power_of_two : WARPSIZE;
    int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int threads_per_block = 128;

    int warps_per_block = (threads_per_block / warp_size);
    int batches_per_block = warps_per_block * batches_per_warp;
    int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
    dim3 threads(warp_size, warps_per_block, 1);

    switch (log2_elements) {
#define LAUNCH_SOFTMAX_WARP_FORWARD(L2E)                                                                          \
  case L2E:                                                                                                       \
    SoftMaxWarpForward<input_t, output_t, acc_t, L2E, is_log_softmax, is_masked><<<blocks, threads, 0, stream>>>( \
      dst, src, batch_count, softmax_elements_stride, softmax_elements, mask, chunk_size, is_transformer_mask);   \
    break;

      LAUNCH_SOFTMAX_WARP_FORWARD(0);   // 1
      LAUNCH_SOFTMAX_WARP_FORWARD(1);   // 2
      LAUNCH_SOFTMAX_WARP_FORWARD(2);   // 4
      LAUNCH_SOFTMAX_WARP_FORWARD(3);   // 8
      LAUNCH_SOFTMAX_WARP_FORWARD(4);   // 16
      LAUNCH_SOFTMAX_WARP_FORWARD(5);   // 32
      LAUNCH_SOFTMAX_WARP_FORWARD(6);   // 64
      LAUNCH_SOFTMAX_WARP_FORWARD(7);   // 128
      LAUNCH_SOFTMAX_WARP_FORWARD(8);   // 256
      LAUNCH_SOFTMAX_WARP_FORWARD(9);   // 512
      LAUNCH_SOFTMAX_WARP_FORWARD(10);  // 1024
      default:
        break;
    }
  }
}

template <typename T, bool is_log_softmax>
cudaError_t Softmax(T *input_, T *output_, size_t dim_size_, size_t outer_size_, size_t inner_size_, size_t device_id,
                    cudaStream_t cuda_stream) {
  using accumulate_t = acc_type<T, true>;
  if (inner_size_ == 1) {
    dim3 grid(outer_size_);
    if (dim_size_ <= 1024 && dim_size_ * sizeof(T) <= 4096) {
      int64_t remaining = outer_size_;
      int64_t chunk_size = (1L << 30L) / dim_size_;
      while (remaining > 0) {
        dispatch_softmax_forward<T, T, accumulate_t, is_log_softmax, false>(output_, input_, dim_size_, dim_size_,
                                                                            std::min<int64_t>(remaining, chunk_size),
                                                                            cuda_stream, nullptr /* not masked */);
        input_ += chunk_size * dim_size_;
        output_ += chunk_size * dim_size_;
        remaining -= chunk_size;
      }
    } else {
      constexpr int InsP = sizeof(float4) / sizeof(T);
      dim3 block = SoftMaxGetBlockSize(InsP, dim_size_);
      cunn_SoftMaxForward<InsP, T, accumulate_t, is_log_softmax>
        <<<grid, block, block.x * sizeof(accumulate_t), cuda_stream>>>(output_, input_, dim_size_);
    }
  } else {
    uint32_t smem_size;
    dim3 grid, block;
    SpatialSoftMaxGetLaunchSizes<T>(&SpatialSoftMaxForward<T, accumulate_t, T, is_log_softmax>, outer_size_, dim_size_,
                                    inner_size_, &grid, &block, &smem_size, device_id);
    SpatialSoftMaxForward<T, accumulate_t, T, is_log_softmax>
      <<<grid, block, smem_size, cuda_stream>>>(output_, input_, outer_size_, dim_size_, inner_size_);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t Softmax<double, false>(double *input_, double *output_, size_t dim_,
                                                            size_t outer_size_, size_t inner_size_, size_t device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Softmax<double, true>(double *input_, double *output_, size_t dim_,
                                                           size_t outer_size_, size_t inner_size_, size_t device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Softmax<float, false>(float *input_, float *output_, size_t dim_,
                                                           size_t outer_size_, size_t inner_size_, size_t device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Softmax<float, true>(float *input_, float *output_, size_t dim_,
                                                          size_t outer_size_, size_t inner_size_, size_t device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Softmax<half, false>(half *input_, half *output_, size_t dim_, size_t outer_size_,
                                                          size_t inner_size_, size_t device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t Softmax<half, true>(half *input_, half *output_, size_t dim_, size_t outer_size_,
                                                         size_t inner_size_, size_t device_id,
                                                         cudaStream_t cuda_stream);
