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
#include "trace_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
__device__ void TraceAtomicAdd(T *address, T val) {
  MsAtomicAdd(address, val);
}

template <>
__device__ void TraceAtomicAdd(Complex<float> *address, Complex<float> val) {
  float *realAddr = reinterpret_cast<float *>(address);
  MsAtomicAdd(realAddr, val.real());
  MsAtomicAdd(realAddr + 1, val.imag());
}

template <>
__device__ void TraceAtomicAdd(Complex<double> *address, Complex<double> val) {
  double *realAddr = reinterpret_cast<double *>(address);
  MsAtomicAdd(realAddr, val.real());
  MsAtomicAdd(realAddr + 1, val.imag());
}

template <typename T, int threads_per_block>
__global__ void Trace(const T *input, const int64_t sum_size, const int64_t matrix_col, T *output) {
  *output = ZeroImpl<T>();
  __shared__ T sPartials[threads_per_block];
  T sum = ZeroImpl<T>();
  const int tid = threadIdx.x;
  for (size_t pos = blockIdx.x * blockDim.x + tid; pos < sum_size; pos += blockDim.x * gridDim.x) {
    sum += input[pos * matrix_col + pos];
  }
  sPartials[tid] = sum;
  __syncthreads();

  size_t floorPow2 = blockDim.x;
  if (floorPow2 & (floorPow2 - 1)) {
    while (floorPow2 & (floorPow2 - 1)) {
      floorPow2 &= (floorPow2 - 1);
    }
    if (tid >= floorPow2) {
      sPartials[tid - floorPow2] += sPartials[tid];
    }
    __syncthreads();
  }
  for (size_t activeTrheads = floorPow2 / 2; activeTrheads > 0; activeTrheads /= 2) {
    if (tid < activeTrheads) {
      sPartials[tid] += sPartials[tid + activeTrheads];
    }
    __syncthreads();
  }
  if (tid == 0) {
    TraceAtomicAdd(output, sPartials[0]);
  }
  return;
}

template <typename T>
void CalTrace(const T *input, const int64_t sum_size, const int64_t matrix_col, T *output, const uint32_t &device_id,
              cudaStream_t cuda_stream) {
  constexpr size_t thread_nums = 64;
  size_t block_nums = (sum_size + thread_nums - 1) / thread_nums;
  Trace<T, thread_nums><<<block_nums, thread_nums, 0, cuda_stream>>>(input, sum_size, matrix_col, output);
  return;
}

template CUDA_LIB_EXPORT void CalTrace<uint8_t>(const uint8_t *input, const int64_t sum_size, const int64_t matrix_col,
                                                uint8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<uint16_t>(const uint16_t *input, const int64_t sum_size,
                                                 const int64_t matrix_col, uint16_t *output, const uint32_t &device_id,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<uint32_t>(const uint32_t *input, const int64_t sum_size,
                                                 const int64_t matrix_col, uint32_t *output, const uint32_t &device_id,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<uint64_t>(const uint64_t *input, const int64_t sum_size,
                                                 const int64_t matrix_col, uint64_t *output, const uint32_t &device_id,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<int8_t>(const int8_t *input, const int64_t sum_size, const int64_t matrix_col,
                                               int8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<int16_t>(const int16_t *input, const int64_t sum_size, const int64_t matrix_col,
                                                int16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<int>(const int *input, const int64_t sum_size, const int64_t matrix_col,
                                            int *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<int64_t>(const int64_t *input, const int64_t sum_size, const int64_t matrix_col,
                                                int64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<half>(const half *input, const int64_t sum_size, const int64_t matrix_col,
                                             half *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<float>(const float *input, const int64_t sum_size, const int64_t matrix_col,
                                              float *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<double>(const double *input, const int64_t sum_size, const int64_t matrix_col,
                                               double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<bool>(const bool *input, const int64_t sum_size, const int64_t matrix_col,
                                             bool *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<Complex<float>>(const Complex<float> *input, const int64_t sum_size,
                                                       const int64_t matrix_col, Complex<float> *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalTrace<Complex<double>>(const Complex<double> *input, const int64_t sum_size,
                                                        const int64_t matrix_col, Complex<double> *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
