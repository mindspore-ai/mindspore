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

#include <stdint.h>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/bernoulli_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T, typename S>
__global__ void BernoulliForwardKernel(const T *p, S *output, uint64_t seed, const size_t num_count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_count; i += blockDim.x * gridDim.x) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, i, 0, &state);
    CUDA_KERNEL_ASSERT(p[i] >= 0 && p[i] <= 1);
    output[i] = static_cast<S>(curand_uniform(&state) <= p[i]);
  }
}

// Broadcast comparison
__device__ __forceinline__ size_t Index(const size_t &index, const size_t &dim) { return dim == 1 ? 0 : index; }

template <typename T, typename S>
__global__ void BroadcastBernoulliForwardKernel(const size_t x0, const size_t x1, const size_t x2, const size_t x3,
                                                const size_t x4, const size_t x5, const size_t x6, const size_t p0,
                                                const size_t p1, const size_t p2, const size_t p3, const size_t p4,
                                                const size_t p5, const size_t p6, const T *p, S *output,
                                                uint64_t seed, const size_t num_count) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < num_count; pos += blockDim.x * gridDim.x) {
    size_t i = pos / (x1 * x2 * x3 * x4 * x5 * x6) % x0;
    size_t j = pos / (x2 * x3 * x4 * x5 * x6) % x1;
    size_t k = pos / (x3 * x4 * x5 * x6) % x2;
    size_t l = pos / (x4 * x5 * x6) % x3;
    size_t m = pos / (x5 * x6) % x4;
    size_t n = pos / x6 % x5;
    size_t o = pos % x6;

    size_t p_index = Index(i, p0) * p1 * p2 * p3 * p4 * p5 * p6;
    p_index += Index(j, p1) * p2 * p3 * p4 * p5 * p6;
    p_index += Index(k, p2) * p3 * p4 * p5 * p6;
    p_index += Index(l, p3) * p4 * p5 * p6;
    p_index += Index(m, p4) * p5 * p6;
    p_index += Index(n, p5) * p6;
    p_index += Index(o, p6);

    curandStatePhilox4_32_10_t state;
    curand_init(seed, pos, 0, &state);
    CUDA_KERNEL_ASSERT(p[p_index] >= 0 && p[p_index] <= 1);
    output[pos] = static_cast<S>(curand_uniform(&state) <= p[p_index]);
  }
}

template <typename T, typename S>
void BernoulliForward(const T *input, S *output, uint64_t seed, const size_t num_count, const uint32_t &device_id,
                      cudaStream_t cuda_stream) {
  int block_num = 256 > num_count ? num_count : 256;
  BernoulliForwardKernel<<<CUDA_BLOCKS_CAL(device_id, num_count, block_num), block_num, 0, cuda_stream>>>(
    input, output, seed, num_count);
}

template <typename T, typename S>
void BroadcastBernoulliForward(const std::vector<size_t> &x_dims, const std::vector<size_t> &p_dims, const T *input,
                               S *output, uint64_t seed, const size_t num_count, const uint32_t &device_id,
                               cudaStream_t cuda_stream) {
  int block_num = 256 > num_count ? num_count : 256;
  BroadcastBernoulliForwardKernel<<<CUDA_BLOCKS_CAL(device_id, num_count, block_num), block_num, 0, cuda_stream>>>(
    x_dims[0], x_dims[1], x_dims[2], x_dims[3], x_dims[4], x_dims[5], x_dims[6], p_dims[0], p_dims[1],
    p_dims[2], p_dims[3], p_dims[4], p_dims[5], p_dims[6], input, output, seed, num_count);
}

template CUDA_LIB_EXPORT void BernoulliForward<float, float>(const float *input, float *output, uint64_t seed,
                                                             const size_t num_count, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<float, double>(const float *input, double *output, uint64_t seed,
                                                              const size_t num_count, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<float, bool>(const float *input, bool *output, uint64_t seed,
                                                            const size_t num_count, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BernoulliForward<double, float>(const double *input, float *output, uint64_t seed,
                                                              const size_t num_count, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<double, double>(const double *input, double *output, uint64_t seed,
                                                               const size_t num_count, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<double, bool>(const double *input, bool *output, uint64_t seed,
                                                             const size_t num_count, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BernoulliForward<float, int8_t>(const float *input, int8_t *output, uint64_t seed,
                                                              const size_t num_count, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<float, uint8_t>(const float *input, uint8_t *output, uint64_t seed,
                                                               const size_t num_count, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<float, int16_t>(const float *input, int16_t *output, uint64_t seed,
                                                               const size_t num_count, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<float, int32_t>(const float *input, int32_t *output, uint64_t seed,
                                                               const size_t num_count, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<float, int64_t>(const float *input, int64_t *output, uint64_t seed,
                                                               const size_t num_count, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BernoulliForward<double, int8_t>(const double *input, int8_t *output, uint64_t seed,
                                                               const size_t num_count, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<double, uint8_t>(const double *input, uint8_t *output, uint64_t seed,
                                                                const size_t num_count, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<double, int16_t>(const double *input, int16_t *output, uint64_t seed,
                                                                const size_t num_count, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<double, int32_t>(const double *input, int32_t *output, uint64_t seed,
                                                                const size_t num_count, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BernoulliForward<double, int64_t>(const double *input, int64_t *output, uint64_t seed,
                                                                const size_t num_count, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BroadcastBernoulliForward<float, float>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const float *input, float *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<float, double>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const float *input, double *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<float, bool>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const float *input, bool *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BroadcastBernoulliForward<double, float>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const double *input, float *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<double, double>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const double *input, double *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<double, bool>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const double *input, bool *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id,  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BroadcastBernoulliForward<float, int8_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const float *input, int8_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<float, uint8_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const float *input, uint8_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<float, int16_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const float *input, int16_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<float, int32_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const float *input, int32_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<float, int64_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const float *input, int64_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void BroadcastBernoulliForward<double, int8_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const double *input, int8_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<double, uint8_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const double *input, uint8_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<double, int16_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const double *input, int16_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<double, int32_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const double *input, int32_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void BroadcastBernoulliForward<double, int64_t>(const std::vector<size_t> &x_dims,
  const std::vector<size_t> &p_dims, const double *input, int64_t *output, uint64_t seed, const size_t num_count,
  const uint32_t &device_id, cudaStream_t cuda_stream);
