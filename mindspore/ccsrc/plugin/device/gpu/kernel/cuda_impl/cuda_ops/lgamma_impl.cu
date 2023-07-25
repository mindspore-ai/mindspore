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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/lgamma_impl.cuh"
#include <limits>
#include "include/cuda_fp16.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/elementwise/elementswise_pub_impl.cuh"
constexpr uint kThreadsPerBlock = cuda::elementwise::kThreadsPerBlock;

template <typename T>
inline __device__ T calc_lgamma(const T input) {
  return static_cast<T>(lgamma(static_cast<double>(input)));
}

template <>
inline __device__ float calc_lgamma(const float input) {
  return lgammaf(input);
}

template <uint vec_size, typename T>
__device__ __forceinline__ void VectorizedCall(const T *input, T *output, uint offset) {
  uint tid = threadIdx.x;

  using VecT = cuda::elementwise::AlignVec<T, vec_size>;
  using VecBool = cuda::elementwise::AlignVec<bool, vec_size>;

  auto vec_input = reinterpret_cast<const VecT *>(input + offset);
  auto vec_output = reinterpret_cast<VecT *>(output + offset);
  VecT cache = vec_input[tid];
  VecT out1{0};

#pragma unroll
  for (uint j = 0; j < vec_size; j++) {
    auto output_pair = calc_lgamma(cache.elements_[j]);
    out1.elements_[j] = output_pair;
  }
  vec_output[tid] = out1;
}

template <uint vec_size, typename T>
__device__ __forceinline__ void NormalCall(const T *input, T *output, uint offset, uint remaining) {
  uint loop = UP_DIV(remaining, vec_size);
  for (uint i = threadIdx.x; i < loop; i += blockDim.x) {
#pragma unroll
    for (uint j = 0; j < vec_size; j++) {
      uint index = i * vec_size + j;
      if (index >= remaining) {
        return;
      }
      index += offset;
      auto output_pair = calc_lgamma(input[index]);
      output[index] = output_pair;
    }
  }
}

template <uint vec_size, typename T>
__global__ void CalLgammaKernel(const T *input, T *output, uint num_of_elements) {
  uint elements_per_block = kThreadsPerBlock * vec_size;
  for (uint offset = elements_per_block * blockIdx.x; offset < num_of_elements;
       offset += elements_per_block * gridDim.x) {
    uint remaining = num_of_elements - offset;
    if (remaining < elements_per_block) {
      NormalCall<vec_size, T>(input, output, offset, remaining);
    } else {
      VectorizedCall<vec_size, T>(input, output, offset);
    }
  }
}

template <typename T>
cudaError_t CalLgamma(size_t num_count, const T *input, T *output, const uint32_t &device_id,
                      cudaStream_t cuda_stream) {
  constexpr uint vec_size = cuda::elementwise::VecSize<T>();
  const auto block_x = uint(kThreadsPerBlock);
  const uint elements_per_block = kThreadsPerBlock * vec_size;
  const auto grid_x = uint(UP_DIV(num_count, elements_per_block));
  dim3 block{block_x};
  dim3 grid{grid_x};
  CalLgammaKernel<vec_size, T><<<grid, block, 0, cuda_stream>>>(input, output, num_count);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalLgamma(size_t num_count, const double *input, double *output,
                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLgamma(size_t num_count, const float *input, float *output,
                                               const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalLgamma(size_t num_count, const half *input, half *output,
                                               const uint32_t &device_id, cudaStream_t cuda_stream);
