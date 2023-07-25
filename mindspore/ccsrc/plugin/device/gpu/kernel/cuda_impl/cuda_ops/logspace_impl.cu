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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/logspace_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void LogSpaceKernel(const T *start, const T *end, const int64_t steps, const size_t base, T *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < steps; i += gridDim.x * blockDim.x) {
    T add = (end[0] - start[0]) / (static_cast<T>(steps == 1 ? steps : steps - 1));
    output[i] = pow(static_cast<T>(base), start[0] + (add * i));
  }
  return;
}

template <>
__global__ void LogSpaceKernel(const half *start, const half *end, const int64_t steps, const size_t base,
                               half *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < steps; i += gridDim.x * blockDim.x) {
    float start_float = __half2float(start[0]);
    float add = (__half2float(end[0]) - start_float) / (static_cast<float>(steps == 1 ? steps : steps - 1));
    output[i] = __float2half(pow(static_cast<float>(base), start_float + (add * i)));
  }
  return;
}

template <typename T>
cudaError_t CalLogSpace(const T *start, const T *end, const int64_t steps, const size_t base, T *output,
                        const uint32_t &device_id, cudaStream_t cuda_stream) {
  LogSpaceKernel<<<CUDA_BLOCKS(device_id, steps), CUDA_THREADS(device_id), 0, cuda_stream>>>(start, end, steps, base,
                                                                                             output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalLogSpace<half>(const half *start, const half *end, const int64_t steps,
                                                       const size_t base, half *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalLogSpace<float>(const float *start, const float *end, const int64_t steps,
                                                        const size_t base, float *output, const uint32_t &device_id,
                                                        cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalLogSpace<double>(const double *start, const double *end, const int64_t steps,
                                                         const size_t base, double *output, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
