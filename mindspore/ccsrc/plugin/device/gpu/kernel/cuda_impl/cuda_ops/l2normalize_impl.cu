/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "l2normalize_impl.cuh"
#include "include/cuda_fp16.h"
template <typename T>
__global__ void AssignEps(const size_t size, const float eps, T *value) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    double v = static_cast<double>(value[pos]);
    double max = v > eps ? v : eps;
    value[pos] = static_cast<T>(max);
  }
}

template <typename T>
cudaError_t GetMaxWithEpsAndValue(const size_t size, const float eps, T *value, cudaStream_t cuda_stream) {
  AssignEps<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, eps, value);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t GetMaxWithEpsAndValue<float>(const size_t size, const float eps, float *value,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t GetMaxWithEpsAndValue<half>(const size_t size, const float eps, half *value,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t GetMaxWithEpsAndValue<int>(const size_t size, const float eps, int *value,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t GetMaxWithEpsAndValue<double>(const size_t size, const float eps, double *value,
                                                                   cudaStream_t cuda_stream);
