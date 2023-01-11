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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/zeta_impl.cuh"
#include <math.h>
#include <limits>
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"
#include "unsupported/Eigen/CXX11/Tensor"
template <typename T>
__device__ __forceinline__ T zeta(T x, T q) {
  return Eigen::internal::scalar_zeta_op<T>()(x, q);
}

template <typename T>
__global__ void ZetaKernel(const size_t size, const T *x, const T *dimension, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = zeta(x[pos], dimension[pos]);
  }
  return;
}

template <typename T>
void CalZeta(const size_t size, const T *x, const T *dimension, T *output, const uint32_t &device_id,
             cudaStream_t cuda_stream) {
  ZetaKernel<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, x, dimension, output);
}

template CUDA_LIB_EXPORT void CalZeta<float>(const size_t size, const float *x, const float *dimension, float *output,
                                             const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalZeta<double>(const size_t size, const double *x, const double *dimension,
                                              double *output, const uint32_t &device_id, cudaStream_t cuda_stream);
