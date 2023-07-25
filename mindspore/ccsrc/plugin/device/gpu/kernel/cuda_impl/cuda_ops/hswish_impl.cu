/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/hswish_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void HSwishKernel(size_t size, const T *input, T *output) {
  const auto add_factor = static_cast<T>(3);
  const auto div_factor = static_cast<T>(6);
  const auto max_threshold = static_cast<T>(3);
  const auto min_threshold = static_cast<T>(-3);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const T value = input[pos];
    if (value <= min_threshold) {
      output[pos] = static_cast<T>(0);
    } else if (value >= max_threshold) {
      output[pos] = value;
    } else {
      output[pos] = value * (value + add_factor) / div_factor;
    }
  }
}

template <typename T>
__global__ void HSwishGradKernel(size_t size, const T *dout, const T *x, T *output) {
  auto two = static_cast<T>(2);
  auto three = static_cast<T>(3);
  auto six = static_cast<T>(6);
  const auto max_threshold = static_cast<T>(3);
  const auto min_threshold = static_cast<T>(-3);

  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const T value = x[pos];
    if (value <= min_threshold) {
      output[pos] = static_cast<T>(0);
    } else if (value >= max_threshold) {
      output[pos] = dout[pos];
    } else {
      output[pos] = dout[pos] * (two * value + three) / six;
    }
  }
}

template <typename T>
cudaError_t CalHSwish(const size_t &size, const T *input, T *output, cudaStream_t cuda_stream) {
  HSwishKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalHSwishGrad(const size_t &size, const T *dout, const T *x, T *output, cudaStream_t cuda_stream) {
  HSwishGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dout, x, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalHSwish<int8_t>(const size_t &size, const int8_t *input, int8_t *output,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwish<int16_t>(const size_t &size, const int16_t *input, int16_t *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwish<int32_t>(const size_t &size, const int32_t *input, int32_t *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwish<int64_t>(const size_t &size, const int64_t *input, int64_t *output,
                                                        cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwish<half>(const size_t &size, const half *input, half *output,
                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwish<float>(const size_t &size, const float *input, float *output,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwish<double>(const size_t &size, const double *input, double *output,
                                                       cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalHSwishGrad<int8_t>(const size_t &size, const int8_t *dout, const int8_t *x,
                                                           int8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwishGrad<int16_t>(const size_t &size, const int16_t *dout, const int16_t *x,
                                                            int16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwishGrad<int32_t>(const size_t &size, const int32_t *dout, const int32_t *x,
                                                            int32_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwishGrad<int64_t>(const size_t &size, const int64_t *dout, const int64_t *x,
                                                            int64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwishGrad<half>(const size_t &size, const half *dout, const half *x,
                                                         half *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwishGrad<float>(const size_t &size, const float *dout, const float *x,
                                                          float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalHSwishGrad<double>(const size_t &size, const double *dout, const double *x,
                                                           double *output, cudaStream_t cuda_stream);
