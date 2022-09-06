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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/hsigmoid_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void HsigmoidKernel(size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T value = (input[pos] + static_cast<T>(3)) / static_cast<T>(6);
    value = value > static_cast<T>(1) ? static_cast<T>(1) : value;
    output[pos] = value > static_cast<T>(0) ? value : static_cast<T>(0);
  }
}

template <typename T>
__global__ void HsigmoidGradKernel(size_t size, const T *dout, const T *x, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T value = dout[pos] / static_cast<T>(6);
    output[pos] = (x[pos] > static_cast<T>(-3) && x[pos] < static_cast<T>(3)) ? value : static_cast<T>(0);
  }
}

template <typename T>
void CalHSigmoid(const size_t &size, const T *input, T *output, cudaStream_t cuda_stream) {
  HsigmoidKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
}

template <typename T>
void CalHSigmoidGrad(const size_t &size, const T *dout, const T *x, T *output, cudaStream_t cuda_stream) {
  HsigmoidGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dout, x, output);
}

template CUDA_LIB_EXPORT void CalHSigmoid<int8_t>(const size_t &size, const int8_t *input, int8_t *output,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoid<int16_t>(const size_t &size, const int16_t *input, int16_t *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoid<int32_t>(const size_t &size, const int32_t *input, int32_t *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoid<int64_t>(const size_t &size, const int64_t *input, int64_t *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoid<half>(const size_t &size, const half *input, half *output,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoid<float>(const size_t &size, const float *input, float *output,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoid<double>(const size_t &size, const double *input, double *output,
                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalHSigmoidGrad<int8_t>(const size_t &size, const int8_t *dout, const int8_t *x,
                                                      int8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoidGrad<int16_t>(const size_t &size, const int16_t *dout, const int16_t *x,
                                                       int16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoidGrad<int32_t>(const size_t &size, const int32_t *dout, const int32_t *x,
                                                       int32_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoidGrad<int64_t>(const size_t &size, const int64_t *dout, const int64_t *x,
                                                       int64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoidGrad<half>(const size_t &size, const half *dout, const half *x, half *output,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoidGrad<float>(const size_t &size, const float *dout, const float *x,
                                                     float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalHSigmoidGrad<double>(const size_t &size, const double *dout, const double *x,
                                                      double *output, cudaStream_t cuda_stream);
