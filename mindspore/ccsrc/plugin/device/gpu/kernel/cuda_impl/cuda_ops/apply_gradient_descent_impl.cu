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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/apply_gradient_descent_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
__global__ void ApplyGradientDescent(const size_t size, T *var, const T *alpha, const T *delta, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const T alpha_value = alpha[0];
    var[pos] -= alpha_value * delta[pos];
    output[pos] = var[pos];
  }
}

template <typename T>
cudaError_t CalApplyGradientDescent(const size_t &size, T *var, const T *alpha, const T *delta, T *output,
                                    cudaStream_t cuda_stream) {
  ApplyGradientDescent<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, var, alpha, delta, output);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<float>(const size_t &size, float *var, const float *alpha,
                                                                    const float *delta, float *output,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<half>(const size_t &size, half *var, const half *alpha,
                                                                   const half *delta, half *output,
                                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<int8_t>(const size_t &size, int8_t *var,
                                                                     const int8_t *alpha, const int8_t *delta,
                                                                     int8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<uint8_t>(const size_t &size, uint8_t *var,
                                                                      const uint8_t *alpha, const uint8_t *delta,
                                                                      uint8_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<int16_t>(const size_t &size, int16_t *var,
                                                                      const int16_t *alpha, const int16_t *delta,
                                                                      int16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<uint16_t>(const size_t &size, uint16_t *var,
                                                                       const uint16_t *alpha, const uint16_t *delta,
                                                                       uint16_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<uint32_t>(const size_t &size, uint32_t *var,
                                                                       const uint32_t *alpha, const uint32_t *delta,
                                                                       uint32_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<int64_t>(const size_t &size, int64_t *var,
                                                                      const int64_t *alpha, const int64_t *delta,
                                                                      int64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<uint64_t>(const size_t &size, uint64_t *var,
                                                                       const uint64_t *alpha, const uint64_t *delta,
                                                                       uint64_t *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<double>(const size_t &size, double *var,
                                                                     const double *alpha, const double *delta,
                                                                     double *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<Complex<float>>(const size_t &size, Complex<float> *var,
                                                                             const Complex<float> *alpha,
                                                                             const Complex<float> *delta,
                                                                             Complex<float> *output,
                                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalApplyGradientDescent<Complex<double>>(const size_t &size, Complex<double> *var,
                                                                              const Complex<double> *alpha,
                                                                              const Complex<double> *delta,
                                                                              Complex<double> *output,
                                                                              cudaStream_t cuda_stream);
