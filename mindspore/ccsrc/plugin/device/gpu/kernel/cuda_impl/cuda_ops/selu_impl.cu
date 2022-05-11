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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/selu_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void CalculateSeluKernel(const T *input, const size_t input_elements, T scale_dot_alpha, T scale,
                                    T *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < input_elements; i += blockDim.x * gridDim.x) {
    T input_value = input[i];
    T template_zero = static_cast<T>(0.0);
    output[i] = input_value >= template_zero ? scale * input_value : scale_dot_alpha * expm1(input_value);
  }
}

__global__ void CalculateSeluKernel(const half *input, const size_t input_elements, half scale_dot_alpha, half scale,
                                    half *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < input_elements; i += blockDim.x * gridDim.x) {
    half input_value = input[i];
    half template_zero = static_cast<half>(0.0);
    output[i] = input_value >= template_zero ? scale * input_value
                                             : scale_dot_alpha * static_cast<half>(expm1(__half2float(input_value)));
  }
}

template <typename T>
__global__ void CalculateSeluKernelInteger(const T *input, const size_t input_elements, float scale_dot_alpha,
                                           float scale, T *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < input_elements; i += blockDim.x * gridDim.x) {
    float input_value = static_cast<float>(input[i]);
    float template_zero = 0.0;
    output[i] = input_value >= template_zero ? scale * input_value : scale_dot_alpha * expm1(input_value);
  }
}

template <typename T>
void CalculateSelu(const T *input, size_t input_elements, T scale_dot_alpha, T scale, T *output,
                   const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalculateSeluKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_elements, scale_dot_alpha, scale, output);
}

template <typename T>
void CalculateSeluInteger(const T *input, size_t input_elements, float scale_dot_alpha, float scale, T *output,
                          const uint32_t &device_id, cudaStream_t cuda_stream) {
  CalculateSeluKernelInteger<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_elements, scale_dot_alpha, scale, output);
}

template CUDA_LIB_EXPORT void CalculateSelu<double>(const double *input, size_t input_elements, double scale_dot_alpha,
                                                    double scale, double *output, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateSelu<float>(const float *input, size_t input_elements, float scale_dot_alpha,
                                                   float scale, float *output, const uint32_t &device_id,
                                                   cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateSelu<half>(const half *input, size_t input_elements, half scale_dot_alpha,
                                                  half scale, half *output, const uint32_t &device_id,
                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateSeluInteger<int8_t>(const int8_t *input, size_t input_elements,
                                                           float scale_dot_alpha, float scale, int8_t *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateSeluInteger<int32_t>(const int32_t *input, size_t input_elements,
                                                            float scale_dot_alpha, float scale, int32_t *output,
                                                            const uint32_t &device_id, cudaStream_t cuda_stream);
