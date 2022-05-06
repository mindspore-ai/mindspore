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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/celu_impl.cuh"
#include "include/cuda_runtime.h"
#include "include/cuda_fp16.h"

template <typename T>
__global__ void CalculateCeluKernel(const T *input, const size_t input_elements, T alpha, T *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < input_elements; i += blockDim.x * gridDim.x) {
    T input_value = input[i];
    T template_zero = static_cast<T>(0.0);
    output[i] = input_value > template_zero ? input_value : alpha * expm1(input_value / alpha);
  }
}

__global__ void CalculateCeluKernel(const half *input, const size_t input_elements, half alpha, half *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < input_elements; i += blockDim.x * gridDim.x) {
    half input_value = input[i];
    half template_zero = static_cast<half>(0.0);
    half template_one = static_cast<half>(1.0);
    output[i] = input_value > template_zero ? input_value : alpha * (hexp(input_value / alpha) - template_one);
  }
}

template <typename T>
void CalculateCelu(const T *input, size_t input_elements, T alpha, T *output, const uint32_t &device_id,
                   cudaStream_t cuda_stream) {
  CalculateCeluKernel<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_elements, alpha, output);
}

template CUDA_LIB_EXPORT void CalculateCelu<double>(const double *input, size_t input_elements, double alpha,
                                                    double *output, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateCelu<float>(const float *input, size_t input_elements, float alpha,
                                                   float *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalculateCelu<half>(const half *input, size_t input_elements, half alpha, half *output,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
