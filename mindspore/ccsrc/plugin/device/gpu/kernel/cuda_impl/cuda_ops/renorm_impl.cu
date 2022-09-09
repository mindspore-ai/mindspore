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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/renorm_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void CalNormValFun1(const T *input, size_t input_elements, size_t inner_size, size_t axis_size, int p,
                               float *norm_value) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (input_elements);
       index += blockDim.x * gridDim.x) {
    auto norm_index = index / inner_size;
    norm_index = norm_index % axis_size;
    float pow_value = pow(abs(static_cast<float>(input[index])), static_cast<float>(p));
    MsAtomicAdd(&norm_value[norm_index], pow_value);
  }
}

template <>
__global__ void CalNormValFun1(const Complex<float> *input, size_t input_elements, size_t inner_size, size_t axis_size,
                               int p, float *norm_value) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (input_elements);
       index += blockDim.x * gridDim.x) {
    auto norm_index = index / inner_size;
    norm_index = norm_index % axis_size;
    float pow_value = pow(static_cast<float>(abs(input[index])), static_cast<float>(p));
    MsAtomicAdd(&norm_value[norm_index], pow_value);
  }
}

template <>
__global__ void CalNormValFun1(const Complex<double> *input, size_t input_elements, size_t inner_size, size_t axis_size,
                               int p, float *norm_value) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (input_elements);
       index += blockDim.x * gridDim.x) {
    auto norm_index = index / inner_size;
    norm_index = norm_index % axis_size;
    float pow_value = pow(static_cast<float>(abs(input[index])), static_cast<float>(p));
    MsAtomicAdd(&norm_value[norm_index], pow_value);
  }
}

__global__ void CalNormValFun2(float *norm_value, int p, size_t axis_size, float max_norm) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (axis_size); index += blockDim.x * gridDim.x) {
    float temp = pow(norm_value[index], static_cast<float>(1.0 / p));
    if (temp > max_norm) {
      norm_value[index] = max_norm / temp;
    } else {
      norm_value[index] = static_cast<float>(1.0);
    }
  }
}

template <typename T>
__global__ void CalNormValFun3(const T *input, size_t inner_size, size_t axis_size, size_t output_elements, T *output,
                               const float *norm_value) {
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < (output_elements);
       index += blockDim.x * gridDim.x) {
    auto norm_index = index / inner_size % axis_size;
    if (norm_value[norm_index] < static_cast<float>(1.0)) {
      output[index] = static_cast<T>(norm_value[norm_index]) * input[index];
    } else {
      output[index] = input[index];
    }
  }
}

template <typename T>
void CalRenorm(const T *input, size_t input_elements, size_t inner_size, size_t axis_size,
               int p, float *norm_value, T *output, const uint32_t &device_id,
               cudaStream_t cuda_stream, float max_norm) {
  CalNormValFun1<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, input_elements, inner_size, axis_size, p, norm_value);
  CalNormValFun2<<<CUDA_BLOCKS(device_id, axis_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(norm_value, p,
                                                                                                 axis_size, max_norm);
  CalNormValFun3<<<CUDA_BLOCKS(device_id, input_elements), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, inner_size, axis_size, input_elements, output, norm_value);
}

template CUDA_LIB_EXPORT
void CalRenorm<half>(const half *input, size_t input_elements, size_t inner_size, size_t axis_size,
                     int p, float *norm_value, half *output, const uint32_t &device_id,
                     cudaStream_t cuda_stream, float max_norm);

template CUDA_LIB_EXPORT
void CalRenorm<float>(const float *input, size_t input_elements, size_t inner_size, size_t axis_size,
                      int p, float *norm_value, float *output, const uint32_t &device_id,
                      cudaStream_t cuda_stream, float max_norm);

template CUDA_LIB_EXPORT
void CalRenorm<double>(const double *input, size_t input_elements, size_t inner_size, size_t axis_size,
                       int p, float *norm_value, double *output, const uint32_t &device_id,
                       cudaStream_t cuda_stream, float max_norm);

template CUDA_LIB_EXPORT
void CalRenorm<Complex<float>>(const Complex<float> *input, size_t input_elements, size_t inner_size,
                               size_t axis_size, int p, float *norm_value, Complex<float> *output,
                               const uint32_t &device_id, cudaStream_t cuda_stream, float max_norm);

template CUDA_LIB_EXPORT
void CalRenorm<Complex<double>>(const Complex<double> *input, size_t input_elements, size_t inner_size,
                                size_t axis_size, int p, float *norm_value, Complex<double> *output,
                                const uint32_t &device_id, cudaStream_t cuda_stream, float max_norm);
