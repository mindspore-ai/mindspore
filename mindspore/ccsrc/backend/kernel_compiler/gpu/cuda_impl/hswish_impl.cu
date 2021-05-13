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

#include "backend/kernel_compiler/gpu/cuda_impl/hswish_impl.cuh"

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
  const auto add_factor = static_cast<T>(0.5);
  const auto div_factor = static_cast<T>(3);
  const auto max_threshold  = static_cast<T>(3);
  const auto min_threshold = static_cast<T>(-3);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    const T value = x[pos];
    if (value <= min_threshold) {
      output[pos] = static_cast<T>(0);
    } else if (value >= max_threshold) {
      output[pos] = dout[pos];
    } else {
      output[pos] = (value / div_factor + add_factor) * dout[pos];
    }
  }
}

template <typename T>
void CalHSwish(const size_t &size, const T *input, T *output, cudaStream_t cuda_stream) {
  HSwishKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
}

template <typename T>
void CalHSwishGrad(const size_t &size, const T *dout, const T *x, T *output, cudaStream_t cuda_stream) {
  HSwishGradKernel<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dout, x, output);
}

template void CalHSwish<half>(const size_t &size, const half *input, half *output, cudaStream_t cuda_stream);
template void CalHSwish<float>(const size_t &size, const float *input, float *output, cudaStream_t cuda_stream);

template void CalHSwishGrad<half>(const size_t &size, const half *dout, const half *x, half *output,
                                    cudaStream_t cuda_stream);
template void CalHSwishGrad<float>(const size_t &size, const float *dout, const float *x, float *output,
                                     cudaStream_t cuda_stream);
