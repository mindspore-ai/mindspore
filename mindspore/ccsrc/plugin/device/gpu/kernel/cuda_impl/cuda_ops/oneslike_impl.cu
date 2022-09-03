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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/oneslike_impl.cuh"
#include <cuda_runtime.h>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
__global__ void OnesLike(const size_t size, const T *input, T *output) {
  int one = 1;
  T val = static_cast<T>(one);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = val;
  }
  return;
}
template <typename T>
void CalOnesLike(const size_t size, const T *input, T *output, cudaStream_t cuda_stream) {
  OnesLike<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, output);
  return;
}

template CUDA_LIB_EXPORT void CalOnesLike<bool>(const size_t size, const bool *input, bool *output,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<double>(const size_t size, const double *input, double *output,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<float>(const size_t size, const float *input, float *output,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<half>(const size_t size, const half *input, half *output,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<int8_t>(const size_t size, const int8_t *input, int8_t *output,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<int16_t>(const size_t size, const int16_t *input, int16_t *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<int32_t>(const size_t size, const int32_t *input, int32_t *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<int64_t>(const size_t size, const int64_t *input, int64_t *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<uint8_t>(const size_t size, const uint8_t *input, uint8_t *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<uint16_t>(const size_t size, const uint16_t *input, uint16_t *output,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<uint32_t>(const size_t size, const uint32_t *input, uint32_t *output,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<uint64_t>(const size_t size, const uint64_t *input, uint64_t *output,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<Complex<float>>(const size_t size, const Complex<float> *input,
                                                          Complex<float> *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalOnesLike<Complex<double>>(const size_t size, const Complex<double> *input,
                                                           Complex<double> *output, cudaStream_t cuda_stream);
