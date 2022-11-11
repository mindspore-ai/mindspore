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

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/parallel_concat_impl.cuh"
#include "include/cuda_fp16.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void ParallelConcat(const size_t size, const int input_num, T **inputs, T *output) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int all_size_axis = size / input_num;
    int num = pos % size / all_size_axis;
    int block_pos = pos / size * all_size_axis + pos % all_size_axis;
    output[pos] = inputs[num][block_pos];
  }
  return;
}

template <typename T>
void ParallelConcatKernel(const size_t size, const int input_num, T **inputs, T *output, const uint32_t &device_id,
                          cudaStream_t cuda_stream) {
  ParallelConcat<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input_num, inputs,
                                                                                            output);
  return;
}

template CUDA_LIB_EXPORT void ParallelConcatKernel<double>(const size_t size, const int input_num, double **inputs,
                                                           double *output, const uint32_t &devizce_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ParallelConcatKernel<float>(const size_t size, const int input_num, float **inputs,
                                                          float *output, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ParallelConcatKernel<half>(const size_t size, const int input_num, half **inputs,
                                                         half *output, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ParallelConcatKernel<int8_t>(const size_t size, const int input_num, int8_t **inputs,
                                                           int8_t *output, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ParallelConcatKernel<int16_t>(const size_t size, const int input_num, int16_t **inputs,
                                                            int16_t *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ParallelConcatKernel<int32_t>(const size_t size, const int input_num, int32_t **inputs,
                                                            int32_t *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ParallelConcatKernel<int64_t>(const size_t size, const int input_num, int64_t **inputs,
                                                            int64_t *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ParallelConcatKernel<uint8_t>(const size_t size, const int input_num, uint8_t **inputs,
                                                            uint8_t *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ParallelConcatKernel<uint16_t>(const size_t size, const int input_num, uint16_t **inputs,
                                                             uint16_t *output, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ParallelConcatKernel<uint32_t>(const size_t size, const int input_num, uint32_t **inputs,
                                                             uint32_t *output, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ParallelConcatKernel<uint64_t>(const size_t size, const int input_num, uint64_t **inputs,
                                                             uint64_t *output, const uint32_t &device_id,
                                                             cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ParallelConcatKernel<bool>(const size_t size, const int input_num, bool **inputs,
                                                         bool *output, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ParallelConcatKernel<Complex<float>>(const size_t size, const int input_num,
                                                                   Complex<float> **inputs, Complex<float> *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ParallelConcatKernel<Complex<double>>(const size_t size, const int input_num,
                                                                    Complex<double> **inputs, Complex<double> *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
