/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/split_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "include/cuda_fp16.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void Split(const size_t size, const int axis_step, const int all_size_before_axis, const int all_size_axis,
                      const T *input, T **outputs) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int num = pos % all_size_before_axis / all_size_axis;
    int block = num / axis_step;
    int block_pos =
      pos / all_size_before_axis * axis_step * all_size_axis + num % axis_step * all_size_axis + pos % all_size_axis;
    outputs[block][block_pos] = input[pos];
  }
  return;
}

template <typename T>
void SplitKernel(const size_t size, const int axis_step, const int all_size_before_axis, const int all_size_axis,
                 const T *input, T **outputs, cudaStream_t cuda_stream) {
  Split<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, axis_step, all_size_before_axis, all_size_axis, input,
                                                           outputs);
  return;
}

template CUDA_LIB_EXPORT void SplitKernel<half>(const size_t size, const int axis_step, const int all_size_before_axis,
                                                const int all_size_axis, const half *input, half **outputs,
                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<float>(const size_t size, const int axis_step, const int all_size_before_axis,
                                                 const int all_size_axis, const float *input, float **outputs,
                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<double>(const size_t size, const int axis_step,
                                                  const int all_size_before_axis, const int all_size_axis,
                                                  const double *input, double **outputs, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<int8_t>(const size_t size, const int axis_step,
                                                  const int all_size_before_axis, const int all_size_axis,
                                                  const int8_t *input, int8_t **outputs, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<int16_t>(const size_t size, const int axis_step,
                                                   const int all_size_before_axis, const int all_size_axis,
                                                   const int16_t *input, int16_t **outputs, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<int32_t>(const size_t size, const int axis_step,
                                                   const int all_size_before_axis, const int all_size_axis,
                                                   const int32_t *input, int32_t **outputs, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<int64_t>(const size_t size, const int axis_step,
                                                   const int all_size_before_axis, const int all_size_axis,
                                                   const int64_t *input, int64_t **outputs, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<uint8_t>(const size_t size, const int axis_step,
                                                   const int all_size_before_axis, const int all_size_axis,
                                                   const uint8_t *input, uint8_t **outputs, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<uint16_t>(const size_t size, const int axis_step,
                                                    const int all_size_before_axis, const int all_size_axis,
                                                    const uint16_t *input, uint16_t **outputs,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<uint32_t>(const size_t size, const int axis_step,
                                                    const int all_size_before_axis, const int all_size_axis,
                                                    const uint32_t *input, uint32_t **outputs,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<uint64_t>(const size_t size, const int axis_step,
                                                    const int all_size_before_axis, const int all_size_axis,
                                                    const uint64_t *input, uint64_t **outputs,
                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<Complex<float>>(const size_t size, const int axis_step,
                                                          const int all_size_before_axis, const int all_size_axis,
                                                          const Complex<float> *input, Complex<float> **outputs,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<Complex<double>>(const size_t size, const int axis_step,
                                                           const int all_size_before_axis, const int all_size_axis,
                                                           const Complex<double> *input, Complex<double> **outputs,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void SplitKernel<bool>(const size_t size, const int axis_step, const int all_size_before_axis,
                                                const int all_size_axis, const bool *input, bool **outputs,
                                                cudaStream_t cuda_stream);
