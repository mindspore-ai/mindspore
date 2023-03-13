/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/concatv2_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void Concat(const size_t size, const int input_num, const int all_size_before_axis, const int all_size_axis,
                       int *len_axis, T **inputs, T *output) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int num = pos % all_size_before_axis / all_size_axis;
    int block = -1;
    int axis_inc = 0;
    int block_len = 0;
    for (int i = 0; i < input_num; i++) {
      if (axis_inc <= num) {
        block++;
        axis_inc += len_axis[i];
      } else {
        break;
      }
    }
    block_len = len_axis[block];
    axis_inc -= len_axis[block];
    int block_pos =
      pos / all_size_before_axis * block_len * all_size_axis + (num - axis_inc) * all_size_axis + pos % all_size_axis;
    output[pos] = inputs[block][block_pos];
  }
  return;
}

template <typename T>
void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis, const int all_size_axis,
                  int *len_axis, T **inputs, T *output, cudaStream_t cuda_stream) {
  Concat<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input_num, all_size_before_axis, all_size_axis,
                                                            len_axis, inputs, output);
  return;
}

template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, Complex<double> **inputs,
                                           Complex<double> *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, Complex<float> **inputs,
                                           Complex<float> *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, double **inputs, double *output,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, float **inputs, float *output,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, half **inputs, half *output,
                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, int64_t **inputs, int64_t *output,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, int **inputs, int *output,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, short **inputs,
                                           short *output,  // NOLINT
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, int8_t **inputs, int8_t *output,
                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, uint64_t **inputs, uint64_t *output,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, uint32_t **inputs, uint32_t *output,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, uint16_t **inputs, uint16_t *output,
                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, uint8_t **inputs, uint8_t *output,
                                           cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void ConcatKernel(const size_t size, const int input_num, const int all_size_before_axis,
                                           const int all_size_axis, int *len_axis, bool **inputs, bool *output,
                                           cudaStream_t cuda_stream);
