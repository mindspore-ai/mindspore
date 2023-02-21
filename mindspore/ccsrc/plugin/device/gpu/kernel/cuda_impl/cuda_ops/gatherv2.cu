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

#include <iostream>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gatherv2.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
__global__ void GatherV2Kernel(T *input, S *indices, T *output, size_t output_dim0, size_t output_dim1,
                               size_t output_dim2, size_t input_dim1) {
  size_t num = output_dim0 * output_dim1 * output_dim2;
  size_t i, j, k;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num;
       write_index += blockDim.x * gridDim.x) {
    i = write_index / (output_dim1 * output_dim2) % output_dim0;
    j = write_index / output_dim2 % output_dim1;
    k = write_index % output_dim2;

    if ((indices[j] >= 0) && (indices[j] < input_dim1)) {
      size_t read_index = i * input_dim1 * output_dim2 + indices[j] * output_dim2 + k;
      output[write_index] = input[read_index];
    } else {
      output[write_index] = 0;
    }
  }

  return;
}
template <typename T, typename S>
__global__ void GatherV2WithBatchDimsKernel(T *input, S *indices, T *output, size_t batch_size, size_t output_dim0,
                                            size_t output_dim1, size_t output_dim2, size_t input_dim1) {
  size_t num = batch_size * output_dim0 * output_dim1 * output_dim2;
  size_t i, j, k, n;
  for (size_t write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num;
       write_index += blockDim.x * gridDim.x) {
    i = write_index / (output_dim0 * output_dim1 * output_dim2) % batch_size;
    j = i * output_dim1 + write_index / output_dim2 % output_dim1;
    n = write_index / (output_dim1 * output_dim2) % output_dim0;
    k = write_index % output_dim2;

    if ((indices[j] >= 0) && (indices[j] < input_dim1)) {
      size_t read_index =
        i * output_dim0 * input_dim1 * output_dim2 + n * input_dim1 * output_dim2 + indices[j] * output_dim2 + k;
      output[write_index] = input[read_index];
    } else {
      output[write_index] = 0;
    }
  }

  return;
}

template <typename T, typename S>
void GatherV2(T *input, S *indices, T *output, size_t batch_size, size_t output_dim0, size_t output_dim1,
              size_t output_dim2, size_t input_dim1, cudaStream_t stream) {
  size_t size = batch_size * output_dim0 * output_dim1 * output_dim2;
  if (batch_size > 1) {
    GatherV2WithBatchDimsKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(
      input, indices, output, batch_size, output_dim0, output_dim1, output_dim2, input_dim1);
  } else {
    GatherV2Kernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                                 output_dim2, input_dim1);
  }
  return;
}

template CUDA_LIB_EXPORT void GatherV2<Complex<float>, int>(Complex<float> *input, int *indices, Complex<float> *output,
                                                            size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                            size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<Complex<float>, int64_t>(Complex<float> *input, int64_t *indices,
                                                                Complex<float> *output, size_t batch_size,
                                                                size_t output_dim0, size_t output_dim1,
                                                                size_t output_dim2, size_t input_dim1,
                                                                cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<Complex<double>, int>(Complex<double> *input, int *indices,
                                                             Complex<double> *output, size_t batch_size,
                                                             size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                             size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<Complex<double>, int64_t>(Complex<double> *input, int64_t *indices,
                                                                 Complex<double> *output, size_t batch_size,
                                                                 size_t output_dim0, size_t output_dim1,
                                                                 size_t output_dim2, size_t input_dim1,
                                                                 cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<float, int>(float *input, int *indices, float *output, size_t batch_size,
                                                   size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                   size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<float, int64_t>(float *input, int64_t *indices, float *output, size_t batch_size,
                                                       size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                       size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<half, int>(half *input, int *indices, half *output, size_t batch_size,
                                                  size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                  size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<half, int64_t>(half *input, int64_t *indices, half *output, size_t batch_size,
                                                      size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                      size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<double, int>(double *input, int *indices, double *output, size_t batch_size,
                                                    size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                    size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<double, int64_t>(double *input, int64_t *indices, double *output,
                                                        size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                        size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<int64_t, int>(int64_t *input, int *indices, int64_t *output, size_t batch_size,
                                                     size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                     size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<int64_t, int64_t>(int64_t *input, int64_t *indices, int64_t *output,
                                                         size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                         size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<int, int>(int *input, int *indices, int *output, size_t batch_size,
                                                 size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                 size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<int, int64_t>(int *input, int64_t *indices, int *output, size_t batch_size,
                                                     size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                     size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<int16_t, int>(int16_t *input, int *indices, int16_t *output, size_t batch_size,
                                                     size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                     size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<int16_t, int64_t>(int16_t *input, int64_t *indices, int16_t *output,
                                                         size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                         size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<int8_t, int>(int8_t *input, int *indices, int8_t *output, size_t batch_size,
                                                    size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                    size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<int8_t, int64_t>(int8_t *input, int64_t *indices, int8_t *output,
                                                        size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                        size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<uint64_t, int>(uint64_t *input, int *indices, uint64_t *output,
                                                      size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                      size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<uint64_t, int64_t>(uint64_t *input, int64_t *indices, uint64_t *output,
                                                          size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                          size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<uint32_t, int>(uint32_t *input, int *indices, uint32_t *output,
                                                      size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                      size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<uint32_t, int64_t>(uint32_t *input, int64_t *indices, uint32_t *output,
                                                          size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                          size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<uint16_t, int>(uint16_t *input, int *indices, uint16_t *output,
                                                      size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                      size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<uint16_t, int64_t>(uint16_t *input, int64_t *indices, uint16_t *output,
                                                          size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                          size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<uint8_t, int>(uint8_t *input, int *indices, uint8_t *output, size_t batch_size,
                                                     size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                     size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<uint8_t, int64_t>(uint8_t *input, int64_t *indices, uint8_t *output,
                                                         size_t batch_size, size_t output_dim0, size_t output_dim1,
                                                         size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<bool, int>(bool *input, int *indices, bool *output, size_t batch_size,
                                                  size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                  size_t input_dim1, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherV2<bool, int64_t>(bool *input, int64_t *indices, bool *output, size_t batch_size,
                                                      size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                      size_t input_dim1, cudaStream_t stream);
