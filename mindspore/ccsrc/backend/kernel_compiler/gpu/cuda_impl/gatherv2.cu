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
#include "backend/kernel_compiler/gpu/cuda_impl/gatherv2.cuh"
#include "runtime/device/gpu/cuda_common.h"
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
void GatherV2(T *input, S *indices, T *output, size_t output_dim0, size_t output_dim1, size_t output_dim2,
              size_t input_dim1, cudaStream_t stream) {
  size_t size = output_dim0 * output_dim1 * output_dim2;
  GatherV2Kernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                               output_dim2, input_dim1);
  return;
}

template void GatherV2<float, int>(float *input, int *indices, float *output, size_t output_dim0, size_t output_dim1,
                                   size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<float, int64_t>(float *input, int64_t *indices, float *output, size_t output_dim0,
                                       size_t output_dim1, size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<half, int>(half *input, int *indices, half *output, size_t output_dim0, size_t output_dim1,
                                  size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<half, int64_t>(half *input, int64_t *indices, half *output, size_t output_dim0,
                                      size_t output_dim1, size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<double, int>(double *input, int *indices, double *output, size_t output_dim0, size_t output_dim1,
                                    size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<double, int64_t>(double *input, int64_t *indices, double *output, size_t output_dim0,
                                        size_t output_dim1, size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<int, int>(int *input, int *indices, int *output, size_t output_dim0, size_t output_dim1,
                                 size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<int, int64_t>(int *input, int64_t *indices, int *output, size_t output_dim0, size_t output_dim1,
                                     size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<int16_t, int>(int16_t *input, int *indices, int16_t *output, size_t output_dim0,
                                     size_t output_dim1, size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<int16_t, int64_t>(int16_t *input, int64_t *indices, int16_t *output, size_t output_dim0,
                                         size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                         cudaStream_t stream);
template void GatherV2<int8_t, int>(int8_t *input, int *indices, int8_t *output, size_t output_dim0, size_t output_dim1,
                                    size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<int8_t, int64_t>(int8_t *input, int64_t *indices, int8_t *output, size_t output_dim0,
                                        size_t output_dim1, size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<uint8_t, int>(uint8_t *input, int *indices, uint8_t *output, size_t output_dim0,
                                     size_t output_dim1, size_t output_dim2, size_t input_dim1, cudaStream_t stream);
template void GatherV2<uint8_t, int64_t>(uint8_t *input, int64_t *indices, uint8_t *output, size_t output_dim0,
                                         size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                         cudaStream_t stream);
