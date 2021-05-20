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

#include "backend/kernel_compiler/gpu/cuda_impl/gatherv2.cuh"
#include "backend/kernel_compiler/gpu/cuda_impl/embedding_lookup_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
__global__ void SubOffset(T *indices, size_t size, int64_t offset) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    indices[pos] -= static_cast<T>(offset);
  }
  return;
}

template <typename T, typename S>
void CalEmbeddingLookup(T *input, S *indices, T *output, size_t output_dim0, size_t output_dim1, size_t output_dim2,
                        size_t input_dim1, int64_t offset, cudaStream_t stream) {
  size_t size = output_dim0 * output_dim1 * output_dim2;
  SubOffset<<<GET_BLOCKS(output_dim1), GET_THREADS, 0, stream>>>(indices, output_dim1, offset);
  GatherV2Kernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                               output_dim2, input_dim1);
  // restore indices
  SubOffset<<<GET_BLOCKS(output_dim1), GET_THREADS, 0, stream>>>(indices, output_dim1, -offset);
  return;
}

template void CalEmbeddingLookup<float, int>(float *input, int *indices, float *output, size_t output_dim0,
                                             size_t output_dim1, size_t output_dim2, size_t input_dim1, int64_t offset,
                                             cudaStream_t stream);
template void CalEmbeddingLookup<float, int64_t>(float *input, int64_t *indices, float *output, size_t output_dim0,
                                                 size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                                 int64_t offset, cudaStream_t stream);
template void CalEmbeddingLookup<half, int>(half *input, int *indices, half *output, size_t output_dim0,
                                            size_t output_dim1, size_t output_dim2, size_t input_dim1, int64_t offset,
                                            cudaStream_t stream);
template void CalEmbeddingLookup<half, int64_t>(half *input, int64_t *indices, half *output, size_t output_dim0,
                                                size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                                int64_t offset, cudaStream_t stream);
template void CalEmbeddingLookup<double, int>(double *input, int *indices, double *output, size_t output_dim0,
                                              size_t output_dim1, size_t output_dim2, size_t input_dim1, int64_t offset,
                                              cudaStream_t stream);
template void CalEmbeddingLookup<double, int64_t>(double *input, int64_t *indices, double *output, size_t output_dim0,
                                                  size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                                  int64_t offset, cudaStream_t stream);
template void CalEmbeddingLookup<int, int>(int *input, int *indices, int *output, size_t output_dim0,
                                           size_t output_dim1, size_t output_dim2, size_t input_dim1, int64_t offset,
                                           cudaStream_t stream);
template void CalEmbeddingLookup<int, int64_t>(int *input, int64_t *indices, int *output, size_t output_dim0,
                                               size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                               int64_t offset, cudaStream_t stream);
template void CalEmbeddingLookup<int16_t, int>(int16_t *input, int *indices, int16_t *output, size_t output_dim0,
                                               size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                               int64_t offset, cudaStream_t stream);
template void CalEmbeddingLookup<int16_t, int64_t>(int16_t *input, int64_t *indices, int16_t *output,
                                                   size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                   size_t input_dim1, int64_t offset, cudaStream_t stream);
template void CalEmbeddingLookup<int8_t, int>(int8_t *input, int *indices, int8_t *output, size_t output_dim0,
                                              size_t output_dim1, size_t output_dim2, size_t input_dim1, int64_t offset,
                                              cudaStream_t stream);
template void CalEmbeddingLookup<int8_t, int64_t>(int8_t *input, int64_t *indices, int8_t *output, size_t output_dim0,
                                                  size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                                  int64_t offset, cudaStream_t stream);
template void CalEmbeddingLookup<uint8_t, int>(uint8_t *input, int *indices, uint8_t *output, size_t output_dim0,
                                               size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                               int64_t offset, cudaStream_t stream);
template void CalEmbeddingLookup<uint8_t, int64_t>(uint8_t *input, int64_t *indices, uint8_t *output,
                                                   size_t output_dim0, size_t output_dim1, size_t output_dim2,
                                                   size_t input_dim1, int64_t offset, cudaStream_t stream);
template void CalEmbeddingLookup<bool, int>(bool *input, int *indices, bool *output, size_t output_dim0,
                                            size_t output_dim1, size_t output_dim2, size_t input_dim1, int64_t offset,
                                            cudaStream_t stream);
template void CalEmbeddingLookup<bool, int64_t>(bool *input, int64_t *indices, bool *output, size_t output_dim0,
                                                size_t output_dim1, size_t output_dim2, size_t input_dim1,
                                                int64_t offset, cudaStream_t stream);
