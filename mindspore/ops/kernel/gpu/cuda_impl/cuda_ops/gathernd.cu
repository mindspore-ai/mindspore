/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "include/cuda_fp16.h"
#include <cuComplex.h>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/gathernd.cuh"

template <typename T>
using Complex = mindspore::utils::Complex<T>;
template <typename T, typename S>
__global__ void GatherNdKernel(T *input, S *indices, T *output, const size_t output_dim0, const size_t output_dim1,
                               const size_t indices_dim1, const GatherNdInfo<S> info) {
  int num = output_dim0 * output_dim1;
  int i, j;
  const S *batch_indices = info.indices;
  const S *batch_strides = info.strides;
  for (int write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num;
       write_index += blockDim.x * gridDim.x) {
    i = write_index / output_dim1;
    j = write_index % output_dim1;
    int read_index = 0;
    int indices_i = 0;
    for (size_t k = 0; k < indices_dim1; k++) {
      size_t ind = indices_dim1 * i + k;
      indices_i = indices[ind];
      if (indices_i >= batch_strides[k] || indices_i < 0) {
        continue;
      }
      read_index += indices_i * batch_indices[k];
    }
    read_index += j;
    output[write_index] = input[read_index];
  }
  return;
}
template <typename T, typename S>
cudaError_t GatherNd(T *input, S *indices, T *output, const size_t &output_dim0, const size_t &output_dim1,
                     const size_t &indices_dim1, const GatherNdInfo<S> &info, cudaStream_t stream) {
  int size = output_dim0 * output_dim1;
  GatherNdKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                               indices_dim1, info);
  return GetCudaStatus();
}

#define SPECIALIZE_GATHER_ND(T, S)                                                                                   \
  template CUDA_LIB_EXPORT cudaError_t GatherNd<T, S>(T * input, S * indices, T * output, const size_t &output_dim0, \
                                                      const size_t &output_dim1, const size_t &indices_dim1,         \
                                                      const GatherNdInfo<S> &info, cudaStream_t stream);

// S = int
SPECIALIZE_GATHER_ND(double, int)
SPECIALIZE_GATHER_ND(float, int)
SPECIALIZE_GATHER_ND(half, int)
SPECIALIZE_GATHER_ND(int64_t, int)
SPECIALIZE_GATHER_ND(int, int)
SPECIALIZE_GATHER_ND(int16_t, int)
SPECIALIZE_GATHER_ND(int8_t, int)
SPECIALIZE_GATHER_ND(char, int)
SPECIALIZE_GATHER_ND(uint64_t, int)
SPECIALIZE_GATHER_ND(uint16_t, int)
SPECIALIZE_GATHER_ND(unsigned char, int)
SPECIALIZE_GATHER_ND(unsigned int, int)
SPECIALIZE_GATHER_ND(bool, int)
SPECIALIZE_GATHER_ND(cuComplex, int)
SPECIALIZE_GATHER_ND(cuDoubleComplex, int)
SPECIALIZE_GATHER_ND(Complex<float>, int)
SPECIALIZE_GATHER_ND(Complex<double>, int)

// S = int64_t

SPECIALIZE_GATHER_ND(double, int64_t)
SPECIALIZE_GATHER_ND(float, int64_t)
SPECIALIZE_GATHER_ND(half, int64_t)
SPECIALIZE_GATHER_ND(int64_t, int64_t)
SPECIALIZE_GATHER_ND(int, int64_t)
SPECIALIZE_GATHER_ND(int16_t, int64_t)
SPECIALIZE_GATHER_ND(int8_t, int64_t)
SPECIALIZE_GATHER_ND(char, int64_t)
SPECIALIZE_GATHER_ND(uint64_t, int64_t)
SPECIALIZE_GATHER_ND(uint16_t, int64_t)
SPECIALIZE_GATHER_ND(unsigned int, int64_t)
SPECIALIZE_GATHER_ND(unsigned char, int64_t)
SPECIALIZE_GATHER_ND(bool, int64_t)
SPECIALIZE_GATHER_ND(cuComplex, int64_t)
SPECIALIZE_GATHER_ND(cuDoubleComplex, int64_t)
SPECIALIZE_GATHER_ND(Complex<float>, int64_t)
SPECIALIZE_GATHER_ND(Complex<double>, int64_t)

#undef SPECIALIZE_GATHER_ND
