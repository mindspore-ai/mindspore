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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gathernd.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
template <typename T, typename S>
__global__ void GatherNdKernel(T *input, S *indices, T *output, const size_t output_dim0, const size_t output_dim1,
                               const size_t indices_dim1, S *batch_indices, S *batch_strides) {
  int num = output_dim0 * output_dim1;
  int i, j;
  for (int write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num;
       write_index += blockDim.x * gridDim.x) {
    i = write_index / output_dim1 % output_dim0;
    j = write_index % output_dim1;

    bool out_of_bound = false;
    int read_index = 0;
    int indices_i = 0;
    for (size_t k = 0; k < indices_dim1; k++) {
      size_t ind = indices_dim1 * i + k;
      indices_i = indices[ind];
      out_of_bound |= !(indices_i < batch_indices[k] && indices_i >= 0);
      read_index += indices_i * batch_strides[k];
    }
    read_index += j;

    if (!out_of_bound) {
      output[write_index] = input[read_index];
    } else {
      output[write_index] = 0;
    }
  }
  return;
}

template <typename S>
__global__ void GatherNdKernel(cuComplex *input, S *indices, cuComplex *output, const size_t output_dim0,
                               const size_t output_dim1, const size_t indices_dim1, S *batch_indices,
                               S *batch_strides) {
  int num = output_dim0 * output_dim1;
  int i, j;
  for (int write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num;
       write_index += blockDim.x * gridDim.x) {
    i = write_index / output_dim1 % output_dim0;
    j = write_index % output_dim1;

    bool out_of_bound = false;
    int read_index = 0;
    int indices_i = 0;
    for (size_t k = 0; k < indices_dim1; k++) {
      size_t ind = indices_dim1 * i + k;
      indices_i = indices[ind];
      out_of_bound |= !(indices_i < batch_indices[k] && indices_i >= 0);
      read_index += indices_i * batch_strides[k];
    }
    read_index += j;

    if (!out_of_bound) {
      output[write_index] = input[read_index];
    } else {
      output[write_index] = make_cuComplex(0, 0);
    }
  }
  return;
}

template <typename S>
__global__ void GatherNdKernel(cuDoubleComplex *input, S *indices, cuDoubleComplex *output, const size_t output_dim0,
                               const size_t output_dim1, const size_t indices_dim1, S *batch_indices,
                               S *batch_strides) {
  int num = output_dim0 * output_dim1;
  int i, j;
  for (int write_index = blockIdx.x * blockDim.x + threadIdx.x; write_index < num;
       write_index += blockDim.x * gridDim.x) {
    i = write_index / output_dim1 % output_dim0;
    j = write_index % output_dim1;

    bool out_of_bound = false;
    int read_index = 0;
    int indices_i = 0;
    for (size_t k = 0; k < indices_dim1; k++) {
      size_t ind = indices_dim1 * i + k;
      indices_i = indices[ind];
      out_of_bound |= !(indices_i < batch_indices[k] && indices_i >= 0);
      read_index += indices_i * batch_strides[k];
    }
    read_index += j;

    if (!out_of_bound) {
      output[write_index] = input[read_index];
    } else {
      output[write_index] = make_cuDoubleComplex(0, 0);
    }
  }
  return;
}

template <typename T, typename S>
void GatherNd(T *input, S *indices, T *output, const size_t &output_dim0, const size_t &output_dim1,
              const size_t &indices_dim1, S *batch_indices, S *batch_strides, cudaStream_t stream) {
  int size = output_dim0 * output_dim1;
  GatherNdKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                               indices_dim1, batch_indices, batch_strides);
  return;
}

template <typename S>
void GatherNd(cuComplex *input, S *indices, cuComplex *output, const size_t &output_dim0, const size_t &output_dim1,
              const size_t &indices_dim1, S *batch_indices, S *batch_strides, cudaStream_t stream) {
  int size = output_dim0 * output_dim1;
  GatherNdKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                               indices_dim1, batch_indices, batch_strides);
  return;
}

template <typename S>
void GatherNd(cuDoubleComplex *input, S *indices, cuDoubleComplex *output, const size_t &output_dim0,
              const size_t &output_dim1, const size_t &indices_dim1, S *batch_indices, S *batch_strides,
              cudaStream_t stream) {
  int size = output_dim0 * output_dim1;
  GatherNdKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                               indices_dim1, batch_indices, batch_strides);
  return;
}

template CUDA_LIB_EXPORT void GatherNd<double, int>(double *input, int *indices, double *output,
                                                    const size_t &output_dim0, const size_t &output_dim1,
                                                    const size_t &indices_dim1, int *batch_indices, int *batch_strides,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<float, int>(float *input, int *indices, float *output, const size_t &output_dim0,
                                                   const size_t &output_dim1, const size_t &indices_dim1,
                                                   int *batch_indices, int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<half, int>(half *input, int *indices, half *output, const size_t &output_dim0,
                                                  const size_t &output_dim1, const size_t &indices_dim1,
                                                  int *batch_indices, int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<int64_t, int>(int64_t *input, int *indices, int64_t *output,
                                                     const size_t &output_dim0, const size_t &output_dim1,
                                                     const size_t &indices_dim1, int *batch_indices, int *batch_strides,
                                                     cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<int, int>(int *input, int *indices, int *output, const size_t &output_dim0,
                                                 const size_t &output_dim1, const size_t &indices_dim1,
                                                 int *batch_indices, int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<short, int>(short *input, int *indices, short *output,  // NOLINT
                                                   const size_t &output_dim0, const size_t &output_dim1,
                                                   const size_t &indices_dim1, int *batch_indices, int *batch_strides,
                                                   cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<unsigned int, int>(unsigned int *input, int *indices, unsigned int *output,
                                                          const size_t &output_dim0, const size_t &output_dim1,
                                                          const size_t &indices_dim1, int *batch_indices,
                                                          int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<uint16_t, int>(uint16_t *input, int *indices, uint16_t *output,
                                                      const size_t &output_dim0, const size_t &output_dim1,
                                                      const size_t &indices_dim1, int *batch_indices,
                                                      int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<uint64_t, int>(uint64_t *input, int *indices, uint64_t *output,
                                                      const size_t &output_dim0, const size_t &output_dim1,
                                                      const size_t &indices_dim1, int *batch_indices,
                                                      int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<int8_t, int>(int8_t *input, int *indices, int8_t *output,
                                                    const size_t &output_dim0, const size_t &output_dim1,
                                                    const size_t &indices_dim1, int *batch_indices, int *batch_strides,
                                                    cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<char, int>(char *input, int *indices, char *output, const size_t &output_dim0,
                                                  const size_t &output_dim1, const size_t &indices_dim1,
                                                  int *batch_indices, int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<unsigned char, int>(unsigned char *input, int *indices, unsigned char *output,
                                                           const size_t &output_dim0, const size_t &output_dim1,
                                                           const size_t &indices_dim1, int *batch_indices,
                                                           int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<bool, int>(bool *input, int *indices, bool *output, const size_t &output_dim0,
                                                  const size_t &output_dim1, const size_t &indices_dim1,
                                                  int *batch_indices, int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<cuComplex, int>(cuComplex *input, int *indices, cuComplex *output,
                                                       const size_t &output_dim0, const size_t &output_dim1,
                                                       const size_t &indices_dim1, int *batch_indices,
                                                       int *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<cuDoubleComplex, int>(cuDoubleComplex *input, int *indices,
                                                             cuDoubleComplex *output, const size_t &output_dim0,
                                                             const size_t &output_dim1, const size_t &indices_dim1,
                                                             int *batch_indices, int *batch_strides,
                                                             cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<double, int64_t>(double *input, int64_t *indices, double *output,
                                                        const size_t &output_dim0, const size_t &output_dim1,
                                                        const size_t &indices_dim1, int64_t *batch_indices,
                                                        int64_t *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<float, int64_t>(float *input, int64_t *indices, float *output,
                                                       const size_t &output_dim0, const size_t &output_dim1,
                                                       const size_t &indices_dim1, int64_t *batch_indices,
                                                       int64_t *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<half, int64_t>(half *input, int64_t *indices, half *output,
                                                      const size_t &output_dim0, const size_t &output_dim1,
                                                      const size_t &indices_dim1, int64_t *batch_indices,
                                                      int64_t *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<int, int64_t>(int *input, int64_t *indices, int *output,
                                                     const size_t &output_dim0, const size_t &output_dim1,
                                                     const size_t &indices_dim1, int64_t *batch_indices,
                                                     int64_t *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<short, int64_t>(short *input, int64_t *indices, short *output,  // NOLINT
                                                       const size_t &output_dim0, const size_t &output_dim1,
                                                       const size_t &indices_dim1, int64_t *batch_indices,
                                                       int64_t *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<unsigned int, int64_t>(unsigned int *input, int64_t *indices,
                                                              unsigned int *output, const size_t &output_dim0,
                                                              const size_t &output_dim1, const size_t &indices_dim1,
                                                              int64_t *batch_indices, int64_t *batch_strides,
                                                              cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<char, int64_t>(char *input, int64_t *indices, char *output,
                                                      const size_t &output_dim0, const size_t &output_dim1,
                                                      const size_t &indices_dim1, int64_t *batch_indices,
                                                      int64_t *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<unsigned char, int64_t>(unsigned char *input, int64_t *indices,
                                                               unsigned char *output, const size_t &output_dim0,
                                                               const size_t &output_dim1, const size_t &indices_dim1,
                                                               int64_t *batch_indices, int64_t *batch_strides,
                                                               cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<bool, int64_t>(bool *input, int64_t *indices, bool *output,
                                                      const size_t &output_dim0, const size_t &output_dim1,
                                                      const size_t &indices_dim1, int64_t *batch_indices,
                                                      int64_t *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<int64_t>(cuComplex *input, int64_t *indices, cuComplex *output,
                                                const size_t &output_dim0, const size_t &output_dim1,
                                                const size_t &indices_dim1, int64_t *batch_indices,
                                                int64_t *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<int64_t>(cuDoubleComplex *input, int64_t *indices, cuDoubleComplex *output,
                                                const size_t &output_dim0, const size_t &output_dim1,
                                                const size_t &indices_dim1, int64_t *batch_indices,
                                                int64_t *batch_strides, cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<int>(cuComplex *input, int *indices, cuComplex *output,
                                            const size_t &output_dim0, const size_t &output_dim1,
                                            const size_t &indices_dim1, int *batch_indices, int *batch_strides,
                                            cudaStream_t stream);
template CUDA_LIB_EXPORT void GatherNd<int>(cuDoubleComplex *input, int *indices, cuDoubleComplex *output,
                                            const size_t &output_dim0, const size_t &output_dim1,
                                            const size_t &indices_dim1, int *batch_indices, int *batch_strides,
                                            cudaStream_t stream);
