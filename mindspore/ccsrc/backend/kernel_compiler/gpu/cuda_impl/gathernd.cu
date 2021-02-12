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

#include "backend/kernel_compiler/gpu/cuda_impl/gathernd.cuh"
#include "runtime/device/gpu/cuda_common.h"
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
      out_of_bound |= !(indices_i < batch_indices[k]);
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
template <typename T, typename S>
void GatherNd(T *input, S *indices, T *output, const size_t &output_dim0, const size_t &output_dim1,
              const size_t &indices_dim1, S *batch_indices, S *batch_strides, cudaStream_t stream) {
  int size = output_dim0 * output_dim1;
  GatherNdKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                               indices_dim1, batch_indices, batch_strides);
  return;
}

template void GatherNd<double, int>(double *input, int *indices, double *output, const size_t &output_dim0,
                                    const size_t &output_dim1, const size_t &indices_dim1, int *batch_indices,
                                    int *batch_strides, cudaStream_t stream);
template void GatherNd<float, int>(float *input, int *indices, float *output, const size_t &output_dim0,
                                   const size_t &output_dim1, const size_t &indices_dim1, int *batch_indices,
                                   int *batch_strides, cudaStream_t stream);
template void GatherNd<half, int>(half *input, int *indices, half *output, const size_t &output_dim0,
                                  const size_t &output_dim1, const size_t &indices_dim1, int *batch_indices,
                                  int *batch_strides, cudaStream_t stream);
template void GatherNd<int, int>(int *input, int *indices, int *output, const size_t &output_dim0,
                                 const size_t &output_dim1, const size_t &indices_dim1, int *batch_indices,
                                 int *batch_strides, cudaStream_t stream);
template void GatherNd<short, int>(short *input, int *indices, short *output, const size_t &output_dim0,  // NOLINT
                                   const size_t &output_dim1, const size_t &indices_dim1, int *batch_indices,
                                   int *batch_strides, cudaStream_t stream);
template void GatherNd<unsigned char, int>(unsigned char *input, int *indices, unsigned char *output,
                                           const size_t &output_dim0, const size_t &output_dim1,
                                           const size_t &indices_dim1, int *batch_indices, int *batch_strides,
                                           cudaStream_t stream);
template void GatherNd<bool, int>(bool *input, int *indices, bool *output, const size_t &output_dim0,
                                  const size_t &output_dim1, const size_t &indices_dim1, int *batch_indices,
                                  int *batch_strides, cudaStream_t stream);
template void GatherNd<double, int64_t>(double *input, int64_t *indices, double *output, const size_t &output_dim0,
                                        const size_t &output_dim1, const size_t &indices_dim1, int64_t *batch_indices,
                                        int64_t *batch_strides, cudaStream_t stream);
template void GatherNd<float, int64_t>(float *input, int64_t *indices, float *output, const size_t &output_dim0,
                                       const size_t &output_dim1, const size_t &indices_dim1, int64_t *batch_indices,
                                       int64_t *batch_strides, cudaStream_t stream);
template void GatherNd<half, int64_t>(half *input, int64_t *indices, half *output, const size_t &output_dim0,
                                      const size_t &output_dim1, const size_t &indices_dim1, int64_t *batch_indices,
                                      int64_t *batch_strides, cudaStream_t stream);
template void GatherNd<int, int64_t>(int *input, int64_t *indices, int *output, const size_t &output_dim0,
                                     const size_t &output_dim1, const size_t &indices_dim1, int64_t *batch_indices,
                                     int64_t *batch_strides, cudaStream_t stream);
template void GatherNd<short, int64_t>(short *input, int64_t *indices, short *output,  // NOLINT
                                       const size_t &output_dim0, const size_t &output_dim1, const size_t &indices_dim1,
                                       int64_t *batch_indices, int64_t *batch_strides, cudaStream_t stream);
template void GatherNd<unsigned char, int64_t>(unsigned char *input, int64_t *indices, unsigned char *output,
                                               const size_t &output_dim0, const size_t &output_dim1,
                                               const size_t &indices_dim1, int64_t *batch_indices,
                                               int64_t *batch_strides, cudaStream_t stream);
template void GatherNd<bool, int64_t>(bool *input, int64_t *indices, bool *output, const size_t &output_dim0,
                                      const size_t &output_dim1, const size_t &indices_dim1, int64_t *batch_indices,
                                      int64_t *batch_strides, cudaStream_t stream);
