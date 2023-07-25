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

#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gathernd.cuh"
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

template CUDA_LIB_EXPORT cudaError_t GatherNd<double, int>(double *input, int *indices, double *output,
                                                           const size_t &output_dim0, const size_t &output_dim1,
                                                           const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                           cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<float, int>(float *input, int *indices, float *output,
                                                          const size_t &output_dim0, const size_t &output_dim1,
                                                          const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                          cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<half, int>(half *input, int *indices, half *output,
                                                         const size_t &output_dim0, const size_t &output_dim1,
                                                         const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                         cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<int64_t, int>(int64_t *input, int *indices, int64_t *output,
                                                            const size_t &output_dim0, const size_t &output_dim1,
                                                            const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                            cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<int, int>(int *input, int *indices, int *output,
                                                        const size_t &output_dim0, const size_t &output_dim1,
                                                        const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                        cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<int16_t, int>(int16_t *input, int *indices, int16_t *output,  // NOLINT
                                                            const size_t &output_dim0, const size_t &output_dim1,
                                                            const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                            cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<unsigned int, int>(unsigned int *input, int *indices,
                                                                 unsigned int *output, const size_t &output_dim0,
                                                                 const size_t &output_dim1, const size_t &indices_dim1,
                                                                 const GatherNdInfo<int> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<uint16_t, int>(uint16_t *input, int *indices, uint16_t *output,
                                                             const size_t &output_dim0, const size_t &output_dim1,
                                                             const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                             cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<uint64_t, int>(uint64_t *input, int *indices, uint64_t *output,
                                                             const size_t &output_dim0, const size_t &output_dim1,
                                                             const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                             cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<int8_t, int>(int8_t *input, int *indices, int8_t *output,
                                                           const size_t &output_dim0, const size_t &output_dim1,
                                                           const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                           cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<char, int>(char *input, int *indices, char *output,
                                                         const size_t &output_dim0, const size_t &output_dim1,
                                                         const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                         cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<unsigned char, int>(unsigned char *input, int *indices,
                                                                  unsigned char *output, const size_t &output_dim0,
                                                                  const size_t &output_dim1, const size_t &indices_dim1,
                                                                  const GatherNdInfo<int> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<bool, int>(bool *input, int *indices, bool *output,
                                                         const size_t &output_dim0, const size_t &output_dim1,
                                                         const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                         cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<cuComplex, int>(cuComplex *input, int *indices, cuComplex *output,
                                                              const size_t &output_dim0, const size_t &output_dim1,
                                                              const size_t &indices_dim1, const GatherNdInfo<int> &info,
                                                              cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<cuDoubleComplex, int>(cuDoubleComplex *input, int *indices,
                                                                    cuDoubleComplex *output, const size_t &output_dim0,
                                                                    const size_t &output_dim1,
                                                                    const size_t &indices_dim1,
                                                                    const GatherNdInfo<int> &info, cudaStream_t stream);

template CUDA_LIB_EXPORT cudaError_t GatherNd<double, int64_t>(double *input, int64_t *indices, double *output,
                                                               const size_t &output_dim0, const size_t &output_dim1,
                                                               const size_t &indices_dim1,
                                                               const GatherNdInfo<int64_t> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<float, int64_t>(float *input, int64_t *indices, float *output,
                                                              const size_t &output_dim0, const size_t &output_dim1,
                                                              const size_t &indices_dim1,
                                                              const GatherNdInfo<int64_t> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<half, int64_t>(half *input, int64_t *indices, half *output,
                                                             const size_t &output_dim0, const size_t &output_dim1,
                                                             const size_t &indices_dim1,
                                                             const GatherNdInfo<int64_t> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<int, int64_t>(int *input, int64_t *indices, int *output,
                                                            const size_t &output_dim0, const size_t &output_dim1,
                                                            const size_t &indices_dim1,
                                                            const GatherNdInfo<int64_t> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<int16_t, int64_t>(int16_t *input, int64_t *indices,
                                                                int16_t *output,  // NOLINT
                                                                const size_t &output_dim0, const size_t &output_dim1,
                                                                const size_t &indices_dim1,
                                                                const GatherNdInfo<int64_t> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<unsigned int, int64_t>(
  unsigned int *input, int64_t *indices, unsigned int *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, const GatherNdInfo<int64_t> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<char, int64_t>(char *input, int64_t *indices, char *output,
                                                             const size_t &output_dim0, const size_t &output_dim1,
                                                             const size_t &indices_dim1,
                                                             const GatherNdInfo<int64_t> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<unsigned char, int64_t>(
  unsigned char *input, int64_t *indices, unsigned char *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, const GatherNdInfo<int64_t> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<bool, int64_t>(bool *input, int64_t *indices, bool *output,
                                                             const size_t &output_dim0, const size_t &output_dim1,
                                                             const size_t &indices_dim1,
                                                             const GatherNdInfo<int64_t> &info, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<cuComplex, int64_t>(cuComplex *input, int64_t *indices, cuComplex *output,
                                                                  const size_t &output_dim0, const size_t &output_dim1,
                                                                  const size_t &indices_dim1,
                                                                  const GatherNdInfo<int64_t> &info,
                                                                  cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t GatherNd<cuDoubleComplex, int64_t>(
  cuDoubleComplex *input, int64_t *indices, cuDoubleComplex *output, const size_t &output_dim0,
  const size_t &output_dim1, const size_t &indices_dim1, const GatherNdInfo<int64_t> &info, cudaStream_t stream);
