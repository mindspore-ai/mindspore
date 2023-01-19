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
                               const size_t indices_dim1, S *batch_strides, S *batch_indices, int *device_flag) {
  int num = output_dim0 * output_dim1;
  int i, j;
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
        *device_flag = i;
        return;
      }
      read_index += indices_i * batch_indices[k];
    }
    read_index += j;
    output[write_index] = input[read_index];
  }
  return;
}
template <typename T, typename S>
std::pair<int, std::vector<S>> GatherNd(T *input, S *indices, T *output, const size_t &output_dim0,
                                        const size_t &output_dim1, const size_t &indices_dim1, S *batch_strides,
                                        S *batch_indices, int *device_flag, cudaStream_t stream) {
  int size = output_dim0 * output_dim1;
  int host_flag = -1;
  std::vector<S> error_indice;
  std::pair<int, std::vector<S>> res = {host_flag, error_indice};
  cudaMemsetAsync(device_flag, -1, sizeof(int), stream);
  GatherNdKernel<<<GET_BLOCKS(size), GET_THREADS, 0, stream>>>(input, indices, output, output_dim0, output_dim1,
                                                               indices_dim1, batch_strides, batch_indices, device_flag);
  cudaMemcpyAsync(&host_flag, device_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  if (host_flag != -1) {
    error_indice.resize(indices_dim1);
    cudaMemcpy(error_indice.data(), indices + host_flag * indices_dim1, sizeof(S) * indices_dim1,
               cudaMemcpyDeviceToHost);
    res = {host_flag, error_indice};
  }
  return res;
}

template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<double, int>(
  double *input, int *indices, double *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<float, int>(
  float *input, int *indices, float *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<half, int>(
  half *input, int *indices, half *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<int64_t, int>(
  int64_t *input, int *indices, int64_t *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<int, int>(
  int *input, int *indices, int *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<int16_t, int>(
  int16_t *input, int *indices, int16_t *output,  // NOLINT
  const size_t &output_dim0, const size_t &output_dim1, const size_t &indices_dim1, int *batch_strides,
  int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<unsigned int, int>(
  unsigned int *input, int *indices, unsigned int *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<uint16_t, int>(
  uint16_t *input, int *indices, uint16_t *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<uint64_t, int>(
  uint64_t *input, int *indices, uint64_t *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<int8_t, int>(
  int8_t *input, int *indices, int8_t *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<char, int>(
  char *input, int *indices, char *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<unsigned char, int>(
  unsigned char *input, int *indices, unsigned char *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<bool, int>(
  bool *input, int *indices, bool *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<cuComplex, int>(
  cuComplex *input, int *indices, cuComplex *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int>> GatherNd<cuDoubleComplex, int>(
  cuDoubleComplex *input, int *indices, cuDoubleComplex *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int *batch_strides, int *batch_indices, int *device_flag, cudaStream_t stream);

template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<double, int64_t>(
  double *input, int64_t *indices, double *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<float, int64_t>(
  float *input, int64_t *indices, float *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<half, int64_t>(
  half *input, int64_t *indices, half *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<int, int64_t>(
  int *input, int64_t *indices, int *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<int16_t, int64_t>(
  int16_t *input, int64_t *indices, int16_t *output,  // NOLINT
  const size_t &output_dim0, const size_t &output_dim1, const size_t &indices_dim1, int64_t *batch_strides,
  int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<unsigned int, int64_t>(
  unsigned int *input, int64_t *indices, unsigned int *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<char, int64_t>(
  char *input, int64_t *indices, char *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<unsigned char, int64_t>(
  unsigned char *input, int64_t *indices, unsigned char *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<bool, int64_t>(
  bool *input, int64_t *indices, bool *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<cuComplex, int64_t>(
  cuComplex *input, int64_t *indices, cuComplex *output, const size_t &output_dim0, const size_t &output_dim1,
  const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices, int *device_flag, cudaStream_t stream);
template CUDA_LIB_EXPORT std::pair<int, std::vector<int64_t>> GatherNd<cuDoubleComplex, int64_t>(
  cuDoubleComplex *input, int64_t *indices, cuDoubleComplex *output, const size_t &output_dim0,
  const size_t &output_dim1, const size_t &indices_dim1, int64_t *batch_strides, int64_t *batch_indices,
  int *device_flag, cudaStream_t stream);
