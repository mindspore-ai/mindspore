/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "inplace_update_impl.cuh"

template <typename T>
__global__ void InplaceUpdate(const size_t size, const T *input_v, T *output, const int64_t *indices,
                              const int64_t band_size) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int v_row = pos / band_size;
    int x_row = indices[v_row];
    int offset = pos % band_size;
    int x_offset = x_row * band_size;
    output[x_offset + offset] = input_v[pos];
  }
  return;
}

template <typename T>
void CalInplaceUpdate(const size_t size_v, const T *input_v, T *output, const int64_t *indices, const int64_t band_size,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  InplaceUpdate<<<CUDA_BLOCKS(device_id, size_v), CUDA_THREADS(device_id), 0, cuda_stream>>>(size_v, input_v, output,
                                                                                           indices, band_size);
  return;
}

template CUDA_LIB_EXPORT void CalInplaceUpdate<half>(const size_t size_v, const half *input_v, half *output,
                                                     const int64_t *indices, const int64_t band_size,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceUpdate<float>(const size_t size_v, const float *input_v, float *output,
                                                      const int64_t *indices, const int64_t band_size,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceUpdate<double>(const size_t size_v, const double *input_v, double *output,
                                                       const int64_t *indices, const int64_t band_size,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalInplaceUpdate<int>(const size_t size_v, const int *input_v, int *output,
                                                    const int64_t *indices, const int64_t band_size,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);
