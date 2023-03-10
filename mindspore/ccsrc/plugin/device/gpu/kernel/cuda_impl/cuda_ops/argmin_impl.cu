/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "argmin_impl.cuh"
template <typename T, typename S>
__global__ void Argmin(const T *input, const S bound, const size_t outer_size, const size_t inner_size, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size * inner_size;
       pos += gridDim.x * blockDim.x) {
    size_t x = pos / inner_size % outer_size;
    size_t y = pos % inner_size;
    S idx = 0;
    size_t input_offset = x * bound * inner_size + 0 * inner_size + y;
    T min_data = input[input_offset];
    for (S i = 1; i < bound; i++) {
      input_offset = x * bound * inner_size + i * inner_size + y;
      auto input_data = input[input_offset];
      idx = input_data < min_data ? i : idx;
      min_data = input_data < min_data ? input_data : min_data;
    }
    output[pos] = idx;
  }
  return;
}

template <typename T, typename S>
cudaError_t CalArgmin(const T *input, const S bound, const size_t outer_size, const size_t inner_size, S *output,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  Argmin<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(input, bound, outer_size,
                                                                                          inner_size, output);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalArgmin<double, int>(const double *input, const int bound,
                                                            const size_t outer_size, const size_t inner_size,
                                                            int *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmin<float, int>(const float *input, const int bound, const size_t outer_size,
                                                           const size_t inner_size, int *output,
                                                           const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmin<half, int>(const half *input, const int bound, const size_t outer_size,
                                                          const size_t inner_size, int *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmin<double, int64_t>(const double *input, const int64_t bound,
                                                                const size_t outer_size, const size_t inner_size,
                                                                int64_t *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmin<float, int64_t>(const float *input, const int64_t bound,
                                                               const size_t outer_size, const size_t inner_size,
                                                               int64_t *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmin<half, int64_t>(const half *input, const int64_t bound,
                                                              const size_t outer_size, const size_t inner_size,
                                                              int64_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
