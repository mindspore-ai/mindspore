/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "argmax_impl.cuh"
template <typename T, typename S>
__global__ void Argmax(const T *input, const S bound, const size_t outer_size, const size_t inner_size, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < outer_size * inner_size;
       pos += gridDim.x * blockDim.x) {
    size_t x = pos / inner_size % outer_size;
    size_t y = pos % inner_size;
    S idx = 0;
    size_t input_offset = x * bound * inner_size + 0 * inner_size + y;
    T max_data = input[input_offset];
    for (S i = 1; i < bound; i++) {
      input_offset = x * bound * inner_size + i * inner_size + y;
      auto input_data = input[input_offset];
      idx = input_data > max_data ? i : idx;
      max_data = input_data > max_data ? input_data : max_data;
    }
    output[pos] = idx;
  }
  return;
}

template <typename T, typename S>
cudaError_t CalArgmax(const T *input, const S bound, const size_t outer_size, const size_t inner_size, S *output,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  Argmax<<<CUDA_BLOCKS(device_id, outer_size), CUDA_THREADS(device_id), 0, cuda_stream>>>(input, bound, outer_size,
                                                                                          inner_size, output);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalArgmax<half, int32_t>(const half *input, const int32_t bound,
                                                              const size_t outer_size, const size_t inner_size,
                                                              int32_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<float, int32_t>(const float *input, const int32_t bound,
                                                               const size_t outer_size, const size_t inner_size,
                                                               int32_t *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<double, int32_t>(const double *input, const int32_t bound,
                                                                const size_t outer_size, const size_t inner_size,
                                                                int32_t *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<int8_t, int32_t>(const int8_t *input, const int32_t bound,
                                                                const size_t outer_size, const size_t inner_size,
                                                                int32_t *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<int16_t, int32_t>(const int16_t *input, const int32_t bound,
                                                                 const size_t outer_size, const size_t inner_size,
                                                                 int32_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<int32_t, int32_t>(const int32_t *input, const int32_t bound,
                                                                 const size_t outer_size, const size_t inner_size,
                                                                 int32_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<int64_t, int32_t>(const int64_t *input, const int32_t bound,
                                                                 const size_t outer_size, const size_t inner_size,
                                                                 int32_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<uint8_t, int32_t>(const uint8_t *input, const int32_t bound,
                                                                 const size_t outer_size, const size_t inner_size,
                                                                 int32_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<uint16_t, int32_t>(const uint16_t *input, const int32_t bound,
                                                                  const size_t outer_size, const size_t inner_size,
                                                                  int32_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<uint32_t, int32_t>(const uint32_t *input, const int32_t bound,
                                                                  const size_t outer_size, const size_t inner_size,
                                                                  int32_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<uint64_t, int32_t>(const uint64_t *input, const int32_t bound,
                                                                  const size_t outer_size, const size_t inner_size,
                                                                  int32_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t CalArgmax<half, int64_t>(const half *input, const int64_t bound,
                                                              const size_t outer_size, const size_t inner_size,
                                                              int64_t *output, const uint32_t &device_id,
                                                              cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<float, int64_t>(const float *input, const int64_t bound,
                                                               const size_t outer_size, const size_t inner_size,
                                                               int64_t *output, const uint32_t &device_id,
                                                               cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<double, int64_t>(const double *input, const int64_t bound,
                                                                const size_t outer_size, const size_t inner_size,
                                                                int64_t *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<int8_t, int64_t>(const int8_t *input, const int64_t bound,
                                                                const size_t outer_size, const size_t inner_size,
                                                                int64_t *output, const uint32_t &device_id,
                                                                cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<int16_t, int64_t>(const int16_t *input, const int64_t bound,
                                                                 const size_t outer_size, const size_t inner_size,
                                                                 int64_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<int32_t, int64_t>(const int32_t *input, const int64_t bound,
                                                                 const size_t outer_size, const size_t inner_size,
                                                                 int64_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<int64_t, int64_t>(const int64_t *input, const int64_t bound,
                                                                 const size_t outer_size, const size_t inner_size,
                                                                 int64_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<uint8_t, int64_t>(const uint8_t *input, const int64_t bound,
                                                                 const size_t outer_size, const size_t inner_size,
                                                                 int64_t *output, const uint32_t &device_id,
                                                                 cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<uint16_t, int64_t>(const uint16_t *input, const int64_t bound,
                                                                  const size_t outer_size, const size_t inner_size,
                                                                  int64_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<uint32_t, int64_t>(const uint32_t *input, const int64_t bound,
                                                                  const size_t outer_size, const size_t inner_size,
                                                                  int64_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalArgmax<uint64_t, int64_t>(const uint64_t *input, const int64_t bound,
                                                                  const size_t outer_size, const size_t inner_size,
                                                                  int64_t *output, const uint32_t &device_id,
                                                                  cudaStream_t cuda_stream);
