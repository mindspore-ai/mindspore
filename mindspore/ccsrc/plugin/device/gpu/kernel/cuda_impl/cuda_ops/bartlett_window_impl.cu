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
#include <math.h>
#include "bartlett_window_impl.cuh"

template <typename S>
__global__ void BartlettWindowOne(const size_t size, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<S>(1);
  }
  return;
}

template <typename S>
__global__ void BartlettWindow(const size_t size, const double N, const double M, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    double out = 0;
    if (pos <= M) {
      out = (2 * pos) / (N - 1);
    } else {
      out = 2 - (2 * pos) / (N - 1);
    }
    output[pos] = static_cast<S>(out);
  }
  return;
}

template <typename T, typename S>
cudaError_t CalBartlettWindow(const size_t size, const T *input, const bool periodic, S *output,
                              const uint32_t &device_id, cudaStream_t cuda_stream) {
  T N = 0;
  cudaMemcpy(&N, &input[0], sizeof(T), cudaMemcpyDeviceToHost);
  if (N == 1) {
    BartlettWindowOne<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, output);
  } else {
    N = periodic ? static_cast<double>(N + 1) : static_cast<double>(N);
    double M = (N - 1) / 2;
    BartlettWindow<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, N, M, output);
  }
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalBartlettWindow<int, half>(const size_t size, const int *input,
                                                                  const bool periodic, half *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBartlettWindow<int64_t, half>(const size_t size, const int64_t *input,
                                                                      const bool periodic, half *output,
                                                                      const uint32_t &device_id,
                                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBartlettWindow<int, float>(const size_t size, const int *input,
                                                                   const bool periodic, float *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBartlettWindow<int64_t, float>(const size_t size, const int64_t *input,
                                                                       const bool periodic, float *output,
                                                                       const uint32_t &device_id,
                                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBartlettWindow<int, double>(const size_t size, const int *input,
                                                                    const bool periodic, double *output,
                                                                    const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalBartlettWindow<int64_t, double>(const size_t size, const int64_t *input,
                                                                        const bool periodic, double *output,
                                                                        const uint32_t &device_id,
                                                                        cudaStream_t cuda_stream);
