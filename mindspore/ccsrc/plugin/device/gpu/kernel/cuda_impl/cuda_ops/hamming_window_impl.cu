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
#include "hamming_window_impl.cuh"

template <typename S>
__global__ void HammingWindowOne(const size_t size, const double N, const double PI, const float alpha,
                                 const float beta, S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<S>(1);
  }
  return;
}

template <typename S>
__global__ void HammingWindow(const size_t size, const double N, const double PI, const float alpha, const float beta,
                              S *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    double out = alpha - beta * cos((2 * pos * PI) / (N - 1));
    output[pos] = static_cast<S>(out);
  }
  return;
}

template <typename T, typename S>
cudaError_t HammingWindow(const size_t size, T N, const float alpha, const float beta, const bool periodic, S *output,
                          const uint32_t &device_id, cudaStream_t cuda_stream) {
  const double PI = acos(-1);
  if (N == 1) {
    HammingWindowOne<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, N, PI, alpha,
                                                                                                beta, output);
  } else {
    N = periodic ? static_cast<double>(N + 1) : static_cast<double>(N);
    HammingWindow<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, N, PI, alpha, beta,
                                                                                             output);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t HammingWindow<int8_t, half>(const size_t size, int8_t N, const float alpha,
                                                                 const float beta, const bool periodic, half *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int16_t, half>(const size_t size, int16_t N, const float alpha,
                                                                  const float beta, const bool periodic, half *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int32_t, half>(const size_t size, int32_t N, const float alpha,
                                                                  const float beta, const bool periodic, half *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int64_t, half>(const size_t size, int64_t N, const float alpha,
                                                                  const float beta, const bool periodic, half *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint8_t, half>(const size_t size, uint8_t N, const float alpha,
                                                                  const float beta, const bool periodic, half *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint16_t, half>(const size_t size, uint16_t N, const float alpha,
                                                                   const float beta, const bool periodic, half *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint32_t, half>(const size_t size, uint32_t N, const float alpha,
                                                                   const float beta, const bool periodic, half *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint64_t, half>(const size_t size, uint64_t N, const float alpha,
                                                                   const float beta, const bool periodic, half *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int8_t, float>(const size_t size, int8_t N, const float alpha,
                                                                  const float beta, const bool periodic, float *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int16_t, float>(const size_t size, int16_t N, const float alpha,
                                                                   const float beta, const bool periodic, float *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int32_t, float>(const size_t size, int32_t N, const float alpha,
                                                                   const float beta, const bool periodic, float *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int64_t, float>(const size_t size, int64_t N, const float alpha,
                                                                   const float beta, const bool periodic, float *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint8_t, float>(const size_t size, uint8_t N, const float alpha,
                                                                   const float beta, const bool periodic, float *output,
                                                                   const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint16_t, float>(const size_t size, uint16_t N, const float alpha,
                                                                    const float beta, const bool periodic,
                                                                    float *output, const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint32_t, float>(const size_t size, uint32_t N, const float alpha,
                                                                    const float beta, const bool periodic,
                                                                    float *output, const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint64_t, float>(const size_t size, uint64_t N, const float alpha,
                                                                    const float beta, const bool periodic,
                                                                    float *output, const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int8_t, double>(const size_t size, int8_t N, const float alpha,
                                                                   const float beta, const bool periodic,
                                                                   double *output, const uint32_t &device_id,
                                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int16_t, double>(const size_t size, int16_t N, const float alpha,
                                                                    const float beta, const bool periodic,
                                                                    double *output, const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int32_t, double>(const size_t size, int32_t N, const float alpha,
                                                                    const float beta, const bool periodic,
                                                                    double *output, const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<int64_t, double>(const size_t size, int64_t N, const float alpha,
                                                                    const float beta, const bool periodic,
                                                                    double *output, const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint8_t, double>(const size_t size, uint8_t N, const float alpha,
                                                                    const float beta, const bool periodic,
                                                                    double *output, const uint32_t &device_id,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint16_t, double>(const size_t size, uint16_t N, const float alpha,
                                                                     const float beta, const bool periodic,
                                                                     double *output, const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint32_t, double>(const size_t size, uint32_t N, const float alpha,
                                                                     const float beta, const bool periodic,
                                                                     double *output, const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t HammingWindow<uint64_t, double>(const size_t size, uint64_t N, const float alpha,
                                                                     const float beta, const bool periodic,
                                                                     double *output, const uint32_t &device_id,
                                                                     cudaStream_t cuda_stream);
