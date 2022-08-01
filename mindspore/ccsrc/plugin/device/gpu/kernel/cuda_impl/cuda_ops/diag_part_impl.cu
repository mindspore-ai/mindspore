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

#include <complex>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/diag_part_impl.cuh"


template <typename T>
__global__ void DiagPart(const size_t size, const T *input, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input[(1 + size) * pos];
  }
}

template <typename T>
void CalDiagPart(const size_t size, const T *input, T *output, const uint32_t &device_id,
                 cudaStream_t cuda_stream) {
  DiagPart<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, input, output);
  return;
}

template
CUDA_LIB_EXPORT void CalDiagPart<int32_t>(const size_t size, const int32_t *input, int32_t *output,
                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalDiagPart<int64_t>(const size_t size, const int64_t *input, int64_t *output,
                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalDiagPart<half>(const size_t size, const half *input, half *output,
                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalDiagPart<double>(const size_t size, const double *input, double *output,
                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalDiagPart<float>(const size_t size, const float *input, float *output,
                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalDiagPart<std::complex<float>>(const size_t size, const std::complex<float> *input,
                                                      std::complex<float> *output, const uint32_t &device_id,
                                                      cudaStream_t cuda_stream);
template
CUDA_LIB_EXPORT void CalDiagPart<std::complex<double>>(const size_t size, const std::complex<double> *input,
                                                       std::complex<double> *output, const uint32_t &device_id,
                                                       cudaStream_t cuda_stream);
