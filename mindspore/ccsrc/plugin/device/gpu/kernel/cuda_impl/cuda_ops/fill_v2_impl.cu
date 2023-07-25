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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fill_v2_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T>
__global__ void FillV2Kernel(const int64_t output_size, const T *input, T *output) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < output_size; pos += blockDim.x * gridDim.x) {
    output[pos] = input[0];
  }
  return;
}

template <typename T>
cudaError_t FillV2(const int64_t output_size, const T *input, T *output, const uint32_t device_id,
                   cudaStream_t stream) {
  FillV2Kernel<<<CUDA_BLOCKS(device_id, output_size), CUDA_THREADS(device_id), 0, stream>>>(output_size, input, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const bool *input, bool *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const int8_t *input, int8_t *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const int16_t *input, int16_t *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const int32_t *input, int32_t *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const int64_t *input, int64_t *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const uint8_t *input, uint8_t *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const uint16_t *input, uint16_t *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const uint32_t *input, uint32_t *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const uint64_t *input, uint64_t *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const half *input, half *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const float *input, float *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const double *input, double *output,
                                            const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const Complex<float> *input,
                                            Complex<float> *output, const uint32_t device_id, cudaStream_t stream);
template CUDA_LIB_EXPORT cudaError_t FillV2(const int64_t output_size, const Complex<double> *input,
                                            Complex<double> *output, const uint32_t device_id, cudaStream_t stream);
