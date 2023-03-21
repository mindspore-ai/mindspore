/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "assign_add_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

template <typename T>
__global__ void AssignAdd(const size_t size, T *ref, const T *value, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    output[pos] = ref[pos] + value[pos];
    ref[pos] = output[pos];
  }
  return;
}

template <typename T>
cudaError_t CalAssignAdd(const size_t size, T *ref, const T *value, T *output, const uint32_t &device_id,
                         cudaStream_t cuda_stream) {
  int thread_num = size > 512 ? 512 : size;
  AssignAdd<<<CUDA_BLOCKS_CAL(device_id, size, thread_num), thread_num, 0, cuda_stream>>>(size, ref, value, output);

  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<float>(const size_t size, float *ref, const float *value,
                                                         float *output, const uint32_t &device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<half>(const size_t size, half *ref, const half *value, half *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<double>(const size_t size, double *ref, const double *value,
                                                          double *output, const uint32_t &device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<char>(const size_t size, char *ref, const char *value, char *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<unsigned char>(const size_t size, unsigned char *ref,
                                                                 const unsigned char *value, unsigned char *output,
                                                                 const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<int16_t>(const size_t size, int16_t *ref, const int16_t *value,
                                                           int16_t *output, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<uint16_t>(const size_t size, uint16_t *ref, const uint16_t *value,
                                                            uint16_t *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<int>(const size_t size, int *ref, const int *value, int *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<uint32_t>(const size_t size, uint32_t *ref, const uint32_t *value,
                                                            uint32_t *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<int64_t>(const size_t size, int64_t *ref, const int64_t *value,
                                                           int64_t *output, const uint32_t &device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<uint64_t>(const size_t size, uint64_t *ref, const uint64_t *value,
                                                            uint64_t *output, const uint32_t &device_id,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<Complex<float>>(const size_t size, Complex<float> *ref,
                                                                  const Complex<float> *value, Complex<float> *output,
                                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignAdd<Complex<double>>(const size_t size, Complex<double> *ref,
                                                                   const Complex<double> *value,
                                                                   Complex<double> *output, const uint32_t &device_id,
                                                                   cudaStream_t cuda_stream);
