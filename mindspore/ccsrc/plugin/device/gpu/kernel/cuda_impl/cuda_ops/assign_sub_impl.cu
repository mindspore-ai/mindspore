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

#include "assign_sub_impl.cuh"
template <typename T>
__global__ void AssignSub(const size_t size, T *ref, const T *value, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    output[pos] = ref[pos] - value[pos];
    ref[pos] = output[pos];
  }
  return;
}

template <typename T>
cudaError_t CalAssignSub(const size_t size, T *ref, const T *value, T *output, const uint32_t device_id,
                         cudaStream_t cuda_stream) {
  AssignSub<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(size, ref, value, output);
  CHECK_CUDA_LAUNCH_SUCCESS();
}

template CUDA_LIB_EXPORT cudaError_t CalAssignSub<uint8_t>(const size_t size, uint8_t *ref, const uint8_t *value,
                                                           uint8_t *output, const uint32_t device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignSub<int>(const size_t size, int *ref, const int *value, int *output,
                                                       const uint32_t device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignSub<int8_t>(const size_t size, int8_t *ref, const int8_t *value,
                                                          int8_t *output, const uint32_t device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignSub<int64_t>(const size_t size, int64_t *ref, const int64_t *value,
                                                           int64_t *output, const uint32_t device_id,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignSub<double>(const size_t size, double *ref, const double *value,
                                                          double *output, const uint32_t device_id,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignSub<float>(const size_t size, float *ref, const float *value,
                                                         float *output, const uint32_t device_id,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalAssignSub<half>(const size_t size, half *ref, const half *value, half *output,
                                                        const uint32_t device_id, cudaStream_t cuda_stream);
