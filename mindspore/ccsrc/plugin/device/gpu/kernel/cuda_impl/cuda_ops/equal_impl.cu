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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/equal_impl.cuh"

template <typename T>
__global__ void EqualKernel(const T *input1, const T *input2, T *output, const int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = (input1[pos] - input2[pos] < 1e-6 && input1[pos] - input2[pos] > -1e-6);
  }
}

template <typename T>
void Equal(const size_t input_size, const T *input1, const T *input2, T *output, cudaStream_t stream,
           const uint32_t device_id) {
  EqualKernel<<<CUDA_BLOCKS(device_id, input_size), CUDA_THREADS(device_id), 0, stream>>>(input1, input2, output,
                                                                                          input_size);
  return;
}

template CUDA_LIB_EXPORT void Equal(const size_t input_size, const float *input1, const float *input2, float *output,
                                    cudaStream_t stream, const uint32_t device_id);
template CUDA_LIB_EXPORT void Equal(const size_t input_size, const int *input1, const int *input2, int *output,
                                    cudaStream_t stream, const uint32_t device_id);
