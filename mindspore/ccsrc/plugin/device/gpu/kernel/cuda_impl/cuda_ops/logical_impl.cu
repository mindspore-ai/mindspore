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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/logical_impl.cuh"

template <typename T>
__global__ void LogicalNotKernel(const T *input1, T *output, const int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(input1[pos] == 0);
  }
}

template <>
__global__ void LogicalNotKernel(const bool *input1, bool *output, const int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = !input1[pos];
  }
}

template <typename T>
__global__ void LogicalAndKernel(const T *input_addr1, const T *input_addr2, T *output, const int size) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input_addr1[pos] * input_addr2[pos];
  }
}

template <typename T>
__global__ void LogicalOrKernel(const T *input_addr1, const T *input_addr2, T *output, const int size) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T sum = input_addr1[pos] + input_addr2[pos];
    output[pos] = static_cast<T>(sum > 0);
  }
}

template <typename T>
void LogicalNot(const int element_cnt, const T *input1, T *output, cudaStream_t stream, const uint32_t device_id) {
  LogicalNotKernel<<<CUDA_BLOCKS(device_id, element_cnt), CUDA_THREADS(device_id), 0, stream>>>(input1, output,
                                                                                                element_cnt);
}

template <typename T>
void LogicalNot(const int element_cnt, const T *input1, T *output, cudaStream_t stream) {
  LogicalNotKernel<<<(element_cnt + 255) / 256, 256, 0, stream>>>(input1, output, element_cnt);
}

template <typename T>
void LogicalAnd(const int element_cnt, const T *input1, const T *input2, T *output, cudaStream_t stream,
                const uint32_t device_id) {
  LogicalAndKernel<<<CUDA_BLOCKS(device_id, element_cnt), CUDA_THREADS(device_id), 0, stream>>>(input1, input2, output,
                                                                                                element_cnt);
}

template <typename T>
void LogicalOr(const int element_cnt, const T *input1, const T *input2, T *output, cudaStream_t stream,
               const uint32_t device_id) {
  LogicalOrKernel<<<CUDA_BLOCKS(device_id, element_cnt), CUDA_THREADS(device_id), 0, stream>>>(input1, input2, output,
                                                                                               element_cnt);
}

template CUDA_LIB_EXPORT void LogicalNot(const int element_cnt, const int32_t *input1, int32_t *output,
                                         cudaStream_t stream, const uint32_t device_id);

template CUDA_LIB_EXPORT void LogicalNot(const int element_cnt, const bool *input1, bool *output, cudaStream_t stream,
                                         const uint32_t device_id);

template CUDA_LIB_EXPORT void LogicalNot(const int element_cnt, const bool *input1, bool *output, cudaStream_t stream);

template CUDA_LIB_EXPORT void LogicalAnd(const int element_cnt, const int32_t *input1, const int32_t *input2,
                                         int32_t *output, cudaStream_t stream, const uint32_t device_id);

template CUDA_LIB_EXPORT void LogicalOr(const int element_cnt, const int32_t *input1, const int32_t *input2,
                                        int32_t *output, cudaStream_t stream, const uint32_t device_id);
