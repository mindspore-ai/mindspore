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

#include "src/litert/delegate/tensorrt/cuda_impl/logical.cuh"
#include "src/litert/delegate/tensorrt/cuda_impl/cuda_helper.h"

template <typename T>
__global__ void LogicalNotKernel(const T *input1, T *output, int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = static_cast<T>(input1[pos] == 0);
  }
}

template <typename T>
__global__ void LogicalAndKernel(const T *input_addr1, const T *input_addr2, T *output, int size) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    output[pos] = input_addr1[pos] * input_addr2[pos];
  }
}

template <typename T>
__global__ void LogicalOrKernel(const T *input_addr1, const T *input_addr2, T *output, int size) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    T sum = input_addr1[pos] + input_addr2[pos];
    output[pos] = static_cast<T>(sum > 0);
  }
}

template <typename T>
__global__ void GreaterOrEqualKernal(const T *input1, const T *input2, T *output, int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = (input1[pos] >= input2[pos]);
  }
}

template <typename T>
__global__ void LessOrEqualKernal(const T *input1, const T *input2, T *output, int element_cnt) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < element_cnt; pos += blockDim.x * gridDim.x) {
    output[pos] = (input1[pos] <= input2[pos]);
  }
}

template <typename T>
void LogicalNot(const T *input1, T *output, int element_cnt, cudaStream_t stream) {
  LogicalNotKernel<<<GET_BLOCKS(element_cnt), GET_THREADS, 0, stream>>>(input1, output, element_cnt);
}

template <typename T>
void LogicalAnd(const T *input1, const T *input2, T *output, int element_cnt, cudaStream_t stream) {
  LogicalAndKernel<<<GET_BLOCKS(element_cnt), GET_THREADS, 0, stream>>>(input1, input2, output, element_cnt);
}

template <typename T>
void LogicalOr(const T *input1, const T *input2, T *output, int element_cnt, cudaStream_t stream) {
  LogicalOrKernel<<<GET_BLOCKS(element_cnt), GET_THREADS, 0, stream>>>(input1, input2, output, element_cnt);
}

template <typename T>
void GreaterOrEqual(const T *input1, const T *input2, T *output, int element_cnt, cudaStream_t stream) {
  GreaterOrEqualKernal<<<GET_BLOCKS(element_cnt), GET_THREADS, 0, stream>>>(input1, input2, output, element_cnt);
}

template <typename T>
void LessOrEqual(const T *input1, const T *input2, T *output, int element_cnt, cudaStream_t stream) {
  LessOrEqualKernal<<<GET_BLOCKS(element_cnt), GET_THREADS, 0, stream>>>(input1, input2, output, element_cnt);
}

template void GreaterOrEqual(const float *input1, const float *input2, float *output, int element_cnt,
                             cudaStream_t stream);

template void GreaterOrEqual(const int *input1, const int *input2, int *output, int element_cnt, cudaStream_t stream);

template void LessOrEqual(const float *input1, const float *input2, float *output, int element_cnt,
                          cudaStream_t stream);

template void LessOrEqual(const int *input1, const int *input2, int *output, int element_cnt, cudaStream_t stream);


template void LogicalNot(const int32_t *input1, int32_t *output, int element_cnt, cudaStream_t stream);

template void LogicalAnd(const int32_t *input1, const int32_t *input2, int32_t *output, int element_cnt,
                         cudaStream_t stream);

template void LogicalOr(const int32_t *input1, const int32_t *input2, int32_t *output, int element_cnt,
                        cudaStream_t stream);
