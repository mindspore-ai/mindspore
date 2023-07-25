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

#include "equalcount_impl.cuh"
#include "include/cuda_fp16.h"
template <typename T>
__global__ void EqualCount(const int size, const T *input1, const T *input2, T *output) {
  T equal_count = 0;

  for (int i = 0; i < size; i++) {
    if (input1[i] == input2[i]) {
      equal_count++;
    }
  }

  output[0] = equal_count;
  return;
}
template <typename T>
cudaError_t CalEqualCount(const int size, const T *input1, const T *input2, T *output, cudaStream_t cuda_stream) {
  EqualCount<<<1, 1, 0, cuda_stream>>>(size, input1, input2, output);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalEqualCount<int>(const int size, const int *input1, const int *input2,
                                                        int *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalEqualCount<float>(const int size, const float *input1, const float *input2,
                                                          float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalEqualCount<half>(const int size, const half *input1, const half *input2,
                                                         half *output, cudaStream_t cuda_stream);
