/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "backend/kernel_compiler/gpu/cuda_impl/scatter_add_impl.cuh"

template <typename T>
__global__ void ScatterAdd(const int input_size, const int inner_size, const int indices_size, const T *input,
                           const int *indices, const T *updates, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size; pos += blockDim.x * gridDim.x) {
    output[pos] = input[pos];
    const size_t index = pos / inner_size;
    const size_t offset = pos % inner_size;
      for (size_t i = 0; i < indices_size; i++) {
        const T value = updates[i*inner_size+offset];
        output[pos] += (indices[i] == index ? value : static_cast<T>(0.0));
    }
  }
}

template <typename T>
void CalScatterAdd(const int &input_size, const int &inner_size, const int &indices_size, const T *input,
                   const int *indices, const T *updates, T *output, cudaStream_t cuda_stream) {
  ScatterAdd<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, inner_size, indices_size, input,
                                                                      indices, updates, output);
}

template void CalScatterAdd<float>(const int &input_size, const int &inner_size, const int &indices_size,
                                   const float *input, const int *indices, const float *updates, float *output,
                                   cudaStream_t cuda_stream);
template void CalScatterAdd<half>(const int &input_size, const int &inner_size, const int &indices_size,
                                  const half *input, const int *indices, const half *updates, half *output,
                                  cudaStream_t cuda_stream);
