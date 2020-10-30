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

#include "backend/kernel_compiler/gpu/cuda_impl/scatter_update_impl.cuh"

template <typename T>
__global__ void ScatterUpdate(const int input_size, const int inner_size, const int indices_size, const T *input,
                              const int *indices, const T *updates, T *output) {
  for (int pos = blockIdx.x * blockDim.x + threadIdx.x; pos < input_size; pos += blockDim.x * gridDim.x) {
    output[pos] = input[pos];
    const int index = pos / inner_size;
    const int offset = pos % inner_size;
    for (int i = 0; i < indices_size; i++) {
      const int update_pos = i * inner_size + offset;
      output[pos] = (indices[i] == index ? updates[update_pos] : output[pos]);
    }
  }
}

template <typename T>
void CalScatterUpdate(const int &input_size, const int &inner_size, const int &indices_size, const T *input,
                      const int *indices, const T *updates, T *output, cudaStream_t cuda_stream) {
  ScatterUpdate<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input_size, inner_size, indices_size, input,
                                                                         indices, updates, output);
}

template void CalScatterUpdate<float>(const int &input_size, const int &inner_size, const int &indices_size,
                                      const float *input, const int *indices, const float *updates, float *output,
                                      cudaStream_t cuda_stream);
template void CalScatterUpdate<half>(const int &input_size, const int &inner_size, const int &indices_size,
                                     const half *input, const int *indices, const half *updates, half *output,
                                     cudaStream_t cuda_stream);
