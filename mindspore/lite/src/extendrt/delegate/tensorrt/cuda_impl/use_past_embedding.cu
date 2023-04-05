/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/extendrt/delegate/tensorrt/cuda_impl/use_past_embedding.cuh"
#include "src/extendrt/delegate/tensorrt/cuda_impl/cuda_helper.h"

__global__ void UsePastEmbedding(const int *input_position, int *out_position, int *init_reset, int *valid_length,
                                 int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    if (*init_reset == 0) {
      // copy all input position
      out_position[idx] = input_position[idx];
    } else {
      if (idx == 0) {
        out_position[idx] = *valid_length;
      } else {
        out_position[idx] = input_position[idx];
      }
    }
  }
}

void InvokeUsePastEmbedding(const int *input_position, int *out_position, int *init_reset, int *valid_length,
                            int size, cudaStream_t cuda_stream) {
  UsePastEmbedding<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(input_position, out_position, init_reset,
                                                                       valid_length, size);
}
