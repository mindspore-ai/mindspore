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
#include <cuda_runtime.h>

#include "sequence_mask_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

__global__ void ValidateArgs(int *maxlen, const int lengths_size, const int max_output_size) {
  int maxlen_value = *maxlen;
  if (maxlen_value < 0 || lengths_size * maxlen_value > max_output_size) {
    asm("trap;");
  }
}

template <typename T, typename S>
__global__ void SequenceMask(
    const T *input, T *maxlen, S *output, const size_t output_size) {
  T maxlen_value = *maxlen;

  for (size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < output_size; gt_id += gridDim.x * blockDim.x) {
    T mask_comparison_value = gt_id % maxlen_value;
    T input_comparison_index = (gt_id - mask_comparison_value) / maxlen_value;
    S result = mask_comparison_value < input[input_comparison_index];
    output[gt_id] = result;
  }
}

template <typename T, typename S>
void CalSequenceMask(const T *lengths, T *maxlen, S *output, const size_t output_size, cudaStream_t cuda_stream) {
  SequenceMask<<<GET_BLOCKS(output_size), GET_THREADS, 0, cuda_stream>>>(lengths, maxlen, output, output_size);
}

template void CalSequenceMask<int, bool>(const int *lengths, int *maxlen, bool *output, const size_t output_size,
                                         cudaStream_t cuda_stream);

template void CalSequenceMask<int64_t, bool>(const int64_t *lengths, int64_t *maxlen, bool *output,
                                             const size_t output_size, cudaStream_t cuda_stream);
