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

#include "argmax_impl.cuh"
#include "device/gpu/cuda_common.h"
#include "include/cuda_fp16.h"
template <typename T>
__global__ void Argmax1D(const T* input, const int channel_size, int* output) {
  int max_index = 0;
  T max = input[0];
  for (int pos = 1; pos < channel_size; pos++) {
    if (max < input[pos]) {
      max = input[pos];
      max_index = pos;
    }
  }
  output[0] = max_index;
  return;
}
template <typename T>
__global__ void ArgmaxDefault2D(const T* input, const int batch_size, const int channel_size, int* output) {
  int pos;
  int max_index;
  T max;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch_size; i += blockDim.x * gridDim.x) {
    max = input[i * channel_size];
    max_index = 0;
    for (int j = 1; j < channel_size; j++) {
      pos = i * channel_size + j;
      if (max < input[pos]) {
        max = input[pos];
        max_index = j;
      }
    }

    output[i] = max_index;
  }
  return;
}
template <typename T>
__global__ void ArgmaxAxis2D(const T* input, const int batch_size, const int channel_size, int* output) {
  int pos;
  int max_index;
  T max;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < channel_size; i += blockDim.x * gridDim.x) {
    max = input[i];
    max_index = 0;
    for (int j = 1; j < batch_size; j++) {
      pos = j * channel_size + i;
      if (max < input[pos]) {
        max = input[pos];
        max_index = j;
      }
    }
    output[i] = max_index;
  }
  return;
}
template <typename T>
void CalArgmax(const T* input, const int batch_size, const int channel_size, const int axis, int* output,
               cudaStream_t cuda_stream) {
  if (batch_size == 0) {
    Argmax1D<<<1, 1, 0, cuda_stream>>>(input, channel_size, output);
  } else if (axis == 1) {
    ArgmaxDefault2D<<<GET_BLOCKS(batch_size), GET_THREADS, 0, cuda_stream>>>(input, batch_size, channel_size, output);
  } else {
    ArgmaxAxis2D<<<GET_BLOCKS(channel_size), GET_THREADS, 0, cuda_stream>>>(input, batch_size, channel_size, output);
  }
  return;
}

template void CalArgmax<float>(const float* input, const int batch_size, const int channel_size, const int axis,
                               int* output, cudaStream_t cuda_stream);
template void CalArgmax<half>(const half* input, const int batch_size, const int channel_size, const int axis,
                              int* output, cudaStream_t cuda_stream);
