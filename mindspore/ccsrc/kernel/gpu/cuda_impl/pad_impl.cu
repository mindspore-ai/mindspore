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

#include <stdio.h>
#include <stdint.h>
#include "kernel/gpu/cuda_impl/pad_impl.cuh"

template <typename T>
__global__ void Pad(const size_t size, const T* input, const int num, const int channels, const int old_height,
                    const int old_width, const int padded_height, const int padded_width, const int pad_top,
                    const int pad_left, float pad_value, T* output) {
  T pad_value_ = static_cast<T>(pad_value);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int block_num = pos / padded_width / padded_height;
    const int padded_w = pos % padded_width;
    const int padded_h = pos / padded_width % padded_height;
    if (padded_h - pad_top < 0 || padded_w - pad_left < 0 || padded_h - pad_top >= old_height ||
        padded_w - pad_left >= old_width) {
      output[pos] = pad_value_;
    } else {
      output[pos] = input[(block_num * old_height + padded_h - pad_top) * old_width + padded_w - pad_left];
    }
  }
  return;
}

template <typename T>
__global__ void PadGrad(const size_t size, const T* dy, const int num, const int channels, const int old_height,
                        const int old_width, const int padded_height, const int padded_width, const int pad_top,
                        const int pad_left, T* dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int block_num = pos / old_width / old_height;
    const int padded_w = pos % old_width + pad_left;
    const int padded_h = pos / old_width % old_height + pad_top;
    dx[pos] = dy[(block_num * padded_height + padded_h) * padded_width + padded_w];
  }
  return;
}

template <typename T>
void CalPad(const size_t size, const T* input, const int num, const int channels, const int old_height,
            const int old_width, const int padded_height, const int padded_width, const int pad_top, const int pad_left,
            const float pad_value, T* output, cudaStream_t cuda_stream) {
  Pad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, channels, old_height, old_width,
                                                         padded_height, padded_width, pad_top, pad_left, pad_value,
                                                         output);
  return;
}

template <typename T>
void CalPadGrad(const size_t size, const T* dy, const int num, const int channels, const int old_height,
                const int old_width, const int padded_height, const int padded_width, const int pad_top,
                const int pad_left, T* dx, cudaStream_t cuda_stream) {
  PadGrad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, channels, old_height, old_width,
                                                             padded_height, padded_width, pad_top, pad_left, dx);
  return;
}

template void CalPad<float>(const size_t size, const float* input, const int num, const int channels,
                            const int old_height, const int old_width, const int padded_height, const int padded_width,
                            const int pad_top, const int pad_left, float pad_value, float* output,
                            cudaStream_t cuda_stream);
template void CalPadGrad<float>(const size_t size, const float* dy, const int num, const int channels,
                                const int old_height, const int old_width, const int padded_height,
                                const int padded_width, const int pad_top, const int pad_left, float* dx,
                                cudaStream_t cuda_stream);
template void CalPad<half>(const size_t size, const half* input, const int num, const int channels,
                           const int old_height, const int old_width, const int padded_height, const int padded_width,
                           const int pad_top, const int pad_left, float pad_value, half* output,
                           cudaStream_t cuda_stream);
template void CalPadGrad<half>(const size_t size, const half* dy, const int num, const int channels,
                               const int old_height, const int old_width, const int padded_height,
                               const int padded_width, const int pad_top, const int pad_left, half* dx,
                               cudaStream_t cuda_stream);
