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
#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"

// For internal OP use, not user facing
template <typename T>
__global__ void Pad(const size_t size, const T* input, const int num, const int channels, const int old_height,
                    const int old_width, const int padded_height, const int padded_width, const int pad_top,
                    const int pad_left, const float pad_value, T* output) {
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

// For internal OP use, not user facing
template <typename T>
__global__ void PadNHWC(const size_t size, const T* input, const int num, const int old_height, const int old_width,
                        const int channels, const int padded_height, const int padded_width, const int pad_top,
                        const int pad_left, float pad_value, T* output) {
  T pad_value_ = static_cast<T>(pad_value);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int block_num = pos / channels / padded_width / padded_height;
    const int padded_w = pos / channels % padded_width;
    const int padded_h = pos / channels / padded_width % padded_height;
    if (padded_h - pad_top < 0 || padded_w - pad_left < 0 || padded_h - pad_top >= old_height ||
        padded_w - pad_left >= old_width) {
      output[pos] = pad_value_;
    } else {
      output[pos] = input[((block_num * old_height + padded_h - pad_top) * old_width + padded_w - pad_left)
                            *channels + pos % channels];
    }
  }
  return;
}

// Used by user facing 'Pad' API
template <typename T>
__global__ void PadGeneral(const size_t size, const T *input, const int num, const int channels_orig,
                           const int pad_channel_before, const int pad_channel_after, const int old_height,
                           const int old_width, const int padded_height, const int padded_width, const int pad_top,
                           const int pad_left, const T pad_value, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
    int block_num = (pos / padded_width) / padded_height;       // total blocks = (batch * channels)
    const int padded_w = pos % padded_width;                  // x coordinate refered to by cur 'pos'
    const int padded_h = (pos / padded_width) % padded_height;  // y coordinate refered to by cur 'pos'

    int channels_new = channels_orig + pad_channel_after + pad_channel_before;  // new number of channels from padding
    int channel_num = block_num % channels_new;                                 // current channel
    int batch_item = block_num / channels_new;                                  // current item in batch
    int equiv_block_num = 0;  // init variable to select equivalent block to copy data from from input

    if (padded_h - pad_top < 0 || padded_w - pad_left < 0 || padded_h - pad_top >= old_height ||
        padded_w - pad_left >= old_width || channel_num <= pad_channel_before - 1 ||
        channel_num > channels_orig + pad_channel_before - 1) {
      output[pos] = pad_value;
    } else {
      // on a block/x,y positon that isn't padding, copy data from the correct block/x,y pos the input
      // calculate from number of blocks of padding (due to channel padding) inserted prior
      equiv_block_num = block_num - (batch_item * (pad_channel_before + pad_channel_after)) - pad_channel_before;
      output[pos] = input[(equiv_block_num * old_height + padded_h - pad_top) * old_width + padded_w - pad_left];
    }
  }
  return;
}

template <typename T>
__global__ void PadGradNHWC(const size_t size, const T* dy, const int num, const int old_height, const int old_width,
                        const int channels, const int padded_height, const int padded_width, const int pad_top,
                        const int pad_left, T* dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int block_num = pos / channels / old_width / old_height;
    const int padded_w = pos / channels % old_width + pad_left;
    const int padded_h = pos / channels / old_width % old_height + pad_top;
    dx[pos] = dy[((block_num * padded_height + padded_h) * padded_width + padded_w)*channels+pos%channels];
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
void CalPadNHWC(const size_t size, const T* input, const int num, const int old_height, const int old_width,
                const int channels, const int padded_height, const int padded_width, const int pad_top,
                const int pad_left, const float pad_value, T* output, cudaStream_t cuda_stream) {
  PadNHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, old_height, old_width, channels,
      padded_height, padded_width, pad_top, pad_left, pad_value, output);
  return;
}

template <typename T>
void CalPadGeneral(const size_t size, const T *input, const int num, const int channels_orig,
                   const int pad_channel_before, const int pad_channel_after, const int old_height, const int old_width,
                   const int padded_height, const int padded_width, const int pad_top, const int pad_left,
                   const T pad_value, T *output, cudaStream_t cuda_stream) {
  PadGeneral<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, channels_orig, pad_channel_before,
                                                                pad_channel_after, old_height, old_width, padded_height,
                                                                padded_width, pad_top, pad_left, pad_value, output);
  return;
}

template <typename T>
void CalPadGradNHWC(const size_t size, const T* dy, const int num, const int old_height, const int old_width,
                 const int channels, const int padded_height, const int padded_width, const int pad_top,
                const int pad_left, T* dx, cudaStream_t cuda_stream) {
  PadGradNHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, old_height, old_width, channels,
      padded_height, padded_width, pad_top, pad_left, dx);
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
template void CalPadNHWC<float>(const size_t size, const float* input, const int num, const int old_height,
                                const int old_width, const int channels, const int padded_height,
                                const int padded_width, const int pad_top, const int pad_left, float pad_value,
                                float* output, cudaStream_t cuda_stream);
template void CalPadNHWC<half>(const size_t size, const half* input, const int num, const int old_height,
                               const int old_width, const int channels, const int padded_height,
                               const int padded_width, const int pad_top, const int pad_left, float pad_value,
                               half* output, cudaStream_t cuda_stream);
template void CalPadGradNHWC<float>(const size_t size, const float* dy, const int num, const int old_height,
                                    const int old_width, const int channels, const int padded_height,
                                    const int padded_width, const int pad_top, const int pad_left, float* dx,
                                    cudaStream_t cuda_stream);
template void CalPadGradNHWC<half>(const size_t size, const half* dy, const int num, const int old_height,
                                   const int old_width, const int channels, const int padded_height,
                                   const int padded_width, const int pad_top, const int pad_left, half* dx,
                                   cudaStream_t cuda_stream);
template void CalPadGeneral<float>(const size_t size, const float *input, const int num, const int channels_orig,
                                   const int pad_channel_before, const int pad_channel_after, const int old_height,
                                   const int old_width, const int padded_height, const int padded_width,
                                   const int pad_top, const int pad_left, const float pad_value, float *output,
                                   cudaStream_t cuda_stream);
template void CalPadGeneral<half>(const size_t size, const half *input, const int num, const int channels_orig,
                                  const int pad_channel_before, const int pad_channel_after, const int old_height,
                                  const int old_width, const int padded_height, const int padded_width,
                                  const int pad_top, const int pad_left, const half pad_value, half *output,
                                  cudaStream_t cuda_stream);
template void CalPadGeneral<int>(const size_t size, const int *input, const int num, const int channels_orig,
                                  const int pad_channel_before, const int pad_channel_after, const int old_height,
                                  const int old_width, const int padded_height, const int padded_width,
                                  const int pad_top, const int pad_left, const int pad_value, int *output,
                                  cudaStream_t cuda_stream);
