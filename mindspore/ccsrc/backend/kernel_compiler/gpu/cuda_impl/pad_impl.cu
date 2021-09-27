/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
}

template <typename T>
__global__ void PadGeneral(const T *input, T *output, const size_t *input_shape, const size_t *strides,
                           const int *paddings, const int input_size, const size_t input_rank) {
  for (size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < input_size; gt_id += blockDim.x * gridDim.x) {
    size_t linear_index = gt_id;
    size_t padded_linear_index = 0;
    for (int i = input_rank - 1; i >= 0; i--) {
      size_t unravel_dimension = input_shape[i];
      size_t unraveled_index = linear_index % unravel_dimension;
      padded_linear_index += ((unraveled_index + paddings[2 * i]) * strides[i]);
      linear_index -= unraveled_index;
      linear_index /= unravel_dimension;
    }
    output[padded_linear_index] = input[gt_id];
  }
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
}

// For internal OP use, not user facing
template <typename T>
__global__ void Pad3d(const size_t size, const T* input, const int num, const int channels, const int old_depth,
                      const int old_height, const int old_width, const int old_dhw, const int old_hw,
                      const int padded_depth, const int padded_height, const int padded_width, const int padded_dhw,
                      const int padded_hw, const int pad_head, const int pad_top, const int pad_left,
                      const float pad_value, T* output) {
  T pad_value_ = static_cast<T>(pad_value);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    const int pos_d = pos / padded_hw % padded_depth;
    const int pos_h = pos / padded_width % padded_height;
    const int pos_w = pos % padded_width;
    const int block_num = pos / padded_dhw;

    if (pos_d - pad_head < 0 || pos_h - pad_top < 0 || pos_w - pad_left < 0 || pos_d - pad_head >= old_depth ||
        pos_h - pad_top >= old_height || pos_w - pad_left >= old_width) {
      output[pos] = pad_value_;
    } else {
      int index = block_num * old_dhw + old_hw * (pos_d - pad_head) + old_width * (pos_h - pad_top) + pos_w - pad_left;
      output[pos] = input[index];
    }
  }
}

template <typename T>
__global__ void PadGrad3d(const size_t size, const T* dy, const int num, const int channels, const int old_depth,
                          const int old_height, const int old_width, const int old_dhw, const int old_hw,
                          const int padded_depth, const int padded_height, const int padded_width,
                          const int padded_dhw, const int padded_hw, const int pad_head, const int pad_top,
                          const int pad_left, T* dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    const int block_num = pos / old_dhw;
    const int pos_d = pos / old_hw % old_depth + pad_head;
    const int pos_h = pos / old_width % old_height + pad_top;
    const int pos_w = pos % old_width + pad_left;
    const int index = block_num * padded_dhw + pos_d * padded_hw + pos_h * padded_width + pos_w;
    dx[pos] = dy[index];
  }
}

// For internal OP use, not user facing
template <typename T>
__global__ void PadNDHWC(const size_t size, const T *input, const int num, const int old_depth, const int old_height,
                         const int old_width, const int channels, const int padded_depth, const int padded_height,
                         const int padded_width, const int pad_head, const int pad_top, const int pad_left,
                         float pad_value, T *output) {
  T pad_value_ = static_cast<T>(pad_value);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int block_num = pos / (channels * padded_width * padded_height * padded_depth);
    const int padded_w = pos / channels % padded_width;
    const int padded_h = pos / (channels * padded_width) % padded_height;
    const int padded_d = pos / (channels * padded_width * padded_height) % padded_depth;
    if (padded_d - pad_head < 0 || padded_h - pad_top < 0 || padded_w - pad_left < 0 ||
        padded_d - pad_head >= old_depth || padded_h - pad_top >= old_height || padded_w - pad_left >= old_width) {
      output[pos] = pad_value_;
    } else {
      output[pos] =
        input[(((block_num * old_depth + padded_d - pad_head) * old_height + padded_h - pad_top) * old_width +
               padded_w - pad_left) *
                channels +
              pos % channels];
    }
  }
}

template <typename T>
__global__ void PadGradNDHWC(const size_t size, const T *dy, const int num, const int old_depth, const int old_height,
                             const int old_width, const int channels, const int padded_depth, const int padded_height,
                             const int padded_width, const int pad_head, const int pad_top, const int pad_left, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int block_num = pos / (channels * old_width * old_height);
    const int padded_w = pos / channels % old_width + pad_left;
    const int padded_h = pos / (channels * old_width) % old_height + pad_top;
    const int padded_d = pos / (channels * old_width * old_height) % old_depth + pad_head;
    dx[pos] =
      dy[(((block_num * padded_depth + padded_d) * padded_height + padded_h) * padded_width + padded_w) * channels +
         pos % channels];
  }
}

template <typename T>
void CalPad(const size_t size, const T* input, const int num, const int channels, const int old_height,
            const int old_width, const int padded_height, const int padded_width, const int pad_top, const int pad_left,
            const float pad_value, T* output, cudaStream_t cuda_stream) {
  Pad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, channels, old_height, old_width,
                                                         padded_height, padded_width, pad_top, pad_left, pad_value,
                                                         output);
}

template <typename T>
void CalPadNHWC(const size_t size, const T* input, const int num, const int old_height, const int old_width,
                const int channels, const int padded_height, const int padded_width, const int pad_top,
                const int pad_left, const float pad_value, T* output, cudaStream_t cuda_stream) {
  PadNHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, old_height, old_width, channels,
      padded_height, padded_width, pad_top, pad_left, pad_value, output);
}

template <typename T>
void CalPadGeneral(const T *input, T *output, const size_t *input_shape, const size_t *strides,
                   const int *paddings, const int input_size, const size_t input_rank, cudaStream_t cuda_stream) {
  PadGeneral<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input, output, input_shape, strides, paddings,
                                                                      input_size, input_rank);
}

template <typename T>
void CalPadGradNHWC(const size_t size, const T* dy, const int num, const int old_height, const int old_width,
                 const int channels, const int padded_height, const int padded_width, const int pad_top,
                const int pad_left, T* dx, cudaStream_t cuda_stream) {
  PadGradNHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, old_height, old_width, channels,
      padded_height, padded_width, pad_top, pad_left, dx);
}

template <typename T>
void CalPadGrad(const size_t size, const T* dy, const int num, const int channels, const int old_height,
                const int old_width, const int padded_height, const int padded_width, const int pad_top,
                const int pad_left, T* dx, cudaStream_t cuda_stream) {
  PadGrad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, channels, old_height, old_width,
                                                             padded_height, padded_width, pad_top, pad_left, dx);
}

template <typename T>
void CalPad3d(const size_t size, const T* input, const int num, const int channels, const int old_depth,
              const int old_height, const int old_width, const int padded_depth, const int padded_height,
              const int padded_width, const int pad_head, const int pad_top, const int pad_left, const float pad_value,
              T* output, cudaStream_t cuda_stream) {
  const int old_hw = old_height * old_width;
  const int old_dhw = old_depth * old_hw;
  const int padded_hw = padded_height * padded_width;
  const int padded_dhw = padded_depth * padded_hw;
  Pad3d<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, channels, old_depth, old_height,
                                                           old_width, old_dhw, old_hw, padded_depth, padded_height,
                                                           padded_width, padded_dhw, padded_hw, pad_head, pad_top,
                                                           pad_left, pad_value, output);
}

template <typename T>
void CalPadGrad3d(const size_t size, const T* dy, const int num, const int channels, const int old_depth,
                  const int old_height, const int old_width, const int padded_depth, const int padded_height,
                  const int padded_width, const int pad_head, const int pad_top, const int pad_left, T* dx,
                  cudaStream_t cuda_stream) {
  const int old_hw = old_height * old_width;
  const int old_dhw = old_depth * old_hw;
  const int padded_hw = padded_height * padded_width;
  const int padded_dhw = padded_depth * padded_hw;
  PadGrad3d<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, channels, old_depth, old_height,
                                                               old_width, old_dhw, old_hw, padded_depth, padded_height,
                                                               padded_width, padded_dhw, padded_hw, pad_head, pad_top,
                                                               pad_left, dx);
}

template <typename T>
void CalPadNDHWC(const size_t size, const T *input, const int num, const int old_depth, const int old_height,
                 const int old_width, const int channels, const int padded_depth, const int padded_height,
                 const int padded_width, const int pad_head, const int pad_top, const int pad_left,
                 const float pad_value, T *output, cudaStream_t cuda_stream) {
  PadNDHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, old_depth, old_height, old_width,
                                                              channels, padded_depth, padded_height, padded_width,
                                                              pad_head, pad_top, pad_left, pad_value, output);
}

template <typename T>
void CalPadGradNDHWC(const size_t size, const T *dy, const int num, const int old_depth, const int old_height,
                     const int old_width, const int channels, const int padded_depth, const int padded_height,
                     const int padded_width, const int pad_head, const int pad_top, const int pad_left, T *dx,
                     cudaStream_t cuda_stream) {
  PadGradNDHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, old_depth, old_height, old_width,
                                                                  channels, padded_depth, padded_height, padded_width,
                                                                  pad_head, pad_top, pad_left, dx);
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
template void CalPadGeneral<float>(const float *input, float *output, const size_t *input_shape, const size_t *strides,
                                   const int *paddings, const int input_size, const size_t input_rank,
                                   cudaStream_t cuda_stream);
template void CalPadGeneral<half>(const half *input, half *output, const size_t *input_shape, const size_t *strides,
                                  const int *paddings, const int input_size, const size_t input_rank,
                                  cudaStream_t cuda_stream);
template void CalPadGeneral<int>(const int *input, int *output, const size_t *input_shape, const size_t *strides,
                                 const int *paddings, const int input_size, const size_t input_rank,
                                 cudaStream_t cuda_stream);
template void CalPad3d<float>(const size_t size, const float* input, const int num, const int channels,
                              const int old_depth, const int old_height, const int old_width, const int padded_depth,
                              const int padded_height, const int padded_width, const int pad_head, const int pad_top,
                              const int pad_left, const float pad_value, float* output, cudaStream_t cuda_stream);
template void CalPad3d<half>(const size_t size, const half* input, const int num, const int channels,
                             const int old_depth, const int old_height, const int old_width, const int padded_depth,
                             const int padded_height, const int padded_width, const int pad_head, const int pad_top,
                             const int pad_left, const float pad_value, half* output, cudaStream_t cuda_stream);
template void CalPadGrad3d<float>(const size_t size, const float* dy, const int num, const int channels,
                                  const int old_depth, const int old_height, const int old_width,
                                  const int padded_depth, const int padded_height, const int padded_width,
                                  const int pad_head, const int pad_top, const int pad_left, float* dx,
                                  cudaStream_t cuda_stream);
template void CalPadGrad3d<half>(const size_t size, const half* dy, const int num, const int channels,
                                 const int old_depth, const int old_height, const int old_width,
                                 const int padded_depth, const int padded_height, const int padded_width,
                                 const int pad_head, const int pad_top, const int pad_left, half* dx,
                                 cudaStream_t cuda_stream);
template void CalPadGradNDHWC<float>(const size_t size, const float *dy, const int num, const int old_depth,
                                     const int old_height, const int old_width, const int channels,
                                     const int padded_depth, const int padded_height, const int padded_width,
                                     const int pad_head, const int pad_top, const int pad_left, float *dx,
                                     cudaStream_t cuda_stream);
template void CalPadGradNDHWC<half>(const size_t size, const half *dy, const int num, const int old_depth,
                                    const int old_height, const int old_width, const int channels,
                                    const int padded_depth, const int padded_height, const int padded_width,
                                    const int pad_head, const int pad_top, const int pad_left, half *dx,
                                    cudaStream_t cuda_stream);
