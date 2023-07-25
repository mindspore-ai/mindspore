/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <stdint.h>
#include <stdio.h>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"

template <typename T>
using Complex = mindspore::utils::Complex<T>;

// For internal OP use, not user facing
template <typename T>
__global__ void Pad(const size_t size, const T *input, const int num, const int channels, const int old_height,
                    const int old_width, const int padded_height, const int padded_width, const int pad_top,
                    const int pad_left, const float pad_value, T *output) {
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
__global__ void PadNHWC(const size_t size, const T *input, const int num, const int old_height, const int old_width,
                        const int channels, const int padded_height, const int padded_width, const int pad_top,
                        const int pad_left, float pad_value, T *output) {
  T pad_value_ = static_cast<T>(pad_value);
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int block_num = pos / channels / padded_width / padded_height;
    const int padded_w = pos / channels % padded_width;
    const int padded_h = pos / channels / padded_width % padded_height;
    if (padded_h - pad_top < 0 || padded_w - pad_left < 0 || padded_h - pad_top >= old_height ||
        padded_w - pad_left >= old_width) {
      output[pos] = pad_value_;
    } else {
      output[pos] = input[((block_num * old_height + padded_h - pad_top) * old_width + padded_w - pad_left) * channels +
                          pos % channels];
    }
  }
}

template <typename T>
__global__ void PadGeneral(const T *input, T *output, const PadInfo info, const int input_size,
                           const size_t input_rank) {
  const int *input_shape = info.shape;
  const int *strides = info.strides;
  const int *paddings = info.paddings;
  for (size_t gt_id = blockIdx.x * blockDim.x + threadIdx.x; gt_id < input_size; gt_id += blockDim.x * gridDim.x) {
    int linear_index = gt_id;
    int padded_linear_index = 0;
    for (int i = input_rank - 1; i >= 0; i--) {
      int unravel_dimension = input_shape[i];
      int unraveled_index = linear_index % unravel_dimension;
      padded_linear_index += ((unraveled_index + paddings[2 * i]) * strides[i]);
      linear_index -= unraveled_index;
      linear_index /= unravel_dimension;
    }
    output[padded_linear_index] = input[gt_id];
  }
}

template <typename T>
__global__ void PadGradNHWC(const size_t size, const T *dy, const int num, const int old_height, const int old_width,
                            const int channels, const int padded_height, const int padded_width, const int pad_top,
                            const int pad_left, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int block_num = pos / channels / old_width / old_height;
    const int padded_w = pos / channels % old_width + pad_left;
    const int padded_h = pos / channels / old_width % old_height + pad_top;
    dx[pos] = dy[((block_num * padded_height + padded_h) * padded_width + padded_w) * channels + pos % channels];
  }
}

template <typename T>
__global__ void PadGrad(const size_t size, const T *dy, const int num, const int channels, const int old_height,
                        const int old_width, const int padded_height, const int padded_width, const int pad_top,
                        const int pad_left, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int block_num = pos / old_width / old_height;
    const int padded_w = pos % old_width + pad_left;
    const int padded_h = pos / old_width % old_height + pad_top;
    dx[pos] = dy[(block_num * padded_height + padded_h) * padded_width + padded_w];
  }
}

// For internal OP use, not user facing
template <typename T>
__global__ void Pad3d(const size_t size, const T *input, const int num, const int channels, const int old_depth,
                      const int old_height, const int old_width, const int old_dhw, const int old_hw,
                      const int padded_depth, const int padded_height, const int padded_width, const int padded_dhw,
                      const int padded_hw, const int pad_head, const int pad_top, const int pad_left,
                      const float pad_value, T *output) {
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
__global__ void PadGrad3d(const size_t size, const T *dy, const int num, const int channels, const int old_depth,
                          const int old_height, const int old_width, const int old_dhw, const int old_hw,
                          const int padded_depth, const int padded_height, const int padded_width, const int padded_dhw,
                          const int padded_hw, const int pad_head, const int pad_top, const int pad_left, T *dx) {
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
cudaError_t CalPad(const size_t size, const T *input, const int num, const int channels, const int old_height,
                   const int old_width, const int padded_height, const int padded_width, const int pad_top,
                   const int pad_left, const float pad_value, T *output, cudaStream_t cuda_stream) {
  Pad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, channels, old_height, old_width,
                                                         padded_height, padded_width, pad_top, pad_left, pad_value,
                                                         output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalPadNHWC(const size_t size, const T *input, const int num, const int old_height, const int old_width,
                       const int channels, const int padded_height, const int padded_width, const int pad_top,
                       const int pad_left, const float pad_value, T *output, cudaStream_t cuda_stream) {
  PadNHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, old_height, old_width, channels,
                                                             padded_height, padded_width, pad_top, pad_left, pad_value,
                                                             output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalPadGeneral(const T *input, T *output, const PadInfo &info, const int input_size, const size_t input_rank,
                          cudaStream_t cuda_stream) {
  PadGeneral<<<GET_BLOCKS(input_size), GET_THREADS, 0, cuda_stream>>>(input, output, info, input_size, input_rank);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalPadGradNHWC(const size_t size, const T *dy, const int num, const int old_height, const int old_width,
                           const int channels, const int padded_height, const int padded_width, const int pad_top,
                           const int pad_left, T *dx, cudaStream_t cuda_stream) {
  PadGradNHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, old_height, old_width, channels,
                                                                 padded_height, padded_width, pad_top, pad_left, dx);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalPadGrad(const size_t size, const T *dy, const int num, const int channels, const int old_height,
                       const int old_width, const int padded_height, const int padded_width, const int pad_top,
                       const int pad_left, T *dx, cudaStream_t cuda_stream) {
  PadGrad<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, channels, old_height, old_width,
                                                             padded_height, padded_width, pad_top, pad_left, dx);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalPad3d(const size_t size, const T *input, const int num, const int channels, const int old_depth,
                     const int old_height, const int old_width, const int padded_depth, const int padded_height,
                     const int padded_width, const int pad_head, const int pad_top, const int pad_left,
                     const float pad_value, T *output, cudaStream_t cuda_stream) {
  const int old_hw = old_height * old_width;
  const int old_dhw = old_depth * old_hw;
  const int padded_hw = padded_height * padded_width;
  const int padded_dhw = padded_depth * padded_hw;
  Pad3d<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, input, num, channels, old_depth, old_height, old_width, old_dhw, old_hw, padded_depth, padded_height,
    padded_width, padded_dhw, padded_hw, pad_head, pad_top, pad_left, pad_value, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalPadGrad3d(const size_t size, const T *dy, const int num, const int channels, const int old_depth,
                         const int old_height, const int old_width, const int padded_depth, const int padded_height,
                         const int padded_width, const int pad_head, const int pad_top, const int pad_left, T *dx,
                         cudaStream_t cuda_stream) {
  const int old_hw = old_height * old_width;
  const int old_dhw = old_depth * old_hw;
  const int padded_hw = padded_height * padded_width;
  const int padded_dhw = padded_depth * padded_hw;
  PadGrad3d<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(
    size, dy, num, channels, old_depth, old_height, old_width, old_dhw, old_hw, padded_depth, padded_height,
    padded_width, padded_dhw, padded_hw, pad_head, pad_top, pad_left, dx);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalPadNDHWC(const size_t size, const T *input, const int num, const int old_depth, const int old_height,
                        const int old_width, const int channels, const int padded_depth, const int padded_height,
                        const int padded_width, const int pad_head, const int pad_top, const int pad_left,
                        const float pad_value, T *output, cudaStream_t cuda_stream) {
  PadNDHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, input, num, old_depth, old_height, old_width,
                                                              channels, padded_depth, padded_height, padded_width,
                                                              pad_head, pad_top, pad_left, pad_value, output);
  return GetCudaStatus();
}

template <typename T>
cudaError_t CalPadGradNDHWC(const size_t size, const T *dy, const int num, const int old_depth, const int old_height,
                            const int old_width, const int channels, const int padded_depth, const int padded_height,
                            const int padded_width, const int pad_head, const int pad_top, const int pad_left, T *dx,
                            cudaStream_t cuda_stream) {
  PadGradNDHWC<<<GET_BLOCKS(size), GET_THREADS, 0, cuda_stream>>>(size, dy, num, old_depth, old_height, old_width,
                                                                  channels, padded_depth, padded_height, padded_width,
                                                                  pad_head, pad_top, pad_left, dx);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t CalPad<float>(const size_t size, const float *input, const int num,
                                                   const int channels, const int old_height, const int old_width,
                                                   const int padded_height, const int padded_width, const int pad_top,
                                                   const int pad_left, float pad_value, float *output,
                                                   cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGrad<float>(const size_t size, const float *dy, const int num,
                                                       const int channels, const int old_height, const int old_width,
                                                       const int padded_height, const int padded_width,
                                                       const int pad_top, const int pad_left, float *dx,
                                                       cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPad<half>(const size_t size, const half *input, const int num,
                                                  const int channels, const int old_height, const int old_width,
                                                  const int padded_height, const int padded_width, const int pad_top,
                                                  const int pad_left, float pad_value, half *output,
                                                  cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGrad<half>(const size_t size, const half *dy, const int num,
                                                      const int channels, const int old_height, const int old_width,
                                                      const int padded_height, const int padded_width,
                                                      const int pad_top, const int pad_left, half *dx,
                                                      cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadNHWC<float>(const size_t size, const float *input, const int num,
                                                       const int old_height, const int old_width, const int channels,
                                                       const int padded_height, const int padded_width,
                                                       const int pad_top, const int pad_left, float pad_value,
                                                       float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadNHWC<half>(const size_t size, const half *input, const int num,
                                                      const int old_height, const int old_width, const int channels,
                                                      const int padded_height, const int padded_width,
                                                      const int pad_top, const int pad_left, float pad_value,
                                                      half *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGradNHWC<float>(const size_t size, const float *dy, const int num,
                                                           const int old_height, const int old_width,
                                                           const int channels, const int padded_height,
                                                           const int padded_width, const int pad_top,
                                                           const int pad_left, float *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGradNHWC<half>(const size_t size, const half *dy, const int num,
                                                          const int old_height, const int old_width, const int channels,
                                                          const int padded_height, const int padded_width,
                                                          const int pad_top, const int pad_left, half *dx,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<double>(const double *input, double *output, const PadInfo &info,
                                                           const int input_size, const size_t input_rank,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<float>(const float *input, float *output, const PadInfo &info,
                                                          const int input_size, const size_t input_rank,
                                                          cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<half>(const half *input, half *output, const PadInfo &info,
                                                         const int input_size, const size_t input_rank,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<int8_t>(const int8_t *input, int8_t *output, const PadInfo &info,
                                                           const int input_size, const size_t input_rank,
                                                           cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<int16_t>(const int16_t *input, int16_t *output, const PadInfo &info,
                                                            const int input_size, const size_t input_rank,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<int32_t>(const int32_t *input, int32_t *output, const PadInfo &info,
                                                            const int input_size, const size_t input_rank,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<int64_t>(const int64_t *input, int64_t *output, const PadInfo &info,
                                                            const int input_size, const size_t input_rank,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<uint8_t>(const uint8_t *input, uint8_t *output, const PadInfo &info,
                                                            const int input_size, const size_t input_rank,
                                                            cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<uint16_t>(const uint16_t *input, uint16_t *output,
                                                             const PadInfo &info, const int input_size,
                                                             const size_t input_rank, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<uint32_t>(const uint32_t *input, uint32_t *output,
                                                             const PadInfo &info, const int input_size,
                                                             const size_t input_rank, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<uint64_t>(const uint64_t *input, uint64_t *output,
                                                             const PadInfo &info, const int input_size,
                                                             const size_t input_rank, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<bool>(const bool *input, bool *output, const PadInfo &info,
                                                         const int input_size, const size_t input_rank,
                                                         cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<Complex<float>>(const Complex<float> *input, Complex<float> *output,
                                                                   const PadInfo &info, const int input_size,
                                                                   const size_t input_rank, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGeneral<Complex<double>>(const Complex<double> *input,
                                                                    Complex<double> *output, const PadInfo &info,
                                                                    const int input_size, const size_t input_rank,
                                                                    cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPad3d<float>(const size_t size, const float *input, const int num,
                                                     const int channels, const int old_depth, const int old_height,
                                                     const int old_width, const int padded_depth,
                                                     const int padded_height, const int padded_width,
                                                     const int pad_head, const int pad_top, const int pad_left,
                                                     const float pad_value, float *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPad3d<half>(const size_t size, const half *input, const int num,
                                                    const int channels, const int old_depth, const int old_height,
                                                    const int old_width, const int padded_depth,
                                                    const int padded_height, const int padded_width, const int pad_head,
                                                    const int pad_top, const int pad_left, const float pad_value,
                                                    half *output, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGrad3d<float>(const size_t size, const float *dy, const int num,
                                                         const int channels, const int old_depth, const int old_height,
                                                         const int old_width, const int padded_depth,
                                                         const int padded_height, const int padded_width,
                                                         const int pad_head, const int pad_top, const int pad_left,
                                                         float *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGrad3d<half>(const size_t size, const half *dy, const int num,
                                                        const int channels, const int old_depth, const int old_height,
                                                        const int old_width, const int padded_depth,
                                                        const int padded_height, const int padded_width,
                                                        const int pad_head, const int pad_top, const int pad_left,
                                                        half *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGradNDHWC<float>(
  const size_t size, const float *dy, const int num, const int old_depth, const int old_height, const int old_width,
  const int channels, const int padded_depth, const int padded_height, const int padded_width, const int pad_head,
  const int pad_top, const int pad_left, float *dx, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT cudaError_t CalPadGradNDHWC<half>(
  const size_t size, const half *dy, const int num, const int old_depth, const int old_height, const int old_width,
  const int channels, const int padded_depth, const int padded_height, const int padded_width, const int pad_head,
  const int pad_top, const int pad_left, half *dx, cudaStream_t cuda_stream);
