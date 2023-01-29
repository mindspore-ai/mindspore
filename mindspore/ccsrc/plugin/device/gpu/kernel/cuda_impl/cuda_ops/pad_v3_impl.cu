/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "pad_v3_impl.cuh"
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
template <typename T>
using Complex = mindspore::utils::Complex<T>;

__host__ __device__ __forceinline__ int imin(int a, int b) { return a > b ? b : a; }

__host__ __device__ __forceinline__ int imax(int a, int b) { return a > b ? a : b; }

template <typename T>
__global__ void ConstantPad3d(const size_t size, const T *input, const int64_t num, const int64_t channels,
                              const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                              const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                              const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                              const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                              const int64_t pad_left, const T *pad_value, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    const int pos_d = pos / padded_hw % padded_depth;
    const int pos_h = pos / padded_width % padded_height;
    const int pos_w = pos % padded_width;
    const int block_num = pos / padded_dhw;

    if (pos_d - pad_head < 0 || pos_h - pad_top < 0 || pos_w - pad_left < 0 || pos_d - pad_head >= old_depth ||
        pos_h - pad_top >= old_height || pos_w - pad_left >= old_width) {
      output[pos] = pad_value[0];
    } else {
      int index = block_num * old_dhw + old_hw * (pos_d - pad_head) + old_width * (pos_h - pad_top) + pos_w - pad_left;
      output[pos] = input[index];
    }
  }
}

// size is the length of dx, which is the origin shape before pad forward.
template <typename T>
__global__ void ConstantPadGrad3d(const size_t size, const T *dy, const int64_t num, const int64_t channels,
                                  const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                                  const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                                  const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                                  const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                                  const int64_t pad_left, T *dx) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    const int block_num = pos / old_dhw;
    const int pos_d = pos / old_hw % old_depth + pad_head;
    const int pos_h = pos / old_width % old_height + pad_top;
    const int pos_w = pos % old_width + pad_left;
    const int index = block_num * padded_dhw + pos_d * padded_hw + pos_h * padded_width + pos_w;
    dx[pos] = dy[index];
  }
}

template <typename T>
__global__ void CircularPad3d(const size_t size, const T *input, const int64_t old_depth, const int64_t old_height,
                              const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
                              const int64_t padded_width, const int64_t pad_head, const int64_t pad_top,
                              const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
                              const int64_t pad_right, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int64_t nc = pos / padded_width;
    const int64_t out_w = pos % padded_width;
    const int64_t out_h = nc % padded_height;
    nc /= padded_height;
    const int64_t out_d = nc % padded_depth;
    nc /= padded_depth;

    int in_d = ((out_d - pad_head) % old_depth + old_depth) % old_depth;
    int in_h = ((out_h - pad_top) % old_height + old_height) % old_height;
    int in_w = ((out_w - pad_left) % old_width + old_width) % old_width;
    if (out_d < pad_head) {
      in_d = (in_d + imin(0, pad_back) + old_depth) % old_depth;
    }
    if (out_d >= old_depth + pad_head) {
      in_d = (in_d + imax(0, -pad_head) + old_depth) % old_depth;
    }
    if (out_h < pad_top) {
      in_h = (in_h + imin(0, pad_down) + old_height) % old_height;
    }
    if (out_h >= old_height + pad_top) {
      in_h = (in_h + imax(0, -pad_top) + old_height) % old_height;
    }
    if (out_w < pad_left) {
      in_w = (in_w + imin(0, pad_right) + old_width) % old_width;
    }
    if (out_w >= old_width + pad_left) {
      in_w = (in_w + imax(0, -pad_left) + old_width) % old_width;
    }
    output[pos] = input[(nc * old_depth * old_height + in_d * old_height + in_h) * old_width + in_w];
  }
}

template <typename T>
__global__ void CircularPadGrad3d(const size_t size, const T *input, const int64_t old_depth, const int64_t old_height,
                                  const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
                                  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top,
                                  const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
                                  const int64_t pad_right, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int nc = pos / old_width;
    const int out_w = pos % old_width;
    const int out_h = nc % old_height;
    nc /= old_height;
    const int out_d = nc % old_depth;
    nc /= old_depth;

    int in_d = ((out_d - pad_head) % padded_depth + padded_depth) % padded_depth;
    int in_h = ((out_h - pad_top) % padded_height + padded_height) % padded_height;
    int in_w = ((out_w - pad_left) % padded_width + padded_width) % padded_width;

    if (out_d < pad_head) {
      in_d = (in_d + imin(0, pad_back) + padded_depth) % padded_depth;
    }
    if (out_d >= padded_depth + pad_head) {
      in_d = (in_d + imax(0, -pad_head) + padded_depth) % padded_depth;
    }
    if (out_h < pad_top) {
      in_h = (in_h + imin(0, pad_down) + padded_height) % padded_height;
    }
    if (out_h >= padded_height + pad_top) {
      in_h = (in_h + imax(0, -pad_top) + padded_height) % padded_height;
    }
    if (out_w < pad_left) {
      in_w = (in_w + imin(0, pad_right) + padded_width) % padded_width;
    }
    if (out_w >= padded_width + pad_left) {
      in_w = (in_w + imax(0, -pad_left) + padded_width) % padded_width;
    }

    int index = (nc * padded_depth * padded_height + in_d * padded_height + in_h) * padded_width + in_w;
    MsAtomicAdd(&output[index], input[pos]);
  }
}

template <typename T>
__global__ void ReflectPad3d(const size_t size, const T *input, const int64_t num, const int64_t channels,
                             const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                             const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                             const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                             const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                             const int64_t pad_left, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int pos_d = pos / padded_hw % padded_depth;      // z
    int pos_h = pos / padded_width % padded_height;  // y
    int pos_w = pos % padded_width;                  // x
    int block_num = pos / padded_dhw;                // n * c

    int i_start_w = imax(0, -pad_left);
    int o_start_w = imax(0, pad_left);
    int i_start_h = imax(0, -pad_top);
    int o_start_h = imax(0, pad_top);
    int i_start_d = imax(0, -pad_head);
    int o_start_d = imax(0, pad_head);

    int map_ori_d = abs(pos_d - pad_head) - abs(pos_d - (old_depth + pad_head - 1)) - pos_d + 2 * pad_head + old_depth -
                    1 - o_start_d + i_start_d;
    int map_ori_h = abs(pos_h - pad_top) - abs(pos_h - (old_height + pad_top - 1)) - pos_h + 2 * pad_top + old_height -
                    1 - o_start_h + i_start_h;
    int map_ori_w = abs(pos_w - pad_left) - abs(pos_w - (old_width + pad_left - 1)) - pos_w + 2 * pad_left + old_width -
                    1 - o_start_w + i_start_w;

    int index = block_num * old_dhw + old_hw * map_ori_d + old_width * map_ori_h + map_ori_w;
    output[pos] = input[index];
  }
}

// input is the origin shape before pad forward
template <typename T>
__global__ void ReflectPadGrad3d(const size_t size, T *input, const int64_t num, const int64_t channels,
                                 const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                                 const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                                 const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                                 const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                                 const int64_t pad_left, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int pos_d = pos / old_hw % old_depth;
    int pos_h = pos / old_width % old_height;
    int pos_w = pos % old_width;
    int block_num = pos / old_dhw;

    int i_start_w = imax(0, -pad_left);
    int o_start_w = imax(0, pad_left);
    int i_start_h = imax(0, -pad_top);
    int o_start_h = imax(0, pad_top);
    int i_start_d = imax(0, -pad_head);
    int o_start_d = imax(0, pad_head);

    int map_ori_d = abs(pos_d - pad_head) - abs(pos_d - (padded_depth + pad_head - 1)) - pos_d + 2 * pad_head +
                    padded_depth - 1 - o_start_d + i_start_d;
    int map_ori_h = abs(pos_h - pad_top) - abs(pos_h - (padded_height + pad_top - 1)) - pos_h + 2 * pad_top +
                    padded_height - 1 - o_start_h + i_start_h;
    int map_ori_w = abs(pos_w - pad_left) - abs(pos_w - (padded_width + pad_left - 1)) - pos_w + 2 * pad_left +
                    padded_width - 1 - o_start_w + i_start_w;

    int index = block_num * padded_dhw + padded_hw * map_ori_d + padded_width * map_ori_h + map_ori_w;
    MsAtomicAdd(&output[index], input[pos]);
  }
}

template <typename T>
__global__ void EdgePad3d(const size_t size, const T *input, const int64_t num, const int64_t channels,
                          const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                          const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                          const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                          const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                          const int64_t pad_left, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int pos_d = pos / padded_hw % padded_depth;
    int pos_h = pos / padded_width % padded_height;
    int pos_w = pos % padded_width;
    int block_num = pos / padded_dhw;

    int i_start_w = imax(0, -pad_left);
    int i_start_h = imax(0, -pad_top);
    int i_start_d = imax(0, -pad_head);
    int o_start_w = imax(0, pad_left);
    int o_start_h = imax(0, pad_top);
    int o_start_d = imax(0, pad_head);

    int map_ori_d = imin(imax(pad_head, pos_d), old_depth + pad_head - 1) - o_start_d + i_start_d;
    int map_ori_h = imin(imax(pad_top, pos_h), old_height + pad_top - 1) - o_start_h + i_start_h;
    int map_ori_w = imin(imax(pad_left, pos_w), old_width + pad_left - 1) - o_start_w + i_start_w;

    int index = block_num * old_dhw + old_hw * map_ori_d + old_width * map_ori_h + map_ori_w;
    output[pos] = input[index];
  }
}

template <typename T>
__global__ void EdgePadGrad3d(const size_t size, T *input, const int64_t num, const int64_t channels,
                              const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                              const int64_t old_dhw, const int64_t old_hw, const int64_t padded_depth,
                              const int64_t padded_height, const int64_t padded_width, const int64_t padded_dhw,
                              const int64_t padded_hw, const int64_t pad_head, const int64_t pad_top,
                              const int64_t pad_left, T *output) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < (size); pos += blockDim.x * gridDim.x) {
    int pos_d = pos / old_hw % old_depth;
    int pos_h = pos / old_width % old_height;
    int pos_w = pos % old_width;
    int block_num = pos / old_dhw;

    int i_start_w = imax(0, -pad_left);
    int i_start_h = imax(0, -pad_top);
    int i_start_d = imax(0, -pad_head);
    int o_start_w = imax(0, pad_left);
    int o_start_h = imax(0, pad_top);
    int o_start_d = imax(0, pad_head);

    int map_ori_d = imin(imax(pad_head, pos_d), padded_depth + pad_head - 1) - o_start_d + i_start_d;
    int map_ori_h = imin(imax(pad_top, pos_h), padded_height + pad_top - 1) - o_start_h + i_start_h;
    int map_ori_w = imin(imax(pad_left, pos_w), padded_width + pad_left - 1) - o_start_w + i_start_w;

    int index = block_num * padded_dhw + padded_hw * map_ori_d + padded_width * map_ori_h + map_ori_w;
    MsAtomicAdd(&output[index], input[pos]);
  }
}

template <typename T>
void CalConstantPad3d(const size_t size, const T *input, const int64_t num, const int64_t channels,
                      const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                      const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width,
                      const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, const T *pad_value,
                      T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int64_t old_hw = old_height * old_width;
  const int64_t old_dhw = old_depth * old_hw;
  const int64_t padded_hw = padded_height * padded_width;
  const int64_t padded_dhw = padded_depth * padded_hw;
  ConstantPad3d<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, num, channels, old_depth, old_height, old_width, old_dhw, old_hw, padded_depth, padded_height,
    padded_width, padded_dhw, padded_hw, pad_head, pad_top, pad_left, pad_value, output);
}

template <typename T>
void CalConstantPadGrad3d(const size_t size, const T *dy, const int64_t num, const int64_t channels,
                          const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                          const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width,
                          const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, T *dx,
                          const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int64_t old_hw = old_height * old_width;
  const int64_t old_dhw = old_depth * old_hw;
  const int64_t padded_hw = padded_height * padded_width;
  const int64_t padded_dhw = padded_depth * padded_hw;
  ConstantPadGrad3d<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, dy, num, channels, old_depth, old_height, old_width, old_dhw, old_hw, padded_depth, padded_height,
    padded_width, padded_dhw, padded_hw, pad_head, pad_top, pad_left, dx);
}

template <typename T>
void CalCircularPad3d(const size_t size, const T *input, const int64_t old_depth, const int64_t old_height,
                      const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
                      const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
                      const int64_t pad_back, const int64_t pad_down, const int64_t pad_right, T *output,
                      const uint32_t &device_id, cudaStream_t cuda_stream) {
  CircularPad3d<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, old_depth, old_height, old_width, padded_depth, padded_height, padded_width, pad_head, pad_top,
    pad_left, pad_back, pad_down, pad_right, output);
}

template <typename T>
void CalCircularPadGrad3d(const size_t size, const T *input, const int64_t old_depth, const int64_t old_height,
                          const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
                          const int64_t padded_width, const int64_t pad_head, const int64_t pad_top,
                          const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
                          const int64_t pad_right, T *output, const uint32_t &device_id, cudaStream_t cuda_stream) {
  CircularPadGrad3d<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, old_depth, old_height, old_width, padded_depth, padded_height, padded_width, pad_head, pad_top,
    pad_left, pad_back, pad_down, pad_right, output);
}

template <typename T>
void CalReflectPad3d(const size_t size, const T *input, const int64_t num, const int64_t channels,
                     const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                     const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width,
                     const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, T *output,
                     const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int64_t old_hw = old_height * old_width;
  const int64_t old_dhw = old_depth * old_hw;
  const int64_t padded_hw = padded_height * padded_width;
  const int64_t padded_dhw = padded_depth * padded_hw;
  ReflectPad3d<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, num, channels, old_depth, old_height, old_width, old_dhw, old_hw, padded_depth, padded_height,
    padded_width, padded_dhw, padded_hw, pad_head, pad_top, pad_left, output);
}

template <typename T>
void CalReflectPadGrad3d(const size_t size, T *input, const int64_t num, const int64_t channels,
                         const int64_t old_depth, const int64_t old_height, const int64_t old_width,
                         const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width,
                         const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, T *output,
                         const uint32_t &device_id, cudaStream_t cuda_stream) {
  const int64_t old_hw = old_height * old_width;
  const int64_t old_dhw = old_depth * old_hw;
  const int64_t padded_hw = padded_height * padded_width;
  const int64_t padded_dhw = padded_depth * padded_hw;
  ReflectPadGrad3d<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, num, channels, old_depth, old_height, old_width, old_dhw, old_hw, padded_depth, padded_height,
    padded_width, padded_dhw, padded_hw, pad_head, pad_top, pad_left, output);
}

template <typename T>
void CalEdgePad3d(const size_t size, const T *input, const int64_t num, const int64_t channels, const int64_t old_depth,
                  const int64_t old_height, const int64_t old_width, const int64_t padded_depth,
                  const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
                  const int64_t pad_top, const int64_t pad_left, T *output, const uint32_t &device_id,
                  cudaStream_t cuda_stream) {
  const int64_t old_hw = old_height * old_width;
  const int64_t old_dhw = old_depth * old_hw;
  const int64_t padded_hw = padded_height * padded_width;
  const int64_t padded_dhw = padded_depth * padded_hw;
  EdgePad3d<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, num, channels, old_depth, old_height, old_width, old_dhw, old_hw, padded_depth, padded_height,
    padded_width, padded_dhw, padded_hw, pad_head, pad_top, pad_left, output);
}

template <typename T>
void CalEdgePadGrad3d(const size_t size, T *input, const int64_t num, const int64_t channels, const int64_t old_depth,
                      const int64_t old_height, const int64_t old_width, const int64_t padded_depth,
                      const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
                      const int64_t pad_top, const int64_t pad_left, T *output, const uint32_t &device_id,
                      cudaStream_t cuda_stream) {
  const int64_t old_hw = old_height * old_width;
  const int64_t old_dhw = old_depth * old_hw;
  const int64_t padded_hw = padded_height * padded_width;
  const int64_t padded_dhw = padded_depth * padded_hw;
  EdgePadGrad3d<<<CUDA_BLOCKS(device_id, size), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    size, input, num, channels, old_depth, old_height, old_width, old_dhw, old_hw, padded_depth, padded_height,
    padded_width, padded_dhw, padded_hw, pad_head, pad_top, pad_left, output);
}

template CUDA_LIB_EXPORT void CalConstantPad3d<double>(
  const size_t size, const double *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const double *pad_value, double *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<double>(
  const size_t size, const double *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, double *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<double>(const size_t size, const double *input, const int64_t num,
                                                      const int64_t channels, const int64_t old_depth,
                                                      const int64_t old_height, const int64_t old_width,
                                                      const int64_t padded_depth, const int64_t padded_height,
                                                      const int64_t padded_width, const int64_t pad_head,
                                                      const int64_t pad_top, const int64_t pad_left, double *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<double>(const size_t size, const double *input, const int64_t num,
                                                   const int64_t channels, const int64_t old_depth,
                                                   const int64_t old_height, const int64_t old_width,
                                                   const int64_t padded_depth, const int64_t padded_height,
                                                   const int64_t padded_width, const int64_t pad_head,
                                                   const int64_t pad_top, const int64_t pad_left, double *output,
                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<double>(const size_t size, double *input, const int64_t num,
                                                          const int64_t channels, const int64_t old_depth,
                                                          const int64_t old_height, const int64_t old_width,
                                                          const int64_t padded_depth, const int64_t padded_height,
                                                          const int64_t padded_width, const int64_t pad_head,
                                                          const int64_t pad_top, const int64_t pad_left, double *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<double>(const size_t size, double *input, const int64_t num,
                                                       const int64_t channels, const int64_t old_depth,
                                                       const int64_t old_height, const int64_t old_width,
                                                       const int64_t padded_depth, const int64_t padded_height,
                                                       const int64_t padded_width, const int64_t pad_head,
                                                       const int64_t pad_top, const int64_t pad_left, double *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<double>(
  const size_t size, const double *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, double *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<float>(
  const size_t size, const float *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const float *pad_value, float *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<float>(
  const size_t size, const float *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, float *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<float>(const size_t size, const float *input, const int64_t num,
                                                     const int64_t channels, const int64_t old_depth,
                                                     const int64_t old_height, const int64_t old_width,
                                                     const int64_t padded_depth, const int64_t padded_height,
                                                     const int64_t padded_width, const int64_t pad_head,
                                                     const int64_t pad_top, const int64_t pad_left, float *output,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<float>(const size_t size, const float *input, const int64_t num,
                                                  const int64_t channels, const int64_t old_depth,
                                                  const int64_t old_height, const int64_t old_width,
                                                  const int64_t padded_depth, const int64_t padded_height,
                                                  const int64_t padded_width, const int64_t pad_head,
                                                  const int64_t pad_top, const int64_t pad_left, float *output,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<float>(const size_t size, float *input, const int64_t num,
                                                         const int64_t channels, const int64_t old_depth,
                                                         const int64_t old_height, const int64_t old_width,
                                                         const int64_t padded_depth, const int64_t padded_height,
                                                         const int64_t padded_width, const int64_t pad_head,
                                                         const int64_t pad_top, const int64_t pad_left, float *output,
                                                         const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<float>(const size_t size, float *input, const int64_t num,
                                                      const int64_t channels, const int64_t old_depth,
                                                      const int64_t old_height, const int64_t old_width,
                                                      const int64_t padded_depth, const int64_t padded_height,
                                                      const int64_t padded_width, const int64_t pad_head,
                                                      const int64_t pad_top, const int64_t pad_left, float *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<float>(
  const size_t size, const float *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, float *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<half>(
  const size_t size, const half *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const half *pad_value, half *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<half>(
  const size_t size, const half *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, half *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<half>(const size_t size, const half *input, const int64_t num,
                                                    const int64_t channels, const int64_t old_depth,
                                                    const int64_t old_height, const int64_t old_width,
                                                    const int64_t padded_depth, const int64_t padded_height,
                                                    const int64_t padded_width, const int64_t pad_head,
                                                    const int64_t pad_top, const int64_t pad_left, half *output,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<half>(const size_t size, const half *input, const int64_t num,
                                                 const int64_t channels, const int64_t old_depth,
                                                 const int64_t old_height, const int64_t old_width,
                                                 const int64_t padded_depth, const int64_t padded_height,
                                                 const int64_t padded_width, const int64_t pad_head,
                                                 const int64_t pad_top, const int64_t pad_left, half *output,
                                                 const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<half>(const size_t size, half *input, const int64_t num,
                                                        const int64_t channels, const int64_t old_depth,
                                                        const int64_t old_height, const int64_t old_width,
                                                        const int64_t padded_depth, const int64_t padded_height,
                                                        const int64_t padded_width, const int64_t pad_head,
                                                        const int64_t pad_top, const int64_t pad_left, half *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<half>(const size_t size, half *input, const int64_t num,
                                                     const int64_t channels, const int64_t old_depth,
                                                     const int64_t old_height, const int64_t old_width,
                                                     const int64_t padded_depth, const int64_t padded_height,
                                                     const int64_t padded_width, const int64_t pad_head,
                                                     const int64_t pad_top, const int64_t pad_left, half *output,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<half>(
  const size_t size, const half *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, half *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<int64_t>(
  const size_t size, const int64_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const int64_t *pad_value, int64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<int64_t>(
  const size_t size, const int64_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, int64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<int64_t>(const size_t size, const int64_t *input, const int64_t num,
                                                       const int64_t channels, const int64_t old_depth,
                                                       const int64_t old_height, const int64_t old_width,
                                                       const int64_t padded_depth, const int64_t padded_height,
                                                       const int64_t padded_width, const int64_t pad_head,
                                                       const int64_t pad_top, const int64_t pad_left, int64_t *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<int64_t>(const size_t size, const int64_t *input, const int64_t num,
                                                    const int64_t channels, const int64_t old_depth,
                                                    const int64_t old_height, const int64_t old_width,
                                                    const int64_t padded_depth, const int64_t padded_height,
                                                    const int64_t padded_width, const int64_t pad_head,
                                                    const int64_t pad_top, const int64_t pad_left, int64_t *output,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<int64_t>(
  const size_t size, int64_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, int64_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<int64_t>(const size_t size, int64_t *input, const int64_t num,
                                                        const int64_t channels, const int64_t old_depth,
                                                        const int64_t old_height, const int64_t old_width,
                                                        const int64_t padded_depth, const int64_t padded_height,
                                                        const int64_t padded_width, const int64_t pad_head,
                                                        const int64_t pad_top, const int64_t pad_left, int64_t *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<int64_t>(
  const size_t size, const int64_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, int64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<int32_t>(
  const size_t size, const int32_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const int32_t *pad_value, int32_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<int32_t>(
  const size_t size, const int32_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, int32_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<int32_t>(const size_t size, const int32_t *input, const int64_t num,
                                                       const int64_t channels, const int64_t old_depth,
                                                       const int64_t old_height, const int64_t old_width,
                                                       const int64_t padded_depth, const int64_t padded_height,
                                                       const int64_t padded_width, const int64_t pad_head,
                                                       const int64_t pad_top, const int64_t pad_left, int32_t *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<int32_t>(const size_t size, const int32_t *input, const int64_t num,
                                                    const int64_t channels, const int64_t old_depth,
                                                    const int64_t old_height, const int64_t old_width,
                                                    const int64_t padded_depth, const int64_t padded_height,
                                                    const int64_t padded_width, const int64_t pad_head,
                                                    const int64_t pad_top, const int64_t pad_left, int32_t *output,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<int32_t>(
  const size_t size, int32_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, int32_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<int32_t>(const size_t size, int32_t *input, const int64_t num,
                                                        const int64_t channels, const int64_t old_depth,
                                                        const int64_t old_height, const int64_t old_width,
                                                        const int64_t padded_depth, const int64_t padded_height,
                                                        const int64_t padded_width, const int64_t pad_head,
                                                        const int64_t pad_top, const int64_t pad_left, int32_t *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<int32_t>(
  const size_t size, const int32_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, int32_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<int16_t>(
  const size_t size, const int16_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const int16_t *pad_value, int16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<int16_t>(
  const size_t size, const int16_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, int16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<int16_t>(const size_t size, const int16_t *input, const int64_t num,
                                                       const int64_t channels, const int64_t old_depth,
                                                       const int64_t old_height, const int64_t old_width,
                                                       const int64_t padded_depth, const int64_t padded_height,
                                                       const int64_t padded_width, const int64_t pad_head,
                                                       const int64_t pad_top, const int64_t pad_left, int16_t *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<int16_t>(const size_t size, const int16_t *input, const int64_t num,
                                                    const int64_t channels, const int64_t old_depth,
                                                    const int64_t old_height, const int64_t old_width,
                                                    const int64_t padded_depth, const int64_t padded_height,
                                                    const int64_t padded_width, const int64_t pad_head,
                                                    const int64_t pad_top, const int64_t pad_left, int16_t *output,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<int16_t>(
  const size_t size, int16_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, int16_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<int16_t>(const size_t size, int16_t *input, const int64_t num,
                                                        const int64_t channels, const int64_t old_depth,
                                                        const int64_t old_height, const int64_t old_width,
                                                        const int64_t padded_depth, const int64_t padded_height,
                                                        const int64_t padded_width, const int64_t pad_head,
                                                        const int64_t pad_top, const int64_t pad_left, int16_t *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<int16_t>(
  const size_t size, const int16_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, int16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<int8_t>(
  const size_t size, const int8_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const int8_t *pad_value, int8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<int8_t>(
  const size_t size, const int8_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, int8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<int8_t>(const size_t size, const int8_t *input, const int64_t num,
                                                      const int64_t channels, const int64_t old_depth,
                                                      const int64_t old_height, const int64_t old_width,
                                                      const int64_t padded_depth, const int64_t padded_height,
                                                      const int64_t padded_width, const int64_t pad_head,
                                                      const int64_t pad_top, const int64_t pad_left, int8_t *output,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<int8_t>(const size_t size, const int8_t *input, const int64_t num,
                                                   const int64_t channels, const int64_t old_depth,
                                                   const int64_t old_height, const int64_t old_width,
                                                   const int64_t padded_depth, const int64_t padded_height,
                                                   const int64_t padded_width, const int64_t pad_head,
                                                   const int64_t pad_top, const int64_t pad_left, int8_t *output,
                                                   const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<int8_t>(const size_t size, int8_t *input, const int64_t num,
                                                          const int64_t channels, const int64_t old_depth,
                                                          const int64_t old_height, const int64_t old_width,
                                                          const int64_t padded_depth, const int64_t padded_height,
                                                          const int64_t padded_width, const int64_t pad_head,
                                                          const int64_t pad_top, const int64_t pad_left, int8_t *output,
                                                          const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<int8_t>(const size_t size, int8_t *input, const int64_t num,
                                                       const int64_t channels, const int64_t old_depth,
                                                       const int64_t old_height, const int64_t old_width,
                                                       const int64_t padded_depth, const int64_t padded_height,
                                                       const int64_t padded_width, const int64_t pad_head,
                                                       const int64_t pad_top, const int64_t pad_left, int8_t *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<int8_t>(
  const size_t size, const int8_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, int8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<uint64_t>(
  const size_t size, const uint64_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const uint64_t *pad_value, uint64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<uint64_t>(
  const size_t size, const uint64_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, uint64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<uint64_t>(const size_t size, const uint64_t *input, const int64_t num,
                                                        const int64_t channels, const int64_t old_depth,
                                                        const int64_t old_height, const int64_t old_width,
                                                        const int64_t padded_depth, const int64_t padded_height,
                                                        const int64_t padded_width, const int64_t pad_head,
                                                        const int64_t pad_top, const int64_t pad_left, uint64_t *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<uint64_t>(const size_t size, const uint64_t *input, const int64_t num,
                                                     const int64_t channels, const int64_t old_depth,
                                                     const int64_t old_height, const int64_t old_width,
                                                     const int64_t padded_depth, const int64_t padded_height,
                                                     const int64_t padded_width, const int64_t pad_head,
                                                     const int64_t pad_top, const int64_t pad_left, uint64_t *output,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<uint64_t>(
  const size_t size, uint64_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, uint64_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<uint64_t>(
  const size_t size, uint64_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, uint64_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<uint64_t>(
  const size_t size, const uint64_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, uint64_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<uint32_t>(
  const size_t size, const uint32_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const uint32_t *pad_value, uint32_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<uint32_t>(
  const size_t size, const uint32_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, uint32_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<uint32_t>(const size_t size, const uint32_t *input, const int64_t num,
                                                        const int64_t channels, const int64_t old_depth,
                                                        const int64_t old_height, const int64_t old_width,
                                                        const int64_t padded_depth, const int64_t padded_height,
                                                        const int64_t padded_width, const int64_t pad_head,
                                                        const int64_t pad_top, const int64_t pad_left, uint32_t *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<uint32_t>(const size_t size, const uint32_t *input, const int64_t num,
                                                     const int64_t channels, const int64_t old_depth,
                                                     const int64_t old_height, const int64_t old_width,
                                                     const int64_t padded_depth, const int64_t padded_height,
                                                     const int64_t padded_width, const int64_t pad_head,
                                                     const int64_t pad_top, const int64_t pad_left, uint32_t *output,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<uint32_t>(
  const size_t size, uint32_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, uint32_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<uint32_t>(
  const size_t size, uint32_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, uint32_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<uint32_t>(
  const size_t size, const uint32_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, uint32_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<uint16_t>(
  const size_t size, const uint16_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const uint16_t *pad_value, uint16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<uint16_t>(
  const size_t size, const uint16_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, uint16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<uint16_t>(const size_t size, const uint16_t *input, const int64_t num,
                                                        const int64_t channels, const int64_t old_depth,
                                                        const int64_t old_height, const int64_t old_width,
                                                        const int64_t padded_depth, const int64_t padded_height,
                                                        const int64_t padded_width, const int64_t pad_head,
                                                        const int64_t pad_top, const int64_t pad_left, uint16_t *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<uint16_t>(const size_t size, const uint16_t *input, const int64_t num,
                                                     const int64_t channels, const int64_t old_depth,
                                                     const int64_t old_height, const int64_t old_width,
                                                     const int64_t padded_depth, const int64_t padded_height,
                                                     const int64_t padded_width, const int64_t pad_head,
                                                     const int64_t pad_top, const int64_t pad_left, uint16_t *output,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<uint16_t>(
  const size_t size, uint16_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, uint16_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<uint16_t>(
  const size_t size, uint16_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, uint16_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<uint16_t>(
  const size_t size, const uint16_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, uint16_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<uint8_t>(
  const size_t size, const uint8_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const uint8_t *pad_value, uint8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<uint8_t>(
  const size_t size, const uint8_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, uint8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<uint8_t>(const size_t size, const uint8_t *input, const int64_t num,
                                                       const int64_t channels, const int64_t old_depth,
                                                       const int64_t old_height, const int64_t old_width,
                                                       const int64_t padded_depth, const int64_t padded_height,
                                                       const int64_t padded_width, const int64_t pad_head,
                                                       const int64_t pad_top, const int64_t pad_left, uint8_t *output,
                                                       const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<uint8_t>(const size_t size, const uint8_t *input, const int64_t num,
                                                    const int64_t channels, const int64_t old_depth,
                                                    const int64_t old_height, const int64_t old_width,
                                                    const int64_t padded_depth, const int64_t padded_height,
                                                    const int64_t padded_width, const int64_t pad_head,
                                                    const int64_t pad_top, const int64_t pad_left, uint8_t *output,
                                                    const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<uint8_t>(
  const size_t size, uint8_t *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, uint8_t *output,
  const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<uint8_t>(const size_t size, uint8_t *input, const int64_t num,
                                                        const int64_t channels, const int64_t old_depth,
                                                        const int64_t old_height, const int64_t old_width,
                                                        const int64_t padded_depth, const int64_t padded_height,
                                                        const int64_t padded_width, const int64_t pad_head,
                                                        const int64_t pad_top, const int64_t pad_left, uint8_t *output,
                                                        const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<uint8_t>(
  const size_t size, const uint8_t *input, const int64_t old_depth, const int64_t old_height, const int64_t old_width,
  const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width, const int64_t pad_head,
  const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, uint8_t *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<Complex<float>>(
  const size_t size, const Complex<float> *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const Complex<float> *pad_value, Complex<float> *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<Complex<float>>(
  const size_t size, const Complex<float> *input, const int64_t old_depth, const int64_t old_height,
  const int64_t old_width, const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width,
  const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, Complex<float> *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<Complex<float>>(
  const size_t size, const Complex<float> *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  Complex<float> *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<Complex<float>>(
  const size_t size, const Complex<float> *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  Complex<float> *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPadGrad3d<Complex<float>>(
  const size_t size, Complex<float> *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  Complex<float> *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePadGrad3d<Complex<float>>(
  const size_t size, Complex<float> *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  Complex<float> *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPadGrad3d<Complex<float>>(
  const size_t size, const Complex<float> *input, const int64_t old_depth, const int64_t old_height,
  const int64_t old_width, const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width,
  const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, Complex<float> *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalConstantPad3d<Complex<double>>(
  const size_t size, const Complex<double> *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  const Complex<double> *pad_value, Complex<double> *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalCircularPad3d<Complex<double>>(
  const size_t size, const Complex<double> *input, const int64_t old_depth, const int64_t old_height,
  const int64_t old_width, const int64_t padded_depth, const int64_t padded_height, const int64_t padded_width,
  const int64_t pad_head, const int64_t pad_top, const int64_t pad_left, const int64_t pad_back, const int64_t pad_down,
  const int64_t pad_right, Complex<double> *output, const uint32_t &device_id, cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT void CalReflectPad3d<Complex<double>>(
  const size_t size, const Complex<double> *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  Complex<double> *output, const uint32_t &device_id, cudaStream_t cuda_stream);
template CUDA_LIB_EXPORT void CalEdgePad3d<Complex<double>>(
  const size_t size, const Complex<double> *input, const int64_t num, const int64_t channels, const int64_t old_depth,
  const int64_t old_height, const int64_t old_width, const int64_t padded_depth, const int64_t padded_height,
  const int64_t padded_width, const int64_t pad_head, const int64_t pad_top, const int64_t pad_left,
  Complex<double> *output, const uint32_t &device_id, cudaStream_t cuda_stream);
