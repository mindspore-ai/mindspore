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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/deformable_offsets_impl.cuh"
#include <stdio.h>
#include <stdint.h>
#include "include/cuda_fp16.h"

constexpr int OFFSET_NUM = 3;

template <typename T>
__device__ T DefromableBilinear(const T *input, const uint width, const uint height, const T x, const T y) {
  if (y <= static_cast<T>(-1) || y >= static_cast<T>(height) || x <= static_cast<T>(-1) || x >= static_cast<T>(width)) {
    return 0;
  }

  int left = floorf(x);
  int top = floorf(y);
  int right = left + 1;
  int bottom = top + 1;

  T l = x - static_cast<T>(left);
  T t = y - static_cast<T>(top);
  T r = static_cast<T>(1) - l;
  T b = static_cast<T>(1) - t;

  T lt = 0;
  T lb = 0;
  if (left >= 0) {
    if (top >= 0) {
      lt = input[top * width + left];
    }
    if (bottom <= height - 1) {
      lb = input[bottom * width + left];
    }
  }
  T rt = 0;
  T rb = 0;
  if (right <= width - 1) {
    if (top >= 0) {
      rt = input[top * width + right];
    }
    if (bottom <= height - 1) {
      rb = input[bottom * width + right];
    }
  }

  T w_lt = r * b;
  T w_rt = l * b;
  T w_lb = r * t;
  T w_rb = l * t;
  T val = (w_lt * lt + w_rt * rt + w_lb * lb + w_rb * rb);
  return val;
}
__global__ void GenPositionGridKernel(const uint kernel_h, const uint kernel_w, const uint stride_h,
                                      const uint stride_w, const uint dilations_h, const uint dilations_w,
                                      const uint pad_l, const uint pad_t, const uint output_w, const uint num,
                                      int32_t *position_grid) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
    uint y = i / output_w;
    uint x = i % output_w;
    uint pixel_y = y / kernel_h;
    uint pixel_x = x / kernel_w;
    uint kernel_y = y % kernel_h;
    uint kernel_x = x % kernel_w;
    uint index = i * 2;
    position_grid[index] = pixel_x * stride_w + kernel_x * dilations_w - pad_l;
    position_grid[index + 1] = pixel_y * stride_h + kernel_y * dilations_h - pad_t;
  }
}

template <class T>
__global__ void DeformableOffsetsKernel(const T *input, const T *offsets, const int32_t *position_grid, const uint c,
                                        const uint output_n_dim, const uint output_c_dim, const uint output_w,
                                        const uint c_size_per_dfm_group, const uint offset_n_dim,
                                        const uint offset_mask_dim, const uint offset_group_dim,
                                        const uint offset_kh_dim, const uint offset_kw_dim, const uint pixel_w,
                                        const uint input_n_dim, const uint input_c_dim, const uint input_h,
                                        const uint input_w, const uint kernel_h, const uint kernel_w, const uint num,
                                        T *output) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x) {
    // Get original input position
    const uint hw_idx = i % output_c_dim;
    const uint position_grid_idx = hw_idx * 2;
    const int input_x = position_grid[position_grid_idx];
    const int input_y = position_grid[position_grid_idx + 1];
    // Get offsets
    const uint n_index = i / output_n_dim;
    const uint c_index = i / output_c_dim % c;
    const uint x = hw_idx % output_w;
    const uint y = hw_idx / output_w;
    const uint dfm_group_index = c_index / c_size_per_dfm_group;
    const uint pixel_x = x / kernel_w;
    const uint pixel_y = y / kernel_h;
    const uint kernel_x = x % kernel_w;
    const uint kernel_y = y % kernel_h;
    const uint x_offsets_offset = n_index * offset_n_dim  // + 0 * offset_mask_dim
                                  + dfm_group_index * offset_group_dim + kernel_y * offset_kh_dim +
                                  kernel_x * offset_kw_dim + pixel_y * pixel_w + pixel_x;
    T x_offsets = offsets[x_offsets_offset];
    const int y_offsets_offset = x_offsets_offset + offset_mask_dim;
    T y_offsets = offsets[y_offsets_offset];
    const int mask_offset = y_offsets_offset + offset_mask_dim;
    T mask = offsets[mask_offset];
    // Deform
    T deformed_x = static_cast<T>(input_x) + x_offsets;
    T deformed_y = static_cast<T>(input_y) + y_offsets;
    const T *input_base = input + n_index * input_n_dim + c_index * input_c_dim;
    T bilinear_val = DefromableBilinear(input_base, input_w, input_h, deformed_x, deformed_y);
    output[i] = bilinear_val * mask;
  }
}

template <class T>
cudaError_t DeformableOffsets(const T *input, const T *offsets, const int32_t *position_grid, uint n, uint c,
                              uint input_h, uint input_w, uint dfm_group, uint kernel_h, uint kernel_w, uint output_h,
                              uint output_w, T *output, uint32_t device_id, cudaStream_t cuda_stream) {
  const uint pixel_w = output_w / kernel_w;
  const uint pixel_h = output_h / kernel_h;
  const uint output_c_dim = output_h * output_w;
  const uint output_n_dim = c * output_c_dim;
  const uint num = n * output_n_dim;
  const uint c_size_per_dfm_group = c / dfm_group;
  const uint offset_kw_dim = pixel_h * pixel_w;
  const uint offset_kh_dim = offset_kw_dim * kernel_w;
  const uint offset_group_dim = offset_kh_dim * kernel_h;
  const uint offset_mask_dim = offset_group_dim * dfm_group;
  const uint offset_n_dim = offset_mask_dim * OFFSET_NUM;
  const uint input_c_dim = input_h * input_w;
  const uint input_n_dim = input_c_dim * c;
  DeformableOffsetsKernel<<<CUDA_BLOCKS(device_id, num), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    input, offsets, position_grid, c, output_n_dim, output_c_dim, output_w, c_size_per_dfm_group, offset_n_dim,
    offset_mask_dim, offset_group_dim, offset_kh_dim, offset_kw_dim, pixel_w, input_n_dim, input_c_dim, input_h,
    input_w, kernel_h, kernel_w, num, output);
  return GetCudaStatus();
}

cudaError_t GenPositionGrid(const uint kernel_h, const uint kernel_w, const uint stride_h, const uint stride_w,
                            const uint dilations_h, const uint dilations_w, const uint pad_l, const uint pad_t,
                            const uint output_w, const uint num, int32_t *position_grid, const uint32_t device_id,
                            cudaStream_t cuda_stream) {
  GenPositionGridKernel<<<CUDA_BLOCKS(device_id, num), CUDA_THREADS(device_id), 0, cuda_stream>>>(
    kernel_h, kernel_w, stride_h, stride_w, dilations_h, dilations_w, pad_l, pad_t, output_w, num, position_grid);
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t DeformableOffsets<float>(const float *input, const float *offsets,
                                                              const int32_t *position_grid, uint n, uint c,
                                                              uint input_h, uint input_w, uint dfm_group, uint kernel_h,
                                                              uint kernel_w, uint output_h, uint output_w,
                                                              float *output, uint32_t device_id,
                                                              cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t DeformableOffsets<half>(const half *input, const half *offsets,
                                                             const int32_t *position_grid, uint n, uint c, uint input_h,
                                                             uint input_w, uint dfm_group, uint kernel_h, uint kernel_w,
                                                             uint output_h, uint output_w, half *output,
                                                             uint32_t device_id, cudaStream_t cuda_stream);
