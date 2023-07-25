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

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/deformable_offsets_grad_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/util.cuh"
#include "include/cuda_fp16.h"

template <typename T>
__device__ void DeformableOffsetGradKernel(const uint offset_position_stride,
                                           const uint input_x_deformable_group_channel_stride,
                                           const uint input_x_w_stride, const uint input_x_h_stride,
                                           const uint grad_deformable_group_channel_stride, const uint dim_x_h,
                                           const uint dim_x_w, const uint dim_deformable_group_channel, float input_x_i,
                                           float input_x_j, const uint offset_index_base_pos,
                                           const uint input_grad_base_pos, const uint input_x_base_pos, T *input_grad,
                                           T *input_x, T *input_offset, T *output_grad_x, T *output_grad_offset) {
  const uint offset_index_i = offset_index_base_pos + offset_position_stride;
  const uint offset_index_weight = offset_index_base_pos + 2 * offset_position_stride;
  float offset_i = static_cast<float>(input_offset[offset_index_i]);
  float offset_j = static_cast<float>(input_offset[offset_index_base_pos]);
  float scale_weight = static_cast<float>(input_offset[offset_index_weight]);

  float floor_offset_i = floorf(offset_i);
  float floor_offset_j = floorf(offset_j);
  float ceil_offset_i = floor_offset_i + 1;
  float ceil_offset_j = floor_offset_j + 1;

  float floor_i = input_x_i + floor_offset_i;
  float floor_j = input_x_j + floor_offset_j;
  float ceil_i = input_x_i + ceil_offset_i;
  float ceil_j = input_x_j + ceil_offset_j;

  float ceil_weight_i = offset_i + 1 - ceil_offset_i;
  float ceil_weight_j = offset_j + 1 - ceil_offset_j;
  float floor_weight_i = 1 - ceil_weight_i;
  float floor_weight_j = 1 - ceil_weight_j;

  float floor_floor_weight = floor_weight_i * floor_weight_j;
  float ceil_floor_weight = ceil_weight_i * floor_weight_j;
  float floor_ceil_weight = floor_weight_i * ceil_weight_j;
  float ceil_ceil_weight = ceil_weight_i * ceil_weight_j;

  bool floor_floor_valid = false;
  bool ceil_floor_valid = false;
  bool floor_ceil_valid = false;
  bool ceil_ceil_valid = false;
  if (floor_i >= 0 && floor_i < dim_x_h) {
    if (floor_j >= 0 && floor_j < dim_x_w) {
      floor_floor_valid = true;
    }
    if (ceil_j >= 0 && ceil_j < dim_x_w) {
      floor_ceil_valid = true;
    }
  }

  if (ceil_i >= 0 && ceil_i < dim_x_h) {
    if (floor_j >= 0 && floor_j < dim_x_w) {
      ceil_floor_valid = true;
    }
    if (ceil_j >= 0 && ceil_j < dim_x_w) {
      ceil_ceil_valid = true;
    }
  }

  for (uint channel = 0; channel < dim_deformable_group_channel; ++channel) {
    float grad = static_cast<float>(input_grad[input_grad_base_pos + channel * grad_deformable_group_channel_stride]);
    float grad_scale = grad * scale_weight;
    uint tmp_input_x_base_pos = input_x_base_pos + channel * input_x_deformable_group_channel_stride;
    float current_x_pos;
    float floor_floor_value = 0;
    float ceil_floor_value = 0;
    float floor_ceil_value = 0;
    float ceil_ceil_value = 0;
    uint input_x_pos = 0;
    if (floor_floor_valid) {
      current_x_pos = tmp_input_x_base_pos + floor_i * input_x_h_stride + floor_j * input_x_w_stride;
      input_x_pos = static_cast<uint>(current_x_pos);
      floor_floor_value = static_cast<float>(input_x[input_x_pos]);
      MsAtomicAdd(output_grad_x + input_x_pos, static_cast<T>(grad_scale * floor_floor_weight));
    }

    if (ceil_floor_valid) {
      current_x_pos = tmp_input_x_base_pos + ceil_i * input_x_h_stride + floor_j * input_x_w_stride;
      input_x_pos = static_cast<uint>(current_x_pos);
      ceil_floor_value = static_cast<float>(input_x[input_x_pos]);
      MsAtomicAdd(output_grad_x + input_x_pos, static_cast<T>(grad_scale * ceil_floor_weight));
    }

    if (floor_ceil_valid) {
      current_x_pos = tmp_input_x_base_pos + floor_i * input_x_h_stride + ceil_j * input_x_w_stride;
      input_x_pos = static_cast<uint>(current_x_pos);
      floor_ceil_value = static_cast<float>(input_x[input_x_pos]);
      MsAtomicAdd(output_grad_x + input_x_pos, static_cast<T>(grad_scale * floor_ceil_weight));
    }

    if (ceil_ceil_valid) {
      current_x_pos = tmp_input_x_base_pos + ceil_i * input_x_h_stride + ceil_j * input_x_w_stride;
      input_x_pos = static_cast<uint>(current_x_pos);
      ceil_ceil_value = static_cast<float>(input_x[input_x_pos]);
      MsAtomicAdd(output_grad_x + input_x_pos, static_cast<T>(grad_scale * ceil_ceil_weight));
    }

    float delta = -floor_floor_value * floor_weight_j + ceil_floor_value * floor_weight_j -
                  floor_ceil_value * ceil_weight_j + ceil_ceil_value * ceil_weight_j;
    delta *= grad_scale;
    output_grad_offset[offset_index_i] += static_cast<T>(delta);

    delta = -floor_floor_value * floor_weight_i - ceil_floor_value * ceil_weight_i + floor_ceil_value * floor_weight_i +
            ceil_ceil_value * ceil_weight_i;
    delta *= grad_scale;
    output_grad_offset[offset_index_base_pos] += static_cast<T>(delta);

    delta = floor_floor_value * floor_floor_weight + ceil_floor_value * ceil_floor_weight +
            floor_ceil_value * floor_ceil_weight + ceil_ceil_value * ceil_ceil_weight;
    delta *= grad;
    output_grad_offset[offset_index_weight] += static_cast<T>(delta);
  }
}

template <typename T>
__global__ void DeformableOffsetGradNHWCKernel(const uint num_kernels, const uint dim_x_n, const uint dim_x_h,
                                               const uint dim_x_w, const uint dim_offset_h, const uint dim_offset_w,
                                               const uint dim_kernel_h, const uint dim_kernel_w, const uint dim_pad_top,
                                               const uint dim_pad_left, const uint dim_stride_h,
                                               const uint dim_stride_w, const uint dim_dilation_h,
                                               const uint dim_dilation_w, const uint dim_deformable_group,
                                               const uint dim_deformable_group_channel, T *input_grad, T *input_x,
                                               T *input_offset, T *output_grad_x, T *output_grad_offset) {
  const uint offset_kernel_w_stride = 1;
  const uint offset_kernel_h_stride = dim_kernel_w * offset_kernel_w_stride;
  const uint offset_deformable_group_stride = dim_kernel_h * offset_kernel_h_stride;
  const uint offset_position_stride = dim_deformable_group * offset_deformable_group_stride;
  const uint offset_offset_w_stride = 3 * offset_position_stride;
  const uint offset_offset_h_stride = dim_offset_w * offset_offset_w_stride;
  const uint offset_n_stride = dim_offset_h * offset_offset_h_stride;

  const uint grad_deformable_group_channel_stride = 1;
  const uint grad_deformable_group_stride = dim_deformable_group_channel * grad_deformable_group_channel_stride;
  const uint grad_kernel_w_stride = dim_deformable_group * grad_deformable_group_stride;
  const uint grad_offset_w_stride = dim_kernel_w * grad_kernel_w_stride;
  const uint grad_kernel_h_stride = dim_offset_w * grad_offset_w_stride;
  const uint grad_offset_h_stride = dim_kernel_h * grad_kernel_h_stride;
  const uint grad_n_stride = dim_offset_h * grad_offset_h_stride;

  const uint input_x_deformable_group_channel_stride = 1;
  const uint input_x_deformable_group_stride = dim_deformable_group_channel * input_x_deformable_group_channel_stride;
  const uint input_x_w_stride = dim_deformable_group * input_x_deformable_group_stride;
  const uint input_x_h_stride = dim_x_w * input_x_w_stride;
  const uint input_x_n_stride = dim_x_h * input_x_h_stride;

  for (uint index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels; index += gridDim.x * blockDim.x) {
    const uint offset_index_kernel_j = index % dim_kernel_w;
    uint tmp = index / dim_kernel_w;
    const uint offset_index_kernel_i = tmp % dim_kernel_h;
    tmp = tmp / dim_kernel_h;
    const uint offset_index_deformable_group_i = tmp % dim_deformable_group;
    tmp = tmp / dim_deformable_group;
    const uint offset_index_offset_j = tmp % dim_offset_w;
    tmp = tmp / dim_offset_w;
    const uint offset_index_offset_i = tmp % dim_offset_h;
    const uint offset_index_n_i = tmp / dim_offset_h;

    const uint offset_index_base_pos =
      offset_index_n_i * offset_n_stride + offset_index_deformable_group_i * offset_deformable_group_stride +
      offset_index_kernel_i * offset_kernel_h_stride + offset_index_kernel_j * offset_kernel_w_stride +
      offset_index_offset_i * offset_offset_h_stride + offset_index_offset_j * offset_offset_w_stride;
    const uint input_grad_base_pos =
      offset_index_n_i * grad_n_stride + offset_index_offset_i * grad_offset_h_stride +
      offset_index_offset_j * grad_offset_w_stride + offset_index_kernel_i * grad_kernel_h_stride +
      offset_index_kernel_j * grad_kernel_w_stride + offset_index_deformable_group_i * grad_deformable_group_stride;
    const uint input_x_base_pos =
      offset_index_n_i * input_x_n_stride + offset_index_deformable_group_i * input_x_deformable_group_stride;
    float input_x_i = -1.0 * dim_pad_top;
    float input_x_j = -1.0 * dim_pad_left;
    input_x_i += offset_index_offset_i * dim_stride_h + offset_index_kernel_i * dim_dilation_h;
    input_x_j += offset_index_offset_j * dim_stride_w + offset_index_kernel_j * dim_dilation_w;

    DeformableOffsetGradKernel(offset_position_stride, input_x_deformable_group_channel_stride, input_x_w_stride,
                               input_x_h_stride, grad_deformable_group_channel_stride, dim_x_h, dim_x_w,
                               dim_deformable_group_channel, input_x_i, input_x_j, offset_index_base_pos,
                               input_grad_base_pos, input_x_base_pos, input_grad, input_x, input_offset, output_grad_x,
                               output_grad_offset);
  }
}

template <typename T>
__global__ void DeformableOffsetGradNCHWKernel(const uint num_kernels, const uint dim_x_n, const uint dim_x_h,
                                               const uint dim_x_w, const uint dim_offset_h, const uint dim_offset_w,
                                               const uint dim_kernel_h, const uint dim_kernel_w, const uint dim_pad_top,
                                               const uint dim_pad_left, const uint dim_stride_h,
                                               const uint dim_stride_w, const uint dim_dilation_h,
                                               const uint dim_dilation_w, const uint dim_deformable_group,
                                               const uint dim_deformable_group_channel, T *input_grad, T *input_x,
                                               T *input_offset, T *output_grad_x, T *output_grad_offset) {
  const uint offset_offset_w_stride = 1;
  const uint offset_offset_h_stride = dim_offset_w * offset_offset_w_stride;
  const uint offset_kernel_w_stride = dim_offset_h * offset_offset_h_stride;
  const uint offset_kernel_h_stride = dim_kernel_w * offset_kernel_w_stride;
  const uint offset_deformable_group_stride = dim_kernel_h * offset_kernel_h_stride;
  const uint offset_position_stride = dim_deformable_group * offset_deformable_group_stride;
  const uint offset_n_stride = 3 * offset_position_stride;

  const uint grad_kernel_w_stride = 1;
  const uint grad_offset_w_stride = dim_kernel_w * grad_kernel_w_stride;
  const uint grad_kernel_h_stride = dim_offset_w * grad_offset_w_stride;
  const uint grad_offset_h_stride = dim_kernel_h * grad_kernel_h_stride;
  const uint grad_deformable_group_channel_stride = dim_offset_h * grad_offset_h_stride;
  const uint grad_deformable_group_stride = dim_deformable_group_channel * grad_deformable_group_channel_stride;
  const uint grad_n_stride = dim_deformable_group * grad_deformable_group_stride;

  const uint input_x_w_stride = 1;
  const uint input_x_h_stride = dim_x_w * input_x_w_stride;
  const uint input_x_deformable_group_channel_stride = dim_x_h * input_x_h_stride;
  const uint input_x_deformable_group_stride = dim_deformable_group_channel * input_x_deformable_group_channel_stride;
  const uint input_x_n_stride = dim_deformable_group * input_x_deformable_group_stride;

  for (uint index = blockIdx.x * blockDim.x + threadIdx.x; index < num_kernels; index += gridDim.x * blockDim.x) {
    const uint offset_index_offset_j = index % dim_offset_w;
    uint tmp = index / dim_offset_w;
    const uint offset_index_offset_i = tmp % dim_offset_h;
    tmp = tmp / dim_offset_h;
    const uint offset_index_kernel_j = tmp % dim_kernel_w;
    tmp = tmp / dim_kernel_w;
    const uint offset_index_kernel_i = tmp % dim_kernel_h;
    tmp = tmp / dim_kernel_h;
    const uint offset_index_deformable_group_i = tmp % dim_deformable_group;
    const uint offset_index_n_i = tmp / dim_deformable_group;

    float input_x_i = -1.0 * dim_pad_top;
    float input_x_j = -1.0 * dim_pad_left;
    input_x_i += offset_index_offset_i * dim_stride_h + offset_index_kernel_i * dim_dilation_h;
    input_x_j += offset_index_offset_j * dim_stride_w + offset_index_kernel_j * dim_dilation_w;

    const uint offset_index_base_pos =
      offset_index_n_i * offset_n_stride + offset_index_deformable_group_i * offset_deformable_group_stride +
      offset_index_kernel_i * offset_kernel_h_stride + offset_index_kernel_j * offset_kernel_w_stride +
      offset_index_offset_i * offset_offset_h_stride + offset_index_offset_j * offset_offset_w_stride;
    const uint input_grad_base_pos =
      offset_index_n_i * grad_n_stride + offset_index_offset_i * grad_offset_h_stride +
      offset_index_offset_j * grad_offset_w_stride + offset_index_kernel_i * grad_kernel_h_stride +
      offset_index_kernel_j * grad_kernel_w_stride + offset_index_deformable_group_i * grad_deformable_group_stride;
    const uint input_x_base_pos =
      offset_index_n_i * input_x_n_stride + offset_index_deformable_group_i * input_x_deformable_group_stride;

    DeformableOffsetGradKernel(offset_position_stride, input_x_deformable_group_channel_stride, input_x_w_stride,
                               input_x_h_stride, grad_deformable_group_channel_stride, dim_x_h, dim_x_w,
                               dim_deformable_group_channel, input_x_i, input_x_j, offset_index_base_pos,
                               input_grad_base_pos, input_x_base_pos, input_grad, input_x, input_offset, output_grad_x,
                               output_grad_offset);
  }
}

template <typename T>
cudaError_t ApplyDeformableOffsetGrad(const uint dim_x_n, const uint dim_x_h, const uint dim_x_w,
                                      const uint dim_offset_h, const uint dim_offset_w, const uint dim_kernel_h,
                                      const uint dim_kernel_w, const uint dim_pad_top, const uint dim_pad_left,
                                      const uint dim_stride_h, const uint dim_stride_w, const uint dim_dilation_h,
                                      const uint dim_dilation_w, const uint dim_deformable_group,
                                      const uint dim_deformable_group_channel, bool nchw, T *input_grad, T *input_x,
                                      T *input_offset, T *output_grad_x, T *output_grad_offset, const uint device_id,
                                      cudaStream_t cuda_stream) {
  const uint num_kernels = dim_x_n * dim_offset_h * dim_offset_w * dim_kernel_h * dim_kernel_w * dim_deformable_group;
  if (nchw) {
    DeformableOffsetGradNCHWKernel<<<CUDA_BLOCKS(device_id, num_kernels), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      num_kernels, dim_x_n, dim_x_h, dim_x_w, dim_offset_h, dim_offset_w, dim_kernel_h, dim_kernel_w, dim_pad_top,
      dim_pad_left, dim_stride_h, dim_stride_w, dim_dilation_h, dim_dilation_w, dim_deformable_group,
      dim_deformable_group_channel, input_grad, input_x, input_offset, output_grad_x, output_grad_offset);
  } else {
    DeformableOffsetGradNHWCKernel<<<CUDA_BLOCKS(device_id, num_kernels), CUDA_THREADS(device_id), 0, cuda_stream>>>(
      num_kernels, dim_x_n, dim_x_h, dim_x_w, dim_offset_h, dim_offset_w, dim_kernel_h, dim_kernel_w, dim_pad_top,
      dim_pad_left, dim_stride_h, dim_stride_w, dim_dilation_h, dim_dilation_w, dim_deformable_group,
      dim_deformable_group_channel, input_grad, input_x, input_offset, output_grad_x, output_grad_offset);
  }
  return GetCudaStatus();
}

template CUDA_LIB_EXPORT cudaError_t ApplyDeformableOffsetGrad<float>(
  const uint dim_x_n, const uint dim_x_h, const uint dim_x_w, const uint dim_offset_h, const uint dim_offset_w,
  const uint dim_kernel_h, const uint dim_kernel_w, const uint dim_pad_top, const uint dim_pad_left,
  const uint dim_stride_h, const uint dim_stride_w, const uint dim_dilation_h, const uint dim_dilation_w,
  const uint dim_deformable_group, const uint dim_deformable_group_channel, bool nchw, float *input_grad,
  float *input_x, float *input_offset, float *output_grad_x, float *output_grad_offset, const uint device_id,
  cudaStream_t cuda_stream);

template CUDA_LIB_EXPORT cudaError_t ApplyDeformableOffsetGrad<half>(
  const uint dim_x_n, const uint dim_x_h, const uint dim_x_w, const uint dim_offset_h, const uint dim_offset_w,
  const uint dim_kernel_h, const uint dim_kernel_w, const uint dim_pad_top, const uint dim_pad_left,
  const uint dim_stride_h, const uint dim_stride_w, const uint dim_dilation_h, const uint dim_dilation_w,
  const uint dim_deformable_group, const uint dim_deformable_group_channel, bool nchw, half *input_grad, half *input_x,
  half *input_offset, half *output_grad_x, half *output_grad_offset, const uint device_id, cudaStream_t cuda_stream);
