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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_DEFORMABLE_OFFSETS_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_DEFORMABLE_OFFSETS_GRAD_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T>
CUDA_LIB_EXPORT void ApplyDeformableOffsetGrad(
  const uint dim_x_n, const uint dim_x_h, const uint dim_x_w, const uint dim_offset_h, const uint dim_offset_w,
  const uint dim_kernel_h, const uint dim_kernel_w, const uint dim_pad_top, const uint dim_pad_left,
  const uint dim_stride_h, const uint dim_stride_w, const uint dim_dilation_h, const uint dim_dilation_w,
  const uint dim_deformable_group, const uint dim_deformable_group_channel, bool nchw, T *input_grad, T *input_x,
  T *input_offset, T *output_grad_x, T *output_grad_offset, const uint device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_DEFORMABLE_OFFSETS_GRAD_IMPL_CUH_
