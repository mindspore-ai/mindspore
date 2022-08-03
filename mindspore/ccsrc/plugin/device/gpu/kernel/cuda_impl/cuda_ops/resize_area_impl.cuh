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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RESIZE_AREA_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RESIZE_AREA_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "include/cuda_fp16.h"
#include "mindapi/base/types.h"
struct ResizeAreaCachedInterpolation {
  size_t start;
  size_t end;
  float start_scale;
  float end_minus_one_scale;
  bool needs_bounding = true;
};
template <typename T>
CUDA_LIB_EXPORT void CalResizeArea(const T *images, ResizeAreaCachedInterpolation *x_interps,
  ResizeAreaCachedInterpolation *y_interps, float *output, int32_t batch_size, const int32_t channels,
  const int32_t out_height, const int32_t out_width, const int32_t in_height, const int32_t in_width,
  bool align_corners, const uint32_t &device_id, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RESIZE_AREA_IMPL_CUH_

