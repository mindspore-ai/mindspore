/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PSROI_POOLING_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PSROI_POOLING_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t PSROIPoolForwardLauncher(const T *input, const T spatial_scale, const int rois_number,
                                                     const int feature_height, const int feature_width,
                                                     const int feature_channels, const int pooled_height,
                                                     const int pooled_width, const T *roi_boxes, const int group_size,
                                                     const int output_channels, T *output_data, int *mapping_channel,
                                                     cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t PSROIPoolBackwardLauncher(
  const T *input_diff, const int *mapping_channel, const int batch_size, const int rois_number, const T spatial_scale,
  const int feature_channels, const int feature_height, const int feature_width, const int pooled_width,
  const int pooled_height, const int output_channels, T *output_diff, const T *roi_boxes, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_PSROI_POOLING_IMPL_CUH_
