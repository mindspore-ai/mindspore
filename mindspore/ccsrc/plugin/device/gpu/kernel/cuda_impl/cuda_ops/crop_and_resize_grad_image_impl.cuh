/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CROP_AND_RESIZE_GRAD_IMAGE_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CROP_AND_RESIZE_GRAD_IMAGE_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
template <typename T, typename G>
CUDA_LIB_EXPORT cudaError_t CalCropAndResizeGradImage(const int32_t size, const T *grads, const T *boxes,
                                                      const int *box_ind, int32_t num_boxes, int32_t batch,
                                                      int32_t image_height, int32_t image_width, int32_t crop_height,
                                                      int32_t crop_width, int32_t depth, int method, G *grad_image,
                                                      const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CROP_AND_RESIZE_GRAD_IMAGE_IMPL_CUH_
