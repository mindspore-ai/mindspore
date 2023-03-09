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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RESIZE_LINEAR_1D_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RESIZE_LINEAR_1D_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

enum ResizeLinearCoordinateTransformationMode { ALIGN_CORNERS = 0, HALF_PIXEL = 1, INVALID_MODE = 255 };

template <typename T>
CUDA_LIB_EXPORT void ResizeLinear1D(const enum ResizeLinearCoordinateTransformationMode mode, const int64_t output_size,
                                    const int64_t in_width, const int64_t out_width, const T *input, T *output,
                                    const uint32_t device_id, cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT void ResizeLinear1DGrad(const enum ResizeLinearCoordinateTransformationMode mode,
                                        const int64_t output_size, const int64_t in_width, const int64_t out_width,
                                        const T *grad_output, T *grad_input, const uint32_t device_id,
                                        cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_RESIZE_LINEAR_1D_CUH_
