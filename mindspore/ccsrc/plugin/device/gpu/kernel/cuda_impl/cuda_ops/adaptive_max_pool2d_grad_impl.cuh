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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADAPTIVE_MAX_POOL2D_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADAPTIVE_MAX_POOL2D_GRAD_IMPL_CUH_

#include <cuda_runtime.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t CalAdaptiveMaxPool2DGrad(const T *input_data, const S *max_index, const int n, const int c,
                                                     const uint input_height, const uint input_width,
                                                     const uint output_height, const uint output_width, T *output_data,
                                                     const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADAPTIVE_MAX_POOL2D_GRAD_IMPL_CUH_
