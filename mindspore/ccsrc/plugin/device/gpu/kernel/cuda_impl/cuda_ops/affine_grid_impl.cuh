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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_AFFINE_GRID_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_AFFINE_GRID_IMPL_CUH_
#include <cudnn.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T>
CUDA_LIB_EXPORT cudnnStatus_t CalculateAffineGrid4D(const T *theta_ptr, T *workspace_ptr, T *grid_ptr, const int32_t &N,
                                                    const int32_t &C, const int32_t &H, const int32_t &W,
                                                    const bool &align_corners, const uint32_t &device_id,
                                                    cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalculateAffineGrid5D(const T *theta_ptr, T *workspace_ptr, T *grid_ptr, const int32_t &N,
                                                  const int32_t &C, const int32_t &D, const int32_t &H,
                                                  const int32_t &W, const bool &align_corners,
                                                  const uint32_t &device_id, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_AFFINE_GRID_IMPL_CUH_
