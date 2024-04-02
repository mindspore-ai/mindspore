/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GROUP_NORM_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GROUP_NORM_GRAD_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename T>
CUDA_LIB_EXPORT cudaError_t GroupNormGrad(const int batch, const int num_channel, const int HxW,
                                          const int num_groups, const T *dy, const T *x, const T *mean,
                                          const T *rstd, const T *gamma, T *dx, T *dg, T *db,
                                          float *dscale, float *dbias, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_GROUP_NORM_GRAD_IMPL_CUH_
