/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SMOOTH_L1_LOSS_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SMOOTH_L1_LOSS_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

enum SmoothL1LossReductionMode { NONE = 0, MEAN = 1, SUM = 2, INVALID_MODE = 255 };

template <typename T>
CUDA_LIB_EXPORT void SmoothL1Loss(const SmoothL1LossReductionMode mode, const int64_t input_size, const float beta,
                                  const T *prediction, const T *target, T *loss, double *tmp_loss,
                                  const uint32_t device_id, cudaStream_t stream);
template <typename T>
CUDA_LIB_EXPORT void SmoothL1LossGrad(const SmoothL1LossReductionMode mode, const int64_t input_size, const float beta,
                                      const T *prediction, const T *target, const T *dloss, T *dx,
                                      const uint32_t device_id, cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_SMOOTH_L1_LOSS_IMPL_CUH_
