/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADAM_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADAM_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT void ApplyAdam(const size_t size, const int64_t batch_size, const T *gradient, const T *beta1_power,
                               const T *beta2_power, const T *learning_rate, const T *beta1, const T *beta2,
                               const T *epsilon, T *variable, T *m, T *v,
                               const bool use_nesterov, cudaStream_t cuda_stream);
template <typename T, typename S>
CUDA_LIB_EXPORT void AdamWeightDecayOp(const size_t size, const S *gradient, const float *learning_rate,
                                       const float *beta1, const float *beta2, const float *epsilon, const float *decay,
                                       S *variable, T *m, T *v, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_ADAM_IMPL_CUH_
