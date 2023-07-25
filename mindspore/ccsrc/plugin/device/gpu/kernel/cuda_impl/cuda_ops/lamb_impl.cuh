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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LAMB_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LAMB_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
template <typename T>
CUDA_LIB_EXPORT cudaError_t ApplyLambEraly(const size_t size, T *variable, T *m, T *v, const float *beta1,
                                           const float *beta2, const float *epsilon, const float *decay,
                                           const int32_t *global_step, const T *gradient, float *update,
                                           float *var_float, float *grad_float, float *g_hat_var,
                                           cudaStream_t cuda_stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t ApplyLambLater(const size_t size, T *variable, const float *lr, const float *update,
                                           const float *trust_ratio, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LAMB_IMPL_CUH_
