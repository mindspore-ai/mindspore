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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_APPLY_ADAGRAD_D_A_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_APPLY_ADAGRAD_D_A_IMPL_CUH_

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

template <typename T, typename T1, typename T2, typename T3, typename T4>
CUDA_LIB_EXPORT cudaError_t ApplyAdagradDA(const size_t batch_size, const size_t size, T *var, T *accum,
                                           T *squared_accum, const T *grad, const T1 *lr, const T2 *l1, const T3 *l2,
                                           const T4 *global_step, T *output_var, T *output_accum,
                                           T *output_squared_accum, const uint32_t &device_id,
                                           cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_APPLY_ADAGRAD_D_A_IMPL_CUH_
