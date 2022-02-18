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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_INSTANCE_NORM_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_INSTANCE_NORM_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
CUDA_LIB_EXPORT void CopyMemDevice2Device(const size_t N, const size_t C, float *gamma_addr, float *beta_addr,
                                          float *runing_mean_addr, float *runnig_variance_addr, float *ws_gamma,
                                          float *ws_beta, float *ws_mean, float *ws_var, cudaStream_t cuda_stream);
CUDA_LIB_EXPORT void ComputeMean(const size_t N, const size_t C, float *dgamma, float *dbeta, const float *ws_dgamma,
                                 const float *ws_dbeta, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_INSTANCE_NORM_IMPL_CUH_
