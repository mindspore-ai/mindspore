// /**
//  * Copyright 2021 Huawei Technologies Co., Ltd
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SYNC_BATCH_NORM_GRAD_IMPL_CUH
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SYNC_BATCH_NORM_GRAD_IMPL_CUH
#include "runtime/device/gpu/cuda_common.h"
template <typename T, typename G>
void CalSyncBatchNormGradPre(size_t N, size_t C, size_t H, size_t W, const T *x_input, const T *dy, G *saved_mean,
                             G *invstd_saved, float *dy_sum_local, float *dot_p_local, cudaStream_t cuda_stream);
template <typename T, typename S, typename G>
void CalSyncBatchNormGradPost(size_t N, size_t C, size_t H, size_t W, const T *x_input, const T *dy, T *dx,
                              G *saved_mean, G *invstd_saved, float *dy_sum_red, float *dot_p_red, S *scale, S *dscale,
                              S *dbias, float epsilon, cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SYNC_BATCH_NORM_GRAD_IMPL_CUH
