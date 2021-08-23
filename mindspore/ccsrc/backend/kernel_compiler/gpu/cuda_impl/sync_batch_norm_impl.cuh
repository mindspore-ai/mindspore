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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SYNC_BATCH_NORM_IMPL_CUH
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SYNC_BATCH_NORM_IMPL_CUH
#include "runtime/device/gpu/cuda_common.h"
template <typename T>
void CalSyncBatchNormPre(size_t N, size_t C, size_t H, size_t W, const T *input, int *output_n, float *means_local,
                         float *invstds_local, float epsilon, cudaStream_t cuda_stream);
template <typename T, typename G>
void CalSyncBatchNormGather(size_t N, size_t C, size_t H, size_t W, int *counts_global, float *means_global,
                            float *invstds_global, int *counts_local, float *means_local, float *invstds_local,
                            T *running_mean_output, T *running_var_output, G *running_mean_input, G *running_var_input,
                            float epsilon, float momentum, size_t group_rank, size_t group_size,
                            cudaStream_t cuda_stream);
template <typename T, typename S>
void CalSyncBatchNormPost(size_t N, size_t C, size_t H, size_t W, const T *input, T *output, float *means_local,
                          float *invstds_local, S *scale, S *bias, S *output_scale, S *output_bias, float epsilon,
                          cudaStream_t cuda_stream);
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_CUDA_IMPL_SYNC_BATCH_NORM_IMPL_CUH
