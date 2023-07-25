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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CTCLOSS_V2_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CTCLOSS_V2_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

template <typename S, typename T>
cudaError_t CalCTCLossV2(const S *log_probs_p, const T *target_p, const T *input_len_p, const T *target_len_p,
                         int64_t batch_size, int64_t target_stride, int64_t time_series, T blank, dim3 log_probs_shape,
                         dim3 log_alpha_shape, S *neg_log_p, S *log_alpha_p, uint32_t device_id,
                         cudaStream_t cuda_stream);

template <typename S, typename T>
cudaError_t CalCTCLossGradV2(const S *grad_out, const S *log_probs, const T *targets, const T *input_lengths,
                             const T *target_lengths, const S *neg_log_likelihood, const S *log_alpha, S *log_beta,
                             int64_t batch_size, int64_t time_series, int64_t num_labels, int64_t max_target_length,
                             bool zero_infinity, T blank, dim3 log_probs_shape, dim3 log_alpha_shape, S *grad,
                             uint32_t device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_CTCLOSS_V2_IMPL_CUH_
