/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LOSS_WITH_REDUCTION_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LOSS_WITH_REDUCTION_IMPL_CUH_
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

enum class ReductionMode { kNone, kMean, kSum };

static std::map<std::string, ReductionMode> kReductionModeMap{
  {"none", ReductionMode::kNone}, {"mean", ReductionMode::kMean}, {"sum", ReductionMode::kSum}};

template <typename T>
CUDA_LIB_EXPORT void BinaryCrossEntropyLoss(const int &input_size, const ReductionMode &reduction, const T *input_x,
                                            const T *input_y, const T *weight, T *loss, T *tmp_loss,
                                            cudaStream_t stream);
template <typename T>
CUDA_LIB_EXPORT void BinaryCrossEntropyLossGrad(const int &input_size, const ReductionMode &reduction, const T *input_x,
                                                const T *input_y, const T *weight, const T *dloss, T *dx,
                                                cudaStream_t stream);
template <typename T>
CUDA_LIB_EXPORT void KLDivLoss(const int &input_size, const ReductionMode &reduction, const T *input_x,
                               const T *input_y, T *loss, T *tmp_loss, cudaStream_t stream);
template <typename T>
CUDA_LIB_EXPORT void KLDivLossGrad(const int &input_size, const ReductionMode &reduction, const T *input_x,
                                   const T *input_y, const T *dloss, T *dx, cudaStream_t stream);
template <typename T, typename S>
CUDA_LIB_EXPORT void NLLLoss(const int n, const int c, const ReductionMode reduction, const T *input,
                             const int32_t *target, const S *weight, T *loss, S *total_weight, T *tmp_loss,
                             S *tmp_target_weight, cudaStream_t stream);
template <typename T, typename S>
CUDA_LIB_EXPORT void NLLLossGrad(const int n, const int c, const ReductionMode reduction, const T *input,
                                 const int32_t *target, const S *weight, const S *total_weight, const T *dloss,
                                 T *dinput, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_LOSS_WITH_REDUCTION_IMPL_CUH_
