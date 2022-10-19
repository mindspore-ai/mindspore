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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_NN_MUTILABEL_MARGIN_LOSS_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_NN_MUTILABEL_MARGIN_LOSS_GRAD_IMPL_CUH_

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

#ifdef __cplusplus
extern "C" {
#endif
CUDA_LIB_EXPORT void CalMultilabelMarginLossGradFp16(const half *input_grad, const half *input, const int *target,
                                                     int *is_target, const int batch_size, const int class_num,
                                                     int64_t reduction, half *output_grad, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);

CUDA_LIB_EXPORT void CalMultilabelMarginLossGradFp32(const float *input_grad, const float *input, const int *target,
                                                     int *is_target, const int batch_size, const int class_num,
                                                     int64_t reduction, float *output_grad, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);

CUDA_LIB_EXPORT void CalMultilabelMarginLossGradFp64(const double *input_grad, const double *input, const int *target,
                                                     int *is_target, const int batch_size, const int class_num,
                                                     int64_t reduction, double *output_grad, const uint32_t &device_id,
                                                     cudaStream_t cuda_stream);
#ifdef __cplusplus
}
#endif

template <typename T>
CUDA_LIB_EXPORT void CalMultilabelMarginLossGrad(const T *input_grad, const T *input, const int *target, int *is_target,
                                                 const int batch_size, const int class_num, int64_t reduction,
                                                 T *output_grad, const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_NN_MUTILABEL_MARGIN_LOSS_GRAD_IMPL_CUH_
