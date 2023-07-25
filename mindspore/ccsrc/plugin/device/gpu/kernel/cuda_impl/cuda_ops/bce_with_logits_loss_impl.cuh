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

#include <vector>
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BCE_WITH_LOGITS_LOSS_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BCE_WITH_LOGITS_LOSS_IMPL_CUH_
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#define MAX_LOGITS_DIMENSION 8

template <typename T>
CUDA_LIB_EXPORT cudaError_t CalBCEWithLogitsLoss(const size_t input_size, const T *predict, const T *target,
                                                 const std::vector<int64_t> &input_shape, const size_t shape_size,
                                                 const T *weight, const std::vector<int64_t> &weight_shape,
                                                 const bool weight_need_broadcast, const T *pos_weight,
                                                 const std::vector<int64_t> &pos_weight_shape,
                                                 const bool pos_weight_need_broadcast, T *shape_broadcasted, T *output,
                                                 cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BCE_WITH_LOGITS_LOSS_IMPL_CUH_
