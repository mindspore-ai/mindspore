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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRIPLET_MARGIN_LOSS_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRIPLET_MARGIN_LOSS_IMPL_CUH_
#include <string>
#include <algorithm>
#include "include/cuda_fp16.h"
#include "mindapi/base/types.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
template <typename T>
using Complex = mindspore::utils::Complex<T>;

template <typename T, typename S>
CUDA_LIB_EXPORT cudaError_t CalTripletMarginLoss(
  const T *anchor, const T *positive, const T *negative, T *anchor_broadcast, T *positive_broadcast,
  T *negative_broadcast, S *output, float *tem_output, const int64_t *tensor_shapes, const int64_t *dst_shape,
  const size_t outer_size, const size_t inner_size, const size_t *bound_list, const size_t bound,
  const size_t shape_size, float *margin, const int64_t p, const float eps, const std::string reduction,
  const bool swap, const bool need_broadcast, const uint32_t &device_id, cudaStream_t cuda_stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_TRIPLET_MARGIN_LOSS_IMPL_CUH_
