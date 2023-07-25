/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BROADCAST_GRAD_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BROADCAST_GRAD_IMPL_CUH_

#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_device_info.h"

enum BroadcastGradOpType {
  BROADCAST_GRAD_TYPE_MAXIMUM = 0,
  BROADCAST_GRAD_TYPE_MINIMUM = 1,
  BROADCAST_GRAD_TYPE_INVALID = 0xffffffff,
};

template <typename T>
CUDA_LIB_EXPORT cudaError_t BroadcastGrad(const std::vector<size_t> &x_shape, const std::vector<size_t> &x2_shape,
                                          const std::vector<size_t> &dy_shape, const size_t &nums, const bool &grad_x1,
                                          const bool &grad_x2, BroadcastGradOpType op, const T *x1, const T *x2,
                                          const T *dy, T *dx1, T *dx2, const uint32_t &device_id, cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT cudaError_t NoBroadcastGrad(const size_t &nums, const bool &grad_x1, const bool &grad_x2,
                                            BroadcastGradOpType op, const T *x1, const T *x2, const T *dy, T *dx1,
                                            T *dx2, const uint32_t &device_id, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BROADCAST_GRAD_IMPL_CUH_
