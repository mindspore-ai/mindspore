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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BROADCAST_GRAD_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BROADCAST_GRAD_H_

#include "device/gpu/cuda_common.h"

enum BroadcastGradOpType {
  BROADCAST_GRAD_TYPE_MAXIMUM = 0,
  BROADCAST_GRAD_TYPE_MINIMUM = 1,
  BROADCAST_GRAD_TYPE_INVALID = 0xffffffff,
};

template <typename T>
void BroadcastGrad(const int &l0, const int &l1, const int &l2, const int &l3, const int &r0, const int &r1,
                   const int &r2, const int &r3, const int &d0, const int &d1, const int &d2, const int &d3,
                   enum BroadcastGradOpType op, const T *x1, const T *x2, const T *dy, T *dx1, T *dx2,
                   cudaStream_t stream);

template <typename T>
void NoBroadcastGrad(const int &nums, enum BroadcastGradOpType op, const T *x1, const T *x2, const T *dy, T *dx1,
                     T *dx2, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BROADCAST_GRAD_H_
