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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BROADCAST_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BROADCAST_H_

#include <vector>
#include "runtime/device/gpu/cuda_common.h"

enum BroadcastOpType {
  BROADCAST_TYPE_GREATER = 0,
  BROADCAST_TYPE_LESS = 1,
  BROADCAST_TYPE_MAXIMUM = 2,
  BROADCAST_TYPE_MINIMUM = 3,
  BROADCAST_TYPE_POWER = 4,
  BROADCAST_TYPE_REALDIV = 5,
  BROADCAST_TYPE_MUL = 6,
  BROADCAST_TYPE_SUB = 7,
  BROADCAST_TYPE_ADD = 8,
  BROADCAST_TYPE_FLOORDIV = 9,
  BROADCAST_TYPE_ABSGRAD = 10,
  BROADCAST_TYPE_DIV = 11,
  BROADCAST_TYPE_INVALID = 0xffffffff,
};

template <typename T, typename S>
void Broadcast(const std::vector<int> &lhs_shape, const std::vector<int> &rhs_shape,
               const std::vector<int> &output_shape, enum BroadcastOpType op, const T *input0, const T *input1,
               S *output, cudaStream_t stream);

template <typename T, typename S>
void NoBroadcast(const int &size, enum BroadcastOpType op, const T *input0, const T *input1, S *output,
                 cudaStream_t stream);

template <typename T>
void BroadcastTo(const int &i0, const int &i1, const int &i2, const int &i3, const int &o0, const int &o1,
                 const int &o2, const int &o3, const T *input_addr, T *output_addr, cudaStream_t stream);

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_BROADCAST_H_
