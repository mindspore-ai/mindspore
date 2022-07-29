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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BROADCAST_IMPL_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BROADCAST_IMPL_CUH_
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

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
  BROADCAST_TYPE_DIVNONAN = 12,
  BROADCAST_TYPE_EQUAL = 13,
  BROADCAST_TYPE_SQUARED_DIFFERENCE = 14,
  BROADCAST_TYPE_MOD = 15,
  BROADCAST_TYPE_FLOORMOD = 16,
  BROADCAST_TYPE_ATAN2 = 17,
  BROADCAST_TYPE_GREATER_EQUAL = 18,
  BROADCAST_TYPE_LESS_EQUAL = 19,
  BROADCAST_TYPE_NOT_EQUAL = 20,
  BROADCAST_TYPE_LOGICAL_AND = 21,
  BROADCAST_TYPE_LOGICAL_OR = 22,
  BROADCAST_TYPE_TRUNCATEDIV = 23,
  BROADCAST_TYPE_TRUNCATEMOD = 24,
  BROADCAST_TYPE_COMPLEX = 25,
  BROADCAST_TYPE_XDIVY = 26,
  BROADCAST_TYPE_BITWISEAND = 27,
  BROADCAST_TYPE_BITWISEOR = 28,
  BROADCAST_TYPE_BITWISEXOR = 29,
  BROADCAST_TYPE_MULNONAN = 30,
  BROADCAST_TYPE_INVALID = 0xffffffff,
  BROADCAST_TYPE_XLOGY = 31,
};

template <typename T>
CUDA_LIB_EXPORT void ElewiseCmp(const int &nums, enum BroadcastOpType op, const T *x0, const T *x1, bool *y,
                                cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT void ElewiseArith(const int &nums, enum BroadcastOpType op, const T *x0, const T *x1, T *y,
                                  cudaStream_t stream);

template <typename T1, typename T2, typename T3>
CUDA_LIB_EXPORT void ElewiseComplexArith(const int &nums, enum BroadcastOpType op, const T1 *x0, const T2 *x1,
                                         Complex<T3> *y, cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT void BroadcastCmp(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                  const std::vector<size_t> &y_dims, enum BroadcastOpType op, const T *x0, const T *x1,
                                  bool *y, cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT void BroadcastArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                    const std::vector<size_t> &y_dims, enum BroadcastOpType op, const T *x0,
                                    const T *x1, T *y, cudaStream_t stream);

template <typename T1, typename T2, typename T3>
CUDA_LIB_EXPORT void BroadcastComplexArith(const std::vector<size_t> &x0_dims, const std::vector<size_t> &x1_dims,
                                           const std::vector<size_t> &y_dims, enum BroadcastOpType op, const T1 *x0,
                                           const T2 *x1, Complex<T3> *y, cudaStream_t stream);

template <typename T>
CUDA_LIB_EXPORT void BroadcastTo(const size_t &i0, const size_t &i1, const size_t &i2, const size_t &i3,
                                 const size_t &i4, const size_t &i5, const size_t &i6, const size_t &i7,
                                 const size_t &o0, const size_t &o1, const size_t &o2, const size_t &o3,
                                 const size_t &o4, const size_t &o5, const size_t &o6, const size_t &o7,
                                 const T *input_addr, T *output_addr, cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BROADCAST_IMPL_CUH_
