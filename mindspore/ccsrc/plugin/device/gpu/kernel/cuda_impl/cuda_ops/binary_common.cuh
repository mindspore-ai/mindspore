/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_COMMON_CUH_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_COMMON_CUH_

#include <limits.h>
#include <cmath>
#include <type_traits>
#include "include/cuda_fp16.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/binary_types.cuh"
template <typename T>
__device__ __host__ __forceinline__ T Eps();

template <typename T>
__device__ __host__ __forceinline__ T Eps() {
  return 0;
}
template <>
__device__ __host__ __forceinline__ float Eps() {
  return 2e-7;
}
template <>
__device__ __host__ __forceinline__ double Eps() {
  return 2e-15;
}
template <>
__device__ __host__ __forceinline__ half Eps() {
  return 6.105e-5;
}
template <enum BinaryOpType op, typename In0_t, typename In1_t, typename Out_t, typename Enabled = std::true_type>
struct BinaryFunc {
  __device__ __host__ __forceinline__ BinaryFunc() {}
  __device__ __forceinline__ Out_t operator()(In0_t val0, In1_t val1) const { return Out_t(0.0); }
};

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_OPS_BINARY_COMMON_CUH_
