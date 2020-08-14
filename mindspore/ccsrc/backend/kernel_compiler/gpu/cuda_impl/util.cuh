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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UTIL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UTIL_H_

#include <cuda_fp16.h>

inline __device__ float ms_atomic_add(float *address, float val) { return atomicAdd(address, val); }

inline __device__ int ms_atomic_add(int *address, int val) { return atomicAdd(address, val); }

inline __device__ half ms_atomic_add(half *address, half val) {
  unsigned int *aligned =
    reinterpret_cast<unsigned int *>(reinterpret_cast<size_t>(address) - (reinterpret_cast<size_t>(address) & 2));
  unsigned int old = *aligned;
  unsigned int assumed;
  unsigned short old_as_us;  // NOLINT
  do {
    assumed = old;
    old_as_us = static_cast<unsigned short>(reinterpret_cast<size_t>(address) & 2 ? old >> 16 : old & 0xffff);  // NOLINT
    half sum = __float2half_rn(__half2float(__ushort_as_half(old_as_us)) + static_cast<float>(val));
    unsigned short sum_as_us = __half_as_ushort(sum);  // NOLINT
    unsigned int sum_as_ui =
      reinterpret_cast<size_t>(address) & 2 ? (sum_as_us << 16) | (old & 0xffff) : (old & 0xffff0000) | sum_as_us;
    old = atomicCAS(aligned, assumed, sum_as_ui);
  } while (assumed != old);
  __half_raw raw = {old_as_us};
  return half(raw);
}

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_UTIL_H_
