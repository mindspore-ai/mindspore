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

#include <cuda_runtime.h>
#include <curand_kernel.h>

#define FINAL_MASK 0xffffffff

template <typename T>
__device__ T warpedReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  }
  return val;
}

template <typename T>
__device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int warped = threadIdx.x & 0x1f;
  val = warpedReduceSum<T>(val);
  if (warped == 0) shared[threadIdx.x >> 5] = val;
  __syncthreads();
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[warped] : static_cast<T>(0.0);
  val = warpedReduceSum<T>(val);
  return val;
}
