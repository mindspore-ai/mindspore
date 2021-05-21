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
#include <iostream>

#include "backend/kernel_compiler/gpu/cuda_impl/logical_not_impl.cuh"
#include "runtime/device/gpu/cuda_common.h"

template <typename T>
struct LogicalNotFunc {
  __device__ __host__ __forceinline__ bool operator()(const T &x) { return !x; }
};

template <typename T, typename Func>
__global__ void LogicalNotKernel(const int nums, const T *x, bool *y) {
  for (size_t pos = blockIdx.x * blockDim.x + threadIdx.x; pos < nums; pos += blockDim.x * gridDim.x) {
    y[pos] = Func()(x[pos]);
  }
}

template <typename T>
void LogicalNotImpl(const int &nums, const T *x, bool *y, cudaStream_t stream) {
  return LogicalNotKernel<T, LogicalNotFunc<T>><<<(nums + 255) / 256, 256, 0, stream>>>(nums, x, y);
}

template void LogicalNotImpl(const int &nums, const bool *x, bool *y, cudaStream_t stream);
