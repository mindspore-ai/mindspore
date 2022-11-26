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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_COMMON_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_COMMON_H_

#include <cuda/std/atomic>
#include "runtime/device/hash_table.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace gpu {
#define CHECK_CUDA_RET(expression, message)                                                \
  do {                                                                                     \
    cudaError_t cuda_ret = (expression);                                                   \
    if (cuda_ret != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << cuda_ret << " " \
                    << cudaGetErrorString(cuda_ret);                                       \
    }                                                                                      \
  } while (false)

#define CHECK_CUDA_RET_WITH_RETURN_FALSE(expression, message)                              \
  do {                                                                                     \
    cudaError_t cuda_ret = (expression);                                                   \
    if (cuda_ret != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << cuda_ret << " " \
                    << cudaGetErrorString(cuda_ret);                                       \
      return false;                                                                        \
    }                                                                                      \
  } while (false)

#define CHECK_CUDA_RET_WITH_EXCEPTION(expression, message)                                     \
  do {                                                                                         \
    cudaError_t cuda_ret = (expression);                                                       \
    if (cuda_ret != cudaSuccess) {                                                             \
      MS_LOG(EXCEPTION) << "CUDA Error: " << message << " | Error Number: " << cuda_ret << " " \
                        << cudaGetErrorString(cuda_ret);                                       \
    }                                                                                          \
  } while (false)

using Status = HashTableElementStatus;

using CudaAtomicSize = cuda::atomic<std::size_t, cuda::thread_scope_device>;
using CudaAtomicInt = cuda::atomic<int32_t, cuda::thread_scope_device>;

// The empty key, empty value(index) and erased key of CucoDynamicMap.
constexpr static int kEmptyKey = -1;
constexpr static int kEmptyValue = -1;
constexpr static int kErasedKey = -2;

constexpr static uint64_t kMinPermitThreshold = 1;
constexpr static uint64_t kMaxEvictThreshold = INT64_MAX;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_HAL_DEVICE_GPU_HASH_TABLE_COMMON_H_
