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

#include "src/delegate/parameter_cache/gpu/gpu_cache_mem.h"
#include <cuda_runtime.h>
#include "src/delegate/tensorrt/cuda_impl/hash.cuh"
#include "runtime/device/gpu/cuda_driver.h"
#include "src/common/log_adapter.h"

namespace mindspore {
namespace gpu {
#define CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(expression, message)                    \
  do {                                                                                   \
    cudaError_t status = (expression);                                                   \
    if (status != cudaSuccess) {                                                         \
      MS_LOG(ERROR) << "CUDA Error: " << message << " | Error Number: " << status << " " \
                    << cudaGetErrorString(status);                                       \
      return false;                                                                      \
    }                                                                                    \
  } while (0)

#define MS_ERROR_IF_NULL_W_RET_VAL(ptr, val)                     \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return val;                                                \
    }                                                            \
  } while (0)

#define MS_ERROR_IF_NULL(ptr)                                    \
  do {                                                           \
    if ((ptr) == nullptr) {                                      \
      MS_LOG(ERROR) << ": The pointer[" << #ptr << "] is null."; \
      return false;                                              \
    }                                                            \
  } while (0)

bool GPUCacheMem::InitDevice(uint32_t device_id, const void *context) {
  auto ret = cudaSetDevice(static_cast<int>(device_id));
  if (ret != cudaSuccess) {
    MS_LOG(ERROR) << "Failed to set device id:" << device_id;
    return false;
  }
  if (context != nullptr) {
    stream_ = *(reinterpret_cast<const cudaStream_t *>(context));
    return true;
  }
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaStreamCreate(reinterpret_cast<CUstream_st **>(&stream_)),
                                           "Cuda create stream failed");
  return true;
}

void *GPUCacheMem::MallocMemory(size_t size) {
  void *device_ptr = nullptr;
  auto cuda_ret = cudaMalloc(&device_ptr, size);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda Malloc failed for size:" << size;
    return nullptr;
  }
  MS_LOG(INFO) << "cudaMalloc size: " << size;
  return device_ptr;
}

void GPUCacheMem::FreeMemory(void *device_addr) {
  auto cuda_ret = cudaFree(device_addr);
  if (cuda_ret != cudaSuccess && cuda_ret != cudaErrorCudartUnloading) {
    MS_LOG(WARNING) << "free cuda failed for " << cudaGetErrorName(cuda_ret);
  }
}

bool GPUCacheMem::RecordEvent() {
  event_.reset(new cudaEvent_t());
  MS_ERROR_IF_NULL_W_RET_VAL(event_, false);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaEventCreate(&(*event_)), "Cuda create event failed");
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaEventRecord(*event_, reinterpret_cast<cudaStream_t>(stream_)),
                                           "Cuda record event failed");
  return true;
}

bool GPUCacheMem::SynchronizeEvent() {
  MS_ERROR_IF_NULL_W_RET_VAL(event_, false);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaEventSynchronize(*event_), "Cuda sync event failed");
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaEventDestroy(*event_), "Cuda destroy event failed");
  return true;
}

bool GPUCacheMem::SynchronizeStream() {
  MS_ERROR_IF_NULL_W_RET_VAL(stream_, false);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
                                           "Cuda sync stream failed");
  return true;
}

bool GPUCacheMem::CopyHostMemToDevice(void *dst, const void *src, size_t size) {
  MS_ERROR_IF_NULL(dst);
  MS_ERROR_IF_NULL(src);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_)),
    "Cuda memcpy failed");
  return true;
}

bool GPUCacheMem::CopyDeviceMemToHost(void *dst, const void *src, size_t size) {
  MS_ERROR_IF_NULL(dst);
  MS_ERROR_IF_NULL(src);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_)),
    "Cuda memcpy failed");
  return true;
}

bool GPUCacheMem::HashSwapOut(void *hash_table_addr, void *swap_out_value_addr, void *swap_out_index_addr, size_t,
                              size_t embedding_size, size_t swap_out_size) {
  MS_ERROR_IF_NULL(hash_table_addr);
  MS_ERROR_IF_NULL(swap_out_value_addr);
  MS_ERROR_IF_NULL(swap_out_index_addr);
  DoHashSwapOut(reinterpret_cast<float *>(hash_table_addr), reinterpret_cast<float *>(swap_out_value_addr),
                reinterpret_cast<int *>(swap_out_index_addr), swap_out_size, embedding_size,
                reinterpret_cast<cudaStream_t>(stream_));
  return true;
}

bool GPUCacheMem::HashSwapIn(void *hash_table_addr, void *swap_in_value_addr, void *swap_in_index_addr, size_t,
                             size_t embedding_size, size_t swap_in_size) {
  MS_ERROR_IF_NULL(hash_table_addr);
  MS_ERROR_IF_NULL(swap_in_value_addr);
  MS_ERROR_IF_NULL(swap_in_index_addr);
  DoHashSwapIn(reinterpret_cast<float *>(hash_table_addr), reinterpret_cast<float *>(swap_in_value_addr),
               reinterpret_cast<int *>(swap_in_index_addr), swap_in_size, embedding_size,
               reinterpret_cast<cudaStream_t>(stream_));
  return true;
}
}  // namespace gpu
}  // namespace mindspore
