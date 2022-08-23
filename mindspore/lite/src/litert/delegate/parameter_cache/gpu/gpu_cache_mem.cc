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

#include "src/litert/delegate/parameter_cache/gpu/gpu_cache_mem.h"
#include <cuda_runtime.h>
#include <string>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/hash_impl.cuh"
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "src/common/log_adapter.h"
#include "src/litert/delegate/parameter_cache/factory_mgr_base.h"
namespace mindspore {
namespace cache {
namespace gpu {
RET_COMMON_PRODUCT_REGISTRAR(std::string, cache::CacheMemBase, cache::gpu::GPUCacheMem, "gpu", GPUCacheMem);
bool GPUCacheMem::InitDevice(uint32_t device_id, const void *context) {
  device_id_ = device_id;
  auto cuda_ret = cudaSetDevice(static_cast<int>(device_id));
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Failed to set device id " << device_id << ", cuda_ret " << cuda_ret << " "
                  << cudaGetErrorString(cuda_ret);
    return false;
  }
  if (context != nullptr) {
    stream_ = *(reinterpret_cast<const cudaStream_t *>(context));
    return true;
  }

  cuda_ret = cudaStreamCreate(&stream_);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda create stream failed, cuda_ret " << cuda_ret << " " << cudaGetErrorString(cuda_ret);
    return false;
  }

  return true;
}

void *GPUCacheMem::MallocMemory(size_t size) {
  void *device_ptr = nullptr;
  auto cuda_ret = cudaMalloc(&device_ptr, size);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda Malloc failed for size:" << size << ", cuda_ret " << cuda_ret << " "
                  << cudaGetErrorString(cuda_ret);
    return nullptr;
  }
  MS_LOG(DEBUG) << "cudaMalloc size: " << size;
  return device_ptr;
}

void GPUCacheMem::FreeMemory(void *device_addr) {
  auto cuda_ret = cudaFree(device_addr);
  if (cuda_ret != cudaSuccess && cuda_ret != cudaErrorCudartUnloading) {
    MS_LOG(WARNING) << "free cuda memory failed, "
                    << ", cuda_ret " << cuda_ret << " " << cudaGetErrorString(cuda_ret);
  }
}

bool GPUCacheMem::SynchronizeStream() {
  auto cuda_ret = cudaStreamSynchronize(stream_);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda sync stream failed, cuda_ret " << cuda_ret << " " << cudaGetErrorString(cuda_ret);
    return false;
  }

  return true;
}

bool GPUCacheMem::CopyHostMemToDevice(void *dst, const void *src, size_t size) {
  if (dst == nullptr) {
    MS_LOG(ERROR) << "dst is nullptr";
    return false;
  }
  if (src == nullptr) {
    MS_LOG(ERROR) << "src is nullptr";
    return false;
  }

  auto cuda_ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream_);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda memcpy failed, cuda_ret " << cuda_ret << " " << cudaGetErrorString(cuda_ret);
    return false;
  }

  return true;
}

bool GPUCacheMem::CopyDeviceMemToHost(void *dst, const void *src, size_t size) {
  if (dst == nullptr) {
    MS_LOG(ERROR) << "dst is nullptr";
    return false;
  }
  if (src == nullptr) {
    MS_LOG(ERROR) << "src is nullptr";
    return false;
  }

  auto cuda_ret = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream_);
  if (cuda_ret != cudaSuccess) {
    MS_LOG(ERROR) << "Cuda memcpy failed, cuda_ret " << cuda_ret << " " << cudaGetErrorString(cuda_ret);
    return false;
  }

  return true;
}

bool GPUCacheMem::HashSwapOut(void *hash_table_addr, void *swap_out_value_addr, void *swap_out_index_addr, size_t,
                              size_t embedding_size, size_t swap_out_size) {
  if (hash_table_addr == nullptr) {
    MS_LOG(ERROR) << "hash_table_addr is nullptr";
    return false;
  }
  if (swap_out_value_addr == nullptr) {
    MS_LOG(ERROR) << "swap_out_value_addr is nullptr";
    return false;
  }
  if (swap_out_index_addr == nullptr) {
    MS_LOG(ERROR) << "swap_out_index_addr is nullptr";
    return false;
  }

  DoHashSwapOut(reinterpret_cast<float *>(hash_table_addr), reinterpret_cast<float *>(swap_out_value_addr),
                reinterpret_cast<int *>(swap_out_index_addr), swap_out_size, embedding_size, stream_, device_id_);
  return true;
}

bool GPUCacheMem::HashSwapIn(void *hash_table_addr, void *swap_in_value_addr, void *swap_in_index_addr, size_t,
                             size_t embedding_size, size_t swap_in_size) {
  if (hash_table_addr == nullptr) {
    MS_LOG(ERROR) << "hash_table_addr is nullptr";
    return false;
  }
  if (swap_in_value_addr == nullptr) {
    MS_LOG(ERROR) << "swap_in_value_addr is nullptr";
    return false;
  }
  if (swap_in_index_addr == nullptr) {
    MS_LOG(ERROR) << "swap_in_index_addr is nullptr";
    return false;
  }

  DoHashSwapIn(reinterpret_cast<float *>(hash_table_addr), reinterpret_cast<float *>(swap_in_value_addr),
               reinterpret_cast<int *>(swap_in_index_addr), swap_in_size, embedding_size, stream_, device_id_);
  return true;
}
}  // namespace gpu
}  // namespace cache
}  // namespace mindspore
