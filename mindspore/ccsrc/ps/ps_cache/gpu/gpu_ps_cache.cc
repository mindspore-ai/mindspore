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

#include "ps/ps_cache/gpu/gpu_ps_cache.h"
#include "ps/ps_cache/ps_cache_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/hash_impl.cuh"
#include "runtime/device/gpu/gpu_common.h"
#include "runtime/device/gpu/gpu_memory_allocator.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ps {
namespace gpu {
MS_REG_PS_CACHE(kGPUDevice, GPUPsCache);
bool GPUPsCache::InitDevice(uint32_t device_id, const void *) {
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaSetDevice(device_id), "Cuda set device failed")
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaStreamCreate(reinterpret_cast<CUstream_st **>(&stream_)),
                                           "Cuda create stream failed");
  return true;
}

void *GPUPsCache::MallocMemory(size_t size) {
  return device::gpu::GPUMemoryAllocator::GetInstance().AllocTensorMem(size);
}

bool GPUPsCache::RecordEvent() {
  event_.reset(new cudaEvent_t());
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaEventCreate(&(*event_)), "Cuda create event failed");
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaEventRecord(*event_, reinterpret_cast<cudaStream_t>(stream_)),
                                           "Cuda record event failed");
  return true;
}

bool GPUPsCache::SynchronizeEvent() {
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaEventSynchronize(*event_), "Cuda sync event failed");
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaEventDestroy(*event_), "Cuda destroy event failed");
  return true;
}

bool GPUPsCache::SynchronizeStream() {
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
                                           "Cuda sync stream failed");
  return true;
}

bool GPUPsCache::CopyHostMemToDevice(void *dst, const void *src, size_t size) {
  MS_ERROR_IF_NULL(dst);
  MS_ERROR_IF_NULL(src);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_)),
    "Cuda memcpy failed");
  return true;
}

bool GPUPsCache::CopyDeviceMemToHost(void *dst, const void *src, size_t size) {
  MS_ERROR_IF_NULL(dst);
  MS_ERROR_IF_NULL(src);
  CHECK_CUDA_RET_WITH_RETURN_ERROR_NOTRACE(
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_)),
    "Cuda memcpy failed");
  return true;
}

bool GPUPsCache::HashSwapOut(void *hash_table_addr, void *swap_out_value_addr, void *swap_out_index_addr, size_t,
                             size_t embedding_size, size_t swap_out_size) {
  MS_ERROR_IF_NULL(hash_table_addr);
  MS_ERROR_IF_NULL(swap_out_value_addr);
  MS_ERROR_IF_NULL(swap_out_index_addr);
  DoHashSwapOut(reinterpret_cast<float *>(hash_table_addr), reinterpret_cast<float *>(swap_out_value_addr),
                reinterpret_cast<int *>(swap_out_index_addr), swap_out_size, embedding_size,
                reinterpret_cast<cudaStream_t>(stream_));
  return true;
}

bool GPUPsCache::HashSwapIn(void *hash_table_addr, void *swap_in_value_addr, void *swap_in_index_addr, size_t,
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
}  // namespace ps
}  // namespace mindspore
