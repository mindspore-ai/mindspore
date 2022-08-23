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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_GPU_GPU_CACHE_MEM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_GPU_GPU_CACHE_MEM_H_

#include <cuda_runtime_api.h>
#include <memory>
#include "src/litert/delegate/parameter_cache/cache_mem_base.h"

namespace mindspore {
namespace cache {
namespace gpu {
class GPUCacheMem : public cache::CacheMemBase {
 public:
  GPUCacheMem() = default;
  ~GPUCacheMem() override = default;
  bool InitDevice(uint32_t device_id, const void *context) override;
  void *MallocMemory(size_t size) override;
  void FreeMemory(void *buf) override;
  bool SynchronizeStream() override;
  bool CopyHostMemToDevice(void *dst, const void *src, size_t size) override;
  bool CopyDeviceMemToHost(void *dst, const void *src, size_t size) override;
  bool HashSwapOut(void *hash_table_addr, void *swap_out_value_addr, void *swap_out_index_addr, size_t cache_vocab_size,
                   size_t embedding_size, size_t swap_out_size) override;
  bool HashSwapIn(void *hash_table_addr, void *swap_in_value_addr, void *swap_in_index_addr, size_t cache_vocab_size,
                  size_t embedding_size, size_t swap_in_size) override;

 private:
  cudaStream_t stream_;
  uint32_t device_id_{0};
};
}  // namespace gpu
}  // namespace cache
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_GPU_GPU_CACHE_MEM_H_
