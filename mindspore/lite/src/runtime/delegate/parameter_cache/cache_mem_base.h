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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_CACHE_MEM_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_CACHE_MEM_BASE_H_
#include <utility>
#include <memory>

namespace mindspore {
namespace cache {
class CacheMemBase {
 public:
  CacheMemBase() = default;
  virtual ~CacheMemBase() = default;
  virtual bool InitDevice(uint32_t device_id, const void *context) = 0;
  virtual void *MallocMemory(size_t size) = 0;
  virtual void FreeMemory(void *buf) = 0;
  virtual bool SynchronizeStream() = 0;
  virtual bool CopyHostMemToDevice(void *dst, const void *src, size_t size) = 0;
  virtual bool CopyDeviceMemToHost(void *dst, const void *src, size_t size) = 0;
  virtual bool HashSwapOut(void *hash_table_addr, void *swap_out_value_addr, void *swap_out_index_addr,
                           size_t cache_vocab_size, size_t embedding_size, size_t swap_out_size) = 0;
  virtual bool HashSwapIn(void *hash_table_addr, void *swap_in_value_addr, void *swap_in_index_addr,
                          size_t cache_vocab_size, size_t embedding_size, size_t swap_in_size) = 0;
};
}  // namespace cache
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_PARAMETER_CACHE_CACHE_MEM_BASE_H_
