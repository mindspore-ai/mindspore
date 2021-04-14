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
#ifndef MINDSPORE_CCSRC_PS_PS_CACHE_PS_CACHE_BASIC_H
#define MINDSPORE_CCSRC_PS_PS_CACHE_PS_CACHE_BASIC_H

#include <utility>
#include <memory>

namespace mindspore {
namespace ps {
#define RETURN_IF_FALSE(condition) \
  do {                             \
    if (!(condition)) {            \
      return false;                \
    }                              \
  } while (false)

class PsCacheBasic {
 public:
  PsCacheBasic() = default;
  virtual ~PsCacheBasic() = default;
  virtual bool InitDevice(uint32_t device_id, const void *context) = 0;
  virtual void *MallocMemory(size_t size) = 0;
  virtual bool MallocConstantMemory(size_t cache_vocab_size) { return true; }
  virtual bool RecordEvent() = 0;
  virtual bool SynchronizeEvent() = 0;
  virtual bool SynchronizeStream() = 0;
  virtual bool CopyHostMemToDevice(void *dst, const void *src, size_t size) = 0;
  virtual bool CopyDeviceMemToHost(void *dst, const void *src, size_t size) = 0;
  virtual bool HashSwapOut(void *hash_table_addr, void *swap_out_value_addr, void *swap_out_index_addr,
                           size_t cache_vocab_size, size_t embedding_size, size_t swap_out_size) = 0;
  virtual bool HashSwapIn(void *hash_table_addr, void *swap_in_value_addr, void *swap_in_index_addr,
                          size_t cache_vocab_size, size_t embedding_size, size_t swap_in_size) = 0;

 protected:
  void *stream_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_PS_CACHE_PS_CACHE_BASIC_H
