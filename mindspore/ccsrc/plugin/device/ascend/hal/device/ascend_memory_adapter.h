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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_ADAPTER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_ADAPTER_H_

#include <mutex>
#include <string>
#include <memory>
#include <vector>
#include "utils/ms_context.h"
#include "ir/anf.h"

namespace mindspore {
namespace device {
namespace ascend {
class AscendMemAdapter {
 public:
  static AscendMemAdapter &GetInstance() {
    static AscendMemAdapter instance{};
    return instance;
  }

  bool Initialize();
  bool DeInitialize();

  uint8_t *MallocStaticDevMem(size_t size, const std::string &tag = "");
  uint8_t *MallocDynamicDevMem(size_t size, const std::string &tag = "");
  uint8_t *MallocOverflowMem();
  bool FreeStaticDevMem(void *) const { return true; }
  void ResetDynamicMemory();

  static size_t GetRoundUpAlignSize(size_t input_size);

  [[nodiscard]] uint64_t FreeDevMemSize() const { return static_mem_offset_ - max_dynamic_mem_offset_; }
  [[nodiscard]] uint64_t MaxHbmSizeForMs() const { return max_available_ms_hbm_size_; }
  [[nodiscard]] uint64_t GetMsUsedHbmSize() const { return ms_used_hbm_size_; }
  std::string DevMemStatistics() const;
  std::string DevMemDetailInfo() const;

 private:
  AscendMemAdapter() = default;
  struct MemoryBlock {
    MemoryBlock(void *ptr, const size_t size, const std::string &tag) {
      mem_ptr = ptr;
      mem_size = size;
      mem_tag = tag;
    }

    void *mem_ptr{nullptr};
    size_t mem_size{0};
    std::string mem_tag;
  };

  uint8_t *MallocFromRts(size_t size) const;
  bool FreeToRts(void *devPtr) const;
  size_t GetDeviceMemSizeFromContext() const;

  bool initialized_{false};

  // Support multi-thread.
  std::mutex mutex_;

  // Support overflow case.
  std::mutex overflow_mutex_;

  // rts Memory INFO
  size_t device_hbm_total_size_{0};
  size_t device_hbm_free_size_{0};
  size_t max_available_ms_hbm_size_{0};
  uint8_t *device_mem_base_addr_{nullptr};
  uint64_t ms_used_hbm_size_{0};

  // dynamic memory info, from a low address to a high address
  uint64_t cur_dynamic_mem_offset_{0};
  // Maximum dynamic memory have already allocated, dynamically updated
  uint64_t max_dynamic_mem_offset_{0};
  std::vector<std::shared_ptr<MemoryBlock>> dynamic_memory_block_list_;

  // static memory info, from a high address to a low address
  uint64_t static_mem_offset_{0};
  std::vector<std::shared_ptr<MemoryBlock>> static_memory_block_list_;
  static size_t GetRoundDownAlignSize(size_t input_size);

  // overflow memory info, key is kernel, val is memory ptr
  mindspore::HashMap<std::string, uint8_t *> overflow_memory_info_map_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_ADAPTER_H_
