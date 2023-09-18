/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_MEMORY_MANAGER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_MEMORY_MANAGER_H_

#include <string>
#include <memory>
#include <vector>

namespace mindspore {

class GeMemoryManager {
 public:
  GeMemoryManager();
  ~GeMemoryManager();

  GeMemoryManager(const GeMemoryManager &) = delete;
  GeMemoryManager &operator=(const GeMemoryManager &) = delete;

  uint8_t *MallocDeviceMemory(const std::string &purpose, size_t size);
  uint8_t *MallocHostMemory(const std::string &purpose, size_t size);

  bool MemcpyHost2Device(void *dst_addr, size_t dst_max_size, const void *src_addr, size_t src_size);
  bool MemcpyDevice2Host(void *dst_addr, size_t dst_max_size, const void *src_addr, size_t src_size);

  void FreeDeviceMemory(void *mem);
  void FreeHostMemory(void *mem);

  void FreeAllMemory();

 private:
  struct GeMemoryInfo {
    std::string purpose;
    void *malloc_addr = nullptr;
    void *use_addr = nullptr;
    size_t malloc_size = 0;
    size_t use_size = 0;
  };
  std::vector<GeMemoryInfo> device_memories_;
  std::vector<GeMemoryInfo> host_memories_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_ASCEND_GE_GE_MEMORY_MANAGER_H_
