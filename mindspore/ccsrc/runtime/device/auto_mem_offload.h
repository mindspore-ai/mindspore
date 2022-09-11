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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_AUTO_MEM_OFFLOAD_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_AUTO_MEM_OFFLOAD_H_

#include <utility>
#include <queue>
#include <map>
#include <vector>
#include <memory>

#include "runtime/device/memory_manager.h"
#include "utils/hash_map.h"
#include "utils/hash_set.h"

namespace mindspore {
namespace device {
class MemHandler {
 public:
  explicit MemHandler(std::shared_ptr<MemoryManager> memory_manager) : memory_manager_(std::move(memory_manager)) {}
  ~MemHandler() = default;
  size_t GetAvailableMemSize() { return memory_manager_->GetAvailableMemSize(); }
  void *MallocDevice(size_t mem_size) { return memory_manager_->MallocMemFromMemPool(mem_size, false); }
  void FreeDevice(void *ptr) { memory_manager_->FreeMemFromMemPool(ptr); }
  void *MallocHost(size_t mem_size);
  void FreeHost(void *ptr);
  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) {
    memory_manager_->SwapIn(host_ptr, device_ptr, mem_size, stream);
  }
  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) {
    memory_manager_->SwapOut(device_ptr, host_ptr, mem_size, stream);
  }
  std::vector<void *> MallocContinuousMemFromMemPool(const std::vector<size_t> &size_list) {
    return memory_manager_->MallocContinuousMemFromMemPool(size_list);
  }

 private:
  std::shared_ptr<MemoryManager> memory_manager_;
  std::map<size_t, std::queue<void *>> cached_host_mem_;
  std::map<void *, std::shared_ptr<std::vector<uint8_t>>> host_mem_block_map_;
};

class BACKEND_EXPORT AutoMemoryOffload {
 public:
  explicit AutoMemoryOffload(std::shared_ptr<MemHandler> mem_handler) : mem_handler_(std::move(mem_handler)) {}
  ~AutoMemoryOffload() = default;
  void *Get(const void *key, void *stream = nullptr, const HashSet<const void *> &not_offload = {});
  void *Malloc(const void *key, size_t mem_size, void *stream, const HashSet<const void *> &not_offload);
  bool MallocContinuous(const std::vector<const void *> &keys, const std::vector<size_t> &size_list, void *stream,
                        const HashSet<const void *> &pinned_memory);
  void Free(const void *key);
  void Clear();
  void SetInitHostPtr(const void *key, void *host_ptr, size_t mem_size);
  void UpdateHighPriorityMem(const void *key);

  void SwapOut(const void *key, void *stream);
  // Return the device ptr where the data is copied to
  void *SwapIn(const void *key, void *stream);

 private:
  size_t GetMemSize(const void *key);
  void GetHostPtr(const void *key, void **host_ptr, bool *from_init);
  void GetOrMallocHostPtr(const void *key, size_t mem_size, void **host_ptr, bool *from_init);
  template <typename MallocInfo>
  bool TryAllocMemory(
    const MallocInfo &info, size_t total_size, void *stream, const HashSet<const void *> &pinned_memory,
    const std::function<bool(const MallocInfo &, const std::shared_ptr<MemHandler> &, HashMap<const void *, void *> *,
                             HashMap<const void *, size_t> *)> &alloc_func);
  std::shared_ptr<MemHandler> mem_handler_;
  HashMap<const void *, void *> mem_result_;
  HashMap<const void *, size_t> mem_size_;
  HashSet<const void *> init_from_host_keys_;
  HashSet<const void *> updated_device_mem_;
  HashSet<const void *> continuous_mem_key_;
  HashMap<const void *, void *> init_host_ptr_;
  HashMap<const void *, void *> swap_host_ptr_;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_AUTO_MEM_OFFLOAD_H_
