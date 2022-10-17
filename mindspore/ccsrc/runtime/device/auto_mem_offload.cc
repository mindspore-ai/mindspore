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

#include "runtime/device/auto_mem_offload.h"
#include <memory>
#include <vector>
#include <queue>
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_offload_strategy.h"

namespace mindspore {
namespace device {
void *OffloadedMemPool::MallocHost(size_t mem_size) {
  auto &mem_que = cached_host_mem_[mem_size];
  if (!mem_que.empty()) {
    auto ret = mem_que.front();
    mem_que.pop();
    return ret;
  }
  auto block = std::make_shared<std::vector<uint8_t>>();
  try {
    block->resize(mem_size, 0);
    auto ptr = block->data();
    host_mem_block_map_[ptr] = block;
    return ptr;
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Malloc memory failed: size " << mem_size;
  }
}

void OffloadedMemPool::FreeHost(void *ptr) {
  MS_EXCEPTION_IF_NULL(ptr);
  auto iter = host_mem_block_map_.find(ptr);
  if (iter == host_mem_block_map_.end()) {
    MS_LOG(DEBUG) << "Free ptr not be created from here, abort";
    return;
  }
  MS_EXCEPTION_IF_NULL(iter->second);
  auto mem_size = iter->second->size();
  (void)cached_host_mem_[mem_size].emplace(iter->first);
}

void AutoMemoryOffload::SetInitHostPtr(const void *key, void *host_ptr, size_t mem_size) {
  (void)init_from_host_keys_.insert(key);
  init_host_ptr_[key] = host_ptr;
  mem_size_[key] = mem_size;
}

void AutoMemoryOffload::Free(const void *key) {
  const auto &iter = mem_result_.find(key);
  if (iter == mem_result_.end()) {
    return;
  }
  auto ptr = iter->second;
  MS_EXCEPTION_IF_NULL(mem_handler_);
  mem_handler_->FreeDevice(ptr);
  (void)mem_result_.erase(key);
}

void *AutoMemoryOffload::Get(const void *key, void *stream, const HashSet<const void *> &pinned_memory) {
  auto iter = mem_result_.find(key);
  if (iter != mem_result_.end()) {
    return iter->second;
  }
  if (stream == nullptr) {
    return nullptr;
  }
  void *host_ptr = nullptr;
  bool from_init = false;
  GetHostPtr(key, &host_ptr, &from_init);
  if (host_ptr == nullptr) {
    return nullptr;
  }
  const auto mem_size = GetMemSize(key);
  auto device_ptr = Malloc(key, mem_size, stream, pinned_memory);
  if (device_ptr == nullptr) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(mem_handler_);
  mem_handler_->SwapIn(host_ptr, device_ptr, mem_size, stream);
  if (!from_init) {
    (void)swap_host_ptr_.erase(key);
    mem_handler_->FreeHost(host_ptr);
  }
  mem_result_[key] = device_ptr;
  return device_ptr;
}

bool AutoMemoryOffload::MallocContinuous(const std::vector<const void *> &keys, const std::vector<size_t> &size_list,
                                         void *stream, const HashSet<const void *> &pinned_memory) {
  MS_EXCEPTION_IF_NULL(mem_handler_);
  const size_t total_size = std::accumulate(size_list.begin(), size_list.end(), static_cast<size_t>(0));
  using MallocInfo = std::pair<const std::vector<const void *> &, const std::vector<size_t> &>;
  std::function<bool(const MallocInfo &, const std::shared_ptr<MemHandler> &mem_handler,
                     HashMap<const void *, void *> *, HashMap<const void *, size_t> *)>
    malloc_func = [](const MallocInfo &info, const std::shared_ptr<MemHandler> &mem_handler,
                     HashMap<const void *, void *> *mem_result, HashMap<const void *, size_t> *mem_size) {
      const auto keys = info.first;
      const auto size_list = info.second;
      auto device_ptr = mem_handler->MallocContinuousMemFromMemPool(size_list);
      if (device_ptr.size() != keys.size()) {
        return false;
      }
      for (size_t i = 0; i < device_ptr.size(); i += 1) {
        (*mem_result)[keys[i]] = device_ptr[i];
        (*mem_size)[keys[i]] = size_list[i];
      }
      return true;
    };
  if (!TryAllocMemory<MallocInfo>(std::make_pair(keys, size_list), total_size, stream, pinned_memory, malloc_func)) {
    return false;
  }
  for (auto key : keys) {
    (void)continuous_mem_key_.insert(key);
  }
  return true;
}

void *AutoMemoryOffload::Malloc(const void *key, size_t mem_size, void *stream,
                                const HashSet<const void *> &pinned_memory) {
  auto iter = mem_result_.find(key);
  if (iter != mem_result_.end()) {
    return iter->second;
  }

  using MallocInfo = std::pair<const void *, size_t>;
  std::function<bool(const MallocInfo &, const std::shared_ptr<MemHandler> &mem_handler,
                     HashMap<const void *, void *> *, HashMap<const void *, size_t> *)>
    malloc_func = [](const MallocInfo &info, const std::shared_ptr<MemHandler> &mem_handler,
                     HashMap<const void *, void *> *mem_result, HashMap<const void *, size_t> *mem_size) {
      MS_EXCEPTION_IF_NULL(mem_handler);
      const auto key = info.first;
      const auto size = info.second;
      auto device_ptr = mem_handler->MallocDevice(size);
      if (device_ptr == nullptr) {
        return false;
      }
      (*mem_result)[key] = device_ptr;
      (*mem_size)[key] = size;
      return true;
    };
  return TryAllocMemory<MallocInfo>(std::make_pair(key, mem_size), mem_size, stream, pinned_memory, malloc_func)
           ? mem_result_[key]
           : nullptr;
}

template <typename MallocInfo>
bool AutoMemoryOffload::TryAllocMemory(
  const MallocInfo &info, size_t total_size, void *stream, const HashSet<const void *> &pinned_memory,
  const std::function<bool(const MallocInfo &, const std::shared_ptr<MemHandler> &, HashMap<const void *, void *> *,
                           HashMap<const void *, size_t> *)> &alloc_func) {
  if (alloc_func(info, mem_handler_, &mem_result_, &mem_size_)) {
    return true;
  }
  if (stream == nullptr) {
    return false;
  }
  using KeySizePair = std::pair<const void *, size_t>;
  auto less = [](const KeySizePair &a, const KeySizePair &b) -> bool { return a.second < b.second; };
  std::priority_queue<KeySizePair, std::vector<KeySizePair>, decltype(less)> mem_can_offload(less);
  for (const auto &i : mem_result_) {
    const auto offload_key = i.first;
    if (pinned_memory.count(offload_key) != 0) {
      continue;
    }
    const auto device_mem_size = GetMemSize(offload_key);
    if (device_mem_size >= total_size) {
      SwapOut(offload_key, stream);
      Free(offload_key);
      if (alloc_func(info, mem_handler_, &mem_result_, &mem_size_)) {
        return true;
      }
    }
    mem_can_offload.push({offload_key, device_mem_size});
  }
  while (!mem_can_offload.empty()) {
    const auto &max_mem_in_device = mem_can_offload.top();
    const auto offload_mem_key = max_mem_in_device.first;
    auto offload_device_ptr = mem_result_[offload_mem_key];
    MS_EXCEPTION_IF_NULL(offload_device_ptr);
    SwapOut(offload_mem_key, stream);
    Free(offload_mem_key);
    if (alloc_func(info, mem_handler_, &mem_result_, &mem_size_)) {
      return true;
    }
    mem_can_offload.pop();
  }
  return false;
}

void AutoMemoryOffload::SwapOut(const void *key, void *stream) {
  const auto iter = mem_result_.find(key);
  void *host_ptr = nullptr;
  bool from_init = false;
  const auto mem_size = GetMemSize(key);
  if (iter == mem_result_.end()) {
    GetHostPtr(key, &host_ptr, &from_init);
    if (host_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Can not find device ptr for key " << key;
    }
    return;
  }
  const auto device_ptr = iter->second;
  GetOrMallocHostPtr(key, mem_size, &host_ptr, &from_init);
  MS_EXCEPTION_IF_NULL(host_ptr);
  auto updated_iter = from_init ? updated_device_mem_.find(key) : updated_device_mem_.end();
  if (!from_init || updated_iter != updated_device_mem_.end()) {
    mem_handler_->SwapOut(device_ptr, host_ptr, mem_size, stream);
    if (updated_iter != updated_device_mem_.end()) {
      (void)updated_device_mem_.erase(updated_iter);
    }
  }
}

void *AutoMemoryOffload::SwapIn(const void *key, void *stream) {
  MS_EXCEPTION_IF_NULL(mem_handler_);
  const size_t mem_size = GetMemSize(key);
  const auto &iter = mem_result_.find(key);
  if (iter == mem_result_.end()) {
    MS_LOG(EXCEPTION) << "Can not find device ptr for key " << key;
  }
  bool from_init = true;
  void *host_ptr = nullptr;
  GetHostPtr(key, &host_ptr, &from_init);
  MS_EXCEPTION_IF_NULL(host_ptr);
  mem_handler_->SwapIn(host_ptr, iter->second, mem_size, stream);
  if (!from_init) {
    mem_handler_->FreeHost(host_ptr);
    (void)swap_host_ptr_.erase(key);
  }
  return iter->second;
}

size_t AutoMemoryOffload::GetMemSize(const void *key) {
  const auto &iter = mem_size_.find(key);
  if (iter == mem_size_.end()) {
    MS_LOG(EXCEPTION) << "Can not find memory size for key " << key;
  }
  return iter->second;
}

void AutoMemoryOffload::GetOrMallocHostPtr(const void *key, size_t mem_size, void **host_ptr, bool *from_init) {
  MS_EXCEPTION_IF_NULL(host_ptr);
  MS_EXCEPTION_IF_NULL(mem_handler_);
  GetHostPtr(key, host_ptr, from_init);
  if (*host_ptr != nullptr) {
    return;
  }
  *host_ptr = mem_handler_->MallocHost(mem_size);
  *from_init = false;
  swap_host_ptr_[key] = *host_ptr;
}

void AutoMemoryOffload::GetHostPtr(const void *key, void **host_ptr, bool *from_init) {
  *from_init = init_from_host_keys_.count(key) != 0;
  if (*from_init) {
    const auto iter = init_host_ptr_.find(key);
    if (iter == init_host_ptr_.end()) {
      MS_LOG(EXCEPTION) << "Can not find host ptr for key " << key;
    }
    *host_ptr = iter->second;
  } else {
    auto iter = swap_host_ptr_.find(key);
    if (iter != swap_host_ptr_.end()) {
      *host_ptr = iter->second;
    }
  }
}

void AutoMemoryOffload::Clear() {
  if (mem_handler_ == nullptr) {
    return;
  }
  for (auto &item : mem_result_) {
    mem_handler_->FreeDevice(item.second);
  }
  mem_result_.clear();
  for (const auto &item : swap_host_ptr_) {
    const auto host_ptr = item.second;
    if (host_ptr != nullptr) {
      mem_handler_->FreeHost(host_ptr);
    }
  }
  swap_host_ptr_.clear();
  init_host_ptr_.clear();
  init_from_host_keys_.clear();
}

void AutoMemoryOffload::UpdateHighPriorityMem(const void *key) { (void)updated_device_mem_.insert(key); }

bool MindRTAutoOffloadAdapter::Malloc(DeviceAddress *device_address) {
  if (device_address->GetPtr() != nullptr) {
    return true;
  }
  const auto original_size = device_address->GetSize();
  constexpr size_t kAlignBytes = 32;
  const size_t align_size = ((original_size + kMemAlignSize + kAlignBytes - 1) / kMemAlignSize) * kMemAlignSize;
  const auto &pinned_mem = MemoryOffloadConflict::GetInstance().GetConflictMap(device_address);
  const auto &device_ptr = Malloc(align_size, pinned_mem);
  if (device_ptr == nullptr) {
    return false;
  }
  device_address->set_ptr(device_ptr);
  device_address->set_from_mem_pool(true);
  std::unique_lock<std::shared_mutex> unq_lock(all_mem_mutex_);
  all_mem_.insert(device_address);
  return true;
}

void *MindRTAutoOffloadAdapter::Malloc(size_t size, const HashSet<const void *> &pinned_mem) {
  const auto malloc_func = [](size_t size, DynamicMemPoolBestFit *mem_pool, void **device_ptr) {
    *device_ptr = mem_pool->AllocTensorMem(size);
    return *device_ptr != nullptr;
  };
  void *device_ptr = nullptr;
  return TryAllocMemory<size_t, void *>(size, size, pinned_mem, malloc_func, &device_ptr) ? device_ptr : nullptr;
}

std::vector<void *> MindRTAutoOffloadAdapter::MallocContinuousMem(const std::vector<size_t> &size_list) {
  const auto malloc_func = [](const std::vector<size_t> &size_list, DynamicMemPoolBestFit *mem_pool,
                              std::vector<void *> *ptr_list) {
    *ptr_list = std::move(mem_pool->AllocContinuousTensorMem(size_list));
    return !ptr_list->empty();
  };
  size_t total_size = std::accumulate(size_list.cbegin(), size_list.cend(), size_t(0));
  std::vector<void *> ptr_list;
  if (!TryAllocMemory<const std::vector<size_t> &, std::vector<void *>>(size_list, total_size, {}, malloc_func,
                                                                        &ptr_list)) {
    return ptr_list;
  }
  if (ptr_list.size() != size_list.size()) {
    MS_LOG(EXCEPTION) << "Size of ptr list[" << ptr_list.size() << "] and size list[" << size_list.size()
                      << "] should be same.";
  }
  return ptr_list;
}

template <typename MallocInfo, typename ReturnType>
bool MindRTAutoOffloadAdapter::TryAllocMemory(
  const MallocInfo &info, size_t total_size, const HashSet<const void *> &pinned_mem,
  const std::function<bool(const MallocInfo &, DynamicMemPoolBestFit *, ReturnType *)> &alloc_func, ReturnType *ret) {
  if (alloc_func(info, mem_pool_, ret)) {
    return true;
  }
  using KeySizePair = std::pair<DeviceAddress *, size_t>;
  auto less = [](const KeySizePair &a, const KeySizePair &b) -> bool { return a.second < b.second; };
  std::priority_queue<KeySizePair, std::vector<KeySizePair>, decltype(less)> mem_can_offload(less);
  {
    std::shared_lock<std::shared_mutex> shd_lock(all_mem_mutex_);
    for (const auto &mem : all_mem_) {
      if (!MemoryOffloadConflict::GetInstance().CanBeOffloaded(mem) || mem->mem_offloaded() ||
          mem->GetPtr() == nullptr || pinned_mem.count(mem) != 0) {
        continue;
      }
      const auto device_mem_size = mem->GetSize();
      if (device_mem_size >= total_size) {
        SwapOut(mem);

        if (alloc_func(info, mem_pool_, ret)) {
          return true;
        }
      } else {
        mem_can_offload.push({mem, device_mem_size});
      }
    }
  }
  while (!mem_can_offload.empty()) {
    const auto &max_mem_in_device = mem_can_offload.top();
    const auto offload_mem = max_mem_in_device.first;
    SwapOut(offload_mem);

    if (alloc_func(info, mem_pool_, ret)) {
      return true;
    }
    mem_can_offload.pop();
  }
  return false;
}

void MindRTAutoOffloadAdapter::SwapOut(DeviceAddress *device_address) {
  if (device_address->mem_offloaded()) {
    return;
  }
  if (!device_address->Offload(stream_id_)) {
    MS_LOG(EXCEPTION) << "Offload failed, size: " << device_address->GetSize() << ", stream id: " << stream_id_;
  }
}
}  // namespace device
}  // namespace mindspore
