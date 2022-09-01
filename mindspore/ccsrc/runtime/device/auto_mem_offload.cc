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

namespace mindspore {
namespace device {
void *MemHandler::MallocHost(size_t mem_size) {
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

void MemHandler::FreeHost(void *ptr) {
  MS_EXCEPTION_IF_NULL(ptr);
  auto iter = host_mem_block_map_.find(ptr);
  if (iter == host_mem_block_map_.end()) {
    MS_LOG(EXCEPTION) << "Free ptr not be created from manager!";
  }
  MS_EXCEPTION_IF_NULL(iter->second);
  auto mem_size = iter->second->size();
  cached_host_mem_[mem_size].emplace(iter->first);
}

void AutoMemoryOffload::SetInitHostPtr(const void *key, void *host_ptr, size_t mem_size) {
  init_from_host_keys_.insert(key);
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

void *AutoMemoryOffload::Get(const void *key, void *stream, const HashSet<const void *> &not_offload) {
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
  auto device_ptr = Malloc(key, mem_size, stream, not_offload);
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

std::vector<void *> AutoMemoryOffload::MallocContinuous(const std::vector<const void *> &keys,
                                                        const std::vector<size_t> &size_list, void *stream,
                                                        const HashSet<const void *> &not_offload) {
  MS_EXCEPTION_IF_NULL(mem_handler_);
  auto device_ptr = mem_handler_->MallocContinuousMemFromMemPool(size_list);
  if (device_ptr.size() == keys.size() || stream == nullptr) {
    for (size_t i = 0; i < device_ptr.size(); i += 1) {
      mem_result_[keys[i]] = device_ptr[i];
      mem_size_[keys[i]] = size_list[i];
      continuous_mem_key_.insert(keys[i]);
    }
    return device_ptr;
  }
  const size_t total_size = std::accumulate(size_list.begin(), size_list.end(), 0);
  using KeySizePair = std::pair<const void *, size_t>;
  auto less = [](const KeySizePair &a, const KeySizePair &b) -> bool { return a.second < b.second; };
  std::priority_queue<KeySizePair, std::vector<KeySizePair>, decltype(less)> mem_can_offload(less);
  for (const auto &mem : mem_result_) {
    const auto offload_key = mem.first;
    if (not_offload.count(offload_key) != 0 || continuous_mem_key_.count(offload_key) != 0) {
      continue;
    }
    const auto device_mem_size = GetMemSize(offload_key);
    if (device_mem_size >= total_size) {
      SwapOut(offload_key, stream);
      Free(offload_key);
      device_ptr = mem_handler_->MallocContinuousMemFromMemPool(size_list);
      if (device_ptr.size() != keys.size()) {
        continue;
      }
      for (size_t i = 0; i < device_ptr.size(); i += 1) {
        mem_result_[keys[i]] = device_ptr[i];
        mem_size_[keys[i]] = size_list[i];
        continuous_mem_key_.insert(keys[i]);
      }
      return device_ptr;
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
    device_ptr = mem_handler_->MallocContinuousMemFromMemPool(size_list);
    if (device_ptr.size() != keys.size()) {
      mem_can_offload.pop();
      continue;
    }
    for (size_t i = 0; i < keys.size(); i += 1) {
      mem_result_[keys[i]] = device_ptr[i];
      mem_size_[keys[i]] = size_list[i];
      continuous_mem_key_.insert(keys[i]);
    }
    return device_ptr;
  }
  return {};
}

void *AutoMemoryOffload::Malloc(const void *key, size_t mem_size, void *stream,
                                const HashSet<const void *> &not_offload) {
  MS_EXCEPTION_IF_NULL(mem_handler_);
  auto iter = mem_result_.find(key);
  if (iter != mem_result_.end()) {
    return iter->second;
  }
  auto device_ptr = mem_handler_->MallocDevice(mem_size);
  if (device_ptr != nullptr || stream == nullptr) {
    mem_result_[key] = device_ptr;
    mem_size_[key] = mem_size;
    return device_ptr;
  }
  using KeySizePair = std::pair<const void *, size_t>;
  auto less = [](const KeySizePair &a, const KeySizePair &b) -> bool { return a.second < b.second; };
  std::priority_queue<KeySizePair, std::vector<KeySizePair>, decltype(less)> mem_can_offload(less);
  for (const auto &i : mem_result_) {
    const auto offload_key = i.first;
    if (not_offload.count(offload_key) != 0) {
      continue;
    }
    const auto device_mem_size = GetMemSize(offload_key);
    if (device_mem_size >= mem_size) {
      SwapOut(offload_key, stream);
      Free(offload_key);
      device_ptr = mem_handler_->MallocDevice(mem_size);
      mem_result_[key] = device_ptr;
      mem_size_[key] = mem_size;
      return device_ptr;
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
    device_ptr = mem_handler_->MallocDevice(mem_size);
    if (device_ptr != nullptr) {
      mem_result_[key] = device_ptr;
      mem_size_[key] = mem_size;
      return device_ptr;
    }
    mem_can_offload.pop();
  }
  return nullptr;
}

void *AutoMemoryOffload::SwapOut(const void *key, void *stream) {
  const auto iter = mem_result_.find(key);
  void *host_ptr = nullptr;
  bool from_init = false;
  const auto mem_size = GetMemSize(key);
  if (iter == mem_result_.end()) {
    GetHostPtr(key, &host_ptr, &from_init);
    if (host_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Can not find device ptr for key " << key;
    }
    return host_ptr;
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
  return host_ptr;
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

void AutoMemoryOffload::UpdateHighPriorityMem(const void *key) { updated_device_mem_.insert(key); }
}  // namespace device
}  // namespace mindspore
