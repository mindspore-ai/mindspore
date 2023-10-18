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

#include "extendrt/delegate/ascend_ge/ge_memory_manager.h"
#include "src/common/log_adapter.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "acl/acl.h"
#include "runtime/mem.h"

namespace mindspore {
size_t ALIGN_OFFSET(void *addr) {
  auto extra = (reinterpret_cast<uint64_t>(addr) & 0xff);
  if (extra == 0) {
    return 0;
  }
  return kMemAlignSize - extra;
}

constexpr size_t kAlignSizeMulti = 2;
size_t ALIGN_UP(size_t size) { return ((size + kAlignSizeMulti * kMemAlignSize - 1) / kMemAlignSize) * kMemAlignSize; }

GeMemoryManager::GeMemoryManager() = default;
GeMemoryManager::~GeMemoryManager() { FreeAllMemory(); }

uint8_t *GeMemoryManager::MallocDeviceMemory(const std::string &purpose, size_t size) {
  GeMemoryInfo info;
  info.malloc_size = ALIGN_UP(size);
  info.purpose = purpose;
  auto ret = rtMalloc(&info.malloc_addr, size, RT_MEMORY_HBM, 0);
  if (ret != ACL_RT_SUCCESS || info.malloc_addr == nullptr) {
    MS_LOG(ERROR) << "Malloc device memory failed, malloc size " << info.malloc_size << ", real size " << size
                  << ", memory purpose " << purpose;
    return nullptr;
  }
  info.use_addr = reinterpret_cast<uint8_t *>(info.malloc_addr) + ALIGN_OFFSET(info.malloc_addr);
  info.use_size = size;
  device_memories_.push_back(info);
  MS_LOG(INFO) << "Malloc device memory success, malloc size " << info.malloc_size << ", real size " << size
               << ", memory purpose " << purpose;
  return reinterpret_cast<uint8_t *>(info.use_addr);
}

uint8_t *GeMemoryManager::MallocHostMemory(const std::string &purpose, size_t size) {
  GeMemoryInfo info;
  info.malloc_size = ALIGN_UP(size);
  info.purpose = purpose;
  auto ret = aclrtMallocHost(&info.malloc_addr, size);
  if (ret != ACL_ERROR_NONE || info.malloc_addr == nullptr) {
    MS_LOG(INFO) << "Malloc host memory success, malloc size " << info.malloc_size << ", real size " << size
                 << ", memory purpose " << purpose;
    return nullptr;
  }
  info.use_addr = reinterpret_cast<uint8_t *>(info.malloc_addr) + ALIGN_OFFSET(info.malloc_addr);
  info.use_size = size;
  host_memories_.push_back(info);
  MS_LOG(INFO) << "Malloc host memory success, malloc size " << info.malloc_size << ", real size " << size
               << ", memory purpose " << purpose;
  return reinterpret_cast<uint8_t *>(info.use_addr);
}

bool GeMemoryManager::MemcpyHost2Device(void *dst_addr, size_t dst_max_size, const void *src_addr, size_t src_size) {
  auto ret = aclrtMemcpy(dst_addr, dst_max_size, src_addr, src_size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMemcpy data from host to device failed, dst size: " << dst_max_size
                  << ", src size: " << src_size;
    return false;
  }
  return true;
}

bool GeMemoryManager::MemcpyDevice2Host(void *dst_addr, size_t dst_max_size, const void *src_addr, size_t src_size) {
  auto ret = aclrtMemcpy(dst_addr, dst_max_size, src_addr, src_size, ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Call aclrtMemcpy data from device to host failed, dst size: " << dst_max_size
                  << ", src size: " << src_size;
    return false;
  }
  return true;
}

void GeMemoryManager::FreeDeviceMemory(void *mem) {
  auto it =
    std::find_if(device_memories_.begin(), device_memories_.end(), [mem](auto &info) { return info.use_addr == mem; });
  if (it == device_memories_.end()) {
    MS_LOG(ERROR) << "Failed to free device memory, memory not found";
    return;
  }
  auto info = *it;
  auto ret = aclrtFree(info.malloc_addr);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Free device memory failed, malloc size " << info.malloc_size << ", real size " << info.use_size
                  << ", memory purpose " << info.purpose;
    return;
  }
  device_memories_.erase(it);
  MS_LOG(INFO) << "Free device memory success, malloc size " << info.malloc_size << ", real size " << info.use_size
               << ", memory purpose " << info.purpose;
}

void GeMemoryManager::FreeHostMemory(void *mem) {
  auto it =
    std::find_if(host_memories_.begin(), host_memories_.end(), [mem](auto &info) { return info.use_addr == mem; });
  if (it == host_memories_.end()) {
    MS_LOG(ERROR) << "Failed to free host memory, memory not found";
    return;
  }
  auto info = *it;
  auto ret = aclrtFreeHost(info.malloc_addr);
  if (ret != ACL_ERROR_NONE) {
    MS_LOG(ERROR) << "Free host memory failed, malloc size " << info.malloc_size << ", real size " << info.use_size
                  << ", memory purpose " << info.purpose;
    return;
  }
  host_memories_.erase(it);
  MS_LOG(INFO) << "Free host memory success, malloc size " << info.malloc_size << ", real size " << info.use_size
               << ", memory purpose " << info.purpose;
}

void GeMemoryManager::FreeAllMemory() {
  for (auto &info : device_memories_) {
    MS_LOG(INFO) << "Free device memory, malloc size " << info.malloc_size << ", real size " << info.use_size
                 << ", memory purpose " << info.purpose;
    auto ret = aclrtFree(info.malloc_addr);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Free device memory failed, malloc size " << info.malloc_size << ", real size " << info.use_size
                    << ", memory purpose " << info.purpose;
      continue;
    }
    MS_LOG(INFO) << "Free device memory success, malloc size " << info.malloc_size << ", real size " << info.use_size
                 << ", memory purpose " << info.purpose;
  }
  device_memories_.clear();
  for (auto &info : host_memories_) {
    MS_LOG(INFO) << "Free host memory, malloc size " << info.malloc_size << ", real size " << info.use_size
                 << ", memory purpose " << info.purpose;
    auto ret = aclrtFreeHost(info.malloc_addr);
    if (ret != ACL_ERROR_NONE) {
      MS_LOG(ERROR) << "Free host memory failed, malloc size " << info.malloc_size << ", real size " << info.use_size
                    << ", memory purpose " << info.purpose;
      continue;
    }
    MS_LOG(INFO) << "Free host memory success, malloc size " << info.malloc_size << ", real size " << info.use_size
                 << ", memory purpose " << info.purpose;
  }
  host_memories_.clear();
}
}  // namespace mindspore
