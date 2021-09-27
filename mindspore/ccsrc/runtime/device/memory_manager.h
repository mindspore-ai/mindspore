/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_MANAGER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_MANAGER_H_
#include <memory>
#include <utility>
#include <vector>
#include <map>
#include <queue>
#include "backend/optimizer/mem_reuse/mem_reuse.h"
#include "backend/optimizer/somas/somas.h"
#include "runtime/device/memory_scheduler.h"
namespace mindspore {
namespace device {
enum MemType { kStaticMem, kDynamicMem, kSomasReuseDynamicMem };
constexpr int kGetAllOuts = -1;
constexpr uint64_t kMemAlignSize = 512;
constexpr uint64_t kTwiceMemAlignSize = kMemAlignSize << 1;
using SomasPtr = mindspore::somas::SomasPtr;

class MemoryManager : public MemHandler {
 public:
  MemoryManager() = default;
  virtual ~MemoryManager() = default;

  virtual void MallocDeviceMemory() = 0;
  virtual void FreeDeviceMemory() = 0;
  virtual void ResetDynamicMemory() {
    total_dynamic_size_ = 0;
    dynamic_mem_offset_ = 0;
  }
  virtual void ClearGlobalIdleMem() {}

  virtual void MallocSomasDynamicMem(const session::KernelGraph &graph);
  uint8_t *MallocOutputMem(const AnfNodePtr &node, size_t index, MemType type, size_t size,
                           const DeviceAddressPtr &address, bool comm_mem);
  uint8_t *MallocWorkSpaceMem(const AnfNodePtr &node, size_t index, MemType type, size_t size);
  virtual uint8_t *MallocMem(MemType type, size_t size, const DeviceAddressPtr &address,
                             uint32_t graph_id = kInvalidGraphId);

  virtual bool MallocMemFromMemPool(const DeviceAddressPtr address, size_t size);
  virtual void *MallocMemFromMemPool(size_t size);
  virtual uint8_t *MallocCommunicationMemFromMemPool(size_t size) { return nullptr; }
  virtual void FreeMemFromMemPool(const DeviceAddressPtr address);
  virtual void FreeMemFromMemPool(void *device_ptr);
  virtual bool MallocContinuousMemFromMemPool(const DeviceAddressPtrList &addr_list, size_t total_size,
                                              std::vector<size_t> size_list);
  virtual std::vector<void *> MallocContinuousMemFromMemPool(size_t total_size, std::vector<size_t> size_list);

  static size_t GetCommonAlignSize(size_t input_size);
  static size_t GetCommunicationAlignSize(size_t input_size);

  // swap manager interface
  void *MallocDevice(size_t mem_size) override { return MallocMemFromMemPool(mem_size); }
  void FreeDevice(void *ptr) override {
    MS_EXCEPTION_IF_NULL(ptr);
    FreeMemFromMemPool(ptr);
  }
  void *MallocHost(size_t mem_size) override {
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
  void FreeHost(void *ptr) override {
    MS_EXCEPTION_IF_NULL(ptr);
    auto iter = host_mem_block_map_.find(ptr);
    if (iter == host_mem_block_map_.end()) {
      MS_LOG(ERROR) << "Free ptr not be created from manager!";
    }
    auto mem_size = iter->second->size();
    cached_host_mem_[mem_size].emplace(iter->first);
  }
  void SwapIn(const void *host_ptr, void *device_ptr, size_t mem_size, void *stream) override {
    MS_LOG(INFO) << "Call default swap in " << host_ptr << "," << device_ptr << "," << mem_size << "," << stream;
  }
  void SwapOut(const void *device_ptr, void *host_ptr, size_t mem_size, void *stream) override {
    MS_LOG(INFO) << "Call default swap out " << host_ptr << "," << device_ptr << "," << mem_size << "," << stream;
  }
  size_t GetAvailableMemSize() override {
    MS_LOG(ERROR) << "Return default 0 mem size!";
    return 0;
  }

 protected:
  virtual uint8_t *MallocStaticMem(size_t size, bool communication_mem, uint32_t graph_id = kInvalidGraphId) = 0;
  virtual uint8_t *MallocDynamicMem(size_t size, bool communication_mem);
  uint8_t *device_mem_base_{nullptr};
  uint64_t device_mem_size_{0};
  uint64_t dynamic_mem_offset_{0};
  uint64_t static_mem_offset_{0};
  size_t total_static_size_ = 0;
  size_t total_dynamic_size_ = 0;
  SomasPtr somas_reuse_util_ptr_{nullptr};
  std::map<size_t, std::queue<void *>> cached_host_mem_;
  std::map<void *, std::shared_ptr<std::vector<uint8_t>>> host_mem_block_map_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_MEMORY_MANAGER_H_
