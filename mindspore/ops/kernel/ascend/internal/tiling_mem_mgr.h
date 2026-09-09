/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_TILING_MEN_MGR_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_TILING_MEN_MGR_H_

#include <string>
#include <vector>
#include <utility>
#include <atomic>
#include <set>
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace kernel {
constexpr size_t kTilingMemPoolBlockSize = 32;
constexpr size_t kTilingMemPoolDeviceBlockNum = 3 * 1024 * 1024;
constexpr size_t kTilingMemPoolHostBlockNum = 8 * 1024 * 1024;

enum MemoryType : int {
  kMemoryUndefined = 0,
  kMemoryCached,
  kMemoryOneOff,
};

struct Slot {
  size_t offset_{0};
  size_t length_{0};
};

class TilingMemPool {
 public:
  TilingMemPool(size_t block_size, size_t block_num);
  virtual ~TilingMemPool() = default;
  virtual int Init();

  size_t GetAlignedSize(size_t size);

  void *Malloc(size_t size);
  void Free(void *addr, size_t size);
  void Rearrange();

  void SetName(const std::string &name) { name_ = name; }

  std::string GetName() const { return name_; }

  inline bool IsOneOffMem(void *addr) const { return addr < mem_base_ptr_ || addr >= mem_base_ptr_ + total_size_; }

 protected:
  virtual void *MallocInner(size_t size) { return nullptr; }
  virtual void FreeInner(void *addr) {}
  void FreeMemPtrs();

 private:
  inline void *MallocOneOffMem(size_t size) {
    auto addr = MallocInner(size);
    MS_EXCEPTION_IF_NULL(addr);
    one_off_mem_ptrs_.insert(addr);
    return addr;
  }

  inline size_t RoundAdd(size_t idx) { return (idx + 1) % block_num_; }

  inline size_t RoundSub(size_t idx) { return (idx + block_num_ - 1) % block_num_; }

  size_t block_size_{0};
  size_t block_num_{0};
  size_t total_size_{0};
  int8_t *mem_base_ptr_{nullptr};
  std::set<void *> one_off_mem_ptrs_;

  std::vector<Slot> mem_slots_;
  size_t head_{0};
  size_t tail_{0};
  std::string name_;
};

class TilingMemPoolHost : public TilingMemPool {
 public:
  TilingMemPoolHost(size_t block_size, size_t block_num);
  ~TilingMemPoolHost() override { FreeMemPtrs(); }

 protected:
  void *MallocInner(size_t size) override;
  void FreeInner(void *addr) override;
};

class TilingMemPoolDevice : public TilingMemPool {
 public:
  TilingMemPoolDevice(size_t block_size, size_t block_num);
  ~TilingMemPoolDevice() override { FreeMemPtrs(); }

 protected:
  void *MallocInner(size_t size) override;
  void FreeInner(void *addr) override;
};

class TilingMemMgr {
 public:
  TilingMemMgr();
  ~TilingMemMgr() = default;

  static TilingMemMgr &GetInstance() {
    static TilingMemMgr mgr;
    return mgr;
  }

  void CopyAsync(void *host_ptr, void *device_ptr, size_t size);

  void CopyAsyncD2H(void *host_ptr, void *device_ptr, size_t size);

  TilingMemPoolHost pool_host_{kTilingMemPoolBlockSize, kTilingMemPoolHostBlockNum};
  TilingMemPoolDevice pool_device_{kTilingMemPoolBlockSize, kTilingMemPoolDeviceBlockNum};

 private:
  device::DeviceContext *device_context_{nullptr};
  void *default_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_TILING_MEN_MGR_H_
