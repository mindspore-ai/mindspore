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
#include "mindspore/ccsrc/runtime/hardware/device_context.h"

namespace mindspore {
namespace kernel {
constexpr size_t kTilingMemPoolBlockSize = 32;
constexpr size_t kTilingMemPoolDeviceBlockNum = 2 * 1024 * 1024;
constexpr size_t kTilingMemPoolHostBlockNum = 512 * 1024 * 1024;

enum MemoryType : int {
  kMemoryUndefined = 0,
  kMemoryCached,
  kMemoryOneOff,
};

class TilingMemPool {
 public:
  TilingMemPool(size_t block_size, size_t block_num) : block_size_(block_size), block_num_(block_num) {
    total_size_ = block_size * block_num;
  }
  virtual ~TilingMemPool();
  virtual int Init();

  std::pair<MemoryType, void *> Malloc(size_t size);
  void Free(MemoryType mem_type, void *addr);

  void Free(void *addr);
  size_t GetAlignedSize(size_t size);

 protected:
  virtual void *MallocInner(size_t size) { return nullptr; }
  virtual void FreeInner(void *addr) {}

 private:
  inline std::pair<MemoryType, void *> MallocOneOffMem(size_t size) {
    auto addr = MallocInner(size);
    MS_EXCEPTION_IF_NULL(addr);
    return std::make_pair(kMemoryOneOff, addr);
  }

  size_t block_size_{0};
  size_t block_num_{0};
  size_t total_size_{0};
  int8_t *mem_base_ptr_{nullptr};
  size_t offset{0};
  std::atomic<bool> pool_mem_created_{false};
  std::atomic<size_t> offset_{0};
};

class TilingMemPoolHost : public TilingMemPool {
 public:
  TilingMemPoolHost(size_t block_size, size_t block_num) : TilingMemPool(block_size, block_num) {}
  ~TilingMemPoolHost() override = default;

  int Init() override;

 protected:
  void *MallocInner(size_t size) override;
  void FreeInner(void *addr) override;
};

class TilingMemPoolDevice : public TilingMemPool {
 public:
  TilingMemPoolDevice(size_t block_size, size_t block_num) : TilingMemPool(block_size, block_num) {}
  ~TilingMemPoolDevice() override = default;

  int Init() override;

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

  void CopySync(void *host_ptr, void *device_ptr, size_t size);

  void CopySyncD2H(void *host_ptr, void *device_ptr, size_t size);

  TilingMemPoolHost pool_host_{kTilingMemPoolBlockSize, kTilingMemPoolHostBlockNum};
  TilingMemPoolDevice pool_device_{kTilingMemPoolBlockSize, kTilingMemPoolDeviceBlockNum};

 private:
  device::DeviceContext *device_context_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_TILING_MEN_MGR_H_
