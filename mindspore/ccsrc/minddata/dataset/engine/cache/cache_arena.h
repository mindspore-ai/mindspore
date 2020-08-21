/**
 * Copyright 2020 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_ARENA_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_ARENA_H_

#include <memory>
#include <mutex>
#include <string>
#include "minddata/dataset/util/arena.h"
#include "minddata/dataset/engine/cache/cache_common.h"
namespace mindspore {
namespace dataset {
/// This is a derived class of Arena but resides in shared memory
class CachedSharedMemoryArena : public MemoryPool {
 public:
  // Disable copy and assignment constructor
  CachedSharedMemoryArena(const CachedSharedMemoryArena &) = delete;
  CachedSharedMemoryArena &operator=(const CachedSharedMemoryArena &) = delete;
  ~CachedSharedMemoryArena() override;
  /// \brief Create an Arena in shared memory
  /// \param[out] p_ba Pointer to a unique_ptr
  /// \param shmkey Shared memory key
  /// \param val_in_GB size of shared memory in gigabyte
  /// \return Status object
  static Status CreateArena(std::unique_ptr<CachedSharedMemoryArena> *out, int32_t port, size_t val_in_GB);

  /// \brief This returns where we attach to the shared memory.
  /// Some gRPC requests will ask for a shared memory block, and
  /// we can't return the absolute address as this makes no sense
  /// in the client. So instead we will return an address relative
  /// to the base address of the shared memory where we attach to.
  /// \return Base address of the shared memory.
  const void *SharedMemoryBaseAddr() const { return impl_->get_base_addr(); }

  /// As a derived class of MemoryPool, we have to implement the following
  /// But we simply transfer the call to the implementation class
  Status Allocate(size_t size, void **pVoid) override {
    std::unique_lock<std::mutex> lock(mux_);
    return impl_->Allocate(size, pVoid);
  }
  Status Reallocate(void **pVoid, size_t old_sz, size_t new_sz) override {
    std::unique_lock<std::mutex> lock(mux_);
    return impl_->Reallocate(pVoid, old_sz, new_sz);
  }
  void Deallocate(void *pVoid) override {
    std::unique_lock<std::mutex> lock(mux_);
    impl_->Deallocate(pVoid);
  }
  uint64_t get_max_size() const override { return impl_->get_max_size(); }
  int PercentFree() const override {
    std::unique_lock<std::mutex> lock(mux_);
    return impl_->PercentFree();
  }

  /// \brief Dump the memory allocation block.
  friend std::ostream &operator<<(std::ostream &os, const CachedSharedMemoryArena &s) {
    os << *(s.impl_);
    return os;
  }

 private:
  mutable std::mutex mux_;
  void *ptr_;
  int32_t val_in_GB_;
  int32_t port_;
  int shmid_;
  std::unique_ptr<ArenaImpl> impl_;
  /// Private constructor. Not to be called directly.
  CachedSharedMemoryArena(int32_t port, size_t val_in_GB);
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_CACHE_ARENA_H_
