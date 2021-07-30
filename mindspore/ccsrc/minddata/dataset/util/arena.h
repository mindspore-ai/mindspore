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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_ARENA_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_ARENA_H_

#include <memory>
#include <mutex>
#include <utility>
#include "minddata/dataset/util/allocator.h"
#include "minddata/dataset/util/memory_pool.h"
#include "minddata/dataset/util/treap.h"
#ifdef ENABLE_GPUQUE
#include <cuda_runtime_api.h>
#endif

#define ARENA_LOG_BLK_SZ (6u)
#define ARENA_BLK_SZ (static_cast<uint16_t>(1u << ARENA_LOG_BLK_SZ))
#define ARENA_WALL_OVERHEAD_SZ 32
namespace mindspore {
namespace dataset {
/// This is a memory arena based on a treap data structure.
/// The constructor of the Arena takes the size of the initial memory size (in MB).
/// Internally we divide the memory into multiple blocks. Each block is 64 bytes.
/// The treap contains all the free blocks with the relative memory address as key
/// and the size of the block as priority.
///
/// Initially the treap has only one root which is the whole memory piece.
///
/// For memory suballocation, we pop the root node of the treap which contains the largest free block.
/// We allocate what we need and return the rest back to the treap. We search for the first fit instead
/// of the best fit so to give us a constant time in memory allocation.
///
/// When a block of memory is freed. It is joined with the blocks before and after (if they are available) to
/// form a bigger block.

/// At the lowest level, we don't really care where the memory is coming from.
/// This allows other class to make use of Arena method and override the origin of the
/// memory, say from some unix shared memory instead.
/// \note Implementation class is not thread safe. Caller needs to ensure proper serialization
class ArenaImpl {
 public:
  /// Constructor
  /// \param ptr The start of the memory address
  /// \param sz Size of the memory block we manage
  ArenaImpl(void *ptr, size_t sz);
  ~ArenaImpl() { ptr_ = nullptr; }

  /// \brief Allocate a sub block
  /// \param n Size requested
  /// \param p pointer to where the result is stored
  /// \return Status object.
  Status Allocate(size_t n, void **p);

  /// \brief Enlarge or shrink a sub block
  /// \param old_sz Original size
  /// \param new_sz New size
  /// \return Status object
  Status Reallocate(void **, size_t old_sz, size_t new_sz);

  /// \brief Free a sub block
  /// \param Address of the block to be freed.
  void Deallocate(void *);

  /// \brief Calculate % free of the memory
  /// \return Percent free
  int PercentFree() const;

  /// \brief What is the maximum we can support in allocate.
  /// \return Max value
  uint64_t get_max_size() const { return (size_in_bytes_ - ARENA_WALL_OVERHEAD_SZ); }

  /// \brief Get the start of the address. Read only
  /// \return Start of the address block
  const void *get_base_addr() const { return ptr_; }

  static uint64_t SizeToBlk(uint64_t sz);
  friend std::ostream &operator<<(std::ostream &os, const ArenaImpl &s);

 private:
  size_t size_in_bytes_;
  Treap<uint64_t, uint64_t> tr_;
  void *ptr_;

  void *get_user_addr(void *base_addr) const { return reinterpret_cast<char *>(base_addr) + ARENA_WALL_OVERHEAD_SZ; }
  void *get_base_addr(void *user_addr) const { return reinterpret_cast<char *>(user_addr) - ARENA_WALL_OVERHEAD_SZ; }
  std::pair<std::pair<uint64_t, uint64_t>, bool> FindPrevBlk(uint64_t addr);
  bool BlockEnlarge(uint64_t *addr, uint64_t old_sz, uint64_t new_sz);
  Status FreeAndAlloc(void **pp, size_t old_sz, size_t new_sz);
};

/// \brief This version of Arena allocates from private memory
class Arena : public MemoryPool {
 public:
  // Disable copy and assignment constructor
  Arena(const Arena &) = delete;
  Arena &operator=(const Arena &) = delete;
  ~Arena() override {
#ifdef ENABLE_GPUQUE
    if (is_cuda_malloc_) {
      if (ptr_ != nullptr) {
        (void)cudaFreeHost(ptr_);
      }
    }
#else
    if (ptr_ != nullptr) {
      free(ptr_);
    }
    ptr_ = nullptr;
#endif
  }

  /// As a derived class of MemoryPool, we have to implement the following.
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

  /// \return Return the start of the memory block
  const void *get_base_addr() const { return impl_->get_base_addr(); }

  /// \brief Dump the memory allocation block.
  friend std::ostream &operator<<(std::ostream &os, const Arena &s) {
    os << *(s.impl_);
    return os;
  }

#ifdef ENABLE_GPUQUE
  /// The only method to create an arena.
  static Status CreateArena(std::shared_ptr<Arena> *p_ba, size_t val_in_MB = 4096, bool is_cuda_malloc = false);
#else
  /// The only method to create an arena.
  static Status CreateArena(std::shared_ptr<Arena> *p_ba, size_t val_in_MB = 4096);
#endif

 protected:
  mutable std::mutex mux_;
  std::unique_ptr<ArenaImpl> impl_;
  void *ptr_;
  size_t size_in_MB_;
#ifdef ENABLE_GPUQUE
  bool is_cuda_malloc_;

  explicit Arena(size_t val_in_MB = 4096, bool is_cuda_malloc = false);
#else

  explicit Arena(size_t val_in_MB = 4096);
#endif

  Status Init();
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_ARENA_H_
