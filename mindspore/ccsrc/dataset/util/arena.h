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
#ifndef DATASET_UTIL_ARENA_H_
#define DATASET_UTIL_ARENA_H_

#include <memory>
#include <mutex>
#include <utility>
#include "dataset/util/memory_pool.h"
#include "dataset/util/treap.h"

#define ARENA_LOG_BLK_SZ (6u)
#define ARENA_BLK_SZ (static_cast<uint16_t>(1u << ARENA_LOG_BLK_SZ))
#define ARENA_WALL_OVERHEAD_SZ 32
namespace mindspore {
namespace dataset {
// This is a memory arena based on a treap data structure.
// The constructor of the Arena takes the size of the initial memory size (in MB).
// Internally we divide the memory into multiple blocks. Each block is 64 bytes.
// The treap contains all the free blocks with the relative memory address as key
// and the size of the block as priority.
//
// Initially the treap has only one root which is the whole memory piece.
//
// For memory suballocation, we pop the root node of the treap which contains the largest free block.
// We allocate what we need and return the rest back to the treap. We search for the first fit instead
// of the best fit so to give us a constant time in memory allocation.
//
// When a block of memory is freed. It is joined with the blocks before and after (if they are available) to
// form a bigger block.
class Arena : public MemoryPool {
 public:
  Arena(const Arena &) = delete;

  Arena &operator=(const Arena &) = delete;

  ~Arena() override {
    if (ptr_ != nullptr) {
      free(ptr_);
      ptr_ = nullptr;
    }
  }

  Status Allocate(size_t n, void **p) override;

  Status Reallocate(void **, size_t old_sz, size_t new_sz) override;

  void Deallocate(void *) override;

  uint64_t get_max_size() const override;

  static uint64_t SizeToBlk(uint64_t sz) {
    uint64_t req_blk = sz / ARENA_BLK_SZ;
    if (sz % ARENA_BLK_SZ) {
      ++req_blk;
    }
    return req_blk;
  }

  int PercentFree() const override;

  const void *get_base_addr() const { return ptr_; }

  friend std::ostream &operator<<(std::ostream &os, const Arena &s);

  static Status CreateArena(std::shared_ptr<Arena> *p_ba, size_t val_in_MB = 4096);

 private:
  std::mutex mux_;
  Treap<uint64_t, uint64_t> tr_;
  void *ptr_;
  size_t size_in_MB_;
  size_t size_in_bytes_;

  explicit Arena(size_t val_in_MB = 4096);

  std::pair<std::pair<uint64_t, uint64_t>, bool> FindPrevBlk(uint64_t addr);

  Status Init();

  bool BlockEnlarge(uint64_t *addr, uint64_t old_sz, uint64_t new_sz);

  Status FreeAndAlloc(void **pp, size_t old_sz, size_t new_sz);

  void *get_user_addr(void *base_addr) const { return reinterpret_cast<char *>(base_addr) + ARENA_WALL_OVERHEAD_SZ; }

  void *get_base_addr(void *user_addr) const { return reinterpret_cast<char *>(user_addr) - ARENA_WALL_OVERHEAD_SZ; }
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_ARENA_H_
