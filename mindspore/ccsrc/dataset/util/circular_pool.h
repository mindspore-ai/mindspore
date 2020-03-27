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
#ifndef DATASET_UTIL_CIRCULAR_POOL_H_
#define DATASET_UTIL_CIRCULAR_POOL_H_

#include <atomic>
#include <memory>
#include <vector>
#include "dataset/util/memory_pool.h"
#include "dataset/util/arena.h"
#include "dataset/util/lock.h"

namespace mindspore {
namespace dataset {
using ListOfArenas = std::vector<std::shared_ptr<Arena>>;

// This is a dynamic memory pool built on top of memory
// segment each of which is 4G in size. Initially we start
// with one segment, and gradually add segments (not
// guaranteed contiguous) until we reach 32G in size.  There
// is an assumption about this kind of memory pool.  Allocated
// memory is not held for the whole duration of the pool and
// will be released soon. Based on this assumption, memory is
// obtained from the tail while allocated memory is returned
// to the head of the pool.
class CircularPool : public MemoryPool {
 public:
  class CircularIterator {
    friend class CircularPool;

   public:
    explicit CircularIterator(CircularPool *dp);

    ~CircularIterator() = default;

    bool has_next() const;

    ListOfArenas::iterator Next();

    void Reset();

   private:
    CircularPool *dp_;
    Arena *cur_tail_{};
    uint32_t start_{};
    uint32_t cur_{};
    bool wrap_{};
    bool has_next_{};
  };

  CircularPool(const CircularPool &) = delete;

  CircularPool &operator=(const CircularPool &) = delete;

  ~CircularPool() override;

  Status Allocate(size_t n, void **) override;

  Status Reallocate(void **, size_t old_size, size_t new_size) override;

  void Deallocate(void *) override;

  uint64_t get_max_size() const override;

  int PercentFree() const override;

  friend std::ostream &operator<<(std::ostream &os, const CircularPool &s) {
    int i = 0;
    for (auto it = s.mem_segments_.begin(); it != s.mem_segments_.end(); ++it, ++i) {
      os << "Dumping segment " << i << "\n" << *(it->get());
    }
    return os;
  }

  static Status CreateCircularPool(std::shared_ptr<MemoryPool> *out_pool, int max_size_in_gb = -1,
                                   int arena_size = 4096, bool create_one_arena = false);

 private:
  ListOfArenas mem_segments_;
  std::atomic<Arena *> tail_{};
  bool unlimited_;
  int max_size_in_mb_;
  int arena_size_;
  int cur_size_in_mb_;
  RWLock rw_lock_;

  // We can take negative or 0 as input which means unlimited.
  CircularPool(int max_size_in_gb, int arena_size);

  Status AddOneArena();
};
}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_UTIL_CIRCULAR_POOL_H_
