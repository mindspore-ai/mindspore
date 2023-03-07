/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/util/circular_pool.h"

#include <algorithm>
#include <limits>
#include <utility>
#include "./securec.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/system_pool.h"

namespace mindspore {
namespace dataset {
Status CircularPool::AddOneArena() {
  Status rc;
  std::shared_ptr<Arena> b;
  RETURN_IF_NOT_OK(Arena::CreateArena(&b, arena_size_, is_cuda_malloc_));
  tail_ = b.get();
  cur_size_in_mb_ += arena_size_;
  mem_segments_.push_back(std::move(b));
  return Status::OK();
}

ListOfArenas::iterator CircularPool::CircularIterator::Next() {
  ListOfArenas::iterator it = dp_->mem_segments_.begin();
  uint32_t size = dp_->mem_segments_.size();
  // This is what we return
  it += cur_;
  // Prepare for the next round
  cur_++;
  if (cur_ == size) {
    if (start_ == 0) {
      has_next_ = false;
    } else {
      wrap_ = true;
      cur_ = 0;
    }
  } else if (cur_ == start_) {
    has_next_ = false;
  }
  return it;
}

bool CircularPool::CircularIterator::has_next() const { return has_next_; }

void CircularPool::CircularIterator::Reset() {
  wrap_ = false;
  has_next_ = false;
  if (!dp_->mem_segments_.empty()) {
    // Find the buddy arena that corresponds to the tail.
    cur_tail_ = dp_->tail_;
    auto list_end = dp_->mem_segments_.end();
    auto it = std::find_if(dp_->mem_segments_.begin(), list_end,
                           [this](const std::shared_ptr<Arena> &b) { return b.get() == cur_tail_; });
    MS_ASSERT(it != list_end);
    start_ = std::distance(dp_->mem_segments_.begin(), it);
    cur_ = start_;
    has_next_ = true;
  }
}

CircularPool::CircularIterator::CircularIterator(CircularPool *dp) : dp_(dp) { Reset(); }

Status CircularPool::Allocate(size_t n, void **p) {
  if (p == nullptr) {
    RETURN_STATUS_UNEXPECTED("p is null");
  }
  Status rc;
  void *ptr = nullptr;
  do {
    SharedLock lock_s(&rw_lock_);
    int prevSzInMB = cur_size_in_mb_;
    bool move_tail = false;
    CircularIterator cirIt(this);
    while (cirIt.has_next()) {
      auto it = cirIt.Next();
      Arena *ba = it->get();
      RETURN_UNEXPECTED_IF_NULL(ba);
      if (ba->get_max_size() < n) {
        RETURN_STATUS_OOM("Out of memory.");
      }
      // If we are asked to move forward the tail
      if (move_tail) {
        Arena *expected = cirIt.cur_tail_;
        (void)atomic_compare_exchange_weak(&tail_, &expected, ba);
        move_tail = false;
      }
      rc = ba->Allocate(n, &ptr);
      if (rc.IsOk()) {
        *p = ptr;
        break;
      } else if (rc == StatusCode::kMDOutOfMemory) {
        // Make the next arena a new tail and continue.
        move_tail = true;
      } else {
        return rc;
      }
    }

    // Handle the case we have done one round robin search.
    if (ptr == nullptr) {
      // If we have room to expand.
      if (unlimited_ || cur_size_in_mb_ < max_size_in_mb_) {
        // lock in exclusively mode.
        lock_s.Upgrade();
        // Check again if someone has already expanded.
        if (cur_size_in_mb_ == prevSzInMB) {
          RETURN_IF_NOT_OK(AddOneArena());
        }
        // Re-acquire the shared lock and try again
        lock_s.Downgrade();
      } else {
        RETURN_STATUS_OOM("Out of memory.");
      }
    }
  } while (ptr == nullptr);
  return rc;
}

void CircularPool::Deallocate(void *p) {
  // Lock in the chain in shared mode and find out which
  // segment it comes from
  SharedLock lock(&rw_lock_);
  auto it = std::find_if(mem_segments_.begin(), mem_segments_.end(), [this, p](std::shared_ptr<Arena> &b) -> bool {
    char *q = reinterpret_cast<char *>(p);
    auto *base = reinterpret_cast<const char *>(b->get_base_addr());
    return (q > base && q < base + arena_size_ * 1048576L);
  });
  lock.Unlock();
  MS_ASSERT(it != mem_segments_.end());
  it->get()->Deallocate(p);
}

Status CircularPool::Reallocate(void **pp, size_t old_sz, size_t new_sz) {
  // Lock in the chain in shared mode and find out which
  // segment it comes from
  if (pp == nullptr) {
    RETURN_STATUS_UNEXPECTED("pp is null");
  }
  void *p = *pp;
  SharedLock lock(&rw_lock_);
  auto it = std::find_if(mem_segments_.begin(), mem_segments_.end(), [this, p](std::shared_ptr<Arena> &b) -> bool {
    char *q = reinterpret_cast<char *>(p);
    auto *base = reinterpret_cast<const char *>(b->get_base_addr());
    return (q > base && q < base + arena_size_ * 1048576L);
  });
  lock.Unlock();
  MS_ASSERT(it != mem_segments_.end());
  Arena *ba = it->get();
  Status rc = ba->Reallocate(pp, old_sz, new_sz);
  if (rc == StatusCode::kMDOutOfMemory) {
    // The current arena has no room for the bigger size.
    // Allocate free space from another arena and copy
    // the content over.
    void *q = nullptr;
    rc = this->Allocate(new_sz, &q);
    RETURN_IF_NOT_OK(rc);
    errno_t err = memcpy_s(q, new_sz, p, old_sz);
    if (err) {
      this->Deallocate(q);
      RETURN_STATUS_UNEXPECTED(std::to_string(err));
    }
    *pp = q;
    ba->Deallocate(p);
  }
  return Status::OK();
}

uint64_t CircularPool::get_max_size() const { return mem_segments_.front()->get_max_size(); }

int CircularPool::PercentFree() const {
  int percent_free = 0;
  int num_arena = 0;
  for (auto const &p : mem_segments_) {
    percent_free += p->PercentFree();
    num_arena++;
  }
  if (num_arena) {
    return percent_free / num_arena;
  } else {
    return 100;
  }
}

CircularPool::CircularPool(int max_size_in_gb, int arena_size, bool is_cuda_malloc)
    : unlimited_(max_size_in_gb <= 0),
      max_size_in_mb_(unlimited_ ? std::numeric_limits<int32_t>::max() : max_size_in_gb * 1024),
      arena_size_(arena_size),
      is_cuda_malloc_(is_cuda_malloc),
      cur_size_in_mb_(0) {}

Status CircularPool::CreateCircularPool(std::shared_ptr<MemoryPool> *out_pool, int max_size_in_gb, int arena_size,
                                        bool createOneArena, bool is_cuda_malloc) {
  Status rc;
  if (out_pool == nullptr) {
    RETURN_STATUS_UNEXPECTED("pPool is null");
  }
  auto pool = new (std::nothrow) CircularPool(max_size_in_gb, arena_size, is_cuda_malloc);
  if (pool == nullptr) {
    RETURN_STATUS_OOM("Out of memory.");
  }
  if (createOneArena) {
    rc = pool->AddOneArena();
  }
  if (rc.IsOk()) {
    (*out_pool).reset(pool);
  } else {
    delete pool;
  }
  return rc;
}

CircularPool::~CircularPool() = default;
}  // namespace dataset
}  // namespace mindspore
