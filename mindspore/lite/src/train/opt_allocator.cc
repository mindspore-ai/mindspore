/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "src/train/opt_allocator.h"
#include <limits>
#include "nnacl/op_base.h"

namespace mindspore {
size_t OptAllocator::FindFree(size_t size) {
  size_t min_size = std::numeric_limits<size_t>::max();
  size_t min_addr = std::numeric_limits<size_t>::max();
  for (auto const &itr : arena_) {
    // best fit
    if (itr.second >= size) {
      if (min_size > itr.second) {
        min_size = itr.second;
        min_addr = itr.first;
      }
    }
  }
  return min_addr;
}

void OptAllocator::Reorder(size_t addr) {
  size_t length = arena_[addr];
  size_t post = addr + length;
  // connect to upper block
  auto it = arena_.find(post);
  if (it != arena_.end()) {
    size_t post_size = it->second;
    arena_[addr] = length + post_size;
    arena_.erase(post);
  }
  // connect to lower block
  auto itr = arena_.lower_bound(addr);
  if (itr != arena_.begin()) {
    itr--;
    size_t last = itr->first;
    if ((last + arena_[last]) == addr) {
      arena_[last] = arena_[last] + arena_[addr];
      arena_.erase(addr);
    }
  }
}

size_t OptAllocator::Malloc(size_t size) {
  size = UP_DIV(size, align_size_) * align_size_;
  size_t addr = FindFree(size);
  // free block not found
  if (addr == std::numeric_limits<size_t>::max()) {
    if (!arena_.empty()) {
      addr = arena_.rbegin()->first;
      if (addr + arena_[addr] < heap_) {
        addr = heap_;
      } else {
        arena_.erase(addr);
      }
    } else {
      addr = heap_;
    }
    heap_ = addr + size;
  } else {
    if (arena_[addr] > size) {
      arena_[addr + size] = arena_[addr] - size;
    }
    arena_.erase(addr);
  }
  alloc_[addr] = size;
  return addr;
}

void OptAllocator::Free(size_t addr) {
  arena_[addr] = alloc_[addr];
  alloc_.erase(addr);
  Reorder(addr);
}
}  // namespace mindspore
