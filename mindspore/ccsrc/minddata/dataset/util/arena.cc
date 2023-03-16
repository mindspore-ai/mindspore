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
#include "minddata/dataset/util/arena.h"
#include <utility>
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/system_pool.h"
#ifdef WITH_BACKEND
#include "mindspore/ccsrc/runtime/hardware/device_context_manager.h"
#endif

namespace mindspore {
namespace dataset {
struct MemHdr {
  uint32_t sig;
  uint64_t addr;
  uint64_t blk_size;
  MemHdr(uint64_t a, uint64_t sz) : sig(0xDEADBEEF), addr(a), blk_size(sz) {}
  static void setHdr(void *p, uint64_t addr, uint64_t sz) { new (p) MemHdr(addr, sz); }
  static void getHdr(void *p, MemHdr *hdr) {
    auto *tmp = reinterpret_cast<MemHdr *>(p);
    *hdr = *tmp;
  }
};

ArenaImpl::ArenaImpl(void *ptr, size_t sz) : size_in_bytes_(sz), ptr_(ptr) {
  // Divide the memory into blocks. Ignore the last partial block.
  uint64_t num_blks = size_in_bytes_ / ARENA_BLK_SZ;
  MS_LOG(DEBUG) << "Arena memory pool is created. Number of blocks : " << num_blks << ". Block size : " << ARENA_BLK_SZ
                << ".";
  tr_.Insert(0, num_blks);
}

Status ArenaImpl::Allocate(size_t n, void **p) {
  RETURN_UNEXPECTED_IF_NULL(p);
  if (n == 0) {
    *p = nullptr;
    return Status::OK();
  }
  // Round up n to 1K block
  uint64_t req_size = static_cast<uint64_t>(n) + ARENA_WALL_OVERHEAD_SZ;
  if (req_size > this->get_max_size()) {
    return Status(StatusCode::kMDOutOfMemory);
  }
  uint64_t reqBlk = SizeToBlk(req_size);
  // Do a first fit search
  auto blk = tr_.Top();
  if (blk.second && reqBlk <= blk.first.priority) {
    uint64_t addr = blk.first.key;
    uint64_t size = blk.first.priority;
    // Trim to the required size and return the rest to the tree.
    tr_.Pop();
    if (size > reqBlk) {
      tr_.Insert(addr + reqBlk, size - reqBlk);
    }
    RETURN_UNEXPECTED_IF_NULL(ptr_);
    char *q = static_cast<char *>(ptr_) + addr * ARENA_BLK_SZ;
    MemHdr::setHdr(q, addr, reqBlk);
    *p = get_user_addr(q);
  } else {
    return Status(StatusCode::kMDOutOfMemory);
  }
  return Status::OK();
}

std::pair<std::pair<uint64_t, uint64_t>, bool> ArenaImpl::FindPrevBlk(uint64_t addr) {
  for (auto &it : tr_) {
    if (it.key + it.priority == addr) {
      return std::make_pair(std::make_pair(it.key, it.priority), true);
    } else if (it.key > addr) {
      break;
    }
  }
  return std::make_pair(std::make_pair(0, 0), false);
}

void ArenaImpl::Deallocate(void *p) {
  if (p == nullptr) {
    MS_LOG(ERROR) << "The pointer[p] is null.";
    return;
  }
  auto *q = get_base_addr(p);
  MemHdr hdr(0, 0);
  MemHdr::getHdr(q, &hdr);
  MS_ASSERT(hdr.sig == 0xDEADBEEF);
  // We are going to insert a free block back to the treap. But first, check if we can combine
  // with the free blocks before and after to form a bigger block.
  // Query if we have a free block after us.
  auto nextBlk = tr_.Search(hdr.addr + hdr.blk_size);
  if (nextBlk.second) {
    // Form a bigger block
    hdr.blk_size += nextBlk.first.priority;
    tr_.DeleteKey(nextBlk.first.key);
  }
  // Next find a block in front of us.
  auto result = FindPrevBlk(hdr.addr);
  if (result.second) {
    // We can combine with this block
    hdr.addr = result.first.first;
    hdr.blk_size += result.first.second;
    tr_.DeleteKey(result.first.first);
  }
  // Now we can insert the free node
  tr_.Insert(hdr.addr, hdr.blk_size);
}

bool ArenaImpl::BlockEnlarge(uint64_t *addr, uint64_t old_sz, uint64_t new_sz) {
  uint64_t size = old_sz;
  // The logic is very much identical to Deallocate. We will see if we can combine with the blocks before and after.
  auto next_blk = tr_.Search(*addr + old_sz);
  if (next_blk.second) {
    size += next_blk.first.priority;
    if (size >= new_sz) {
      // In this case, we can just enlarge the block without doing any moving.
      tr_.DeleteKey(next_blk.first.key);
      // Return unused back to the tree.
      if (size > new_sz) {
        tr_.Insert(*addr + new_sz, size - new_sz);
      }
    }
    return true;
  }
  // If we still get here, we have to look at the block before us.
  auto result = FindPrevBlk(*addr);
  if (result.second) {
    // We can combine with this block together with the next block (if any)
    size += result.first.second;
    *addr = result.first.first;
    if (size >= new_sz) {
      // We can combine with this block together with the next block (if any)
      tr_.DeleteKey(*addr);
      if (next_blk.second) {
        tr_.DeleteKey(next_blk.first.key);
      }
      // Return unused back to the tree.
      if (size > new_sz) {
        tr_.Insert(*addr + new_sz, size - new_sz);
      }
      return true;
    }
  }
  return false;
}

Status ArenaImpl::FreeAndAlloc(void **pp, size_t old_sz, size_t new_sz) {
  RETURN_UNEXPECTED_IF_NULL(pp);
  RETURN_UNEXPECTED_IF_NULL(*pp);
  void *p = nullptr;
  void *q = *pp;
  RETURN_IF_NOT_OK(Allocate(new_sz, &p));
  errno_t err = memmove_s(p, new_sz, q, old_sz);
  if (err != EOK) {
    RETURN_STATUS_UNEXPECTED("Error from memmove: " + std::to_string(err));
  }
  *pp = p;
  // Free the old one.
  Deallocate(q);
  return Status::OK();
}

Status ArenaImpl::Reallocate(void **pp, size_t old_sz, size_t new_sz) {
  RETURN_UNEXPECTED_IF_NULL(pp);
  RETURN_UNEXPECTED_IF_NULL(*pp);
  uint64_t actual_size = static_cast<uint64_t>(new_sz) + ARENA_WALL_OVERHEAD_SZ;
  if (actual_size > this->get_max_size()) {
    RETURN_STATUS_UNEXPECTED("Request size too big : " + std::to_string(new_sz));
  }
  uint64_t req_blk = SizeToBlk(actual_size);
  char *oldAddr = reinterpret_cast<char *>(*pp);
  auto *oldHdr = get_base_addr(oldAddr);
  MemHdr hdr(0, 0);
  MemHdr::getHdr(oldHdr, &hdr);
  MS_ASSERT(hdr.sig == 0xDEADBEEF);
  if (hdr.blk_size > req_blk) {
    // Refresh the header with the new smaller size.
    MemHdr::setHdr(oldHdr, hdr.addr, req_blk);
    // Return the unused memory back to the tree. Unlike allocate, we we need to merge with the block after us.
    auto next_blk = tr_.Search(hdr.addr + hdr.blk_size);
    if (next_blk.second) {
      hdr.blk_size += next_blk.first.priority;
      tr_.DeleteKey(next_blk.first.key);
    }
    tr_.Insert(hdr.addr + req_blk, hdr.blk_size - req_blk);
  } else if (hdr.blk_size < req_blk) {
    uint64_t addr = hdr.addr;
    // Attempt a block enlarge. No guarantee it is always successful.
    bool success = BlockEnlarge(&addr, hdr.blk_size, req_blk);
    if (success) {
      RETURN_UNEXPECTED_IF_NULL(ptr_);
      auto *newHdr = static_cast<char *>(ptr_) + addr * ARENA_BLK_SZ;
      MemHdr::setHdr(newHdr, addr, req_blk);
      if (addr != hdr.addr) {
        errno_t err =
          memmove_s(get_user_addr(newHdr), (req_blk * ARENA_BLK_SZ) - ARENA_WALL_OVERHEAD_SZ, oldAddr, old_sz);
        if (err != EOK) {
          RETURN_STATUS_UNEXPECTED("Error from memmove: " + std::to_string(err));
        }
      }
      *pp = get_user_addr(newHdr);
      return Status::OK();
    }
    return FreeAndAlloc(pp, old_sz, new_sz);
  }
  return Status::OK();
}

int ArenaImpl::PercentFree() const {
  uint64_t sz = 0;
  for (auto &it : tr_) {
    sz += it.priority;
  }
  if (size_in_bytes_ == 0) {
    MS_LOG(ERROR) << "size_in_bytes_ can not be zero.";
    return 0;
  }
  double ratio = static_cast<double>(sz * ARENA_BLK_SZ) / static_cast<double>(size_in_bytes_);
  return static_cast<int>(ratio * 100.0);
}

uint64_t ArenaImpl::SizeToBlk(uint64_t sz) {
  uint64_t req_blk = sz / ARENA_BLK_SZ;
  if (sz % ARENA_BLK_SZ) {
    ++req_blk;
  }
  return req_blk;
}

std::ostream &operator<<(std::ostream &os, const ArenaImpl &s) {
  for (auto &it : s.tr_) {
    os << "Address : " << it.key << ". Size : " << it.priority << "\n";
  }
  return os;
}

Status Arena::Init() {
  try {
    int64_t sz = size_in_MB_ * 1048576L;
#ifdef WITH_BACKEND
    if (is_cuda_malloc_) {
      auto ms_context = MsContext::GetInstance();
      RETURN_UNEXPECTED_IF_NULL(ms_context);
      auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET), ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
      RETURN_UNEXPECTED_IF_NULL(device_context);
      RETURN_UNEXPECTED_IF_NULL(device_context->device_res_manager_);
      ptr_ = device_context->device_res_manager_->AllocateHostMemory(sz);
    } else {
      ptr_ = std::shared_ptr<void>(::malloc(sz), ::free);
    }
#else
    ptr_ = std::shared_ptr<void>(::malloc(sz), ::free);
#endif
    if (ptr_ == nullptr) {
      return Status(StatusCode::kMDOutOfMemory);
    }
    impl_ = std::make_unique<ArenaImpl>(ptr_.get(), sz);
    if (impl_ == nullptr) {
      return Status(StatusCode::kMDOutOfMemory);
    }
  } catch (std::bad_alloc &e) {
    return Status(StatusCode::kMDOutOfMemory);
  }
  return Status::OK();
}

Arena::Arena(size_t val_in_MB, bool is_cuda_malloc)
    : ptr_(nullptr), size_in_MB_(val_in_MB), is_cuda_malloc_(is_cuda_malloc) {}

Status Arena::CreateArena(std::shared_ptr<Arena> *p_ba, size_t val_in_MB, bool is_cuda_malloc) {
  RETURN_UNEXPECTED_IF_NULL(p_ba);
  auto ba = new (std::nothrow) Arena(val_in_MB, is_cuda_malloc);
  if (ba == nullptr) {
    return Status(StatusCode::kMDOutOfMemory);
  }
  (*p_ba).reset(ba);
  RETURN_IF_NOT_OK(ba->Init());
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
