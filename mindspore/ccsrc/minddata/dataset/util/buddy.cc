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
#include "minddata/dataset/util/buddy.h"
#include <iomanip>
#include <stdexcept>

#include "minddata/dataset/util/memory_pool.h"
#include "minddata/dataset/util/log_adapter.h"
#include "minddata/dataset/util/system_pool.h"
#include "./securec.h"

inline uint64_t BitLeftShift(uint64_t v, uint64_t n) { return (v << n); }

inline uint64_t BitRightShift(uint64_t v, uint64_t n) { return (v >> n); }

inline uint64_t BitOr(uint64_t rhs, uint64_t lhs) { return rhs | lhs; }

inline uint64_t BitEx(uint64_t rhs, uint64_t lhs) { return rhs ^ lhs; }

inline uint64_t BitAnd(uint64_t rhs, uint64_t lhs) { return rhs & lhs; }

namespace mindspore {
namespace dataset {
Status BuddySpace::Init() {
  if (log_min_ < 0) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "log_min must be positive : " + std::to_string(log_min_));
  }
  if (num_lvl_ < 3 || num_lvl_ > 18) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__,
                  "num_lvl must be between 3 and 18 : " + std::to_string(num_lvl_));
  }
  min_ = BitLeftShift(1, log_min_);
  max_ = BitLeftShift(1, log_min_ + num_lvl_ - 1);
  size_t offset_1 = sizeof(rel_addr_t) * num_lvl_;
  size_t offset_2 = sizeof(int) * num_lvl_ + offset_1;
  size_t offset_3 = sizeof(char) * BitLeftShift(1, num_lvl_ - 3) + offset_2;
  try {
    mem_ = std::make_unique<uint8_t[]>(offset_3);
  } catch (const std::bad_alloc &e) {
    return Status(StatusCode::kMDOutOfMemory);
  }
  (void)memset_s(mem_.get(), offset_3, 0, offset_3);
  auto ptr = mem_.get();
  hint_ = reinterpret_cast<rel_addr_t *>(ptr);
  count_ = reinterpret_cast<int *>((reinterpret_cast<char *>(ptr) + offset_1));
  map_ = reinterpret_cast<char *>(ptr) + offset_2;
  count_[num_lvl_ - 1] = 1;
  map_[0] = BitOr(MORE_BIT, num_lvl_ - 3);
  return Status::OK();
}

Status BuddySpace::Alloc(const uint64_t sz, BSpaceDescriptor *desc, addr_t *p) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  addr_t addr = AllocNoLock(sz, desc);
  if (addr != NOSPACE) {
    *p = addr;
    return Status::OK();
  } else {
    return Status(StatusCode::kMDBuddySpaceFull, "BuddySpace full. Not an error. Please ignore.");
  }
}

addr_t BuddySpace::AllocNoLock(const uint64_t sz, BSpaceDescriptor *desc) noexcept {
  MS_ASSERT(sz <= max_);
  uint32_t reqSize = SizeToBlock(sz);
  rel_addr_t rel_addr = AllocBuddySeg(reqSize);
  if (rel_addr != static_cast<rel_addr_t>(NOSPACE)) {
    (void)memset_s(desc, sizeof(BSpaceDescriptor), 0, sizeof(BSpaceDescriptor));
    desc->sig = static_cast<int>(0xDEADBEEF);
    desc->addr = rel_addr;
    desc->req_size = reqSize;
    desc->blk_size = NextPowerOf2(reqSize);
    return static_cast<addr_t>(rel_addr * min_);
  } else {
    return NOSPACE;
  }
}

void BuddySpace::FreeNoLock(const BSpaceDescriptor *desc) {
  MS_ASSERT(desc->sig == 0XDEADBEEF);
  rel_addr_t rel_addr = desc->addr;
  size_t blk_size = desc->blk_size;
  size_t req_size = desc->req_size;
  FreeBuddySeg(rel_addr, blk_size, req_size);
}

void BuddySpace::Free(const BSpaceDescriptor *desc) {
  std::lock_guard<std::mutex> lock(mutex_);
  return FreeNoLock(desc);
}

std::ostream &operator<<(std::ostream &os, const BuddySpace &s) {
  os << "1 unit = " << s.GetMinSize() << "\n"
     << "Size of buddy space = " << s.GetMaxSize() << "\n"
     << "Number of levels = " << s.num_lvl_ << "\n\n"
     << "Percent free = " << s.PercentFree() << "\n"
     << "Dumping count array : "
     << "\n";
  for (int i = 0; i < s.num_lvl_; i++) {
    os << "[" << i << "] = " << s.count_[i] << " ";
    if (((i + 1) % 4) == 0) {
      os << "\n";
    }
  }
  os << "\n";
  os << "Dumping allocation info:"
     << "\n";
  auto max_addr = static_cast<rel_addr_t>(BitLeftShift(1, s.num_lvl_ - 1));
  rel_addr_t addr = 0;
  while (addr < max_addr) {
    size_t sz = 0;
    BuddySpace::STATE st;
    s.GetBuddySegState(addr, &sz, &st);
    os << "Address : " << std::left << std::setw(8) << addr << " Size : " << std::setw(8) << sz << " State : "
       << ((st == BuddySpace::STATE::kAlloc) ? "ALLOC" : ((st == BuddySpace::STATE::kFree) ? "FREE" : "Unknown"))
       << "\n";
    addr += sz;
  }
  return os;
}

void BuddySpace::GetBuddySegState(const rel_addr_t rel_addr, size_t *rel_sz, STATE *st) const {
  char byte;
  int pos;
  int offset;
  uint64_t val = 0;
  int shift;
  pos = BitRightShift(rel_addr, 2);
  offset = rel_addr % 4;
  shift = offset * 2;
  byte = map_[pos];
  switch (offset) {
    case 0:
      val = byte;
      break;
    case 1:
    case 3:
      if (offset == 1) {
        val = BitLeftShift(BitAnd(byte, 0x30), shift);
      } else {
        val = BitLeftShift(BitAnd(byte, 0x03), shift);
      }
      break;
    case 2:
      val = BitLeftShift(BitAnd(byte, 0x0F), shift);
      break;
  }
  if (BitAnd(val, ONE_BIT)) {
    *rel_sz = 1;
  } else if (BitAnd(val, TWO_BIT)) {
    *rel_sz = 2;
  } else if (BitAnd(val, MORE_BIT)) {
    log_t lg = BitAnd(val, 0x0F);
    *rel_sz = BitLeftShift(1, lg + 2);
  } else {
    *st = STATE::kEmpty;
    return;
  }
  *st = BitAnd(val, ALLOC_BIT) ? STATE::kAlloc : STATE::kFree;
}

void BuddySpace::SetBuddySegState(rel_addr_t rel_addr, size_t rel_sz, STATE st) {
  int clr;
  int mask;
  int pos;
  int offset;
  int val = 0;
  int shift;
  auto log_sz = static_cast<log_t>(Log2(rel_sz));
  pos = BitRightShift(rel_addr, 2);
  offset = rel_addr % 4;
  shift = offset * 2;
  if (rel_sz == 1) {
    val = ONE_BIT;
    mask = 0xC0;
  } else if (rel_sz == 2) {
    val = TWO_BIT;
    mask = 0xF0;
  } else {
    val = BitOr(log_sz - 2, MORE_BIT);
    mask = 0xFF;
  }
  if (st == STATE::kAlloc) {
    val = BitOr(val, ALLOC_BIT);
  } else if (st == STATE::kFree) {
    val = BitAnd(val, ~(static_cast<uint64_t>(ALLOC_BIT)));
  } else if (st == STATE::kEmpty) {
    val = 0;
  }
  clr = static_cast<int>(~(BitRightShift(mask, shift)));
  map_[pos] = static_cast<char>(BitAnd(map_[pos], clr));
  map_[pos] = static_cast<char>(BitOr(map_[pos], BitRightShift(val, shift)));
  if (st == STATE::kAlloc) {
    count_[log_sz]--;
  } else if (st == STATE::kFree) {
    count_[log_sz]++;
    if (rel_addr < hint_[log_sz]) {
      hint_[log_sz] = rel_addr;
    }
  }
}

void BuddySpace::JoinBuddySeg(rel_addr_t addr, size_t blk_sz) {
  while (blk_sz < BitLeftShift(1, num_lvl_)) {
    rel_addr_t buddy = BitEx(addr, blk_sz);
    size_t sz = 0;
    STATE st;
    GetBuddySegState(buddy, &sz, &st);
    if (st == STATE::kFree && sz == blk_sz) {
      auto log_sz = static_cast<log_t>(Log2(blk_sz));
      rel_addr_t left = (buddy < addr) ? buddy : addr;
      rel_addr_t right = left + blk_sz;
      MS_ASSERT(count_[log_sz] >= 2);
      count_[log_sz] -= 2;
      SetBuddySegState(right, blk_sz, STATE::kEmpty);
      SetBuddySegState(left, BitLeftShift(blk_sz, 1), STATE::kFree);
      for (int i = 0; i < log_sz; i++) {
        if (hint_[i] == right) {
          hint_[i] = left;
        }
      }
      addr = left;
      blk_sz <<= 1u;
    } else {
      break;
    }
  }
}

void BuddySpace::TrimBuddySeg(rel_addr_t addr, size_t blk_sz, size_t ask_sz) {
  MS_ASSERT(ask_sz < blk_sz);
  uint32_t inx = Log2(blk_sz);
  size_t remaining_sz = ask_sz;
  for (int i = inx; i > 0; i--) {
    size_t b_size = BitLeftShift(1, i);
    size_t half_sz = BitRightShift(b_size, 1);
    count_[i]--;
    SetBuddySegState(addr, half_sz, STATE::kFree);
    SetBuddySegState(addr + half_sz, half_sz, STATE::kFree);
    if (remaining_sz >= half_sz) {
      SetBuddySegState(addr, half_sz, STATE::kAlloc);
      remaining_sz -= half_sz;
      if (remaining_sz == 0) {
        break;
      }
      addr += half_sz;
    }
  }
}

void BuddySpace::UnTrimBuddySeg(rel_addr_t addr, size_t blk_sz, size_t ask_sz) {
  MS_ASSERT(ask_sz < blk_sz);
  uint32_t inx = Log2(blk_sz);
  size_t remaining_sz = ask_sz;
  for (int i = inx; i > 0; i--) {
    size_t b_size = BitLeftShift(1, i);
    size_t half_sz = BitRightShift(b_size, 1);
    if (remaining_sz >= half_sz) {
#ifdef DEBUG
      {
        size_t sz = 0;
        STATE st;
        GetBuddySegState(addr, &sz, &st);
        MS_ASSERT(sz == half_sz && st == STATE::kAlloc);
      }
#endif
      SetBuddySegState(addr, half_sz, STATE::kFree);
      remaining_sz -= half_sz;
      if (remaining_sz == 0) {
        JoinBuddySeg(addr, half_sz);
        break;
      }
      addr += half_sz;
    }
  }
}

rel_addr_t BuddySpace::AllocBuddySeg(uint32_t req_size) noexcept {
  uint32_t blk_size = NextPowerOf2(req_size);
  int start_inx = static_cast<int>(Log2(blk_size));
  bool found = false;
  rel_addr_t ask_addr = 0;
  auto max_addr = static_cast<rel_addr_t>(BitLeftShift(1, num_lvl_ - 1));
  STATE st;
  size_t sz = 0;
  for (int i = start_inx; !found && i < num_lvl_; i++) {
    MS_ASSERT(count_[i] >= 0);
    if (count_[i] == 0) {
      continue;
    }
    auto blk_sz = static_cast<size_t>(BitLeftShift(1, i));
    ask_addr = hint_[i];
    while (ask_addr < max_addr && !found) {
      GetBuddySegState(ask_addr, &sz, &st);
      if (st == STATE::kFree && sz == blk_sz) {
        found = true;
      } else {
        MS_ASSERT(st != STATE::kEmpty);
        ask_addr += ((sz > blk_sz) ? sz : blk_sz);
      }
    }
  }
  if (found) {
    if (sz > req_size) {
      TrimBuddySeg(ask_addr, sz, req_size);
    } else {
      SetBuddySegState(ask_addr, sz, STATE::kAlloc);
      hint_[start_inx] = ask_addr;
    }
    return ask_addr;
  } else {
    return static_cast<rel_addr_t>(NOSPACE);
  }
}

void BuddySpace::FreeBuddySeg(rel_addr_t addr, size_t blk_size, size_t req_size) {
  if (req_size == blk_size) {
#ifdef DEBUG
    {
      size_t sz = 0;
      STATE st;
      GetBuddySegState(addr, &sz, &st);
    }
#endif
    SetBuddySegState(addr, blk_size, STATE::kFree);
    JoinBuddySeg(addr, blk_size);
  } else {
    UnTrimBuddySeg(addr, blk_size, req_size);
  }
}

int BuddySpace::PercentFree() const {
  uint64_t total_free_sz = 0;
  uint64_t max_sz_in_unit = BitLeftShift(1, num_lvl_ - 1);
  // Go through the count array without lock
  for (int i = 0; i < num_lvl_; i++) {
    int cnt = count_[i];
    if (cnt == 0) {
      continue;
    }
    uint64_t blk_sz = BitLeftShift(1, i);
    total_free_sz += (blk_sz * cnt);
  }
  return static_cast<int>(static_cast<float>(total_free_sz) / static_cast<float>(max_sz_in_unit) * 100);
}

BuddySpace::BuddySpace(int log_min, int num_lvl)
    : hint_(nullptr), count_(nullptr), map_(nullptr), log_min_(log_min), num_lvl_(num_lvl), min_(0), max_(0) {}

BuddySpace::~BuddySpace() {
  hint_ = nullptr;
  count_ = nullptr;
  map_ = nullptr;
}

Status BuddySpace::CreateBuddySpace(std::unique_ptr<BuddySpace> *out_bs, int log_min, int num_lvl) {
  Status rc;
  auto bs = new (std::nothrow) BuddySpace(log_min, num_lvl);
  if (bs == nullptr) {
    return Status(StatusCode::kMDOutOfMemory);
  }
  rc = bs->Init();
  if (rc.IsOk()) {
    (*out_bs).reset(bs);
  } else {
    delete bs;
  }
  return rc;
}
}  // namespace dataset
}  // namespace mindspore
