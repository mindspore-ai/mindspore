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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_BUDDY_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_BUDDY_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include "minddata/dataset/util/status.h"

using addr_t = int64_t;
using rel_addr_t = int32_t;
using log_t = int;
#define ALLOC_BIT 0x80
#define ONE_BIT 0x40
#define TWO_BIT 0x20
#define MORE_BIT 0x10
#define NOSPACE ((addr_t)(-1))
namespace mindspore {
namespace dataset {
struct BSpaceDescriptor {
  int32_t sig;
  rel_addr_t addr;
  size_t req_size;
  size_t blk_size;
};

class BuddySpace {
 public:
  // C++11 feature. Change STATE into a type safe class with
  // the keyword. Don't take out the keyword 'class'
  enum class STATE { kFree, kAlloc, kEmpty };

  BuddySpace(const BuddySpace &) = delete;

  BuddySpace &operator=(const BuddySpace &) = delete;

  virtual ~BuddySpace();

  Status Alloc(uint64_t sz, BSpaceDescriptor *desc, addr_t *) noexcept;

  void Free(const BSpaceDescriptor *desc);

  uint64_t GetMinSize() const { return min_; }

  uint64_t GetMaxSize() const { return max_; }

  int PercentFree() const;

  friend std::ostream &operator<<(std::ostream &os, const BuddySpace &s);

  static uint64_t NextPowerOf2(uint64_t n) {
    if (n <= 1) {
      return 1;
    }
    n = n - 1;
    while (n & (n - 1)) {
      n = n & (n - 1);
    }
    return n << 1;
  }

  static uint32_t Log2(uint64_t n) {
    uint32_t cnt = 0;
    while (n >>= 1) {
      cnt++;
    }
    return cnt;
  }

  static Status CreateBuddySpace(std::unique_ptr<BuddySpace> *out_bs, int log_min = 15, int num_lvl = 18);

 private:
  rel_addr_t *hint_;
  int *count_;
  char *map_;
  int log_min_;
  int num_lvl_;
  uint64_t min_;
  uint64_t max_;
  std::unique_ptr<uint8_t[]> mem_;
  std::mutex mutex_;

  explicit BuddySpace(int log_min = 15, int num_lvl = 18);

  Status Init();

  addr_t AllocNoLock(const uint64_t sz, BSpaceDescriptor *desc) noexcept;

  void FreeNoLock(const BSpaceDescriptor *desc);

  uint32_t SizeToBlock(const uint64_t sz) const;

  void GetBuddySegState(const rel_addr_t rel_addr, size_t *rel_sz, STATE *st) const;

  void SetBuddySegState(rel_addr_t rel_addr, size_t rel_sz, STATE st);

  void JoinBuddySeg(rel_addr_t addr, size_t blk_sz);

  void TrimBuddySeg(rel_addr_t addr, size_t blk_sz, size_t ask_sz);

  void UnTrimBuddySeg(rel_addr_t addr, size_t blk_sz, size_t ask_sz);

  rel_addr_t AllocBuddySeg(uint32_t req_size) noexcept;

  void FreeBuddySeg(rel_addr_t addr, size_t blk_size, size_t req_size);
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_UTIL_BUDDY_H_
