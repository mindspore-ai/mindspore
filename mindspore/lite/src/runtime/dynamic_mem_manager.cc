/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/dynamic_mem_manager.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"

namespace mindspore {
namespace {
// Alloc memory aligned according to 64 bytes.
static constexpr size_t kMemAlginSize = 64;

// The minimum unit size (512M) of memory block used for dynamic extend.
static constexpr size_t kAllocUnitSize = 536870912;

static constexpr size_t kBlockSize = 1024;
// invalid block index
static constexpr int64_t kInvalidIndex = -1;

size_t Rounded(size_t size) { return (size + kMemAlginSize - 1) & (~(kMemAlginSize - 1)); }

void *Allocate(size_t allocate_size) {
  if (allocate_size > lite::GetMaxMallocSize()) {
    MS_LOG(ERROR) << "MallocData out of max_size, size: " << allocate_size;
    return nullptr;
  }
  void *data = nullptr;
#ifdef _WIN32
  data = _aligned_malloc(allocate_size, kMemAlginSize);
#else
  auto ret = posix_memalign(&data, kMemAlginSize, allocate_size);
  if (UNLIKELY(ret != 0)) {
    MS_LOG(ERROR) << "posix_memalign failed!ret: " << ret;
    return nullptr;
  }
#endif
  if (UNLIKELY(data == nullptr)) {
    MS_LOG(ERROR) << "malloc data failed!";
    return nullptr;
  }
  return data;
}
}  // namespace

DynamicMemManager::DynamicMemManager() {
  blocks_.resize(kBlockSize);
  garbage_block_ = kInvalidIndex;
  auto *block = GetBlock();
  block->data_ = Allocate(kAllocUnitSize);
  if (UNLIKELY(block->data_ == nullptr)) {
    return;
  }
  all_datas_.emplace_back(block->data_);
  block->size_ = kAllocUnitSize;
  free_blocks_.emplace(kAllocUnitSize, block->index_);
}

Block *DynamicMemManager::GetBlock() {
  Block *block;
  if (garbage_block_ != kInvalidIndex) {
    block = &blocks_[garbage_block_];
    garbage_block_ = blocks_[garbage_block_].next_index_;
  } else {
    if (block_count_ >= blocks_.size()) {
      blocks_.resize(blocks_.size() + kBlockSize);
    }
    blocks_[block_count_].index_ = block_count_;
    block = &blocks_[block_count_++];
  }
  block->used_ = false;
  block->ref_count_ = 0;
  block->pre_index_ = kInvalidIndex;
  block->next_index_ = kInvalidIndex;
  return block;
}

void DynamicMemManager::AddGarbageBlock(const int64_t index) {
  blocks_[index].next_index_ = garbage_block_;
  garbage_block_ = index;
}

// malloc memory for data storage
void *DynamicMemManager::Malloc(size_t size) {
  auto rounded_size = Rounded(size);
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = free_blocks_.lower_bound(rounded_size);
  if (iter != free_blocks_.end()) {
    auto index = iter->second;
    free_blocks_.erase(iter);
    auto *block = &blocks_[index];
    block->used_ = true;
    datas_.emplace(block->data_, index);
    if (block->size_ > rounded_size) {
      Block *block_next = GetBlock();
      block_next->size_ = block->size_ - rounded_size;
      block->size_ = rounded_size;
      block_next->data_ = static_cast<int8_t *>(block->data_) + rounded_size;
      block_next->pre_index_ = index;
      auto next_index = block->next_index_;
      block_next->next_index_ = next_index;
      if (next_index != kInvalidIndex) {
        blocks_[next_index].pre_index_ = block_next->index_;
      }
      block->next_index_ = block_next->index_;
      free_blocks_.emplace(block_next->size_, block_next->index_);
    }
    return block->data_;
  }
  // todo kAllocUnitSize can be replaced by config
  auto allocate_size = rounded_size < kAllocUnitSize ? kAllocUnitSize : rounded_size;
  void *data = Allocate(allocate_size);
  if (UNLIKELY(data == nullptr)) {
    return nullptr;
  }
  all_datas_.emplace_back(data);
  Block *block = GetBlock();
  block->size_ = rounded_size;
  block->data_ = data;
  block->used_ = true;
  datas_.emplace(data, block->index_);
  if (allocate_size > rounded_size) {
    Block *block_next = GetBlock();
    block_next->data_ = static_cast<int8_t *>(data) + rounded_size;
    block_next->size_ = allocate_size - rounded_size;
    block_next->pre_index_ = block->index_;
    block->next_index_ = block_next->index_;
    free_blocks_.emplace(block_next->size_, block_next->index_);
  }
  return data;
}

// return memory to the memory pool
void DynamicMemManager::Free(void *ptr) {
  if (UNLIKELY(ptr == nullptr)) {
    return;
  }
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = datas_.find(ptr);
  if (iter == datas_.end()) {
    return;
  }

  auto index = iter->second;
  datas_.erase(iter);
  Block *block = &blocks_[index];
  auto next_index = block->next_index_;
  if (next_index != kInvalidIndex && !blocks_[next_index].used_) {
    EraseFreeBlock(next_index);
    block->size_ += blocks_[next_index].size_;
    auto next_next_index = blocks_[next_index].next_index_;
    if (next_next_index != kInvalidIndex) {
      blocks_[next_next_index].pre_index_ = block->index_;
    }
    block->next_index_ = next_next_index;
    block->used_ = false;
    block->ref_count_ = 0;
    free_blocks_.emplace(block->size_, block->index_);
    AddGarbageBlock(next_index);
  }
  auto pre_index = block->pre_index_;
  if (pre_index != kInvalidIndex && !blocks_[pre_index].used_) {
    EraseFreeBlock(pre_index);
    if (!block->used_) {
      EraseFreeBlock(index);
    }
    blocks_[pre_index].size_ += block->size_;
    next_index = block->next_index_;
    blocks_[pre_index].next_index_ = next_index;
    if (next_index != kInvalidIndex) {
      blocks_[next_index].pre_index_ = pre_index;
    }
    block->used_ = false;
    block->ref_count_ = 0;
    free_blocks_.emplace(blocks_[pre_index].size_, pre_index);
    AddGarbageBlock(index);
  }
  if (block->used_) {
    block->used_ = false;
    block->ref_count_ = 0;
    free_blocks_.emplace(block->size_, block->index_);
  }
}

void DynamicMemManager::EraseFreeBlock(const int64_t index) {
  auto range = free_blocks_.equal_range(blocks_[index].size_);
  for (auto item = range.first; item != range.second; ++item) {
    if (item->second == index) {
      free_blocks_.erase(item);
      break;
    }
  }
}

DynamicMemManager::~DynamicMemManager() {
  MS_LOG(DEBUG) << "~DynamicMemManager() begin.";
  for (auto &&data : all_datas_) {
#ifdef _WIN32
    _aligned_free(data);
#else
    free(data);
#endif
    data = nullptr;
  }
  free_blocks_.clear();
  all_datas_.clear();
  blocks_.clear();
  MS_LOG(DEBUG) << "~DynamicMemManager() end.";
}

int DynamicMemManager::SetRefCount(void *ptr, int ref_count) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = datas_.find(ptr);
  if (iter != datas_.end()) {
    auto index = iter->second;
    blocks_[index].ref_count_ = ref_count;
    return ref_count;
  }
  return -1;
}

int DynamicMemManager::IncRefCount(void *ptr, int ref_count) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = datas_.find(ptr);
  if (iter != datas_.end()) {
    auto index = iter->second;
    blocks_[index].ref_count_ += ref_count;
    return blocks_[index].ref_count_;
  }
  return -1;
}

int DynamicMemManager::DecRefCount(void *ptr, int ref_count) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = datas_.find(ptr);
  if (iter != datas_.end()) {
    auto index = iter->second;
    blocks_[index].ref_count_ -= ref_count;
    return blocks_[index].ref_count_;
  }
  return -1;
}

int DynamicMemManager::RefCount(void *ptr) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = datas_.find(ptr);
  if (iter != datas_.end()) {
    return blocks_[iter->second].ref_count_;
  }
  return -1;
}
}  // namespace mindspore
