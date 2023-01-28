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
#include "src/extendrt/dynamic_mem_manager.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "src/common/common.h"

using mindspore::numa::NUMAAdapter;

using mindspore::numa::MemoryInfo;

namespace mindspore {
namespace {
// Alloc memory aligned according to 64 bytes.
static constexpr size_t kMemAlginSize = 64;

// The default unit size (256M) of memory block used for dynamic extend.
static constexpr size_t kAllocUnitSize = 256 * 1024 * 1024;
// The minimum unit size (64M) of memory block used for dynamic extend.
static constexpr size_t kMinimumAllocUnitSize = 64 * 1024 * 1024;
// 16G
static constexpr size_t kMinimumSysMemory = 17179869184;
static constexpr auto kBlockSize = 2048;
// invalid block index
static constexpr int kInvalidIndex = -1;
// invalid numa node id
static constexpr int kInvalidNodeId = -1;
static constexpr int kInvalidRefCount = -1;
size_t Rounded(size_t size) { return (size + kMemAlginSize - 1) & (~(kMemAlginSize - 1)); }
}  // namespace

void *MemOperator::Allocate(size_t rounded_size, int node_id, size_t *allocate_size) {
  static const auto kMaxMallocSize = lite::GetMaxMallocSize();
  static const auto unit_size = kMaxMallocSize < kMinimumSysMemory ? kMinimumAllocUnitSize : kAllocUnitSize;
  auto allocate_tmp_size = rounded_size < unit_size ? unit_size : rounded_size;
  if (allocate_tmp_size > kMaxMallocSize) {
    MS_LOG(ERROR) << "request invalid memory size " << allocate_tmp_size << ", total: " << kMaxMallocSize;
    return nullptr;
  }

  *allocate_size = allocate_tmp_size;
  void *data = nullptr;
#ifdef _WIN32
  data = _aligned_malloc(allocate_tmp_size, kMemAlginSize);
#else
  if (node_id >= 0) {
    data = numa_instance_->Malloc(node_id, static_cast<size_t>(allocate_tmp_size));
    if (MS_UNLIKELY((data == nullptr && allocate_tmp_size > rounded_size))) {
      MS_LOG(WARNING) << "Malloc memory(" << allocate_tmp_size << ") failed! malloc rounded_size(" << rounded_size
                      << ") memory again. node_id: " << node_id;
      allocate_tmp_size = rounded_size;
      *allocate_size = rounded_size;
      data = numa_instance_->Malloc(node_id, rounded_size);
    }
  } else {
    auto ret = posix_memalign(&data, kMemAlginSize, static_cast<size_t>(allocate_tmp_size));
    if (MS_UNLIKELY(ret == ENOMEM && allocate_tmp_size > rounded_size)) {
      MS_LOG(WARNING) << "Malloc memory(" << allocate_tmp_size << ") failed! malloc rounded_size(" << rounded_size
                      << ") memory again.";
      allocate_tmp_size = rounded_size;
      *allocate_size = rounded_size;
      ret = posix_memalign(&data, kMemAlginSize, rounded_size);
    }
    if (MS_UNLIKELY(ret != 0)) {
      MS_LOG(ERROR) << "posix_memalign failed!ret: " << ret << ", node_id: " << node_id
                    << ", request: " << allocate_tmp_size;
      return nullptr;
    }
  }
#endif
  if (MS_UNLIKELY(data == nullptr)) {
    MS_LOG(ERROR) << "malloc data failed!node_id: " << node_id << ", request: " << allocate_tmp_size;
    return nullptr;
  }

  return data;
}

Block *MemOperator::GetBlock() {
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

void MemOperator::AddGarbageBlock(const int64_t index) {
  blocks_[index].next_index_ = garbage_block_;
  garbage_block_ = index;
}

// malloc memory for data storage
void *MemOperator::Malloc(size_t size) {
  auto rounded_size = size == 0 ? 1 : Rounded(size);
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = free_blocks_.lower_bound(rounded_size);
  if (iter != free_blocks_.end()) {
    auto index = iter->second;
    free_blocks_.erase(iter);
    blocks_[index].used_ = true;
    auto data = blocks_[index].data_;
    datas_.emplace(data, index);
    if (blocks_[index].size_ > rounded_size) {
      Block *block_next = GetBlock();
      auto *block = &blocks_[index];
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
    return data;
  }
  // todo kAllocUnitSize can be replaced by config
  size_t allocate_size;
  void *data = Allocate(rounded_size, node_id_, &allocate_size);
  if (MS_UNLIKELY(data == nullptr)) {
    return nullptr;
  }
  all_datas_.emplace(data, allocate_size);
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
void MemOperator::Free(void *ptr) {
  if (MS_UNLIKELY(ptr == nullptr)) {
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

void MemOperator::EraseFreeBlock(const int64_t index) {
  auto range = free_blocks_.equal_range(blocks_[index].size_);
  for (auto item = range.first; item != range.second; ++item) {
    if (item->second == index) {
      free_blocks_.erase(item);
      break;
    }
  }
}

MemOperator::MemOperator(int node_id) {
  numa_instance_ = NUMAAdapter::GetInstance();
  if (node_id >= 0 && numa_instance_->Available()) {
    node_id_ = node_id;
  }

  blocks_.resize(kBlockSize);
  garbage_block_ = kInvalidIndex;
  auto *block = GetBlock();
  size_t allocate_size;
  block->data_ = Allocate(kAllocUnitSize, node_id, &allocate_size);
  if (MS_UNLIKELY(block->data_ == nullptr)) {
    return;
  }
  all_datas_.emplace(block->data_, allocate_size);
  block->size_ = allocate_size;
  free_blocks_.emplace(allocate_size, block->index_);
}

MemOperator::~MemOperator() {
  MS_LOG(DEBUG) << "~MemOperator() begin.";
  for (auto &&data : all_datas_) {
#ifdef _WIN32
    _aligned_free(data.first);
#else
    if (node_id_ >= 0) {
      numa_instance_->Free(data.first, data.second);
    } else {
      free(data.first);
    }
#endif
  }
  free_blocks_.clear();
  all_datas_.clear();
  blocks_.clear();
  MS_LOG(DEBUG) << "~MemOperator() end.";
}

int MemOperator::SetRefCount(void *ptr, int ref_count) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = datas_.find(ptr);
  if (iter != datas_.end()) {
    auto index = iter->second;
    blocks_[index].ref_count_ = ref_count;
    return ref_count;
  }
  return kInvalidRefCount;
}

int MemOperator::IncRefCount(void *ptr, int ref_count) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = datas_.find(ptr);
  if (iter != datas_.end()) {
    auto index = iter->second;
    blocks_[index].ref_count_ += ref_count;
    return blocks_[index].ref_count_;
  }
  return kInvalidRefCount;
}

int MemOperator::DecRefCount(void *ptr, int ref_count) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = datas_.find(ptr);
  if (iter != datas_.end()) {
    auto index = iter->second;
    blocks_[index].ref_count_ -= ref_count;
    return blocks_[index].ref_count_;
  }
  return kInvalidRefCount;
}

int MemOperator::RefCount(void *ptr) {
  std::lock_guard<std::mutex> locker(mutex_);
  auto iter = datas_.find(ptr);
  if (iter != datas_.end()) {
    return blocks_[iter->second].ref_count_;
  }
  return kInvalidRefCount;
}

std::shared_ptr<MemOperator> DynamicMemManager::GetMemOperator(const int node_id) {
  std::map<int, std::shared_ptr<MemOperator>>::iterator iter;
  int numa_node_id = node_id;
  if (numa_node_id < 0) {
    numa_node_id = kInvalidNodeId;
  }

  std::lock_guard<std::mutex> locker(mutex_);
  std::shared_ptr<MemOperator> mem_oper = nullptr;
  iter = nodes_mem_.find(numa_node_id);
  if (iter == nodes_mem_.end()) {
    mem_oper = std::make_shared<MemOperator>(numa_node_id);
    if (MS_UNLIKELY(mem_oper == nullptr)) {
      MS_LOG(ERROR) << "make_shared MemOperator failed!";
      return nullptr;
    }
    nodes_mem_.insert({numa_node_id, mem_oper});
  } else {
    mem_oper = iter->second;
  }
  return mem_oper;
}
}  // namespace mindspore
