/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "include/runtime/memory/mem_pool/abstract_dynamic_mem_pool.h"

#include <stdio.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <numeric>

#include "include/runtime/memory/mem_pool/mem_pool_util.h"
#include "include/utils/utils.h"

namespace mindspore {
namespace device {
MemBuf::MemBuf(size_t size, void *addr, uint32_t stream_id, MemBlock *mem_block, MemBufStatus status)
    : prev_(nullptr),
      next_(nullptr),
      size_(size),
      addr_(addr),
      stream_id_(stream_id),
      mem_block_(mem_block),
      status_(status) {}

MemBuf::~MemBuf() {}

MemBufAllocator::~MemBufAllocator() {
  MS_LOG(INFO) << "MemBufAllocator[" << this << "] : " << BriefInfo() << " deconstruct.";
  for (auto &mem_block : mem_blocks_) {
    delete mem_block;
  }
  mem_blocks_.clear();
  for (auto mem_buf : free_mem_bufs_) {
    delete mem_buf;
  }
  free_mem_bufs_.clear();
  for (auto mem_buf : eager_free_mem_bufs_) {
    delete mem_buf;
  }
  eager_free_mem_bufs_.clear();
  delete search_key_;
}

void MemBufAllocator::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Release device resource for allocator, " << BriefInfo()
               << ", mem_blocks_ size : " << mem_blocks_.size() << ".";
  for (auto mem_block : mem_blocks_) {
    MS_LOG(INFO) << "Clean mem block : " << mem_block->ToJson() << ".";
    (void)mem_block_cleaner_(mem_block);
  }
  for (auto mem_block : mem_blocks_) {
    MS_LOG(INFO) << "Delete mem block : " << mem_block->ToJson() << ".";
    delete mem_block;
  }
  mem_blocks_.clear();

  MS_LOG(INFO) << "Free mem buf size : " << free_mem_bufs_.size() << ".";
  for (auto mem_buf : free_mem_bufs_) {
    delete mem_buf;
  }
  free_mem_bufs_.clear();

  MS_LOG(INFO) << "Eager free mem buf size : " << eager_free_mem_bufs_.size() << ".";
  for (auto mem_buf : eager_free_mem_bufs_) {
    delete mem_buf;
  }
  eager_free_mem_bufs_.clear();
}

MemBuf *MemBufAllocator::Malloc(size_t size) {
  // Malloc with expand block first.
  if (MS_UNLIKELY(mem_blocks_.empty())) {
    return MallocExpandBlock(size);
  }

  search_key_->size_ = size;
  auto it = free_mem_bufs_.lower_bound(search_key_);
  MemBuf *candidate = nullptr;
  // 1. Try to find in free mem bufs.
  if (MS_LIKELY(it != free_mem_bufs_.end())) {
    candidate = *it;
    (void)free_mem_bufs_.erase(it);
    return MapAndSplitMemBuf(candidate, size);
  }
  // 2. Try to search available buf, free and eager free buf.
  candidate = SearchAvailableMemBuf(size);
  if (MS_UNLIKELY(candidate != nullptr)) {
    return candidate;
  }
  // 3. Try to find in eager free mem bufs.
  it = eager_free_mem_bufs_.lower_bound(search_key_);
  if (it != eager_free_mem_bufs_.end()) {
    candidate = *it;
    EraseEagerFreeBuf(candidate);
    return MapAndSplitMemBuf(candidate, size);
  }

  return nullptr;
}

inline MemBuf *MemBufAllocator::SearchAvailableMemBuf(size_t size) {
  if (!enable_eager_free_ || MS_UNLIKELY(is_customized_)) {
    return nullptr;
  }
  // Search from back to front, because the free mem buf is sorted by size.
  // More efficient way is to search more candidates, do it in the next version.
  for (auto backward_it = free_mem_bufs_.rbegin(); backward_it != free_mem_bufs_.rend(); backward_it++) {
    auto mem_buf = *backward_it;
    auto next_buf = mem_buf->next_;
    if (next_buf == nullptr || next_buf->status_ != MemBufStatus::kMemBufEagerFree ||
        mem_buf->size_ + next_buf->size_ < size) {
      continue;
    }

    // Located candidates, try map and split.
    auto need_map_size = size - mem_buf->size_;
    auto mapped_size = mem_mapper_(need_map_size, next_buf->addr_);
    if (mapped_size != need_map_size) {
      MS_LOG(WARNING) << "Map mem buf : " << mem_buf->ToJson() << ", next buf : " << next_buf->ToJson()
                      << ", size : " << size << ", need_map_size : " << need_map_size
                      << ", mapped_size : " << mapped_size << " failed.";
      return nullptr;
    }
    // Update mem buf.
    free_mem_bufs_.erase(mem_buf);
    mem_buf->size_ = size;
    mem_buf->status_ = MemBufStatus::kMemBufUsed;
    // Remove eager free buf and try update it.
    EraseEagerFreeBuf(next_buf);
    next_buf->addr_ = static_cast<uint8_t *>(next_buf->addr_) + need_map_size;
    next_buf->size_ = next_buf->size_ - need_map_size;
    // If next buf is empty, remove it or update remain eager free mem buf.
    if (next_buf->size_ == 0) {
      mem_buf->next_ = next_buf->next_;
      if (next_buf->next_ != nullptr) {
        next_buf->next_->prev_ = mem_buf;
      }
      delete next_buf;
    } else {
      InsertEagerFreeBuf(next_buf);
    }
    return mem_buf;
  }
  return nullptr;
}

void MemBufAllocator::MergeMemBuf(MemBuf *src, MemBuf *dst) {
  if (dst->prev_ == src) {
    dst->addr_ = src->addr_;
    dst->size_ += src->size_;
    dst->prev_ = src->prev_;
    if (dst->prev_ != nullptr) {
      dst->prev_->next_ = dst;
    }
  } else {
    dst->size_ += src->size_;
    dst->next_ = src->next_;
    if (dst->next_ != nullptr) {
      dst->next_->prev_ = dst;
    }
  }

  const auto status = src->status_;
  if (status == MemBufStatus::kMemBufIdle) {
    const auto ret = free_mem_bufs_.erase(src);
    if (ret == 0) {
      MS_LOG(ERROR) << "Erase next buf : " << src->ToJson() << " failed.";
    }
  } else if (status == MemBufStatus::kMemBufEagerFree) {
    EraseEagerFreeBuf(src);
  }
  delete src;
}

bool MemBufAllocator::Free(MemBuf *mem_buf, MemBufStatus target_status) {
  // Change mem buf status to used by event, and wait for event to free.
  if (MS_UNLIKELY(!mem_buf->IsEventNotUsed())) {
    mem_buf->status_ = MemBufStatus::kMemBufUsedByEvent;
    return false;
  }

  mem_buf->status_ = target_status;
  // Try to merge from prev.
  const auto prev_buf = mem_buf->prev_;
  if (MS_LIKELY(prev_buf != nullptr && prev_buf->status_ == target_status)) {
    MergeMemBuf(prev_buf, mem_buf);
  }
  // Try to merge from next.
  const auto next_buf = mem_buf->next_;
  if (MS_LIKELY(next_buf != nullptr && next_buf->status_ == target_status)) {
    MergeMemBuf(next_buf, mem_buf);
  }

  if (target_status == MemBufStatus::kMemBufIdle) {
    (void)free_mem_bufs_.emplace(mem_buf);
  } else if (target_status == MemBufStatus::kMemBufEagerFree) {
    InsertEagerFreeBuf(mem_buf);
  }

  return true;
}

MemBuf *MemBufAllocator::MallocExpandBlock(size_t size) {
  MemBlock *mem_block = ExpandBlock(size);
  if (mem_block == nullptr) {
    return nullptr;
  }
  MemBuf *candidate = new MemBuf(
    mem_block->size_, mem_block->addr_, mem_block->stream_id_, mem_block,
    MS_LIKELY(!is_customized_) && enable_eager_free_ ? MemBufStatus::kMemBufEagerFree : MemBufStatus::kMemBufIdle);
  if (candidate->size_ < size) {
    if (candidate->status_ == MemBufStatus::kMemBufIdle) {
      (void)free_mem_bufs_.emplace(candidate);
    } else {
      InsertEagerFreeBuf(candidate);
    }
    MS_LOG(WARNING) << "Candidate size: " << candidate->size_ << " is less than required size : " << size << ".";
    return nullptr;
  }

  return MapAndSplitMemBuf(candidate, size);
}

void MemBufAllocator::Initialize(size_t size) {
  MS_LOG(INFO) << "Initialize allocator : " << BriefInfo() << " with size : " << size << ".";
  if (enable_eager_free_ || MS_UNLIKELY(is_customized_)) {
    MS_LOG(INFO) << "Skip initialization of allocator, since vmm is enabled.";
    return;
  }
  MemBlock *mem_block = ExpandBlock(size);
  if (mem_block == nullptr) {
    MS_LOG(WARNING) << "Initialize allocator failed, size : " << size << ".";
    return;
  }
  MemBuf *mem_buf =
    new MemBuf(mem_block->size_, mem_block->addr_, mem_block->stream_id_, mem_block, MemBufStatus::kMemBufIdle);
  (void)free_mem_bufs_.emplace(mem_buf);
}

const std::pair<size_t, size_t> MemBufAllocator::FreeIdleMemsByEagerFree() {
  // Free all idle mem bufs.
  size_t eager_free_size = 0;
  for (auto mem_buf : free_mem_bufs_) {
    eager_free_size += mem_buf->size_;
    Free(mem_buf, MemBufStatus::kMemBufEagerFree);
  }
  free_mem_bufs_.clear();
  // Do eager free on eager free mem bufs.
  size_t real_free_size = 0;
  for (auto mem_buf : eager_free_mem_bufs_) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Eager free mem buf : " << mem_buf << ", details : " << mem_buf->ToJson()
                                         << ".";
    real_free_size += mem_eager_freer_(mem_buf->addr_, mem_buf->size_);
  }
  MS_LOG(INFO) << "Free idle mems by eager free, eager_free_size : " << eager_free_size
               << ", real_free_size : " << real_free_size << ".";
  return std::make_pair(eager_free_size, real_free_size);
}

size_t MemBufAllocator::ReleaseFreeBlocks() {
  size_t release_size = 0;
  for (auto iter = mem_blocks_.begin(); iter != mem_blocks_.end();) {
    auto mem_block = *iter;
    MemBuf mem_buf(mem_block->size_, mem_block->addr_, mem_block->stream_id_, mem_block, MemBufStatus::kMemBufIdle);
    // Judge if mem block in free mem bufs.
    auto &&it = free_mem_bufs_.find(&mem_buf);
    if (it == free_mem_bufs_.end()) {
      iter++;
      continue;
    }
    auto mem_buf_it = *it;
    if (mem_buf_it->addr_ == mem_block->addr_ && mem_buf_it->size_ == mem_block->size_) {
      MS_LOG(INFO) << "Release mem block : " << mem_block->ToJson() << ".";
      bool ret = mem_block_cleaner_(mem_block);
      if (!ret) {
        MS_LOG(WARNING) << "Clean mem block : " << mem_block->ToJson() << " failed.";
        iter++;
        continue;
      }
      free_mem_bufs_.erase(it);
      delete mem_buf_it;
      release_size += mem_block->size_;
      delete mem_block;
      iter = mem_blocks_.erase(iter);
    } else {
      iter++;
    }
  }
  return release_size;
}

std::string MemBufAllocator::DumpStateInfo() const {
  std::stringstream ss;
  ss << "Dump state info for " << BriefInfo() << "\n";
  ss << "\tMem block info - total count : " << mem_blocks_.size() << "\n";
  size_t mem_block_size = 0;
  for (const auto mem_block : mem_blocks_) {
    ss << "\t\t" << mem_block->ToJson() << "\n";
    mem_block_size += mem_block->size_;
  }
  ss << "\tMem block total size : " << mem_block_size << "\n";

  auto mem_buf_accumulator = [](size_t size, MemBuf *const &left) { return left->size_ + size; };
  size_t init_value = 0;
  size_t free_size = std::accumulate(free_mem_bufs_.begin(), free_mem_bufs_.end(), init_value, mem_buf_accumulator);
  ss << "\tFree mem buf total size : " << free_size << ", total count : " << free_mem_bufs_.size() << "\n";

  size_t eager_free_size =
    std::accumulate(eager_free_mem_bufs_.begin(), eager_free_mem_bufs_.end(), init_value, mem_buf_accumulator);
  ss << "\tEager free mem buf total size : " << eager_free_size << ", total count : " << eager_free_mem_bufs_.size()
     << "\n";

  return ss.str();
}

std::string MemBufAllocator::DumpDebugInfo() const {
  std::stringstream ss;
  ss << "Dump debug info for " << BriefInfo() << "\n";
  ss << "\tMem block info - total count : " << mem_blocks_.size() << "\n";
  for (const auto mem_block : mem_blocks_) {
    ss << "\t\t" << mem_block->ToJson() << "\n";
  }
  ss << "\tFree mem buf info - total count : " << free_mem_bufs_.size() << "\n";
  for (const auto mem_buf : free_mem_bufs_) {
    ss << "\t\t" << mem_buf->ToJson() << "\n";
  }
  ss << "\tEager free mem buf info - total count : " << eager_free_mem_bufs_.size() << "\n";
  for (const auto mem_buf : eager_free_mem_bufs_) {
    ss << "\t\t" << mem_buf->ToJson() << "\n";
  }
  return ss.str();
}

inline MemBuf *MemBufAllocator::MapAndSplitMemBuf(MemBuf *candidate, size_t size) {
  size_t remaining_size = candidate->size_ - size;
  // Mmap memory first.
  if (candidate->status_ == MemBufStatus::kMemBufEagerFree) {
    size_t map_size = (remaining_size >= kDynamicMemAlignSize) ? size : candidate->size_;
    auto mapped_size = mem_mapper_(map_size, candidate->addr_);
    if (mapped_size != map_size) {
      MS_LOG(INFO) << "Mapped_size : " << mapped_size << " is not equal to required size : " << map_size
                   << ", mem buf info : " << candidate->ToJson() << ".";
      InsertEagerFreeBuf(candidate);
      return nullptr;
    }
  }

  bool need_split = remaining_size >= kDynamicMemAlignSize;
  // Try to split mem buf.
  if (MS_LIKELY(need_split)) {
    void *remaining_addr = static_cast<uint8_t *>(candidate->addr_) + size;
    auto remaining_buf =
      new MemBuf(remaining_size, remaining_addr, candidate->stream_id_, candidate->mem_block_, candidate->status_);

    auto next = candidate->next_;
    if (next != nullptr) {
      next->prev_ = remaining_buf;
      remaining_buf->next_ = next;
    }
    candidate->next_ = remaining_buf;
    remaining_buf->prev_ = candidate;
    if (remaining_buf->status_ == MemBufStatus::kMemBufIdle) {
      (void)free_mem_bufs_.emplace(remaining_buf);
    } else {
      InsertEagerFreeBuf(remaining_buf);
    }

    // Update candidate size.
    candidate->size_ = size;
  }

  candidate->status_ = MemBufStatus::kMemBufUsed;
  // Update mem block usage.
  candidate->mem_block_->UpdateBorderAddr(candidate);

  return candidate;
}

inline MemBlock *MemBufAllocator::ExpandBlock(size_t size) {
  MemBlock *mem_block = mem_block_expander_(size);
  if (mem_block == nullptr) {
    MS_LOG(WARNING) << "Expand block failed, expand size : " << size << ", memory is not enough.";
    return nullptr;
  }

  if (mem_block->size_ < size) {
    MS_LOG(WARNING) << "Expand block failed, expand size : " << mem_block->size_
                    << " is less than require size : " << size << ".";
  }

  (void)mem_blocks_.emplace_back(mem_block);
  return mem_block;
}

AbstractDynamicMemPool::AbstractDynamicMemPool() {}

void AbstractDynamicMemPool::Initialize(size_t init_size, size_t increase_size, size_t max_size) {
  const auto &value = memory::mem_pool::GetAllocConfigValue(memory::mem_pool::kAllocDefragMemoryStepFreq);
  std::string dump_key = "dump";
  enable_dump_memory_ = value.find(dump_key) != std::string::npos;

  if (init_size == 0) {
    MS_LOG(INFO) << "Skip initialization of memory pool since init size is not configured.";
    return;
  }

  LockGuard lock(lock_);
  MS_LOG(INFO) << "Initialize dynamic memory pool, init size : " << init_size << ", increase size : " << increase_size
               << ", max size : " << max_size << ".";
  init_size_ = init_size >> 1;
  increase_size_ = increase_size;
  max_size_ = max_size;

  // Do initialization with init size.
  auto persistent_allocator = GetMemBufAllocator(init_size_, true, kDefaultStreamIndex);
  persistent_allocator->Initialize(AlignMemorySize(init_size_));
  auto common_allocator = GetMemBufAllocator(init_size_, false, kDefaultStreamIndex);
  common_allocator->Initialize(AlignMemorySize(init_size_));
}

void AbstractDynamicMemPool::ReleaseDeviceRes() {
  LockGuard lock(lock_);
  DumpDynamicMemPoolStateInfo();
  for (const auto &iter : stream_pair_mem_bufs_) {
    auto size = iter.second.size();
    MS_LOG(INFO) << "Event referred stream_pair_mem_bufs_[" << iter.first.first << "-" << iter.first.second
                 << "], size : " << size << ".";
  }
  // Clear map of address to mem buf.
  for (const auto &iter : addr_mem_buf_allocators_) {
    auto mem_buf = iter.second.first;
    delete mem_buf;
  }
  addr_mem_buf_allocators_.clear();

  MS_LOG(INFO) << "Release device resource for " << GetMemoryPoolType() << " : " << mem_stat_ptr_->ToReadableString()
               << ".";
  for (const auto &stream_id_allocator : stream_id_allocators_) {
    const auto &allocator = stream_id_allocator.second;
    allocator->ReleaseDeviceRes();
  }
  for (const auto &customized_allocator : customized_allocators_) {
    const auto &allocator = customized_allocator.second;
    allocator->ReleaseDeviceRes();
  }
  stream_id_allocators_.clear();
  customized_allocators_.clear();
  stream_pair_mem_bufs_.clear();
  mem_stat_ptr_->Reset();
}

/**
 * @brief Alloc tensor mem.
 * Allocation follow steps below:
 *    1 align size
 *    2 find from current allocator, if failed transfer to 3
 *    3 find from another allocator, if failed transfer to 4
 *    4 do eager free and find from current allocator again, if failed transfer to 5
 *    5 expand block
 */
DeviceMemPtr AbstractDynamicMemPool::AllocTensorMem(size_t size, bool from_persistent_mem, bool, uint32_t stream_id) {
  size_t align_size = AlignMemorySize(size);
  LockGuard lock(lock_);
  auto &&mem_buf_allocator = AllocMemBuf(align_size, from_persistent_mem, stream_id);
  if (MS_UNLIKELY(mem_buf_allocator.first == nullptr)) {
    // Dump mem pool state info when alloc tensor failed.
    DumpDynamicMemPoolStateInfo();
    return nullptr;
  }

  (void)addr_mem_buf_allocators_.emplace(mem_buf_allocator.first->addr_, mem_buf_allocator);
  return mem_buf_allocator.first->addr_;
}

/**
 * @brief Alloc mem buf.
 * Strategy when vmm is disable:
 *    Persistent memory:  First malloc form its own pool, if fails, try to malloc from common pool.
 *    Common memory:  First malloc from its own pool, if fails, it will try to expand the pool.
 *                    If the expansion fails, try to malloc from persistent pool.
 */
std::pair<MemBuf *, MemBufAllocator *> AbstractDynamicMemPool::AllocMemBuf(size_t align_size, bool from_persistent_mem,
                                                                           uint32_t stream_id) {
  auto allocator = GetMemBufAllocator(align_size, from_persistent_mem, stream_id);
  auto mem_buf = allocator->Malloc(align_size);
  if (MS_UNLIKELY(mem_buf == nullptr)) {
    // Enable malloc from another allocator when from_persistent_mem is true and vmm is not enabled.
    if (!enable_vmm_ && from_persistent_mem && MS_LIKELY(!enable_custom_allocator_)) {
      auto common_allocator = GetMemBufAllocator(align_size, false, stream_id);
      mem_buf = common_allocator->Malloc(align_size);
      allocator = common_allocator;
    }

    if (MS_UNLIKELY(mem_buf == nullptr)) {
      if ((enable_vmm_ || IsEnableEagerFree()) && MS_LIKELY(!enable_custom_allocator_)) {
        WaitPipelineHelper();
        if (!SyncAllStreams()) {
          MS_LOG(INTERNAL_EXCEPTION) << "Sync all streams failed.";
          return std::make_pair(nullptr, nullptr);
        }
        (void)FreeIdleMemsByEagerFree();
        mem_buf = allocator->Malloc(align_size);
      }
      if (MS_UNLIKELY(mem_buf == nullptr)) {
        mem_buf = allocator->MallocExpandBlock(align_size);
        if (MS_UNLIKELY(mem_buf == nullptr)) {
          if (MS_LIKELY(!from_persistent_mem) && MS_LIKELY(!enable_custom_allocator_)) {
            // Common pool expand block failed, try to malloc from persistent pool.
            auto persistent_allocator = GetMemBufAllocator(align_size, true, stream_id);
            mem_buf = persistent_allocator->Malloc(align_size);
            if (MS_LIKELY(mem_buf != nullptr)) {
              allocator = persistent_allocator;
            }
          }

          if (MS_UNLIKELY(mem_buf == nullptr)) {
            MS_LOG(INFO) << "Alloc tensor mem failed and try to sync all events to release memory.";
            (void)DoSyncAllEvents();
            mem_buf = allocator->Malloc(align_size);
            if (MS_UNLIKELY(mem_buf == nullptr)) {
              return std::make_pair(nullptr, nullptr);
            }
          }
        }
      }
    }
  }

  // Update stat.
  mem_stat_ptr_->used_size_ += mem_buf->size_;
  mem_stat_ptr_->UpdatePeakSize(enable_vmm_, GetVmmUsedMemSize());
  return std::make_pair(mem_buf, allocator);
}

std::vector<DeviceMemPtr> AbstractDynamicMemPool::AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                                           uint32_t stream_id) {
  std::vector<DeviceMemPtr> device_addr_list;
  size_t total_size = std::accumulate(size_list.begin(), size_list.end(), static_cast<size_t>(0));
  // Pre-alloc the one whole piece memory.
  auto device_addr = AbstractDynamicMemPool::AllocTensorMem(total_size, false, false, stream_id);
  if (device_addr == nullptr) {
    return device_addr_list;
  }

  (void)device_addr_list.emplace_back(device_addr);
  if (size_list.size() == 1) {
    return device_addr_list;
  }

  // Try to split mem bufs.
  LockGuard lock(lock_);
  auto &&it = addr_mem_buf_allocators_.find(device_addr);
  if (it != addr_mem_buf_allocators_.end()) {
    auto mem_buf = it->second.first;
    auto allocator = it->second.second;
    mem_buf->size_ = size_list[0];
    MemBuf *prev_mem_buf = mem_buf;
    void *next_addr = static_cast<uint8_t *>(mem_buf->addr_) + size_list[0];
    total_size -= size_list[0];
    for (size_t i = 1; i < size_list.size(); i++) {
      auto new_mem_buf = new MemBuf(size_list[i], next_addr, stream_id, mem_buf->mem_block_, MemBufStatus::kMemBufUsed);
      new_mem_buf->Link(prev_mem_buf, prev_mem_buf->next_);
      (void)addr_mem_buf_allocators_.emplace(new_mem_buf->addr_, std::make_pair(new_mem_buf, allocator));
      // Update result.
      (void)device_addr_list.emplace_back(next_addr);
      // Update next addr and prev mem buf.
      if (i < size_list.size() - 1) {
        next_addr = static_cast<uint8_t *>(next_addr) + size_list[i];
        total_size -= size_list[i];
        prev_mem_buf = new_mem_buf;
      } else {
        // Update last mem buf
        if (total_size != size_list[i]) {
          MS_LOG(WARNING) << "Remain size : " << total_size << " is not equal to last size : " << size_list[i] << ".";
          new_mem_buf->size_ = total_size;
        }
      }
    }
  } else {
    DumpDynamicMemPoolDebugInfo();
    // Unreachable routine.
    MS_LOG(INTERNAL_EXCEPTION) << "Find addr : " << device_addr << " failed.";
  }

  return device_addr_list;
}

// The main program entry of memory free.
void AbstractDynamicMemPool::FreeTensorMem(const DeviceMemPtr &device_addr) {
  LockGuard lock(lock_);
  (void)DoFreeTensorMem(device_addr);
}

bool AbstractDynamicMemPool::IsNotEventUsedTensorMem(const DeviceMemPtr &device_addr) {
  auto &&it = addr_mem_buf_allocators_.find(device_addr);
  if (MS_LIKELY(it != addr_mem_buf_allocators_.end())) {
    auto mem_buf = it->second.first;
    return mem_buf->IsEventNotUsed();
  }
  return false;
}

// The main program entry of memory free.
bool AbstractDynamicMemPool::DoFreeTensorMem(const DeviceMemPtr &device_addr) {
  void *addr = device_addr;
  auto &&it = addr_mem_buf_allocators_.find(device_addr);
  if (MS_LIKELY(it != addr_mem_buf_allocators_.end())) {
    auto allocator = it->second.second;
    auto mem_buf = it->second.first;
    auto free_size = mem_buf->size_;
    if (MS_LIKELY(allocator->Free(mem_buf))) {
      mem_stat_ptr_->used_size_ -= free_size;
      (void)addr_mem_buf_allocators_.erase(it);
      return true;
    }
  } else {
    // This may be normal case.
    MS_LOG(INFO) << "Free tensor mem failed, can not find address : " << addr << ".";
  }
  return false;
}

inline MemBufAllocator *AbstractDynamicMemPool::GetMemBufAllocator(size_t size, bool from_persistent_mem,
                                                                   uint32_t stream_id) {
  const AllocatorInfo key{stream_id, from_persistent_mem, UseSmallPool(size, from_persistent_mem)};
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Get allocator, " << key.ToString() << ".";

  MemBufAllocatorPtr allocator = nullptr;
  // Enable malloc from custom allocator when enable_custom_allocator_ is true
  if (MS_UNLIKELY(enable_custom_allocator_)) {
    auto &&it = customized_allocators_.find(stream_id);
    if (it == customized_allocators_.end()) {
      allocator = GenerateCustomAllocator(stream_id);
      (void)customized_allocators_.emplace(stream_id, allocator);
    } else {
      allocator = it->second;
    }
    return allocator.get();
  }

  auto &&it = stream_id_allocators_.find(key);
  if (it == stream_id_allocators_.end()) {
    allocator = GenerateAllocator(key);
    (void)stream_id_allocators_.emplace(key, allocator);
  } else {
    allocator = it->second;
  }
  return allocator.get();
}

// Keep addrs is in free addrs, so here find mem bufs first.
// And then, traverse keep addrs and spilt candidates.
void AbstractDynamicMemPool::FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                const std::vector<DeviceMemPtr> &keep_addrs,
                                                const std::vector<size_t> &keep_addr_sizes) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Free part tensor mems.";
  LockGuard lock(lock_);
  (void)DoFreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

std::vector<MemBuf *> AbstractDynamicMemPool::DoFreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                                   const std::vector<DeviceMemPtr> &keep_addrs,
                                                                   const std::vector<size_t> &keep_addr_sizes) {
  std::vector<MemBuf *> mem_bufs;
  std::map<void *, std::pair<MemBuf *, MemBufAllocator *>> candidates;
  for (const auto &free_addr : free_addrs) {
    auto &&it = addr_mem_buf_allocators_.find(free_addr);
    if (it != addr_mem_buf_allocators_.end()) {
      (void)candidates.emplace(it->first, it->second);
    } else {
      // This is illegal routine, but level0 case entered.
      MS_LOG(INFO) << "Find address : " << free_addr << " failed.";
    }
  }

  std::set<std::uintptr_t> processed_keep_addrs;
  for (size_t i = 0; i < keep_addrs.size(); i++) {
    auto keep_addr = keep_addrs[i];
    std::uintptr_t keep_addr_to_size = reinterpret_cast<std::uintptr_t>(keep_addr);
    if (processed_keep_addrs.count(keep_addr_to_size) > 0) {
      MS_LOG(INFO) << "Duplicate keep address : " << keep_addr << ".";
      continue;
    }
    (void)processed_keep_addrs.insert(keep_addr_to_size);
    auto &&it = candidates.upper_bound(keep_addr);
    if (it == candidates.begin()) {
      MS_LOG(WARNING) << "Locate keep addr : " << keep_addr << " failed.";
      continue;
    }
    auto iter = --it;
    auto mem_buf = iter->second.first;
    auto allocator = iter->second.second;
    std::uintptr_t base_start = reinterpret_cast<std::uintptr_t>(mem_buf->addr_);
    std::uintptr_t base_end = base_start + mem_buf->size_;
    std::uintptr_t keep_start = keep_addr_to_size;
    std::uintptr_t keep_end = keep_start + keep_addr_sizes[i];
    // Since free part tensor mem may double free keep addr, continue for these keep addrs.
    if (keep_start >= base_end) {
      MS_LOG(WARNING) << "Check range error, base start : " << base_start << ", base end : " << base_end
                      << ", keep start : " << keep_start << ", keep end : " << keep_end << ".";
      continue;
    }
    // Split candidates. If keep start equal to base start, split mem buf into two parts, or three parts.
    // First construct keep mem buf and set it into addr_mem_buf_allocators_, then process head buf and tail buf.
    MemBuf *keep_mem_buf = nullptr;
    if (keep_start == base_start) {
      keep_mem_buf = mem_buf;
      keep_mem_buf->size_ = keep_addr_sizes[i];
      // Remove keep addr since keep start equal to base start, no need to free keep addr any more.
      (void)candidates.erase(mem_buf->addr_);
    } else {
      // Split middle mem buf.
      keep_mem_buf =
        new MemBuf(keep_addr_sizes[i], keep_addr, mem_buf->stream_id_, mem_buf->mem_block_, mem_buf->status_);
      keep_mem_buf->Link(mem_buf, mem_buf->next_);
      (void)addr_mem_buf_allocators_.emplace(keep_addr, std::make_pair(keep_mem_buf, allocator));
      std::uintptr_t prev_remain_size = keep_start - base_start;
      mem_buf->size_ = prev_remain_size;
    }
    (void)mem_bufs.emplace_back(keep_mem_buf);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "keep_mem_buf : " << keep_mem_buf->ToJson() << ".";
    // Process last mem buf.
    if (keep_end < base_end) {
      void *last_addr = static_cast<uint8_t *>(keep_mem_buf->addr_) + keep_mem_buf->size_;
      auto last_mem_buf = new MemBuf(base_end - keep_end, last_addr, keep_mem_buf->stream_id_, keep_mem_buf->mem_block_,
                                     mem_buf->status_);
      last_mem_buf->Link(keep_mem_buf, keep_mem_buf->next_);
      (void)addr_mem_buf_allocators_.emplace(last_mem_buf->addr_, std::make_pair(last_mem_buf, allocator));
      if (candidates.count(last_mem_buf->addr_) > 0) {
        MS_LOG(INFO) << "Duplicate address : " << last_mem_buf->addr_ << ".";
      }
      MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "last mem buf : " << last_mem_buf->ToJson() << ".";
      (void)candidates.emplace(last_mem_buf->addr_, std::make_pair(last_mem_buf, allocator));
    }
  }
  for (const auto &candidate : candidates) {
    auto mem_buf = candidate.second.first;
    if (!AbstractDynamicMemPool::DoFreeTensorMem(mem_buf->addr_)) {
      DumpDynamicMemPoolStateInfo();
      MS_LOG(ERROR) << "Free device address failed : " << mem_buf->addr_ << ", mem_buf : " << mem_buf->ToJson() << ".";
    }
  }
  return mem_bufs;
}

MemBufAllocatorPtr AbstractDynamicMemPool::GenerateAllocator(const AllocatorInfo &allocator_key) {
  const auto is_persistent = allocator_key.from_persistent_mem;
  const auto stream_id = allocator_key.stream_id;
  const auto is_small = allocator_key.use_small_pool;

  MS_LOG(INFO) << "Generate allocator, " << allocator_key.ToString() << ".";
  std::function<MemBlock *(size_t)> mem_block_expander = [&, is_persistent = is_persistent,
                                                          stream_id = stream_id](size_t size) {
    size_t block_size = CalMemBlockAllocSize(size, is_persistent);
    MemBlock *mem_block = nullptr;
    if (block_size == 0) {
      MS_LOG(INFO) << "Malloc mem block failed, is enable eager free : " << IsEnableEagerFree()
                   << ", is enable vmm : " << IsEnableVmm() << ", size : " << size << ", block size is  0.";
      return mem_block;
    }
    DeviceMemPtr addr = nullptr;
    size_t alloc_size;
    MS_LOG(INFO) << "Malloc mem block, is enable eager free : " << IsEnableEagerFree()
                 << ", is enable vmm : " << IsEnableVmm() << ", size : " << size << ", block size : " << block_size
                 << ".";
    if (IsEnableVmm() || IsEnableEagerFree()) {
      // Virtual address is unlimited.
      auto eager_free_size = std::max(block_size, static_cast<size_t>(total_mem_size()));
      alloc_size = AllocDeviceMemByEagerFree(eager_free_size, &addr);
    } else {
      alloc_size = AllocDeviceMem(block_size, &addr);
      if (alloc_size < block_size) {
        MS_LOG(WARNING) << "Alloc device mem failed, alloc size : " << alloc_size << ", block size : " << block_size
                        << ".";
      }
    }
    if (alloc_size == 0) {
      return mem_block;
    }
    mem_stat_ptr_->alloc_size_ += alloc_size;
    mem_block = new MemBlock(alloc_size, addr, stream_id);
    MS_LOG(INFO) << "Malloc mem block : " << mem_block->ToJson() << ".";
    return mem_block;
  };

  std::function<bool(MemBlock *)> mem_block_cleaner = [&](MemBlock *mem_block) {
    mem_stat_ptr_->alloc_size_ -= mem_block->size_;
    // Call free device mem as ascend memory pool would do stat in free operation.
    return FreeDeviceMem(mem_block->addr_);
  };
  std::function<size_t(size_t size, void *addr)> mem_mapper = [&](size_t size, void *addr) {
    return MmapDeviceMem(size, addr);
  };
  std::function<size_t(void *addr, const size_t size)> mem_eager_freer = [&](void *addr, const size_t size) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Eager free addr : " << addr << ", size : " << size << ".";
    return FreeDeviceMemByEagerFree(addr, size);
  };

  return std::make_shared<MemBufAllocator>(mem_block_expander, mem_block_cleaner, mem_mapper, mem_eager_freer,
                                           IsEnableVmm() || IsEnableEagerFree(), is_persistent, stream_id, is_small,
                                           mem_stat_ptr_);
}

// Element in vector : <memory_stream_id, addr>
bool AbstractDynamicMemPool::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                         const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                                         const DeviceEventPtr &event) {
  LockGuard lock(lock_);
  for (auto &[memory_stream_id, addr] : memory_stream_addresses) {
    auto &&it = addr_mem_buf_allocators_.find(addr);
    if (it != addr_mem_buf_allocators_.end()) {
      auto mem_buf = it->second.first;
      if (mem_buf->IsEventNotUsed()) {
        mem_stat_ptr_->used_by_event_size_ += mem_buf->size_;
      }
      (void)mem_buf->RecordEvent(task_id_on_stream, user_stream_id, event);
      (void)stream_pair_mem_bufs_[std::make_pair(user_stream_id, memory_stream_id)].emplace(mem_buf);
    } else {
      // Output of somas sub graph may be used by somas sub graph inner node, address may not be kept in mem pool.
      MS_LOG(INFO) << "Unknown address : " << addr << ".";
    }
  }
  return true;
}

bool AbstractDynamicMemPool::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
  LockGuard lock(lock_);
  auto key = std::make_pair(user_stream_id, memory_stream_id);
  auto iter = stream_pair_mem_bufs_.find(key);
  if (iter == stream_pair_mem_bufs_.end()) {
    return false;
  }

  auto mem_bufs_ = iter->second;
  for (const auto &mem_buf : mem_bufs_) {
    mem_buf->WaitEvent(task_id_on_stream, user_stream_id);
    // Remove event and try to free memory.
    if (mem_buf->IsEventNotUsed()) {
      mem_stat_ptr_->used_by_event_size_ -= mem_buf->size_;
      // Force clear all mem bufs.
      for (auto &stream_pair_mem_bufs : stream_pair_mem_bufs_) {
        (void)stream_pair_mem_bufs.second.erase(mem_buf);
      }
      if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        (void)DoFreeTensorMem(mem_buf->addr_);
      }
    }
  }
  return true;
}

bool AbstractDynamicMemPool::WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) {
  LockGuard lock(lock_);
  for (auto &stream_pair_mem_bufs : stream_pair_mem_bufs_) {
    const auto &[user_stream, memory_stream] = stream_pair_mem_bufs.first;
    if (memory_stream != memory_stream_id) {
      continue;
    }
    auto mem_bufs = stream_pair_mem_bufs.second;
    for (const auto &mem_buf : mem_bufs) {
      mem_buf->WaitEvent(task_id_on_stream, user_stream);
      // Remove event and try to free memory.
      if (mem_buf->IsEventNotUsed()) {
        mem_stat_ptr_->used_by_event_size_ -= mem_buf->size_;
        // Force clear all mem bufs.
        for (auto &kv : stream_pair_mem_bufs_) {
          (void)kv.second.erase(mem_buf);
        }
        if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
          (void)DoFreeTensorMem(mem_buf->addr_);
        }
      }
    }
  }
  return true;
}

bool AbstractDynamicMemPool::SyncAllEvents() {
  LockGuard lock(lock_);
  return DoSyncAllEvents();
}

bool AbstractDynamicMemPool::DoSyncAllEvents() {
  if (stream_pair_mem_bufs_.empty()) {
    return false;
  }

  std::set<MemBuf *> carry_event_mem_bufs;
  for (const auto &stream_pair_mem_buf : stream_pair_mem_bufs_) {
    for (const auto &mem_buf : stream_pair_mem_buf.second) {
      (void)carry_event_mem_bufs.emplace(mem_buf);
    }
  }
  for (auto &mem_buf : carry_event_mem_bufs) {
    if (mem_buf->SyncAllEvents() && mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
      if (DoFreeTensorMem(mem_buf->addr_)) {
        mem_stat_ptr_->used_by_event_size_ -= mem_buf->size_;
      }
    }
  }

  stream_pair_mem_bufs_.clear();
  return true;
}

size_t AbstractDynamicMemPool::CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool) {
  auto device_free_mem_size = free_mem_size();
  // Make sure available mem is enough.
  if (device_free_mem_size < size) {
    MS_LOG(WARNING) << "Memory not enough: current free memory size[" << device_free_mem_size
                    << "] is smaller than required size[" << size << "].";
    return 0;
  }
  auto unit_size = MemAllocUnitSize(from_persistent_mem);
  if (device_free_mem_size < unit_size) {
    MS_LOG(WARNING) << "Device memory size [" << device_free_mem_size << "] is smaller than unit size [" << unit_size
                    << "].";
  }
  // Calculate alloc size.
  size_t alloc_size = unit_size;
  if (size > unit_size) {
    alloc_size = ((size + unit_size - 1) / unit_size) * unit_size;
  }
  return std::min(alloc_size, device_free_mem_size);
}

void AbstractDynamicMemPool::DefragMemory() {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Try to defrag memory.";
  LockGuard lock(lock_);

  if (enable_dump_memory_) {
    MS_LOG(INFO) << "Start dump memory info.";
    DumpDynamicMemPoolDebugInfo();
  }

  if (!enable_vmm_) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Skip defrag memory since vmm is not enabled.";
    return;
  }

  if (eager_free_count_ == 0) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Exit defrag memory since eager free count is 0.";
    return;
  }
  if (last_eager_free_count_ == eager_free_count_) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY)
      << "Exit defrag memory since last eager free count equals to eager free count : " << last_eager_free_count_
      << ".";
    return;
  }

  MS_LOG(INFO) << "Defrag memory start.";
  WaitPipelineHelper();
  if (!SyncAllStreams()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Sync all streams failed.";
    return;
  }
  const auto [eager_free_size, real_free_size] = FreeIdleMemsByEagerFree();
  MS_LOG(INFO) << "Defrag memory, eager_free_size : " << eager_free_size << ", real_free_size : " << real_free_size
               << ".";
  last_eager_free_count_ = eager_free_count_;
}

void AbstractDynamicMemPool::WaitPipelineHelper() {
  if (pipeline_callback_) {
    lock_.unlock();
    pipeline_callback_();
    lock_.lock();
  }
}

namespace {
void SplitAndDumpLog(const std::string &log_str) {
  const char *delim = "\n";
  char *next_token;
#ifdef _MSC_VER
  const char *token = strtok_s(const_cast<char *>(log_str.c_str()), delim, &next_token);
#else
  const char *token = strtok_r(const_cast<char *>(log_str.c_str()), delim, &next_token);
#endif
  size_t count = 0;
  const size_t max_count = 1000;
  while (token && (count++ < max_count)) {
    MS_LOG(INFO) << token;
#ifdef _MSC_VER
    token = strtok_s(NULL, delim, &next_token);
#else
    token = strtok_r(NULL, delim, &next_token);
#endif
  }
}
};  // namespace

void AbstractDynamicMemPool::DumpDynamicMemPoolStateInfo() { SplitAndDumpLog(DynamicMemPoolStateInfo()); }

std::string AbstractDynamicMemPool::DynamicMemPoolStateInfo() const {
  std::stringstream ss;
  // Classify mem buf and stat mem buf state info.
  size_t mem_buf_used_stat[static_cast<int>(memory::mem_pool::MemType::kOther) + 1] = {0};
  struct AddrComparator {
    bool operator()(MemBuf *const &left, MemBuf *const &right) const { return left->addr_ < right->addr_; }
  };
  std::map<MemBufAllocator *, std::set<MemBuf *, AddrComparator>> allocator_mem_bufs;
  for (const auto &addr_mem_buf_allocator : addr_mem_buf_allocators_) {
    const auto allocator = addr_mem_buf_allocator.second.second;
    const auto mem_buf = addr_mem_buf_allocator.second.first;
    mem_buf_used_stat[static_cast<int>(mem_buf->alloc_type_)] += mem_buf->size_;
    auto &mem_bufs = allocator_mem_bufs[allocator];
    (void)mem_bufs.insert(mem_buf);
  }
  for (const auto &[allocator, mem_bufs] : allocator_mem_bufs) {
    ss << "\tIn used mem buf info for " << allocator->BriefInfo() << ", mem_bufs size : " << mem_bufs.size() << "\n";
  }

  for (const auto &[key, buf_allocator_ptr] : stream_id_allocators_) {
    ss << key.ToString() << "\n" << buf_allocator_ptr->DumpStateInfo();
  }

  size_t other_used_size = 0;
  int start = static_cast<int>(memory::mem_pool::MemType::kGraphOutput);
  int end = static_cast<int>(memory::mem_pool::MemType::kOther);
  for (int i = start; i <= end; i++) {
    other_used_size += mem_buf_used_stat[i];
  }

  ss << "The dynamic memory pool[" << GetMemoryPoolType() << "] stat info : " << mem_stat_ptr_->ToReadableString()
     << ", actual peak used mem:" << ActualPeakStatistics() / kMBToByte
     << "M. Weight used size:" << mem_buf_used_stat[static_cast<int>(memory::mem_pool::MemType::kWeight)] / kMBToByte
     << "M, constant value used size:"
     << mem_buf_used_stat[static_cast<int>(memory::mem_pool::MemType::kConstantValue)] / kMBToByte
     << "M, kernel output used size:"
     << mem_buf_used_stat[static_cast<int>(memory::mem_pool::MemType::kKernel)] / kMBToByte
     << "M, other used size:" << other_used_size / kMBToByte << "M.\n";
  return ss.str();
}

void AbstractDynamicMemPool::DumpDynamicMemPoolDebugInfo() {
  std::stringstream ss;
  ss << "The dynamic memory pool stat info : " << mem_stat_ptr_->ToReadableString() << "\n";

  struct AddrComparator {
    bool operator()(MemBuf *const &left, MemBuf *const &right) const { return left->addr_ < right->addr_; }
  };
  std::map<MemBufAllocator *, std::set<MemBuf *, AddrComparator>> allocator_mem_bufs;
  for (const auto &addr_mem_buf_allocator : addr_mem_buf_allocators_) {
    const auto allocator = addr_mem_buf_allocator.second.second;
    const auto mem_buf = addr_mem_buf_allocator.second.first;
    auto &mem_bufs = allocator_mem_bufs[allocator];
    (void)mem_bufs.insert(mem_buf);
  }

  for (const auto &[allocator, mem_bufs] : allocator_mem_bufs) {
    ss << "\tIn used mem buf info for " << allocator->BriefInfo() << ", mem_bufs size : " << mem_bufs.size() << "\n";
    for (const auto mem_buf : mem_bufs) {
      ss << "\t\t" << mem_buf->ToJson() << "\n";
    }
  }

  for (const auto &[key, buf_allocator_ptr] : stream_id_allocators_) {
    ss << key.ToString() << "\n" << buf_allocator_ptr->DumpDebugInfo();
  }

  // Write info file.
  static std::atomic<size_t> file_count = 0;
  auto file_path =
    memory::mem_pool::GeneratePath(rank_id_getter_(), "MemoryDebugInfo" + std::to_string(file_count++), "txt");
  std::ofstream file_stream(file_path);
  file_stream << ss.str();
  file_stream.close();
}

const std::pair<size_t, size_t> AbstractDynamicMemPool::FreeIdleMemsByEagerFree() {
  if (!IsEnableVmm() && !IsEnableEagerFree()) {
    MS_LOG(WARNING) << "FreeIdleMemsByEagerFree is not allowed since vmm is not enabled.";
    return std::make_pair(0L, 0L);
  }

  MS_LOG(INFO) << "Free idle mems by eager free start, allocator size : " << stream_id_allocators_.size() << ".";
  eager_free_count_++;

  size_t total_eager_free_size = 0;
  size_t total_real_free_size = 0;
  for (auto &stream_id_allocator : stream_id_allocators_) {
    const auto [eager_free_size, real_free_size] = stream_id_allocator.second->FreeIdleMemsByEagerFree();
    total_eager_free_size += eager_free_size;
    total_real_free_size += real_free_size;
  }

  size_t not_free_size =
    total_eager_free_size > total_real_free_size ? (total_eager_free_size - total_real_free_size) : 0;
  if (total_real_free_size >= kGBToByte) {
    MS_LOG(WARNING) << "Eager free count : " << eager_free_count_ << ", free memory : " << total_eager_free_size
                    << ", real free : " << total_real_free_size << ", not free : " << not_free_size << ".";
  } else {
    MS_LOG(INFO) << "Eager free count : " << eager_free_count_ << ", free memory : " << total_eager_free_size
                 << ", real free : " << total_real_free_size << ", not free : " << not_free_size << ".";
  }

  return {total_eager_free_size, total_real_free_size};
}

size_t AbstractDynamicMemPool::ReleaseFreeBlocks() {
  MS_LOG(INFO) << "Release free blocks start.";
  size_t release_size = 0;
  for (auto &stream_id_allocator : stream_id_allocators_) {
    release_size += stream_id_allocator.second->ReleaseFreeBlocks();
  }
  MS_LOG(INFO) << "Release free blocks size : " << release_size << ".";
  return release_size;
}

size_t AbstractDynamicMemPool::ReleaseCustomFreeBlocks() {
  MS_LOG(INFO) << "Release custom free blocks start.";
  size_t release_size = 0;
  for (auto &customized_allocator : customized_allocators_) {
    release_size += customized_allocator.second->ReleaseFreeBlocks();
  }
  MS_LOG(INFO) << "Release custom free blocks size : " << release_size << ".";
  return release_size;
}

// The statistics information.
size_t AbstractDynamicMemPool::TotalMemStatistics() const {
  if (IsEnableVmm()) {
    return GetVmmUsedMemSize() + mem_stat_ptr_->custom_alloc_size_;
  }
  return mem_stat_ptr_->alloc_size_ + mem_stat_ptr_->custom_alloc_size_;
}

size_t AbstractDynamicMemPool::TotalUsedMemStatistics() const { return mem_stat_ptr_->used_size_; }

size_t AbstractDynamicMemPool::TotalUsedByEventMemStatistics() const { return mem_stat_ptr_->used_by_event_size_; }

size_t AbstractDynamicMemPool::TotalIdleMemStatistics() const { return mem_stat_ptr_->IdleSize(); }

size_t AbstractDynamicMemPool::TotalEagerFreeMemStatistics() const { return mem_stat_ptr_->eager_free_size_; }

size_t AbstractDynamicMemPool::UsedMemPeakStatistics() const { return mem_stat_ptr_->peak_size_; }

size_t AbstractDynamicMemPool::MaxMemAllocatedStatistics() const { return mem_stat_ptr_->iter_used_peak_size_; }

size_t AbstractDynamicMemPool::MaxMemReservedStatistics() const { return mem_stat_ptr_->iter_alloc_peak_size_; }

size_t AbstractDynamicMemPool::ActualPeakStatistics() const {
  if (IsEnableVmm()) {
    return GetVmmUsedMemSize() + mem_stat_ptr_->custom_alloc_size_;
  }

  size_t peak_size = 0;
  for (auto &stream_id_allocator : stream_id_allocators_) {
    peak_size += stream_id_allocator.second->ActualPeakSize();
  }
  for (auto &customized_allocator : customized_allocators_) {
    peak_size += customized_allocator.second->ActualPeakSize();
  }
  return peak_size;
}

std::unordered_map<std::string, std::size_t> AbstractDynamicMemPool::BlockCountsStatistics() const {
  LockGuard lock(lock_);
  size_t persistent_block_count = 0;
  size_t common_block_count = 0;
  for (const auto &[allocator_info, allocator_ptr] : stream_id_allocators_) {
    if (allocator_info.from_persistent_mem) {
      persistent_block_count += allocator_ptr->mem_blocks_.size();
    } else {
      common_block_count += allocator_ptr->mem_blocks_.size();
    }
  }
  std::unordered_map<std::string, size_t> block_counts;
  block_counts[kPersistentMemPoolType] = persistent_block_count;
  block_counts[kCommonMemPoolType] = common_block_count;
  return block_counts;
}

std::unordered_map<std::string, std::size_t> AbstractDynamicMemPool::BlockUnitSizeStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<std::string, size_t> block_units;
  block_units[kPersistentMemPoolType] = persist_unit_size_;
  block_units[kCommonMemPoolType] = common_unit_size_;
  return block_units;
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AbstractDynamicMemPool::CommonMemBlocksInfoStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> block_infos;
  for (const auto &[allocator_info, allocator_ptr] : stream_id_allocators_) {
    if (!allocator_info.from_persistent_mem) {
      const auto &mem_blocks = allocator_ptr->mem_blocks_;
      for (const auto mem_block : mem_blocks) {
        std::unordered_map<std::string, size_t> block_info;
        block_info[kBlockMemorySize] = mem_block->size_;
        block_info[kBlockStreamId] = mem_block->stream_id_;
        block_infos[(std::string *)(mem_block->addr_)] = block_info;
      }
    }
  }
  return block_infos;
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AbstractDynamicMemPool::PersistentMemBlocksInfoStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> block_infos;
  for (const auto &[allocator_info, allocator_ptr] : stream_id_allocators_) {
    if (allocator_info.from_persistent_mem) {
      const auto &mem_blocks = allocator_ptr->mem_blocks_;
      for (const auto mem_block : mem_blocks) {
        std::unordered_map<std::string, size_t> block_info;
        block_info[kBlockMemorySize] = mem_block->size_;
        block_info[kBlockStreamId] = mem_block->stream_id_;
        block_infos[(std::string *)(mem_block->addr_)] = block_info;
      }
    }
  }
  return block_infos;
}

void AbstractDynamicMemPool::ResetMaxMemReserved() {
  LockGuard lock(lock_);
  mem_stat_ptr_->iter_alloc_peak_size_ = IsEnableVmm() ? GetVmmUsedMemSize() + mem_stat_ptr_->custom_alloc_size_
                                                       : mem_stat_ptr_->alloc_size_ + mem_stat_ptr_->custom_alloc_size_;
}

void AbstractDynamicMemPool::ResetMaxMemAllocated() {
  LockGuard lock(lock_);
  mem_stat_ptr_->iter_used_peak_size_ = mem_stat_ptr_->used_size_;
}

AbstractEnhancedDynamicMemPool::AbstractEnhancedDynamicMemPool() {}

void AbstractEnhancedDynamicMemPool::ReportMemoryPoolInfo() {
  // Report memory data to profiler.
  if (memory_profiler_callback_) {
    memory_profiler_callback_();
  }
}

void AbstractEnhancedDynamicMemPool::ReportMemoryPoolMallocInfoToMstx(void *addr, size_t size) {
  if (memory_malloc_mstx_callback_) {
    memory_malloc_mstx_callback_(addr, size);
  }
}

void AbstractEnhancedDynamicMemPool::ReportMemoryPoolFreeInfoToMstx(void *addr) {
  if (memory_free_mstx_callback_) {
    memory_free_mstx_callback_(addr);
  }
}

MemoryTimeEventPtr AbstractEnhancedDynamicMemPool::GenAllocateMemoryTimeEvent(const void *addr, size_t size,
                                                                              uint32_t stream_id, bool from_persistent,
                                                                              bool is_persistent) {
  auto time_event = std::make_shared<MemoryTimeEvent>();
  time_event->created_at_ = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
      .count());
  time_event->addr_ = const_cast<void *>(addr);
  time_event->size_ = size;
  time_event->from_persistent_ = static_cast<uint8_t>(from_persistent);
  time_event->is_persistent_ = static_cast<uint8_t>(is_persistent);
  time_event->stream_id_ = stream_id;
  time_event->run_mode_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().run_mode_;
  time_event->used_size_ = mem_stat_ptr_->used_size_;
  time_event->peak_size_ = mem_stat_ptr_->peak_size_;
  time_event->alloc_size_ = TotalMemStatistics();
  time_event->used_by_event_size_ = mem_stat_ptr_->used_by_event_size_;
  time_event->eager_free_size_ = mem_stat_ptr_->eager_free_size_;
  time_event->owner_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().name_;
  time_event->alloc_type_ = static_cast<uint8_t>(DynamicMemAllocatorDebugInfo::GetDebugInfo().type_);
  return time_event;
}

MemoryTimeEventPtr AbstractEnhancedDynamicMemPool::GenFreeMemoryTimeEvent(const void *addr) {
  auto time_event = std::make_shared<MemoryTimeEvent>();
  time_event->created_at_ = static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch())
      .count());
  time_event->addr_ = const_cast<void *>(addr);
  const size_t time_event_free_size = -1;
  time_event->size_ = time_event_free_size;
  time_event->used_size_ = mem_stat_ptr_->used_size_;
  time_event->peak_size_ = mem_stat_ptr_->peak_size_;
  time_event->alloc_size_ = TotalMemStatistics();
  time_event->used_by_event_size_ = mem_stat_ptr_->used_by_event_size_;
  time_event->eager_free_size_ = mem_stat_ptr_->eager_free_size_;
  return time_event;
}
}  // namespace device
}  // namespace mindspore
