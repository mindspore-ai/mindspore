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

#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include <string>
#include <algorithm>
#include <numeric>
#include <ostream>
#include <utility>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace device {
static const char kPersistentParamMem[] = "Persistent mem";
static const char kCommonMem[] = "Common mem";
constexpr size_t kGBToByte = 1024 << 20;
// The smallest memory request size, if it is smaller than this size, the device memory request may fail
// Set experience value to 10M
const size_t kMinimumAllocMem = 10 << 20;

thread_local AllocatorDebugInfo DynamicMemAllocatorDebugInfo::debug_info_;

static const std::map<DynamicMemBufStatus, std::string> kBufStatusString = {
  {DynamicMemBufStatus::kMemBufIdle, "idle"},
  {DynamicMemBufStatus::kMemBufUsed, "used"},
  {DynamicMemBufStatus::kMemBufEagerFree, "eager_free"}};

static const std::map<AllocatorType, std::string> kAllocatorTypeString = {
  {AllocatorType::kWeight, "weight"},
  {AllocatorType::kConstantValue, "constant value"},
  {AllocatorType::kKernelOutput, "kernel output"},
  {AllocatorType::kOther, "other"},
};

bool IsMemoryPoolRecycle() {
  static const char kMemoryPoolRecycle[] = "MS_MEMORY_POOL_RECYCLE";
  static const auto memory_pool_recycle = common::GetEnv(kMemoryPoolRecycle);
  return memory_pool_recycle == "1";
}

DynamicMemPoolBestFit::~DynamicMemPoolBestFit() {
  persistent_mem_->clear();
  common_mem_->clear();
}

DeviceMemPtr DynamicMemPoolBestFit::AllocTensorMem(size_t size, bool from_persistent_mem, bool need_recycle) {
  size_t align_size = AlignMemorySize(size);
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  // Find the memory buf by tensor size, if not find, then add new memory block and memory buf.
  DeviceMemPtr device_addr = FindAvailableMemBuf(align_size, from_persistent_mem);
  if (device_addr == nullptr) {
    device_addr = AddMemBlockAndMemBuf(align_size, from_persistent_mem, need_recycle);
  }

  // Alloc memory failed and dump the info.
  if (!device_addr) {
    DumpDynamicMemPoolStateInfo();
  }

  if (common::IsNeedProfileMemory()) {
    MS_LOG(WARNING) << "Need Profile Memory, Memory pool alloc, total mem: " << TotalMemStatistics()
                    << ", peak mem: " << UsedMemPeakStatistics() << ", in use mem: " << TotalUsedMemStatistics()
                    << ", device address addr: " << device_addr << ", size: " << size;
  }

  if (IsMemoryPoolRecycle()) {
    mem_bufs_.insert(device_addr);
  }
  MS_LOG(DEBUG) << "Alloc memory details, name:" << DynamicMemAllocatorDebugInfo::GetDebugInfo().name_
                << ", address:" << device_addr << ", size:" << size << "B, total allocated mem:" << TotalMemStatistics()
                << "B, peak used mem:" << UsedMemPeakStatistics() << "B, in used mem:" << TotalUsedMemStatistics()
                << "B, total idle mem:" << (TotalMemStatistics() - TotalUsedMemStatistics()) << "B.";
  return device_addr;
}

std::vector<DeviceMemPtr> DynamicMemPoolBestFit::AllocContinuousTensorMem(const std::vector<size_t> &size_list) {
  std::vector<DeviceMemPtr> device_addr_list;
  size_t total_size = std::accumulate(size_list.begin(), size_list.end(), IntToSize(0));
  // Pre-alloc the one whole piece memory.
  auto device_addr = AllocTensorMem(total_size, false);
  if (!device_addr) {
    return device_addr_list;
  }
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  // Remove the pre-alloc memory.
  auto mem_block = FindMemBlock(device_addr, common_mem_);
  if (mem_block == nullptr) {
    mem_block = FindMemBlock(device_addr, persistent_mem_);
  }
  MS_EXCEPTION_IF_NULL(mem_block);
  const auto &iter = mem_block->block_all_mem_buf_map_.find(device_addr);
  if (iter == mem_block->block_all_mem_buf_map_.end()) {
    DumpDynamicMemPoolDebugInfo();
    MS_LOG(INTERNAL_EXCEPTION) << "Can't find the device address[" << device_addr << "].";
  }
  auto mem_buf = iter->second;
  MS_EXCEPTION_IF_NULL(mem_buf);
  if (mem_buf->size_ < total_size) {
    DumpDynamicMemPoolDebugInfo();
    MS_LOG(EXCEPTION) << "The size of membuf is less than total_size.";
  }
  auto rest_size = mem_buf->size_ - total_size;
  (void)mem_block->block_all_mem_buf_map_.erase(iter);
  // Split the pre-alloc memory into continuous memory by the size list.
  DynamicMemBufPtr continuous_mem_buf;
  auto buf_addr = device_addr;
  for (size_t i : size_list) {
    continuous_mem_buf = std::make_shared<DynamicMemBuf>(buf_addr, DynamicMemBufStatus::kMemBufUsed, i,
                                                         DynamicMemAllocatorDebugInfo::GetDebugInfo().name_,
                                                         DynamicMemAllocatorDebugInfo::GetDebugInfo().type_);
    MS_EXCEPTION_IF_NULL(continuous_mem_buf);
    (void)mem_block->block_all_mem_buf_map_.emplace(buf_addr, continuous_mem_buf);
    device_addr_list.emplace_back(buf_addr);
    buf_addr = AddressOffset(buf_addr, i);
  }
  // Update the size of the last memory buf.
  continuous_mem_buf->size_ += rest_size;
  return device_addr_list;
}

size_t DynamicMemPoolBestFit::AlignMemorySize(size_t size) const {
  if (size == 0) {
    return kDynamicMemAlignSize;
  }
  return ((size + kDynamicMemAlignSize - 1) / kDynamicMemAlignSize) * kDynamicMemAlignSize;
}

DeviceMemPtr DynamicMemPoolBestFit::FindAvailableMemBuf(size_t size, bool from_persistent_mem) {
  auto addr = FindMemBufByStatus(size, from_persistent_mem, DynamicMemBufStatus::kMemBufIdle);
  if (addr == nullptr && is_trigger_eager_free_) {
    MS_LOG(DEBUG) << "Find idle mem buf failed and eager free is enabled, try to search in eager free bufs.";
    // Check total used max memory limits, since real occupy memory size equals to used mem size plus idle mem size.
    // Eager free mem may occupy some memory, so total_mem_size need multiply by a factor.
    float threshold_factor = 0.8f;
    size_t threshold = static_cast<size_t>(total_mem_size() * threshold_factor);
    if (TotalUsedMemStatistics() + TotalIdleMemStatistics() + size <= threshold) {
      addr = FindMemBufByStatus(size, from_persistent_mem, DynamicMemBufStatus::kMemBufEagerFree);
    }
  }
  return addr;
}

DeviceMemPtr DynamicMemPoolBestFit::FindMemBufByStatus(size_t size, bool from_persistent_mem,
                                                       DynamicMemBufStatus target_status) {
  auto addr = FindMemBufInSpecifiedMng(size, from_persistent_mem, target_status);
  if (addr == nullptr) {
    if (from_persistent_mem && !persistent_mem_->mem_block_list_.empty()) {
      MS_LOG(DEBUG) << "Find mem buf in current pool failed, try to find in another one.";
      addr = FindMemBufInSpecifiedMng(size, !from_persistent_mem, target_status);
    }
  }
  return addr;
}

DeviceMemPtr DynamicMemPoolBestFit::FindMemBufInSpecifiedMng(size_t size, bool from_persistent_mem,
                                                             DynamicMemBufStatus target_status) {
  auto &mem_mng = from_persistent_mem ? persistent_mem_ : common_mem_;
  SizeMapMemBuf &mem_buf_map =
    (target_status == DynamicMemBufStatus::kMemBufIdle) ? mem_mng->idle_mem_buf_map_ : mem_mng->eager_free_mem_buf_map_;
  auto iter = mem_buf_map.lower_bound(size);
  if (iter != mem_buf_map.end()) {
    auto mem_buf = iter->second;
    MS_EXCEPTION_IF_NULL(mem_buf);
    if (mem_buf->status_ != target_status) {
      DumpDynamicMemPoolDebugInfo();
      MS_LOG(EXCEPTION) << "Mem_buf is not " << target_status << ", alloc_size[" << size << "] mem_buf_size["
                        << mem_buf->size_ << "] mem_buf_address[" << mem_buf->device_addr_ << "].";
    }
    mem_buf->allocator_name_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().name_;
    mem_buf->allocator_type_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().type_;
    // Remove map of old idle memory buf
    (void)mem_buf_map.erase(iter);
    // Divide memory buf
    if (IsSplit(size, mem_buf->size_)) {
      SplitMemBuf(size, mem_buf, mem_mng);
    }
    mem_buf->status_ = DynamicMemBufStatus::kMemBufUsed;
    // Memory statistics
    mem_mng->mps_.total_used_mem_size_ += mem_buf->size_;
    if (mem_mng->mps_.total_used_mem_size_ > mem_mng->mps_.used_mem_peak_size_) {
      mem_mng->mps_.used_mem_peak_size_ = mem_mng->mps_.total_used_mem_size_;
    }
    if (target_status == DynamicMemBufStatus::kMemBufIdle) {
      mem_mng->mps_.total_idle_mem_size_ -= mem_buf->size_;
    } else if (target_status == DynamicMemBufStatus::kMemBufEagerFree) {
      mem_mng->mps_.total_eager_free_mem_size_ -= mem_buf->size_;
    }
    return mem_buf->device_addr_;
  }
  return nullptr;
}

size_t DynamicMemPoolBestFit::MemAllocUnitSize(bool from_persistent_mem) const {
  return from_persistent_mem ? persistent_mem_->unit_size_ : common_mem_->unit_size_;
}

void DynamicMemPoolBestFit::SetMemAllocUintSize(size_t common_size, size_t persist_size) {
  persistent_mem_->unit_size_ = persist_size;
  common_mem_->unit_size_ = common_size;
  config_unit_size_ = common_size;
  MS_LOG(INFO) << "Set mem alloc unit size, common " << common_size << " persistent " << persist_size;
}

void *DynamicMemPoolBestFit::GetMinUsedMemoryAddr() const {
  if (mem_bufs_.empty()) {
    return nullptr;
  }
  return *(mem_bufs_.begin());
}

void DynamicMemPoolBestFit::SetMemPoolBlockSize(size_t available_device_mem_size) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  float mem_block_size = ms_context->get_param<float>(MS_CTX_MEMPOOL_BLOCK_SIZE);
  if (mem_block_size == kDefaultMempoolBlockSize) {
    return;
  }

  size_t config_size = FloatToSize(mem_block_size * kGBToByte);
  if (config_size > available_device_mem_size) {
    MS_LOG(WARNING) << "Memory pool block size " << config_size << " is bigger than currently available maximum memory "
                    << available_device_mem_size << ", and the actual effective value will be "
                    << available_device_mem_size;
  }
  // Reserve 1G for persistent_mem
  if (available_device_mem_size > kGBToByte) {
    available_device_mem_size -= kGBToByte;
  }
  size_t real_block_size = std::min(config_size, available_device_mem_size);
  SetMemAllocUintSize(real_block_size);
}

DeviceMemPtr DynamicMemPoolBestFit::AddMemBlockAndMemBuf(size_t size, bool from_persistent_mem, bool need_recycle) {
  if (from_persistent_mem && !need_recycle && !persistent_mem_->mem_block_list_.empty()) {
    from_persistent_mem = false;
  }

  // Try eager free routine.
  if (is_trigger_eager_free_) {
    return AddMemBlockAndMemBufByEagerFree(size, from_persistent_mem);
  }

  size_t alloc_mem_size = CalMemBlockAllocSize(size, from_persistent_mem, need_recycle);
  MS_LOG(DEBUG) << "CalMemBlockAllocSize return : " << size << ", alloc_mem_size : " << alloc_mem_size;
  if (alloc_mem_size == 0) {
    if (IsEnableEagerFree()) {
      is_trigger_eager_free_ = true;
      return AddMemBlockAndMemBufByEagerFree(size, from_persistent_mem);
    }
    return nullptr;
  }

  // Add new memory block
  DeviceMemPtr device_addr = nullptr;
  auto real_alloc_size = AllocDeviceMem(alloc_mem_size, &device_addr);
  if (real_alloc_size < size) {
    MS_LOG(WARNING) << "Memory not enough: alloc size[" << real_alloc_size << "] is smaller than required size[" << size
                    << "].";
    return nullptr;
  }
  // If unit_size is changed by other function(not context), change unit_size back
  MS_EXCEPTION_IF_NULL(common_mem_);
  common_mem_->unit_size_ = config_unit_size_;

  return CreateMemBlockAndMemBuf(size, from_persistent_mem, device_addr, real_alloc_size,
                                 DynamicMemBufStatus::kMemBufIdle);
}

DeviceMemPtr DynamicMemPoolBestFit::AddMemBlockAndMemBufByEagerFree(size_t size, bool from_persistent_mem) {
  // Check used max memory limits.
  if (TotalUsedMemStatistics() + size > total_mem_size()) {
    MS_LOG(ERROR) << "TotalUsedMemStatistics : " << TotalUsedMemStatistics() << " plus alloc size : " << size
                  << " is more than total mem size : " << total_mem_size() << ".";
    return nullptr;
  }

  MS_LOG(DEBUG) << "Try to eager free memory.";
  if (!SyncAllStreams()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Sync all streams failed.";
  }
  FreeIdleMemsByEagerFree();
  auto mem_addr = FindMemBufByStatus(size, from_persistent_mem, DynamicMemBufStatus::kMemBufEagerFree);
  if (mem_addr != nullptr) {
    MS_LOG(DEBUG) << "Find eager free memory success, mem_addr : " << mem_addr << ".";
    return mem_addr;
  }

  auto alloc_size = total_mem_size();
  MS_LOG(INFO) << "Try to alloc eager free mem block, size : " << alloc_size << ".";
  DeviceMemPtr device_addr = nullptr;
  auto real_alloc_size = AllocDeviceMemByEagerFree(alloc_size, &device_addr);
  if (real_alloc_size < alloc_size) {
    MS_LOG(ERROR) << "AllocDeviceMemByEagerFree failed, alloc_size : " << real_alloc_size << ".";
    return nullptr;
  }
  return CreateMemBlockAndMemBuf(size, from_persistent_mem, device_addr, real_alloc_size,
                                 DynamicMemBufStatus::kMemBufEagerFree);
}

DeviceMemPtr DynamicMemPoolBestFit::CreateMemBlockAndMemBuf(size_t size, bool from_persistent_mem,
                                                            DeviceMemPtr source_addr, size_t source_size,
                                                            DynamicMemBufStatus mem_buf_status) {
  auto mem_block = std::make_shared<DynamicMemBlock>(source_addr, source_size);
  auto mem_mng = from_persistent_mem ? persistent_mem_ : common_mem_;
  const auto &iter = std::upper_bound(mem_mng->mem_block_list_.begin(), mem_mng->mem_block_list_.end(),
                                      mem_block->device_addr(), CmpMemBlock);
  (void)mem_mng->mem_block_list_.insert(iter, mem_block);
  // Add new memory buf.
  auto mem_buf = std::make_shared<DynamicMemBuf>(mem_block->device_addr(), mem_buf_status, mem_block->size(),
                                                 DynamicMemAllocatorDebugInfo::GetDebugInfo().name_,
                                                 DynamicMemAllocatorDebugInfo::GetDebugInfo().type_);
  // Add map of new memory buf in the block
  (void)mem_block->block_all_mem_buf_map_.emplace(mem_block->device_addr(), mem_buf);
  // Split memory buf
  if (IsSplit(size, mem_buf->size_)) {
    SplitMemBuf(size, mem_buf, mem_mng);
  }
  mem_buf->status_ = DynamicMemBufStatus::kMemBufUsed;
  // Memory statistics
  mem_mng->mps_.total_mem_size_ += mem_block->size();
  mem_mng->mps_.total_used_mem_size_ += mem_buf->size_;
  if (mem_mng->mps_.total_used_mem_size_ > mem_mng->mps_.used_mem_peak_size_) {
    mem_mng->mps_.used_mem_peak_size_ = mem_mng->mps_.total_used_mem_size_;
  }
  MS_LOG(DEBUG) << "Usage: used size : " << TotalUsedMemStatistics() << ", idle size : " << TotalIdleMemStatistics()
                << ", eager free size : " << TotalEagerFreeMemStatistics() << ".";
  return mem_buf->device_addr_;
}

size_t DynamicMemPoolBestFit::CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool) {
  auto device_free_mem_size = free_mem_size();
  if (device_free_mem_size < size && common::IsNeedProfileMemory()) {
    device_free_mem_size = size;
  }
  if (device_free_mem_size < size) {
    MS_LOG(WARNING) << "Memory not enough: current free memory size[" << device_free_mem_size
                    << "] is smaller than required size[" << size << "].";
    return 0;
  }
  // The memory of the device is too small, which may cause the new application to fail.
  if (device_free_mem_size < kMinimumAllocMem) {
    MS_LOG(WARNING) << "Device memory size [" << device_free_mem_size << "] is smaller than minimum alloc size ["
                    << kMinimumAllocMem << "].";
    return 0;
  }
  auto alloc_mem_size = MemAllocUnitSize(from_persistent_mem);
  // Growing at twice of alloc size
  constexpr size_t kDouble = 2;
  while (alloc_mem_size < size) {
    alloc_mem_size = alloc_mem_size * kDouble;
  }
  alloc_mem_size = std::min(alloc_mem_size, device_free_mem_size);
  return alloc_mem_size;
}

const size_t DynamicMemPoolBestFit::FreeIdleMemsByEagerFree() {
  auto eager_free_mem_func = [&](MemStatusManagerPtr &mem_mng) {
    size_t free_size = 0;
    size_t real_free_size = 0;
    for (auto &[size, mem_buf] : mem_mng->idle_mem_buf_map_) {
      free_size += size;
      real_free_size += FreeDeviceMemByEagerFree(mem_buf->device_addr_, size);
      auto mem_block = FindMemBlock(mem_buf->device_addr_, mem_mng);
      MS_EXCEPTION_IF_NULL(mem_block);
      CombineMemBuf(mem_block, mem_buf->device_addr_, mem_mng, DynamicMemBufStatus::kMemBufIdle,
                    DynamicMemBufStatus::kMemBufEagerFree);
    }
    mem_mng->idle_mem_buf_map_.clear();
    return std::make_pair(free_size, real_free_size);
  };

  const auto [persistent_free_size, persistent_real_free_size] = eager_free_mem_func(persistent_mem_);
  const auto [common_free_size, common_real_free_size] = eager_free_mem_func(common_mem_);
  auto free_size = persistent_free_size + common_free_size;
  auto real_free_size = persistent_real_free_size + common_real_free_size;
  MS_LOG(DEBUG) << "Total eager free memory : " << free_size << ", real free : " << real_free_size << ".";
  return real_free_size;
}

bool DynamicMemPoolBestFit::IsSplit(size_t tensor_size, size_t mem_buf_size) const {
  return mem_buf_size - tensor_size >= kDynamicMemAlignSize;
}

void DynamicMemPoolBestFit::SplitMemBuf(size_t size, const DynamicMemBufPtr &mem_buf,
                                        const MemStatusManagerPtr &mem_mng) {
  MS_EXCEPTION_IF_NULL(mem_buf);
  MS_EXCEPTION_IF_NULL(mem_mng);
  const auto &mem_block = FindMemBlock(mem_buf->device_addr_, mem_mng);
  MS_EXCEPTION_IF_NULL(mem_block);
  // Divide new memory buf
  if (mem_buf->size_ < size) {
    DumpDynamicMemPoolDebugInfo();
    MS_LOG(EXCEPTION) << "The size of membuf is less than size.";
  }
  size_t newbuf_size = mem_buf->size_ - size;
  mem_buf->size_ = size;
  DeviceMemPtr newbuf_addr = AddressOffset(mem_buf->device_addr_, size);
  auto new_mem_buf = std::make_shared<DynamicMemBuf>(newbuf_addr, mem_buf->status_, newbuf_size);
  // Add map of new memory buf in the block
  (void)mem_block->block_all_mem_buf_map_.emplace(newbuf_addr, new_mem_buf);
  if (new_mem_buf->status_ == DynamicMemBufStatus::kMemBufIdle) {
    // Add map of new idle memory buf
    (void)mem_mng->idle_mem_buf_map_.emplace(new_mem_buf->size_, new_mem_buf);
  } else if (new_mem_buf->status_ == DynamicMemBufStatus::kMemBufEagerFree) {
    (void)mem_mng->eager_free_mem_buf_map_.emplace(new_mem_buf->size_, new_mem_buf);
  }
}

bool DynamicMemPoolBestFit::CmpMemBlock(const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block) {
  MS_EXCEPTION_IF_NULL(device_addr);
  MS_EXCEPTION_IF_NULL(mem_block);
  return device_addr < mem_block->device_addr();
}

DynamicMemBlockPtr DynamicMemPoolBestFit::FindMemBlock(const DeviceMemPtr &device_addr,
                                                       const MemStatusManagerPtr &mem_mng) const {
  MS_EXCEPTION_IF_NULL(device_addr);
  MS_EXCEPTION_IF_NULL(mem_mng);
  auto &&iter =
    std::upper_bound(mem_mng->mem_block_list_.begin(), mem_mng->mem_block_list_.end(), device_addr, CmpMemBlock);
  if (iter != mem_mng->mem_block_list_.begin()) {
    return *(--iter);
  }
  return nullptr;
}

void DynamicMemPoolBestFit::FreeTensorMem(const DeviceMemPtr &device_addr) {
  MS_EXCEPTION_IF_NULL(device_addr);
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  auto fn = [this](const MemStatusManagerPtr &mem_mng, const DeviceMemPtr &device_addr) -> DynamicMemBlockPtr {
    auto mem_block = FindMemBlock(device_addr, mem_mng);
    if (mem_block != nullptr) {
      const auto &iter = mem_block->block_all_mem_buf_map_.find(device_addr);
      if (iter != mem_block->block_all_mem_buf_map_.end()) {
        return mem_block;
      }
    }
    return nullptr;
  };
  auto mem_block = fn(common_mem_, device_addr);
  if (mem_block == nullptr) {
    mem_block = fn(persistent_mem_, device_addr);
    if (mem_block == nullptr) {
      // Maybe destroy the memory pool first, then destroy the address, so this is normal case.
      MS_LOG(DEBUG) << "Can't find the mem_block of the device address[" << device_addr << "].";
      return;
    }
    CombineMemBuf(mem_block, device_addr, persistent_mem_, DynamicMemBufStatus::kMemBufUsed,
                  DynamicMemBufStatus::kMemBufIdle);
  } else {
    CombineMemBuf(mem_block, device_addr, common_mem_, DynamicMemBufStatus::kMemBufUsed,
                  DynamicMemBufStatus::kMemBufIdle);
  }

  if (IsMemoryPoolRecycle()) {
    mem_bufs_.erase(device_addr);
  }
  MS_LOG(DEBUG) << "Free memory details, name:" << DynamicMemAllocatorDebugInfo::GetDebugInfo().name_
                << ", address:" << device_addr << ", total allocated mem:" << TotalMemStatistics()
                << "B, peak used mem:" << UsedMemPeakStatistics() << "B, in used mem:" << TotalUsedMemStatistics()
                << "B, total idle mem:" << (TotalMemStatistics() - TotalUsedMemStatistics()) << "B.";
}

void DynamicMemPoolBestFit::CombineMemBuf(const DynamicMemBlockPtr &mem_block, const DeviceMemPtr &device_addr,
                                          const MemStatusManagerPtr &mem_mng, DynamicMemBufStatus origin_status,
                                          DynamicMemBufStatus target_status) {
  MS_EXCEPTION_IF_NULL(mem_block);
  MS_EXCEPTION_IF_NULL(device_addr);
  MS_EXCEPTION_IF_NULL(mem_mng);
  const auto &iter = mem_block->block_all_mem_buf_map_.find(device_addr);
  if (iter == mem_block->block_all_mem_buf_map_.end()) {
    DumpDynamicMemPoolDebugInfo();
    MS_LOG(INTERNAL_EXCEPTION) << "Can't find the device address[" << device_addr << "].";
  }
  auto mem_buf = iter->second;
  MS_EXCEPTION_IF_NULL(mem_buf);

  if (common::IsNeedProfileMemory()) {
    MS_LOG(WARNING) << "Need Profile Memory, Memory pool free, total mem: " << TotalMemStatistics()
                    << ", peak mem: " << UsedMemPeakStatistics() << ", in use mem: " << TotalUsedMemStatistics()
                    << ", device address addr: " << device_addr << ", size: " << mem_buf->size_;
  }

  if (mem_buf->status_ != origin_status) {
    DumpDynamicMemPoolDebugInfo();
    MS_LOG(EXCEPTION) << "Find the mem_buf is not used, mem_buf_address[" << mem_buf->device_addr_ << "].";
  }
  mem_buf->status_ = target_status;
  if (origin_status == DynamicMemBufStatus::kMemBufUsed) {
    if (mem_mng->mps_.total_used_mem_size_ < mem_buf->size_) {
      DumpDynamicMemPoolDebugInfo();
      MS_LOG(EXCEPTION) << "The total used mem size : " << mem_mng->mps_.total_used_mem_size_
                        << " is less than the size of membuf : " << mem_buf->size_ << ".";
    }
    mem_mng->mps_.total_used_mem_size_ -= mem_buf->size_;
  } else if (origin_status == DynamicMemBufStatus::kMemBufIdle) {
    if (mem_mng->mps_.total_idle_mem_size_ < mem_buf->size_) {
      DumpDynamicMemPoolDebugInfo();
      MS_LOG(EXCEPTION) << "The total idle mem size : " << mem_mng->mps_.total_idle_mem_size_
                        << " is less than the size of membuf : " << mem_buf->size_ << ".";
    }
    mem_mng->mps_.total_idle_mem_size_ -= mem_buf->size_;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unsupported origin status : " << origin_status << ".";
  }
  // Combine backward(combine the next_mem_buf to mem_buf)
  auto next_iter = iter;
  (void)next_iter++;
  if (next_iter != mem_block->block_all_mem_buf_map_.end()) {
    auto next_mem_buf = next_iter->second;
    MS_EXCEPTION_IF_NULL(next_mem_buf);
    if (next_mem_buf->status_ == target_status) {
      mem_buf->size_ += next_mem_buf->size_;
      EraseMemBufByStatus(next_mem_buf->size_, next_mem_buf->device_addr_, mem_mng, target_status);
      (void)mem_block->block_all_mem_buf_map_.erase(next_iter);
    }
  }
  // Combine forward(combine the mem_buf to prev_mem_buf)
  bool forward_combine = false;
  DynamicMemBufPtr prev_mem_buf;
  if (iter != mem_block->block_all_mem_buf_map_.begin()) {
    auto prev_iter = iter;
    (void)prev_iter--;
    prev_mem_buf = prev_iter->second;
    MS_EXCEPTION_IF_NULL(prev_mem_buf);
    if (prev_mem_buf->status_ == target_status) {
      EraseMemBufByStatus(prev_mem_buf->size_, prev_mem_buf->device_addr_, mem_mng, target_status);
      prev_mem_buf->size_ += mem_buf->size_;
      (void)mem_block->block_all_mem_buf_map_.erase(iter);
      forward_combine = true;
    }
  }

  // Put back mem buf into specific map.
  auto put_back_mem_buf_func = [&](DynamicMemBufStatus status, const DynamicMemBufPtr &mem_buf) {
    if (status == DynamicMemBufStatus::kMemBufIdle) {
      (void)mem_mng->idle_mem_buf_map_.emplace(mem_buf->size_, mem_buf);
      mem_mng->mps_.total_idle_mem_size_ += mem_buf->size_;
    } else if (status == DynamicMemBufStatus::kMemBufEagerFree) {
      (void)mem_mng->eager_free_mem_buf_map_.emplace(mem_buf->size_, mem_buf);
      mem_mng->mps_.total_eager_free_mem_size_ += mem_buf->size_;
    }
  };
  if (forward_combine) {
    put_back_mem_buf_func(target_status, prev_mem_buf);
  } else {
    put_back_mem_buf_func(target_status, mem_buf);
  }
}

void DynamicMemPoolBestFit::EraseMemBufByStatus(size_t size, const DeviceMemPtr &device_addr,
                                                const MemStatusManagerPtr &mem_mng,
                                                DynamicMemBufStatus target_status) const {
  MS_EXCEPTION_IF_NULL(device_addr);
  MS_EXCEPTION_IF_CHECK_FAIL(target_status != DynamicMemBufStatus::kMemBufUsed, "Illegal used status");
  auto &mem_buf_map = (target_status == DynamicMemBufStatus::kMemBufEagerFree) ? mem_mng->eager_free_mem_buf_map_
                                                                               : mem_mng->idle_mem_buf_map_;
  auto &&iter = mem_buf_map.equal_range(size);
  while (iter.first != iter.second) {
    MS_EXCEPTION_IF_NULL(iter.first->second);
    // Remove map of memory buf by size and device address.
    if (iter.first->second->device_addr_ == device_addr) {
      (void)mem_buf_map.erase(iter.first);
      return;
    }
    (void)iter.first++;
  }
  MS_LOG(ERROR) << "Can't find the size[" << size << "] and device address[" << device_addr << "] in the idle mem_buf.";
}

void DynamicMemPoolBestFit::ReleaseDeviceRes() {
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  DumpDynamicMemPoolStateInfo();

  auto fn = [this](const MemStatusManagerPtr &mem_mng) {
    MS_EXCEPTION_IF_NULL(mem_mng);
    for (auto &iter : mem_mng->mem_block_list_) {
      MS_EXCEPTION_IF_NULL(iter);
      auto &device_addr = iter->device_addr_base_;
      if (device_addr != nullptr) {
        if (!FreeDeviceMem(device_addr)) {
          MS_LOG(ERROR) << "Free device memory[" << device_addr << "] error.";
        }
        device_addr = nullptr;
      }
    }
    mem_mng->clear();
  };
  fn(common_mem_);
  fn(persistent_mem_);
}

void DynamicMemPoolBestFit::DumpDynamicMemPoolStateInfo() {
  size_t total_used_size_list[kAllocatorTypeNum] = {0};
  auto fn = [&](const MemStatusManagerPtr &mem_mng, const std::string &mem_type) {
    MS_EXCEPTION_IF_NULL(mem_mng);
    if (mem_mng->mem_block_list_.empty()) {
      return;
    }

    std::ostringstream buf;
    for (size_t i = 0; i < mem_mng->mem_block_list_.size(); ++i) {
      size_t mem_block_used_size = 0;
      MS_EXCEPTION_IF_NULL(mem_mng->mem_block_list_[i]);
      for (auto mb = mem_mng->mem_block_list_[i]->block_all_mem_buf_map_.begin();
           mb != mem_mng->mem_block_list_[i]->block_all_mem_buf_map_.end(); ++mb) {
        if (mb->second->status_ == DynamicMemBufStatus::kMemBufUsed) {
          mem_block_used_size += mb->second->size_;
          MS_EXCEPTION_IF_CHECK_FAIL((static_cast<int>(mb->second->allocator_type_) < kAllocatorTypeNum),
                                     "Allocator type is out of range.");
          total_used_size_list[static_cast<int>(mb->second->allocator_type_)] += mb->second->size_;
        }
      }
      buf << ", block[" << i << "] block size:" << mem_mng->mem_block_list_[i]->mem_block_size_ / kMBToByte
          << "M idle size:" << (mem_mng->mem_block_list_[i]->mem_block_size_ - mem_block_used_size) / kMBToByte << "M";
    }

    // Dump all the memory buf info
    MS_LOG(INFO) << mem_type << " pool info: Total allocated mem:" << mem_mng->mps_.total_mem_size_ / kMBToByte
                 << "M, peak used mem:" << mem_mng->mps_.used_mem_peak_size_ / kMBToByte
                 << "M, in used mem:" << mem_mng->mps_.total_used_mem_size_ / kMBToByte << "M, total idle mem:"
                 << (mem_mng->mps_.total_mem_size_ - mem_mng->mps_.total_used_mem_size_) / kMBToByte
                 << "M. Block unit size:" << mem_mng->unit_size_ / kMBToByte
                 << "M, block counts:" << mem_mng->mem_block_list_.size() << buf.str();
  };

  fn(common_mem_, std::string(kCommonMem));
  fn(persistent_mem_, std::string(kPersistentParamMem));
  MS_LOG(INFO) << "The dynamic memory pool total allocated mem:" << TotalMemStatistics() / kMBToByte
               << "M, peak used mem:" << UsedMemPeakStatistics() / kMBToByte
               << "M, in used mem:" << TotalUsedMemStatistics() / kMBToByte
               << "M, total idle mem:" << TotalIdleMemStatistics() / kMBToByte
               << "M, total eager free mem:" << TotalEagerFreeMemStatistics() / kMBToByte
               << "M. Weight used size:" << total_used_size_list[static_cast<int>(AllocatorType::kWeight)] / kMBToByte
               << "M, constant value used size:"
               << total_used_size_list[static_cast<int>(AllocatorType::kConstantValue)] / kMBToByte
               << "M, kernel output used size:"
               << total_used_size_list[static_cast<int>(AllocatorType::kKernelOutput)] / kMBToByte
               << "M, other used size:" << total_used_size_list[static_cast<int>(AllocatorType::kOther)] / kMBToByte
               << "M.";
}

void DynamicMemPoolBestFit::DumpDynamicMemPoolDebugInfo() {
  auto fn = [](const MemStatusManagerPtr &mem_mng, const std::string &mem_type) {
    MS_EXCEPTION_IF_NULL(mem_mng);
    size_t total_mem = 0;
    size_t total_used_mem = 0;
    size_t total_idle_mem1 = 0;
    size_t total_idle_mem2 = 0;
    size_t total_eager_free_mem = 0;
    // Dump the memory block info and memory buf info.
    MS_LOG(WARNING) << mem_type << " all mem_block info: counts[" << mem_mng->mem_block_list_.size() << "].";
    for (auto iter = mem_mng->mem_block_list_.begin(); iter != mem_mng->mem_block_list_.end(); ++iter) {
      total_mem += (*iter)->size();
      auto mem_buf_map = (*iter)->block_all_mem_buf_map_;
      MS_LOG(WARNING) << " MemBlock info: number[" << iter - mem_mng->mem_block_list_.begin() << "] mem_buf_counts["
                      << mem_buf_map.size() << "] base_address[" << (*iter)->device_addr() << "] block_size["
                      << (*iter)->size() << "].";
      for (auto iter_mem_buf = mem_buf_map.begin(); iter_mem_buf != mem_buf_map.end(); ++iter_mem_buf) {
        auto mem_buf = iter_mem_buf->second;
        MS_EXCEPTION_IF_NULL(mem_buf);
        if (mem_buf->status_ == DynamicMemBufStatus::kMemBufIdle) {
          total_idle_mem1 += mem_buf->size_;
        } else if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsed) {
          total_used_mem += mem_buf->size_;
        } else if (mem_buf->status_ == DynamicMemBufStatus::kMemBufEagerFree) {
          total_eager_free_mem += mem_buf->size_;
        }
        MS_LOG(INFO) << "  MemBuf info: address[" << mem_buf->device_addr_ << "] size[" << mem_buf->size_ << "] status["
                     << kBufStatusString.at(mem_buf->status_) << "] name[" << mem_buf->allocator_name_ << "] type["
                     << kAllocatorTypeString.at(mem_buf->allocator_type_) << "].";
      }
    }
    // Dump all the idle memory buf info.
    MS_LOG(WARNING) << mem_type << " all idle mem_buf info: counts[" << mem_mng->idle_mem_buf_map_.size() << "].";
    for (auto iter_idle = mem_mng->idle_mem_buf_map_.begin(); iter_idle != mem_mng->idle_mem_buf_map_.end();
         ++iter_idle) {
      auto mem_buf = iter_idle->second;
      MS_EXCEPTION_IF_NULL(mem_buf);
      total_idle_mem2 += mem_buf->size_;
      MS_LOG(INFO) << " Idle mem_buf info: size[" << mem_buf->size_ << "] address[" << mem_buf->device_addr_
                   << "] status[" << kBufStatusString.at(mem_buf->status_) << "].";
    }
    // Dump all the eager free memory buf info.
    size_t total_eager_free_mem_in_map = 0;
    MS_LOG(WARNING) << mem_type << " all eager free mem_buf info: counts[" << mem_mng->eager_free_mem_buf_map_.size()
                    << "].";
    for (auto iter = mem_mng->eager_free_mem_buf_map_.begin(); iter != mem_mng->eager_free_mem_buf_map_.end(); ++iter) {
      auto mem_buf = iter->second;
      MS_EXCEPTION_IF_NULL(mem_buf);
      total_eager_free_mem_in_map += mem_buf->size_;
      MS_LOG(INFO) << " Idle mem_buf info: size[" << mem_buf->size_ << "] address[" << mem_buf->device_addr_
                   << "] status[" << kBufStatusString.at(mem_buf->status_) << "].";
    }
    // Dump the memory statistical info.
    MS_LOG(WARNING) << mem_type << " total allocated memory[" << total_mem << "], used memory[" << total_used_mem
                    << "], idle memory[" << total_idle_mem1 << "].";
    if (total_idle_mem1 != total_idle_mem2) {
      MS_LOG(ERROR) << "Check error: the idle memory in the mem_block is not equal the global idle memory.";
    }
    if (total_eager_free_mem != total_eager_free_mem_in_map) {
      MS_LOG(ERROR) << "Check error: the eager free memory in the mem_block is not equal the global eager free memory.";
    }
    if (total_mem != total_used_mem + total_idle_mem1 + total_eager_free_mem) {
      MS_LOG(ERROR) << "Check error: the the total memory : " << total_mem
                    << " is not equal the sum of used memory : " << total_used_mem
                    << ", idle memory : " << total_idle_mem1 << " and eager free memory : " << total_eager_free_mem
                    << ".";
    }
  };

  MS_LOG(WARNING) << "Start dump dynamic memory pool debug info.";
  fn(common_mem_, std::string(kCommonMem));
  fn(persistent_mem_, std::string(kPersistentParamMem));
  MS_LOG(WARNING) << "Finish dump dynamic memory pool debug info.";
}

// The statistics information.
size_t DynamicMemPoolBestFit::TotalMemStatistics() const {
  return common_mem_->mps_.total_mem_size_ + persistent_mem_->mps_.total_mem_size_;
}
size_t DynamicMemPoolBestFit::TotalUsedMemStatistics() const {
  return common_mem_->mps_.total_used_mem_size_ + persistent_mem_->mps_.total_used_mem_size_;
}
size_t DynamicMemPoolBestFit::TotalIdleMemStatistics() const {
  return common_mem_->mps_.total_idle_mem_size_ + persistent_mem_->mps_.total_idle_mem_size_;
}
size_t DynamicMemPoolBestFit::TotalEagerFreeMemStatistics() const {
  return common_mem_->mps_.total_eager_free_mem_size_ + persistent_mem_->mps_.total_eager_free_mem_size_;
}
size_t DynamicMemPoolBestFit::UsedMemPeakStatistics() const {
  return common_mem_->mps_.used_mem_peak_size_ + persistent_mem_->mps_.used_mem_peak_size_;
}
}  // namespace device
}  // namespace mindspore
