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
#include <algorithm>
#include <numeric>
#include <ostream>
#include <utility>
#include <string>
#include "include/backend/mem_reuse/mem_tracker.h"
#include "include/common/utils/utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_utils.h"
#ifdef ENABLE_DEBUGGER
#include "plugin/device/cpu/hal/profiler/cpu_profiling.h"
#endif

namespace mindspore {
namespace device {
static const char kPersistentParamMem[] = "Persistent mem";
static const char kCommonMem[] = "Common mem";
constexpr size_t kGBToByte = 1024 << 20;
// The smallest memory request size, if it is smaller than this size, the device memory request may fail
// Set experience value to 10M
const size_t kMinimumAllocMem = 10 << 20;

thread_local AllocatorDebugInfo DynamicMemAllocatorDebugInfo::debug_info_;

const char kBlockMemorySize[] = "block_memory_size";
const char kBlockStreamId[] = "block_stream_id";
const char kCommonMemPoolType[] = "common_mem_pool";
const char kPersistentMemPoolType[] = "persistent_mem_pool";

static const std::map<DynamicMemBufStatus, std::string> kBufStatusString = {
  {DynamicMemBufStatus::kMemBufIdle, "idle"},
  {DynamicMemBufStatus::kMemBufUsed, "used"},
  {DynamicMemBufStatus::kMemBufEagerFree, "eager_free"},
  {DynamicMemBufStatus::kMemBufUsedByEvent, "used_by_event"}};

static const std::map<AllocatorType, std::string> kAllocatorTypeString = {
  {AllocatorType::kWeight, "weight"},
  {AllocatorType::kConstantValue, "constant value"},
  {AllocatorType::kKernelOutput, "kernel output"},
  {AllocatorType::kGraphOutput, "graph output"},
  {AllocatorType::kWorkspace, "workspace"},
  {AllocatorType::kOther, "other"},
};

DynamicMemPoolBestFit::~DynamicMemPoolBestFit() {
  persistent_mem_->Clear();
  common_mem_->Clear();
  stream_pair_addresses_.clear();
}

void DynamicMemBlock::update_border_addr(DeviceMemPtr left_addr, DeviceMemPtr right_addr) {
  if (min_addr_ == nullptr) {
    min_addr_ = left_addr;
  } else {
    min_addr_ = std::min(min_addr_, left_addr);
  }
  if (max_addr_ == nullptr) {
    max_addr_ = right_addr;
  } else {
    max_addr_ = std::max(max_addr_, right_addr);
  }
}

size_t DynamicMemBlock::get_actual_peak() {
  if (min_addr_ == nullptr || max_addr_ == nullptr) {
    return 0;
  }
  int64_t actual_memory = reinterpret_cast<uint8_t *>(max_addr_) - reinterpret_cast<uint8_t *>(min_addr_);
  return actual_memory;
}

DeviceMemPtr DynamicMemPoolBestFit::AllocTensorMem(size_t size, bool from_persistent_mem, bool need_recycle,
                                                   uint32_t stream_id) {
  if (stream_id == UINT32_MAX) {
    MS_LOG(DEBUG) << "Rewrite stream id from INT32 MAX to 0.";
    stream_id = kDefaultStreamIndex;
  }
  size_t align_size = AlignMemorySize(size);
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  // Find the memory buf by tensor size, if not find, then add new memory block and memory buf.
  DeviceMemPtr device_addr = FindAvailableMemBuf(align_size, from_persistent_mem, stream_id);
  static bool init_recycle_memory = false;
  if (need_recycle && !init_recycle_memory) {
    // Force persist memory to be reserved when recycle memory is allocated for the first time
    init_recycle_memory = true;
    MS_LOG(INFO) << "Init Recycle Memory";
    device_addr = nullptr;
  }
  if (device_addr == nullptr) {
    device_addr = AddMemBlockAndMemBuf(align_size, from_persistent_mem, need_recycle, stream_id);

    if (device_addr == nullptr) {
      MS_LOG(INFO) << "Alloc tensor mem failed and try to sync all events to release memory.";
      SyncAllEventsInner();
      device_addr = FindAvailableMemBuf(align_size, from_persistent_mem, stream_id);
    }

    // Alloc memory failed and dump the info.
    if (!device_addr) {
      DumpDynamicMemPoolStateInfo();
    }
  }

// report memory data to profiler
#ifdef ENABLE_DEBUGGER
  static auto profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
  if (profiler_inst->GetEnableFlag() && profiler_inst->GetProfileMemoryFlag()) {
    profiler_inst->RecordMemoryPoolInfo(TotalUsedMemStatistics(), TotalMemStatistics(),
                                        TotalUsedByEventMemStatistics());
  }
#endif

  if (common::IsNeedProfileMemory()) {
    MS_LOG(WARNING) << "Need Profile Memory, Memory pool alloc, total mem: " << TotalMemStatistics()
                    << ", peak mem: " << UsedMemPeakStatistics() << ", in use mem: " << TotalUsedMemStatistics()
                    << ", used by event mem: " << TotalUsedByEventMemStatistics()
                    << ", device address addr: " << device_addr << ", size: " << size
                    << ", from persistent mem: " << from_persistent_mem << ", need recycle: " << need_recycle;
  }
  if (device_addr != nullptr) {
    if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
      device::tracker::CALL_MEMORY_TRACKER(AllocMemBlock, device_addr, align_size, GetMemoryPoolType(),
                                           ActualPeakStatistics(), TotalUsedMemStatistics(), TotalMemStatistics(),
                                           stream_id);
    }
    if (IsMemoryPoolRecycle()) {
      (void)mem_bufs_.insert(device_addr);
    }
  }
  MS_LOG(DEBUG) << "Alloc memory details, name:" << DynamicMemAllocatorDebugInfo::GetDebugInfo().name_
                << ", persistent_mem:" << from_persistent_mem << ", stream id: " << stream_id
                << ", address:" << device_addr << ", size:" << size << "B, total allocated mem:" << TotalMemStatistics()
                << "B, peak used mem:" << UsedMemPeakStatistics() << "B, in used mem:" << TotalUsedMemStatistics()
                << "B, used by event mem:" << TotalUsedByEventMemStatistics()
                << "B, actual peak used mem:" << ActualPeakStatistics()
                << "B, total idle mem:" << TotalIdleMemStatistics() << "B.";
  return device_addr;
}

std::vector<DeviceMemPtr> DynamicMemPoolBestFit::AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                                          uint32_t stream_id) {
  std::vector<DeviceMemPtr> device_addr_list;
  size_t total_size = std::accumulate(size_list.begin(), size_list.end(), IntToSize(0));
  // Pre-alloc the one whole piece memory.
  auto device_addr = AllocTensorMem(total_size, false, false, stream_id);
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
    continuous_mem_buf = std::make_shared<DynamicMemBuf>(buf_addr, DynamicMemBufStatus::kMemBufUsed, i, stream_id,
                                                         DynamicMemAllocatorDebugInfo::GetDebugInfo().name_,
                                                         DynamicMemAllocatorDebugInfo::GetDebugInfo().type_);
    MS_EXCEPTION_IF_NULL(continuous_mem_buf);
    (void)mem_block->block_all_mem_buf_map_.emplace(buf_addr, continuous_mem_buf);
    mem_block->update_border_addr(mem_buf->device_addr_, AddressOffset(mem_buf->device_addr_, mem_buf->size_));
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

DeviceMemPtr DynamicMemPoolBestFit::FindAvailableMemBuf(size_t size, bool from_persistent_mem, uint32_t stream_id) {
  auto addr = FindMemBufByStatus(size, from_persistent_mem, DynamicMemBufStatus::kMemBufIdle, stream_id);
  if (addr == nullptr && is_trigger_eager_free_) {
    MS_LOG(DEBUG) << "Find idle mem buf failed and eager free is enabled, try to search in eager free bufs.";
    // Check total used max memory limits, since real occupy memory size equals to used mem size plus idle mem size.
    // Eager free mem may occupy some memory, so total_mem_size need multiply by a factor.
    float threshold_factor = 0.8f;
    size_t threshold = IsEnableVmm() ? total_mem_size() : static_cast<size_t>(total_mem_size() * threshold_factor);
    if (TotalUsedMemStatistics() + TotalUsedByEventMemStatistics() + TotalIdleMemStatistics() + size <= threshold) {
      addr = FindMemBufByStatus(size, from_persistent_mem, DynamicMemBufStatus::kMemBufEagerFree, stream_id);
    }
  }
  return addr;
}

DeviceMemPtr DynamicMemPoolBestFit::FindMemBufByStatus(size_t size, bool from_persistent_mem,
                                                       DynamicMemBufStatus target_status, uint32_t stream_id) {
  auto addr = FindMemBufInSpecifiedMng(size, from_persistent_mem, target_status, stream_id);
  if (addr == nullptr && !IsEnableVmm()) {
    if (from_persistent_mem && !persistent_mem_->mem_block_list_.empty()) {
      MS_LOG(DEBUG) << "Find mem buf in current pool failed, try to find in another one.";
      addr = FindMemBufInSpecifiedMng(size, !from_persistent_mem, target_status, stream_id);
    }
  }
  return addr;
}

DeviceMemPtr DynamicMemPoolBestFit::FindMemBufInSpecifiedMng(size_t size, bool from_persistent_mem,
                                                             DynamicMemBufStatus target_status, uint32_t stream_id) {
  auto &mem_mng = from_persistent_mem ? persistent_mem_ : common_mem_;
  auto &mem_buf_map = mem_mng->GetOrCreateMemBufMap(stream_id, target_status);
  auto iter = mem_buf_map.lower_bound(size);
  if (iter != mem_buf_map.end()) {
    if (IsMemoryPoolRecycle()) {
      // Ensure that the addresses corresponding to the same Tensor for each step are consistent, making the memory pool
      // recycling function more stable.
      auto find_size = iter->first;
      // Can be optimized in the future.
      auto [lb, ub] = mem_buf_map.equal_range(find_size);
      for (auto i = lb; i != ub; ++i) {
        if (i->second->device_addr_ > iter->second->device_addr_) {
          iter = i;
        }
      }
    }
    auto mem_buf = iter->second;
    MS_EXCEPTION_IF_NULL(mem_buf);
    if (mem_buf->status_ != target_status) {
      DumpDynamicMemPoolDebugInfo();
      MS_LOG(EXCEPTION) << "Mem_buf is not " << target_status << ", alloc_size[" << size << "] mem_buf_size["
                        << mem_buf->size_ << "] mem_buf_address[" << mem_buf->device_addr_ << "].";
    }
    mem_buf->allocator_name_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().name_;
    mem_buf->allocator_type_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().type_;
    if (mem_buf->status_ == DynamicMemBufStatus::kMemBufEagerFree && IsEnableVmm()) {
      MS_LOG(DEBUG) << "Find eager free memory, mem_buf_size[" << mem_buf->size_ << "] mem_buf_address["
                    << mem_buf->device_addr_ << "], need size: " << size;
      auto ret = MmapDeviceMem(size, mem_buf->device_addr_);
      if (ret != size) {
        return nullptr;
      }
    }
    // Remove map of old idle memory buf
    (void)mem_buf_map.erase(iter);
    // Divide memory buf
    if (IsSplit(size, mem_buf->size_)) {
      SplitMemBuf(size, mem_buf, mem_mng, stream_id);
    }
    auto mem_block = FindMemBlock(mem_buf->device_addr_, mem_mng);
    MS_EXCEPTION_IF_NULL(mem_block);
    mem_block->update_border_addr(mem_buf->device_addr_, AddressOffset(mem_buf->device_addr_, mem_buf->size_));
    mem_buf->status_ = DynamicMemBufStatus::kMemBufUsed;
    // Memory statistics
    mem_mng->mps_.total_used_mem_size_ += mem_buf->size_;
    mem_mng->mps_.UpdatePeakSize();
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
  MS_LOG(DEBUG) << "Set mem alloc unit size, common " << common_size << " persistent " << persist_size;
}

void *DynamicMemPoolBestFit::GetMinUsingMemoryAddr() const {
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

DeviceMemPtr DynamicMemPoolBestFit::AddMemBlockAndMemBuf(size_t size, bool from_persistent_mem, bool need_recycle,
                                                         uint32_t stream_id) {
  if (from_persistent_mem && !need_recycle && !persistent_mem_->Empty()) {
    from_persistent_mem = false;
  }

  // Try eager free routine.
  if (IsEnableVmm() || is_trigger_eager_free_) {
    is_trigger_eager_free_ = true;
    return AddMemBlockAndMemBufByEagerFree(size, from_persistent_mem, stream_id);
  }

  size_t alloc_mem_size = CalMemBlockAllocSize(size, from_persistent_mem, need_recycle);
  MS_LOG(DEBUG) << "CalMemBlockAllocSize return : " << size << ", alloc_mem_size : " << alloc_mem_size;
  if (alloc_mem_size == 0) {
    if (auto device_addr = FindAvailableMemBuf(size, !from_persistent_mem, stream_id)) {
      return device_addr;
    }
    if (IsEnableEagerFree()) {
      is_trigger_eager_free_ = true;
      return AddMemBlockAndMemBufByEagerFree(size, from_persistent_mem, stream_id);
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
                                 DynamicMemBufStatus::kMemBufIdle, stream_id);
}

DeviceMemPtr DynamicMemPoolBestFit::AddMemBlockAndMemBufByEagerFree(size_t size, bool from_persistent_mem,
                                                                    uint32_t stream_id) {
  // Check used max memory limits.
  if (TotalUsedMemStatistics() + TotalUsedByEventMemStatistics() + size > total_mem_size()) {
    MS_LOG(ERROR) << "TotalUsedMemStatistics : " << TotalUsedMemStatistics()
                  << " plus TotalUsedByEventMemStatistics : " << TotalUsedByEventMemStatistics()
                  << " and plus alloc size : " << size << " is more than total mem size : " << total_mem_size() << ".";
    return nullptr;
  }

  MS_LOG(DEBUG) << "Try to eager free memory.";
  if (!SyncAllStreams()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Sync all streams failed.";
  }
  FreeIdleMemsByEagerFree();
  auto mem_addr = FindMemBufByStatus(size, from_persistent_mem, DynamicMemBufStatus::kMemBufEagerFree, stream_id);
  if (mem_addr != nullptr) {
    MS_LOG(DEBUG) << "Find eager free memory success, mem_addr : " << mem_addr << ".";
    return mem_addr;
  }

  auto alloc_size = std::max(size, static_cast<size_t>(total_mem_size()));
  MS_LOG(INFO) << "Try to alloc eager free mem block, size : " << size << ", alloc_size : " << alloc_size << ".";
  DeviceMemPtr device_addr = nullptr;
  auto real_alloc_size = AllocDeviceMemByEagerFree(alloc_size, &device_addr);
  if (real_alloc_size < alloc_size) {
    MS_LOG(ERROR) << "AllocDeviceMemByEagerFree failed, alloc_size : " << real_alloc_size << ".";
    return nullptr;
  }
  return CreateMemBlockAndMemBuf(size, from_persistent_mem, device_addr, real_alloc_size,
                                 DynamicMemBufStatus::kMemBufEagerFree, stream_id);
}

DeviceMemPtr DynamicMemPoolBestFit::CreateMemBlockAndMemBuf(size_t size, bool from_persistent_mem,
                                                            DeviceMemPtr source_addr, size_t source_size,
                                                            DynamicMemBufStatus mem_buf_status, uint32_t stream_id) {
  auto mem_block = std::make_shared<DynamicMemBlock>(source_addr, source_size, stream_id);
  auto mem_mng = from_persistent_mem ? persistent_mem_ : common_mem_;
  mem_mng->AddMemBlock(mem_block, stream_id);
  // Add new memory buf.
  auto mem_buf = std::make_shared<DynamicMemBuf>(mem_block->device_addr(), mem_buf_status, mem_block->size(), stream_id,
                                                 DynamicMemAllocatorDebugInfo::GetDebugInfo().name_,
                                                 DynamicMemAllocatorDebugInfo::GetDebugInfo().type_);
  if (mem_buf->status_ == DynamicMemBufStatus::kMemBufEagerFree && IsEnableVmm()) {
    MS_LOG(DEBUG) << "Find eager free memory, mem_buf_size[" << mem_buf->size_ << "] mem_buf_address["
                  << mem_buf->device_addr_ << "], need size: " << size;
    auto ret = MmapDeviceMem(size, mem_buf->device_addr_);
    if (ret != size) {
      return nullptr;
    }
  }
  // Add map of new memory buf in the block
  (void)mem_block->block_all_mem_buf_map_.emplace(mem_block->device_addr(), mem_buf);
  // Split memory buf
  if (IsSplit(size, mem_buf->size_)) {
    SplitMemBuf(size, mem_buf, mem_mng, stream_id);
  }
  mem_block->update_border_addr(mem_buf->device_addr_, AddressOffset(mem_buf->device_addr_, mem_buf->size_));
  mem_buf->status_ = DynamicMemBufStatus::kMemBufUsed;
  // Memory statistics
  mem_mng->mps_.total_mem_size_ += mem_block->size();
  mem_mng->mps_.total_used_mem_size_ += mem_buf->size_;
  mem_mng->mps_.UpdatePeakSize();
  if (mem_buf_status == DynamicMemBufStatus::kMemBufIdle) {
    mem_mng->mps_.total_idle_mem_size_ += source_size - mem_buf->size_;
  } else if (mem_buf_status == DynamicMemBufStatus::kMemBufEagerFree) {
    mem_mng->mps_.total_eager_free_mem_size_ += source_size - mem_buf->size_;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unsupported mem_buf_status : " << mem_buf_status << ".";
  }
  MS_LOG(DEBUG) << "Usage: used size : " << TotalUsedMemStatistics()
                << ", used by event size : " << TotalUsedByEventMemStatistics()
                << ", idle size : " << TotalIdleMemStatistics()
                << ", eager free size : " << TotalEagerFreeMemStatistics() << ".";
  return mem_buf->device_addr_;
}

size_t DynamicMemPoolBestFit::CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool) {
  auto device_free_mem_size = free_mem_size();
  if (device_free_mem_size < size && common::IsNeedProfileMemory()) {
    device_free_mem_size = size;
  }
  if (device_free_mem_size < size) {
    MS_LOG(INFO) << "Memory not enough: current free memory size[" << device_free_mem_size
                 << "] is smaller than required size[" << size << "].";
    return 0;
  }
  // The memory of the device is too small, which may cause the new application to fail.
  if (device_free_mem_size < kMinimumAllocMem) {
    MS_LOG(INFO) << "Device memory size [" << device_free_mem_size << "] is smaller than minimum alloc size ["
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
  eager_free_count_++;

  auto eager_free_mem_func = [&](MemStatusManagerPtr &mem_mng) {
    const auto &stream_ids = mem_mng->GetStreamIds();
    for (const auto &stream_id : stream_ids) {
      auto key = std::make_pair(stream_id, DynamicMemBufStatus::kMemBufIdle);
      auto &&iter = mem_mng->mem_bufs_.find(key);
      if (iter == mem_mng->mem_bufs_.end()) {
        continue;
      }
      auto &mem_buf_map = iter->second;
      for (auto &size_mem_buf : mem_buf_map) {
        auto &mem_buf = size_mem_buf.second;
        auto [mem_block, iter, mem_mng] = FindByStrictAddr(mem_buf->device_addr_);
        if (PreCombineMemBuf(mem_buf, mem_mng)) {
          CombineMemBuf(mem_block, iter, mem_mng, DynamicMemBufStatus::kMemBufIdle,
                        DynamicMemBufStatus::kMemBufEagerFree);
        }
      }
      mem_mng->mem_bufs_.erase(iter);
    }
    // After memory free idle, do eager free.
    size_t free_size = 0;
    size_t real_free_size = 0;
    for (const auto &stream_id : stream_ids) {
      auto key = std::make_pair(stream_id, DynamicMemBufStatus::kMemBufEagerFree);
      auto &&iter = mem_mng->mem_bufs_.find(key);
      if (iter == mem_mng->mem_bufs_.end()) {
        continue;
      }
      auto &mem_buf_map = iter->second;
      for (auto &size_mem_buf : mem_buf_map) {
        auto &mem_buf = size_mem_buf.second;
        free_size += mem_buf->size_;
        MS_LOG(DEBUG) << "Eager free address : " << mem_buf->device_addr_ << ".";
        real_free_size += FreeDeviceMemByEagerFree(mem_buf->device_addr_, mem_buf->size_);
      }
    }

    return std::make_pair(free_size, real_free_size);
  };

  const auto [persistent_free_size, persistent_real_free_size] = eager_free_mem_func(persistent_mem_);
  const auto [common_free_size, common_real_free_size] = eager_free_mem_func(common_mem_);
  auto free_size = persistent_free_size + common_free_size;
  auto real_free_size = persistent_real_free_size + common_real_free_size;
  static bool is_enable_memory_statistics = common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat);
  if (is_enable_memory_statistics) {
    std::cout << "Total eager free memory : " << free_size << ", real free : " << real_free_size
              << ", not free size: " << (free_size - real_free_size) << "." << std::endl;
  }
  MS_LOG(INFO) << "Eager free count : " << eager_free_count_ << ", free memory : " << free_size
               << ", real free : " << real_free_size << ", not free size: " << (free_size - real_free_size) << ".";
  return real_free_size;
}

void DynamicMemPoolBestFit::DefragMemory() {
  MS_LOG(DEBUG) << "Start defrag memory.";
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif

  // eager free count initialize with 0, and increase by initializing persistent pool and common pool.
  if (eager_free_count_ <= 2L) {
    MS_LOG(DEBUG) << "Exit defrag memory since eager free count is 0.";
    return;
  }
  if (last_eager_free_count_ == eager_free_count_) {
    MS_LOG(DEBUG) << "Exit defrag memory since last eager free count equals to eager free count : "
                  << last_eager_free_count_ << ".";
    return;
  }

  MS_LOG(INFO) << "Try to defrag memory.";
  if (!SyncAllStreams()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Sync all streams failed.";
  }
  FreeIdleMemsByEagerFree();
  last_eager_free_count_ = eager_free_count_;
}

bool DynamicMemPoolBestFit::IsSplit(size_t tensor_size, size_t mem_buf_size) const {
  return mem_buf_size - tensor_size >= kDynamicMemAlignSize;
}

void DynamicMemPoolBestFit::SplitMemBuf(size_t size, const DynamicMemBufPtr &mem_buf,
                                        const MemStatusManagerPtr &mem_mng, uint32_t stream_id) {
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
  auto new_mem_buf = std::make_shared<DynamicMemBuf>(newbuf_addr, mem_buf->status_, newbuf_size, stream_id);
  // Add map of new memory buf in the block
  (void)mem_block->block_all_mem_buf_map_.emplace(newbuf_addr, new_mem_buf);
  mem_mng->AddMemBuf(new_mem_buf);
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
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  FreeTensorMemInner(device_addr);
}

void DynamicMemPoolBestFit::FreeTensorMemInner(const DeviceMemPtr &device_addr) {
  auto [mem_block, iter, mem_mng] = FindByStrictAddr(device_addr);
  if (mem_block == nullptr) {
    // Maybe destroy the memory pool first, then destroy the address, so this is normal case.
    MS_LOG(DEBUG) << "Can't find the mem_block of the device address[" << device_addr << "].";
    return;
  }
  auto mem_buf = iter->second;
  MS_EXCEPTION_IF_NULL(mem_buf);
  if (PreCombineMemBuf(mem_buf, mem_mng)) {
    CombineMemBuf(mem_block, iter, mem_mng, mem_buf->status_, DynamicMemBufStatus::kMemBufIdle);
    if (IsMemoryPoolRecycle()) {
      (void)mem_bufs_.erase(device_addr);
    }
    MS_LOG(DEBUG) << "Free memory details, name:" << DynamicMemAllocatorDebugInfo::GetDebugInfo().name_
                  << ", address:" << device_addr << ", total allocated mem:" << TotalMemStatistics()
                  << "B, peak used mem:" << UsedMemPeakStatistics() << "B, in used mem:" << TotalUsedMemStatistics()
                  << "B, used by event mem:" << TotalUsedByEventMemStatistics()
                  << "B, actual peak used mem:" << ActualPeakStatistics()
                  << "B, total idle mem:" << TotalIdleMemStatistics() << "B.";
  }
}

// PreCombineMemBuf judge status for mem buf can be combined or not.
// If there are no events recorded on mem buf, return true to release mem buf.
// If there are events recorded on mem buf, change status of mem buf to kMemBufUsedByEvent, and return false.
// Note: Before release mem buf by event, must make share that the status of mem buf is kMemBufUsedByEvent,
// or wait event may release mem buf incorrectly.
bool DynamicMemPoolBestFit::PreCombineMemBuf(const DynamicMemBufPtr &mem_buf, const MemStatusManagerPtr &mem_mng) {
  auto device_addr = mem_buf->device_addr_;
  if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsed && !mem_buf->IsEventNotUsed()) {
    mem_buf->status_ = DynamicMemBufStatus::kMemBufUsedByEvent;
    mem_mng->mps_.total_used_mem_size_ -= mem_buf->size_;
    mem_mng->mps_.total_used_by_event_mem_size_ += mem_buf->size_;
    MS_LOG(DEBUG) << "Combine mem buf exit since mem buf is used by event, device_addr : " << device_addr
                  << ", used by event mem size : " << mem_mng->mps_.total_used_by_event_mem_size_ << ".";
    return false;
  }

  if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent && !mem_buf->IsEventNotUsed()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Combine mem buf failed as mem buf can not be freed, device_addr : " << device_addr
                               << ".";
  }

  MS_LOG(DEBUG) << "Pre combine mem buf address : " << mem_buf->device_addr_ << " success.";
  return true;
}

void DynamicMemPoolBestFit::CombineMemBuf(const DynamicMemBlockPtr &mem_block,
                                          const DeviceAddrMapMemBuf::iterator &iter, const MemStatusManagerPtr &mem_mng,
                                          DynamicMemBufStatus origin_status, DynamicMemBufStatus target_status) {
  const auto &mem_buf = iter->second;
  MS_LOG(DEBUG) << "Combine mem buf release mem buf, device_addr : " << mem_buf->device_addr_ << ".";

// report memory data to profiler
#ifdef ENABLE_DEBUGGER
  static auto profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
  if (profiler_inst->GetEnableFlag() && profiler_inst->GetProfileMemoryFlag()) {
    profiler_inst->RecordMemoryPoolInfo(TotalUsedMemStatistics(), TotalMemStatistics(),
                                        TotalUsedByEventMemStatistics());
  }
#endif

  if (common::IsNeedProfileMemory()) {
    MS_LOG(WARNING) << "Need Profile Memory, Memory pool free, total mem: " << TotalMemStatistics()
                    << ", peak mem: " << UsedMemPeakStatistics() << ", in use mem: " << TotalUsedMemStatistics()
                    << ", used by event mem: " << TotalUsedByEventMemStatistics()
                    << ", device address addr: " << mem_buf->device_addr_ << ", size: " << mem_buf->size_;
  }
  if (device::tracker::MemTrackerManager::GetInstance().IsEnabled() &&
      target_status == DynamicMemBufStatus::kMemBufIdle) {
    device::tracker::CALL_MEMORY_TRACKER(FreeMemBlock, mem_buf->device_addr_, TotalUsedMemStatistics(),
                                         TotalMemStatistics());
  }

  if (mem_buf->status_ != origin_status) {
    DumpDynamicMemPoolDebugInfo();
    MS_LOG(EXCEPTION) << "Find the mem_buf status : " << mem_buf->status_
                      << " is not equal to origin status : " << origin_status << ", mem_buf_address["
                      << mem_buf->device_addr_ << "].";
  }
  mem_buf->status_ = target_status;
  if (origin_status == DynamicMemBufStatus::kMemBufUsed) {
    if (mem_mng->mps_.total_used_mem_size_ < mem_buf->size_) {
      DumpDynamicMemPoolDebugInfo();
      MS_LOG(EXCEPTION) << "The total used mem size : " << mem_mng->mps_.total_used_mem_size_
                        << " is less than the size of membuf : " << mem_buf->size_ << ".";
    }
    mem_mng->mps_.total_used_mem_size_ -= mem_buf->size_;
  } else if (origin_status == DynamicMemBufStatus::kMemBufUsedByEvent) {
    if (mem_mng->mps_.total_used_by_event_mem_size_ < mem_buf->size_) {
      DumpDynamicMemPoolDebugInfo();
      MS_LOG(EXCEPTION) << "The total used by event mem size : " << mem_mng->mps_.total_used_by_event_mem_size_
                        << " is less than the size of membuf : " << mem_buf->size_ << ".";
    }
    mem_mng->mps_.total_used_by_event_mem_size_ -= mem_buf->size_;
    MS_LOG(DEBUG) << "Combime mem buf for addr : " << mem_buf->device_addr_
                  << ", used by event mem size : " << mem_mng->mps_.total_used_by_event_mem_size_ << ".";
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
  if (target_status == DynamicMemBufStatus::kMemBufIdle) {
    mem_mng->mps_.total_idle_mem_size_ += mem_buf->size_;
  } else if (target_status == DynamicMemBufStatus::kMemBufEagerFree) {
    mem_mng->mps_.total_eager_free_mem_size_ += mem_buf->size_;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unsupported target status : " << target_status << ".";
  }
  // Combine backward(combine the next_mem_buf to mem_buf)
  auto next_iter = iter;
  (void)next_iter++;
  if (next_iter != mem_block->block_all_mem_buf_map_.end()) {
    auto next_mem_buf = next_iter->second;
    MS_EXCEPTION_IF_NULL(next_mem_buf);
    if (next_mem_buf->status_ == target_status) {
      mem_buf->size_ += next_mem_buf->size_;
      mem_mng->RemoveMemBuf(next_mem_buf);

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
      mem_mng->RemoveMemBuf(prev_mem_buf);
      prev_mem_buf->size_ += mem_buf->size_;
      (void)mem_block->block_all_mem_buf_map_.erase(iter);
      forward_combine = true;
    }
  }

  if (forward_combine) {
    mem_mng->AddMemBuf(prev_mem_buf);
  } else {
    mem_mng->AddMemBuf(mem_buf);
  }
}

std::tuple<DynamicMemBlockPtr, DeviceAddrMapMemBuf::iterator, MemStatusManagerPtr>
DynamicMemPoolBestFit::FindByStrictAddr(const DeviceMemPtr &device_addr) const {
  MS_EXCEPTION_IF_NULL(device_addr);
  // Find in the common pool.
  auto mem_block = FindMemBlock(device_addr, common_mem_);
  if (mem_block != nullptr) {
    const auto &iter = mem_block->block_all_mem_buf_map_.find(device_addr);
    if (iter != mem_block->block_all_mem_buf_map_.end()) {
      return std::make_tuple(mem_block, iter, common_mem_);
    }
  }

  // Find in the persistent pool.
  mem_block = FindMemBlock(device_addr, persistent_mem_);
  if (mem_block != nullptr) {
    const auto &iter = mem_block->block_all_mem_buf_map_.find(device_addr);
    if (iter != mem_block->block_all_mem_buf_map_.end()) {
      return std::make_tuple(mem_block, iter, persistent_mem_);
    }
  }

  DeviceAddrMapMemBuf empty_map;
  return std::make_tuple(nullptr, empty_map.end(), common_mem_);
}

void DynamicMemPoolBestFit::FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                               const std::vector<DeviceMemPtr> &keep_addrs,
                                               const std::vector<size_t> &keep_addr_sizes) {
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif

  for (auto &free_addr : free_addrs) {
    FreeTensorMemInner(free_addr);
  }

  MS_EXCEPTION_IF_CHECK_FAIL((keep_addrs.size() == keep_addr_sizes.size()), "The keep addrs size is wrong.");
  for (size_t i = 0; i < keep_addrs.size(); ++i) {
    KeepTensorMemByAddr(keep_addrs[i], keep_addr_sizes[i]);
  }
}

void DynamicMemPoolBestFit::KeepTensorMemByAddr(const DeviceMemPtr &device_addr, size_t size) {
  MS_EXCEPTION_IF_NULL(device_addr);
  // Fetch the memblock and membuf by the device address.
  auto [mem_block, mem_buf, mem_mng] = FindByKeepAddr(device_addr);
  if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    device::tracker::CALL_MEMORY_TRACKER(AllocMemBlock, device_addr, size, GetMemoryPoolType(), ActualPeakStatistics(),
                                         TotalUsedMemStatistics(), TotalMemStatistics(), mem_block->stream_id_);
  }
  MS_EXCEPTION_IF_NULL(mem_block);
  MS_EXCEPTION_IF_NULL(mem_buf);
  MS_EXCEPTION_IF_NULL(mem_mng);
  if (mem_buf->status_ != DynamicMemBufStatus::kMemBufIdle) {
    DumpDynamicMemPoolDebugInfo();
    MS_LOG(EXCEPTION) << "The membuf status isn't idle for addr:" << device_addr << ", size:" << size
                      << ", find the mem buf addr:" << mem_buf->device_addr_ << ", size:" << mem_buf->size_;
  }

  // Calculate the size of left and right split membuf.
  size_t split_left_size = CalAddressOffset(device_addr, mem_buf->device_addr_);
  MS_EXCEPTION_IF_CHECK_FAIL((mem_buf->size_ >= (split_left_size + size)), "The split size is wrong.");
  size_t split_right_szie = mem_buf->size_ - split_left_size - size;

  // Split the left membuf.
  mem_mng->RemoveMemBuf(mem_buf);
  if (split_left_size == 0) {
    mem_buf->status_ = DynamicMemBufStatus::kMemBufUsed;
    mem_buf->size_ = size;
    mem_buf->allocator_name_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().name_;
    mem_buf->allocator_type_ = DynamicMemAllocatorDebugInfo::GetDebugInfo().type_;
  } else {
    mem_buf->size_ = split_left_size;
    mem_mng->AddMemBuf(mem_buf);

    auto used_mem_buf = std::make_shared<DynamicMemBuf>(
      device_addr, DynamicMemBufStatus::kMemBufUsed, size, mem_block->stream_id_,
      DynamicMemAllocatorDebugInfo::GetDebugInfo().name_, DynamicMemAllocatorDebugInfo::GetDebugInfo().type_);
    (void)mem_block->block_all_mem_buf_map_.emplace(device_addr, used_mem_buf);
  }

  // Split the right membuf.
  if (split_right_szie > 0) {
    DeviceMemPtr right_buf_addr = AddressOffset(device_addr, size);
    auto right_mem_buf = std::make_shared<DynamicMemBuf>(right_buf_addr, DynamicMemBufStatus::kMemBufIdle,
                                                         split_right_szie, mem_block->stream_id_);
    (void)mem_block->block_all_mem_buf_map_.emplace(right_buf_addr, right_mem_buf);
    mem_mng->AddMemBuf(right_mem_buf);
  }

  // Memory statistics.
  mem_mng->mps_.total_used_mem_size_ += size;
  mem_mng->mps_.UpdatePeakSize();
  mem_mng->mps_.total_idle_mem_size_ -= size;
  MS_LOG(DEBUG) << "Keep memory details, name:" << DynamicMemAllocatorDebugInfo::GetDebugInfo().name_
                << ", address:" << device_addr << ", size:" << size << "B, total allocated mem:" << TotalMemStatistics()
                << "B, peak used mem:" << UsedMemPeakStatistics() << "B, in used mem:" << TotalUsedMemStatistics()
                << "B, used by event mem:" << TotalUsedByEventMemStatistics()
                << "B, actual peak used mem:" << ActualPeakStatistics()
                << "B, total idle mem:" << TotalIdleMemStatistics() << "B.";
}

DynamicMemBufPtr DynamicMemPoolBestFit::FindMemBufByKeepAddr(const DeviceMemPtr &device_addr,
                                                             const DynamicMemBlockPtr &mem_block) const {
  MS_EXCEPTION_IF_NULL(device_addr);
  MS_EXCEPTION_IF_NULL(mem_block);
  auto &&iter = mem_block->block_all_mem_buf_map_.upper_bound(device_addr);
  if (iter != mem_block->block_all_mem_buf_map_.begin()) {
    return (--iter)->second;
  }
  return nullptr;
}

std::tuple<DynamicMemBlockPtr, DynamicMemBufPtr, MemStatusManagerPtr> DynamicMemPoolBestFit::FindByKeepAddr(
  const DeviceMemPtr &device_addr) const {
  MS_EXCEPTION_IF_NULL(device_addr);
  auto is_addr_in_membuf = [](const DeviceMemPtr &device_addr, const DynamicMemBufPtr &mem_buf) {
    return (mem_buf != nullptr) && (device_addr >= mem_buf->device_addr_) &&
           (mem_buf->size_ >= CalAddressOffset(device_addr, mem_buf->device_addr_));
  };

  // Find in the common pool.
  auto mem_block = FindMemBlock(device_addr, common_mem_);
  if (mem_block != nullptr) {
    auto mem_buf = FindMemBufByKeepAddr(device_addr, mem_block);
    if (is_addr_in_membuf(device_addr, mem_buf)) {
      return std::make_tuple(mem_block, mem_buf, common_mem_);
    }
  }

  // Find in the persistent pool.
  mem_block = FindMemBlock(device_addr, persistent_mem_);
  if (mem_block != nullptr) {
    auto mem_buf = FindMemBufByKeepAddr(device_addr, mem_block);
    if (is_addr_in_membuf(device_addr, mem_buf)) {
      return std::make_tuple(mem_block, mem_buf, persistent_mem_);
    }
  }

  return std::make_tuple(nullptr, nullptr, common_mem_);
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
    mem_mng->Clear();
  };
  fn(common_mem_);
  fn(persistent_mem_);

  tracker::MemTrackerManager::GetInstance().Dump();
}

void DynamicMemPoolBestFit::DumpDynamicMemPoolStateInfo() {
  size_t total_used_size_list[kAllocatorTypeNum] = {0};
  static bool is_enable_memory_statistics = common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat) ||
                                            common::IsEnableRuntimeConfig(common::kRuntimeMemoryTrack);
  auto fn = [&](const MemStatusManagerPtr &mem_mng, const std::string &mem_type) {
    MS_EXCEPTION_IF_NULL(mem_mng);
    if (mem_mng->Empty()) {
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
      buf << ", block[" << i << "] stream id:" << mem_mng->mem_block_list_[i]->stream_id_
          << " block size:" << mem_mng->mem_block_list_[i]->mem_block_size_ / kMBToByte
          << "M idle size:" << (mem_mng->mem_block_list_[i]->mem_block_size_ - mem_block_used_size) / kMBToByte
          << "M actual size: " << (mem_mng->mem_block_list_[i]->get_actual_peak()) / kMBToByte << "M.";
    }
    std::ostringstream oss_buf;
    // Dump all the memory buf info
    oss_buf << mem_type << " pool info: Total allocated mem:" << mem_mng->mps_.total_mem_size_ / kMBToByte
            << "M, peak used mem:" << mem_mng->mps_.used_mem_peak_size_ / kMBToByte
            << "M, in used mem:" << mem_mng->mps_.total_used_mem_size_ / kMBToByte
            << "M, total use by event mem:" << mem_mng->mps_.total_used_by_event_mem_size_ / kMBToByte
            << "M, total idle mem:" << mem_mng->mps_.total_idle_mem_size_ / kMBToByte
            << "M. Block unit size:" << mem_mng->unit_size_ / kMBToByte
            << "M, block counts:" << mem_mng->mem_block_list_.size() << buf.str();
    if (is_enable_memory_statistics) {
      std::cout << "[MS_RUNTIME_PROF]" << oss_buf.str() << std::endl;
    }
    MS_LOG(INFO) << oss_buf.str();
  };

  fn(common_mem_, std::string(kCommonMem));
  fn(persistent_mem_, std::string(kPersistentParamMem));
  std::ostringstream oss_mem;
  oss_mem << "The dynamic memory pool total allocated mem:" << TotalMemStatistics() / kMBToByte
          << "M, min addr :" << GetMinUsingMemoryAddr()
          << ", max addr: " << (mem_bufs_.empty() ? nullptr : *(--mem_bufs_.end()))
          << ", peak used mem:" << UsedMemPeakStatistics() / kMBToByte
          << "M, actual peak used mem:" << ActualPeakStatistics() / kMBToByte
          << "M, in used mem:" << TotalUsedMemStatistics() / kMBToByte
          << "M, total used by event mem:" << TotalUsedByEventMemStatistics() / kMBToByte
          << "M, total idle mem:" << TotalIdleMemStatistics() / kMBToByte
          << "M, total eager free mem:" << TotalEagerFreeMemStatistics() / kMBToByte
          << "M. Weight used size:" << total_used_size_list[static_cast<int>(AllocatorType::kWeight)] / kMBToByte
          << "M, constant value used size:"
          << total_used_size_list[static_cast<int>(AllocatorType::kConstantValue)] / kMBToByte
          << "M, kernel output used size:"
          << total_used_size_list[static_cast<int>(AllocatorType::kKernelOutput)] / kMBToByte
          << "M, other used size:" << total_used_size_list[static_cast<int>(AllocatorType::kOther)] / kMBToByte << "M.";
  if (is_enable_memory_statistics) {
    std::cout << "[MS_RUNTIME_PROF]" << oss_mem.str() << std::endl;
  }
  MS_LOG(INFO) << oss_mem.str();
}

void DynamicMemPoolBestFit::DumpDynamicMemPoolDebugInfo() {
  auto fn = [](const MemStatusManagerPtr &mem_mng, const std::string &mem_type) {
    const auto &device_state = mem_mng->DumpMemBlockDebugInfo(mem_type);
    const auto &stream_ids = mem_mng->GetStreamIds();
    // Dump all the idle memory buf info.
    size_t total_idle_mem_in_mem_mng = 0;
    MS_LOG(WARNING) << mem_type << " all idle_mem_bufs info: counts[" << stream_ids.size() << "].";
    for (const auto &stream_id : stream_ids) {
      auto key = std::make_pair(stream_id, DynamicMemBufStatus::kMemBufIdle);
      const auto &&iter = mem_mng->mem_bufs_.find(key);
      if (iter == mem_mng->mem_bufs_.end()) {
        continue;
      }
      const auto &mem_buf_map = iter->second;
      MS_LOG(WARNING) << "  stream id : " << stream_id << ", idle mem buf info : count[]" << mem_buf_map.size() << "].";
      for (auto &&idle_iter = mem_buf_map.begin(); idle_iter != mem_buf_map.end(); idle_iter++) {
        auto &mem_buf = idle_iter->second;
        MS_EXCEPTION_IF_NULL(mem_buf);
        total_idle_mem_in_mem_mng += mem_buf->size_;
        MS_LOG(INFO) << " Idle mem_buf info: size[" << mem_buf->size_ << "] address[" << mem_buf->device_addr_
                     << "] status[" << kBufStatusString.at(mem_buf->status_) << "] stream id[" << mem_buf->stream_id_
                     << "].";
      }
    }
    // Dump all the eager free memory buf info.
    size_t total_eager_free_mem_in_mem_mng = 0;
    MS_LOG(WARNING) << mem_type << " all eager free mem_buf info: counts[" << stream_ids.size() << "].";
    for (const auto &stream_id : stream_ids) {
      auto key = std::make_pair(stream_id, DynamicMemBufStatus::kMemBufEagerFree);
      const auto &&iter = mem_mng->mem_bufs_.find(key);
      if (iter == mem_mng->mem_bufs_.end()) {
        continue;
      }
      const auto &mem_buf_map = iter->second;
      MS_LOG(WARNING) << "  stream id : " << stream_id << ", eager free mem buf info : count[]" << mem_buf_map.size()
                      << "].";
      for (auto &&idle_iter = mem_buf_map.begin(); idle_iter != mem_buf_map.end(); idle_iter++) {
        auto &mem_buf = idle_iter->second;
        MS_EXCEPTION_IF_NULL(mem_buf);
        total_eager_free_mem_in_mem_mng += mem_buf->size_;
        MS_LOG(INFO) << " Eager free mem_buf info: size[" << mem_buf->size_ << "] address[" << mem_buf->device_addr_
                     << "] status[" << kBufStatusString.at(mem_buf->status_) << "] stream id[" << mem_buf->stream_id_
                     << "].";
      }
    }
    // Dump the memory statistical info.
    MS_LOG(WARNING) << mem_type << " total allocated memory[" << device_state.total_mem_size_ << "], used memory["
                    << device_state.total_used_mem_size_ << "], used by event memory["
                    << device_state.total_used_by_event_mem_size_ << "], idle memory["
                    << device_state.total_idle_mem_size_ << "].";
    if (device_state.total_idle_mem_size_ != total_idle_mem_in_mem_mng) {
      MS_LOG(ERROR) << "Check error: the idle memory in the mem_block is not equal the global idle memory.";
    }
    if (device_state.total_used_by_event_mem_size_ != mem_mng->mps_.total_used_by_event_mem_size_) {
      MS_LOG(ERROR) << "Check error: the used by event memory in the mem_block is not equal the global idle memory.";
    }
    if (device_state.total_eager_free_mem_size_ != total_eager_free_mem_in_mem_mng) {
      MS_LOG(ERROR) << "Check error: the eager free memory in the mem_block is not equal the global eager free memory.";
    }
    if (device_state.total_mem_size_ != device_state.total_used_mem_size_ + device_state.total_used_by_event_mem_size_ +
                                          device_state.total_idle_mem_size_ + device_state.total_eager_free_mem_size_) {
      MS_LOG(ERROR) << "Check error: the the total memory : " << device_state.total_mem_size_
                    << " is not equal the sum of used memory : " << device_state.total_used_mem_size_
                    << ", use by event memory : " << device_state.total_used_by_event_mem_size_
                    << ", idle memory : " << device_state.total_idle_mem_size_
                    << " and eager free memory : " << device_state.total_eager_free_mem_size_ << ".";
    }
  };

  MS_LOG(WARNING) << "Start dump dynamic memory pool debug info.";
  fn(common_mem_, std::string(kCommonMem));
  fn(persistent_mem_, std::string(kPersistentParamMem));
  MS_LOG(WARNING) << "Finish dump dynamic memory pool debug info.";
}

// Element in vector : memory_stream_id, address
bool DynamicMemPoolBestFit::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                        const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                                        const DeviceEventPtr &event) {
  MS_LOG(DEBUG) << "Record event for, task_id_on_stream : " << task_id_on_stream
                << ", user_stream_id : " << user_stream_id
                << ", memory_stream_addresses size : " << memory_stream_addresses.size() << ", event : " << event.get()
                << ".";
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  for (auto &[memory_stream_id, address] : memory_stream_addresses) {
    auto &&mem_buf_tuple = FindByStrictAddr(address);
    auto mem_block = std::get<0>(mem_buf_tuple);
    // Output of somas sub graph may be used by somas sub graph inner node, address may not be kept in mem pool.
    if (mem_block == nullptr) {
      MS_LOG(DEBUG) << "Can't find memblock by address in memory pool.";
      continue;
    }
    auto mem_buf = (std::get<1>(mem_buf_tuple))->second;
    (void)mem_buf->RecordEvent(task_id_on_stream, user_stream_id, event);
    (void)stream_pair_addresses_[std::make_pair(user_stream_id, memory_stream_id)].emplace(mem_buf);
  }
  return true;
}

bool DynamicMemPoolBestFit::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  auto key = std::make_pair(user_stream_id, memory_stream_id);
  auto iter = stream_pair_addresses_.find(key);
  if (iter == stream_pair_addresses_.end()) {
    return false;
  }

  auto addresses = iter->second;
  for (const auto &address : addresses) {
    address->WaitEvent(task_id_on_stream, user_stream_id);
    // Remove event and try to free memory.
    if (address->IsEventNotUsed()) {
      iter->second.erase(address);
      if (address->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        FreeTensorMemInner(address->device_addr_);
      }
    }
  }
  MS_LOG(DEBUG) << "After release, bounded addresses size : " << iter->second.size()
                << ", used by event size : " << TotalUsedByEventMemStatistics() << ".";
  return true;
}

// WaitEvent is called before sync stream, so performance may not be the issue.
bool DynamicMemPoolBestFit::WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) {
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  for (auto &stream_pair_addresses : stream_pair_addresses_) {
    const auto &[user_stream, memory_stream] = stream_pair_addresses.first;
    if (memory_stream != memory_stream_id) {
      continue;
    }
    auto addresses = stream_pair_addresses.second;
    for (const auto &address : addresses) {
      address->WaitEvent(task_id_on_stream, user_stream);
      // Remove event and try to free memory.
      if (address->IsEventNotUsed()) {
        stream_pair_addresses.second.erase(address);
        if (address->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
          FreeTensorMemInner(address->device_addr_);
        }
      }
    }
  }
  MS_LOG(DEBUG) << "After release events, task_id_on_stream : " << task_id_on_stream
                << ", memory_stream_id : " << memory_stream_id
                << ", used by event size : " << TotalUsedByEventMemStatistics() << ".";
  return true;
}

bool DynamicMemPoolBestFit::SyncAllEvents() {
#ifdef __APPLE__
  std::lock_guard<SpinLock> spin_lock(spin_lock_);
#else
  std::lock_guard<std::mutex> locker(mutex_);
#endif
  return SyncAllEventsInner();
}

bool DynamicMemPoolBestFit::SyncAllEventsInner() {
  MS_LOG(DEBUG) << "Sync all events, stream_pair_addresses_ size : " << stream_pair_addresses_.size() << ".";
  if (stream_pair_addresses_.empty()) {
    return false;
  }

  std::set<DynamicMemBufPtr> carry_event_addresses;
  for (const auto &stream_pair_address : stream_pair_addresses_) {
    for (const auto &address : stream_pair_address.second) {
      (void)carry_event_addresses.emplace(address);
    }
  }
  for (auto &address : carry_event_addresses) {
    if (address->SyncAllEvents() && address->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
      FreeTensorMemInner(address->device_addr_);
    }
  }

  stream_pair_addresses_.clear();
  return true;
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
DynamicMemPoolBestFit::ExtractBlocksListInfo(const MemStatusManagerPtr &mem_mng) const {
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> blocks_list_info;
  for (auto iter = mem_mng->mem_block_list_.begin(); iter != mem_mng->mem_block_list_.end(); ++iter) {
    std::unordered_map<std::string, size_t> block_info;
    block_info[kBlockMemorySize] = (*iter)->size();
    block_info[kBlockStreamId] = (*iter)->stream_id_;
    blocks_list_info[(std::string *)(*iter)->device_addr()] = block_info;
  }
  return blocks_list_info;
}

// The statistics information.
size_t DynamicMemPoolBestFit::TotalMemStatistics() const {
  return common_mem_->mps_.total_mem_size_ + persistent_mem_->mps_.total_mem_size_;
}
size_t DynamicMemPoolBestFit::TotalUsedMemStatistics() const {
  return common_mem_->mps_.total_used_mem_size_ + persistent_mem_->mps_.total_used_mem_size_;
}
size_t DynamicMemPoolBestFit::TotalUsedByEventMemStatistics() const {
  return common_mem_->mps_.total_used_by_event_mem_size_ + persistent_mem_->mps_.total_used_by_event_mem_size_;
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
size_t DynamicMemPoolBestFit::MaxMemAllocatedStatistics() const {
  return common_mem_->mps_.temp_used_mem_peak_size_ + persistent_mem_->mps_.temp_used_mem_peak_size_;
}
size_t DynamicMemPoolBestFit::MaxMemReservedStatistics() const {
  return common_mem_->mps_.total_mem_size_ + persistent_mem_->mps_.total_mem_size_ -
         common_mem_->mps_.temp_total_mem_size_ - persistent_mem_->mps_.temp_total_mem_size_;
}
size_t DynamicMemPoolBestFit::ActualPeakStatistics() const {
  return common_mem_->CalActualPeak() + persistent_mem_->CalActualPeak();
}
std::unordered_map<std::string, std::size_t> DynamicMemPoolBestFit::BlockCountsStatistics() const {
  size_t common_mem_block_counts = common_mem_->mem_block_list_.size();
  size_t persistent_mem_block_counts = persistent_mem_->mem_block_list_.size();
  std::unordered_map<std::string, std::size_t> block_count_stats;
  block_count_stats[kCommonMemPoolType] = common_mem_block_counts;
  block_count_stats[kPersistentMemPoolType] = persistent_mem_block_counts;
  return block_count_stats;
}
std::unordered_map<std::string, std::size_t> DynamicMemPoolBestFit::BlockUnitSizeStatistics() const {
  size_t common_mem_block_unit_size = common_mem_->unit_size_;
  size_t persistent_mem_block_unit_size = persistent_mem_->unit_size_;
  std::unordered_map<std::string, std::size_t> block_unit_size_stats;
  block_unit_size_stats[kCommonMemPoolType] = common_mem_block_unit_size;
  block_unit_size_stats[kPersistentMemPoolType] = persistent_mem_block_unit_size;
  return block_unit_size_stats;
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
DynamicMemPoolBestFit::CommonMemBlocksInfoStatistics() const {
  return ExtractBlocksListInfo(common_mem_);
}
std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
DynamicMemPoolBestFit::PersistentMemBlocksInfoStatistics() const {
  return ExtractBlocksListInfo(persistent_mem_);
}
void DynamicMemPoolBestFit::ResetMaxMemReserved() const {
  common_mem_->mps_.temp_total_mem_size_ = common_mem_->mps_.total_mem_size_;
  persistent_mem_->mps_.temp_total_mem_size_ = persistent_mem_->mps_.total_mem_size_;
}
void DynamicMemPoolBestFit::ResetMaxMemAllocated() const {
  common_mem_->mps_.temp_total_used_mem_size_ = common_mem_->mps_.total_used_mem_size_;
  persistent_mem_->mps_.temp_total_used_mem_size_ = persistent_mem_->mps_.total_used_mem_size_;
  common_mem_->mps_.temp_total_used_by_event_mem_size_ = common_mem_->mps_.total_used_by_event_mem_size_;
  persistent_mem_->mps_.temp_total_used_by_event_mem_size_ = persistent_mem_->mps_.total_used_by_event_mem_size_;
  common_mem_->mps_.temp_used_mem_peak_size_ = 0;
  persistent_mem_->mps_.temp_used_mem_peak_size_ = 0;
}

size_t MemStatusManager::CalActualPeak() {
  if (mem_block_insertion_order_.empty()) {
    return 0;
  }
  size_t actual_peak = total_block_size_;
  const auto &end_block = mem_block_insertion_order_.back();
  MS_EXCEPTION_IF_NULL(end_block);
  actual_peak -= end_block->size();
  actual_peak += end_block->get_actual_peak();
  return actual_peak;
}

bool DynamicMemBuf::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id, const DeviceEventPtr &event) {
  MS_EXCEPTION_IF_NULL(event);
  if (events_ == nullptr) {
    events_ = std::make_shared<std::unordered_map<uint32_t, std::shared_ptr<std::list<TaskIdOnStreamEvent>>>>();
  }
  std::shared_ptr<std::list<TaskIdOnStreamEvent>> event_list = nullptr;
  auto iter = events_->find(user_stream_id);
  if (iter == events_->end()) {
    event_list = std::make_shared<std::list<TaskIdOnStreamEvent>>();
    (void)events_->emplace(user_stream_id, event_list);
  } else {
    event_list = iter->second;
    MS_EXCEPTION_IF_NULL(event_list);
  }
  (void)event_list->emplace_back(task_id_on_stream, event);
  return true;
}

bool DynamicMemBuf::WaitEvent(uint32_t task_id_on_stream, uint32_t user_stream_id) {
  if (events_ == nullptr) {
    return false;
  }
  auto iter = events_->find(user_stream_id);
  if (iter == events_->end()) {
    return false;
  }
  auto &event_list = iter->second;
  MS_EXCEPTION_IF_NULL(event_list);
  // Pop all element in list that not bigger than task_id_on_stream.
  while (!event_list->empty() && event_list->front().first <= task_id_on_stream) {
    event_list->pop_front();
  }
  // Remove list if event list is empty.
  if (event_list->empty()) {
    events_->erase(iter);
  }
  return true;
}

bool DynamicMemBuf::IsEventNotUsed() { return events_ == nullptr ? true : events_->empty(); }

bool DynamicMemBuf::SyncAllEvents() {
  if (IsEventNotUsed()) {
    return false;
  }

  for (auto iter = events_->begin(); iter != events_->end();) {
    auto &event_list = iter->second;
    MS_EXCEPTION_IF_NULL(event_list);
    for (auto list_iter = event_list->begin(); list_iter != event_list->end();) {
      auto &event = list_iter->second;
      // Sync event if event is not arrived.
      if (!event->QueryEvent()) {
        event->SyncEvent();
      }
      list_iter = event_list->erase(list_iter);
    }
    if (event_list->empty()) {
      // list is empty, erase list in map.
      iter = events_->erase(iter);
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Event list is not empty.";
    }
  }
  return events_->empty();
}

void MemStatusManager::AddMemBlock(const DynamicMemBlockPtr &mem_block, uint32_t stream_id) {
  auto iter = mem_blocks_.find(stream_id);
  if (iter != mem_blocks_.end()) {
    DoAddMemBlock(mem_block, &iter->second);
  } else {
    (void)mem_blocks_.emplace(stream_id, std::vector<DynamicMemBlockPtr>{mem_block});
  }

  DoAddMemBlock(mem_block, &mem_block_list_);
  mem_block_insertion_order_.emplace_back(mem_block);
  total_block_size_ += mem_block->size();
}

void MemStatusManager::DoAddMemBlock(const DynamicMemBlockPtr &mem_block,
                                     std::vector<DynamicMemBlockPtr> *mem_block_list) {
  auto iter = std::upper_bound(mem_block_list->begin(), mem_block_list->end(), mem_block->device_addr(),
                               [](const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block) {
                                 return device_addr < mem_block->device_addr();
                               });
  (void)mem_block_list->insert(iter, mem_block);
}

SizeMapMemBuf &MemStatusManager::GetOrCreateMemBufMap(uint32_t stream_id, DynamicMemBufStatus status) {
  return mem_bufs_[std::make_pair(stream_id, status)];
}

void MemStatusManager::AddMemBuf(const DynamicMemBufPtr &mem_buf) {
  auto key = std::make_pair(mem_buf->stream_id_, mem_buf->status_);
  auto &mem_buf_map = mem_bufs_[key];
  (void)mem_buf_map.emplace(mem_buf->size_, mem_buf);
}

void MemStatusManager::RemoveMemBuf(const DynamicMemBufPtr &mem_buf) {
  auto key = std::make_pair(mem_buf->stream_id_, mem_buf->status_);
  auto &mem_buf_map = mem_bufs_[key];
  auto &&iter = mem_buf_map.equal_range(mem_buf->size_);
  while (iter.first != iter.second) {
    if (iter.first->second->device_addr_ == mem_buf->device_addr_) {
      (void)mem_buf_map.erase(iter.first);
      return;
    }
    (void)iter.first++;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Remove mem buf failed, address : " << mem_buf->device_addr_ << ".";
}

void MemStatusManager::Clear() noexcept {
  mem_blocks_.clear();
  mem_block_list_.clear();
  mem_bufs_.clear();
}

const DeviceState MemStatusManager::DumpMemBlockDebugInfo(const std::string &mem_type) {
  DeviceState device_state;
  // Dump the memory block info and memory buf info.
  MS_LOG(WARNING) << mem_type << " all mem_block info: counts[" << mem_block_list_.size() << "].";
  for (auto iter = mem_block_list_.begin(); iter != mem_block_list_.end(); ++iter) {
    device_state.total_mem_size_ += (*iter)->size();
    auto mem_buf_map = (*iter)->block_all_mem_buf_map_;
    MS_LOG(WARNING) << " MemBlock info: number[" << iter - mem_block_list_.begin() << "] mem_buf_counts["
                    << mem_buf_map.size() << "] base_address[" << (*iter)->device_addr() << "] block_size["
                    << (*iter)->size() << "] stream id[" << (*iter)->stream_id_ << "].";
    for (auto iter_mem_buf = mem_buf_map.begin(); iter_mem_buf != mem_buf_map.end(); ++iter_mem_buf) {
      auto mem_buf = iter_mem_buf->second;
      MS_EXCEPTION_IF_NULL(mem_buf);
      if (mem_buf->status_ == DynamicMemBufStatus::kMemBufIdle) {
        device_state.total_idle_mem_size_ += mem_buf->size_;
      } else if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsed) {
        device_state.total_used_mem_size_ += mem_buf->size_;
      } else if (mem_buf->status_ == DynamicMemBufStatus::kMemBufEagerFree) {
        device_state.total_eager_free_mem_size_ += mem_buf->size_;
      } else if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        device_state.total_used_by_event_mem_size_ += mem_buf->size_;
      } else {
        MS_LOG(INTERNAL_EXCEPTION) << "Unknown mem buf status : " << mem_buf->status_ << ".";
      }
      MS_LOG(INFO) << "  MemBuf info: address[" << mem_buf->device_addr_ << "] size[" << mem_buf->size_ << "] status["
                   << kBufStatusString.at(mem_buf->status_) << "] name["
                   << (mem_buf->allocator_name_.empty() ? "Unknown" : mem_buf->allocator_name_) << "] type["
                   << kAllocatorTypeString.at(mem_buf->allocator_type_) << "] stream id[" << mem_buf->stream_id_
                   << "].";
    }
  }
  return device_state;
}
}  // namespace device
}  // namespace mindspore
