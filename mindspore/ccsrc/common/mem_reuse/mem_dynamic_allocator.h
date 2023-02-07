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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_

#include <memory>
#include <map>
#include <vector>
#include <algorithm>
#include <utility>
#include <thread>
#include <mutex>
#include <string>
#include "utils/ms_utils.h"
#include "include/backend/visible.h"
#ifdef __APPLE__
#include "mindrt/include/async/spinlock.h"
#endif

namespace mindspore {
namespace device {
using DeviceMemPtr = void(*);

// The status of memory buf.
enum class DynamicMemBufStatus : int { kMemBufIdle, kMemBufUsed };

// Memory allocator type is used to record the memory classification statistics information.
enum class AllocatorType : int { kWeight, kConstantValue, kKernelOutput, kOther };
static const int ALLOCATOR_TYPE_NUM = 4;

// Alloc memory aligned according to 512 bytes.
static const size_t DYNAMIC_MEM_ALIGN_SIZE = 512;

// The minimum unit size (1G) of memory block used for dynamic extend.
static const size_t DYNAMIC_MEM_ALLOC_UNIT_SIZE = 1024 << 20;

// The Comparator of device address from small to large.
struct DeviceAddrCmp {
  bool operator()(const DeviceMemPtr &addr1, const DeviceMemPtr &addr2) const { return addr1 < addr2; }
};

// Recording information for debugging the memory allocator.
struct AllocatorDebugInfo {
  std::string name_{"Unknown"};
  AllocatorType type_{AllocatorType::kOther};
  int input_index_{-1};
  int output_index_{-1};
};

// The AllocatorDebugInfo wrapper which is the local thread for the dynamic memory pool.
class DynamicMemAllocatorDebugInfo {
 public:
  static AllocatorDebugInfo &GetDebugInfo() noexcept { return debug_info_; }

  // Set the debug info when memory alloc.
  static void SetDebugInfo(const std::string &name, AllocatorType type, int input_index = -1, int output_index = -1) {
    debug_info_.name_ = name;
    debug_info_.type_ = type;
    debug_info_.input_index_ = input_index;
    debug_info_.output_index_ = output_index;
  }

 private:
  DynamicMemAllocatorDebugInfo() = default;
  virtual ~DynamicMemAllocatorDebugInfo() = default;
  DISABLE_COPY_AND_ASSIGN(DynamicMemAllocatorDebugInfo);

  static thread_local AllocatorDebugInfo debug_info_;
};

// Memory buf is the smallest operation object of dynamic memory pool.
struct DynamicMemBuf {
  DynamicMemBuf(DeviceMemPtr addr, DynamicMemBufStatus status, size_t size,
                const std::string &allocator_name = "Unknown", AllocatorType allocator_type = AllocatorType::kOther)
      : device_addr_(addr),
        status_(status),
        size_(size),
        allocator_name_(allocator_name),
        allocator_type_{allocator_type} {}
  DeviceMemPtr device_addr_;
  DynamicMemBufStatus status_;
  size_t size_;

  // Debug info.
  std::string allocator_name_;
  AllocatorType allocator_type_;
};
using DynamicMemBufPtr = std::shared_ptr<DynamicMemBuf>;
// Multimap key is the tensor size, for finding the idle memory buf by tensor size.
using SizeMapMemBuf = std::multimap<size_t, DynamicMemBufPtr>;
// Map key is the device address, for finding the used memory buf in memory block by device address.
using DeviceAddrMapMemBuf = std::map<DeviceMemPtr, DynamicMemBufPtr, DeviceAddrCmp>;

// Memory block is composed of memory buf.
class DynamicMemBlock {
 public:
  DynamicMemBlock() = default;
  DynamicMemBlock(DeviceMemPtr addr_base, size_t size) : device_addr_base_(addr_base), mem_block_size_(size) {}
  ~DynamicMemBlock() { block_all_mem_buf_map_.clear(); }
  const DeviceMemPtr &device_addr() const { return device_addr_base_; }
  size_t size() const { return mem_block_size_; }

 private:
  friend class DynamicMemPoolBestFit;

  // The map of all memory buf in this memory block by device address.
  DeviceAddrMapMemBuf block_all_mem_buf_map_;

  DeviceMemPtr device_addr_base_{nullptr};
  size_t mem_block_size_{0};
};
using DynamicMemBlockPtr = std::shared_ptr<DynamicMemBlock>;

struct DeviceState {
  // Memory allocated from device
  size_t total_mem_size_{0};
  // Memory in use
  size_t total_used_mem_size_{0};
  // Maximum peak memory usage
  size_t used_mem_peak_size_{0};
};

struct MemStatusManager {
  size_t unit_size_{DYNAMIC_MEM_ALLOC_UNIT_SIZE};
  // Mem pool state
  DeviceState mps_;
  std::vector<DynamicMemBlockPtr> mem_block_list_;
  // The map of all idle memory buf by size.
  SizeMapMemBuf idle_mem_buf_map_;
  void clear() noexcept {
    mem_block_list_.clear();
    idle_mem_buf_map_.clear();
  }
};
using MemStatusManagerPtr = std::shared_ptr<MemStatusManager>;

// The main class of dynamic memory pool.
class BACKEND_EXPORT DynamicMemPoolBestFit {
 public:
  DynamicMemPoolBestFit()
      : persistent_mem_(std::make_shared<MemStatusManager>()), common_mem_(std::make_shared<MemStatusManager>()) {}
  virtual ~DynamicMemPoolBestFit();

  // The main program entry of memory alloc.
  DeviceMemPtr AllocTensorMem(size_t size, bool from_persistent_mem = false);
  // The main program entry of continuous memory alloc.
  std::vector<DeviceMemPtr> AllocContinuousTensorMem(const std::vector<size_t> &size_list);
  // The main program entry of memory free.
  void FreeTensorMem(const DeviceMemPtr &device_addr);

  // Release the real device memory.
  void ReleaseDeviceRes();

  // Get the minimum memory unit size using for dynamic extend.
  size_t MemAllocUnitSize(bool from_persistent_mem = false) const;
  // Set the minimum memory unit size using for dynamic extend.
  void SetMemAllocUintSize(size_t common_size, size_t persist_size = DYNAMIC_MEM_ALLOC_UNIT_SIZE);

  // The statistics information.
  size_t TotalMemStatistics() const {
    return common_mem_->mps_.total_mem_size_ + persistent_mem_->mps_.total_mem_size_;
  }
  size_t TotalUsedMemStatistics() const {
    return common_mem_->mps_.total_used_mem_size_ + persistent_mem_->mps_.total_used_mem_size_;
  }
  size_t UsedMemPeakStatistics() const {
    return common_mem_->mps_.used_mem_peak_size_ + persistent_mem_->mps_.used_mem_peak_size_;
  }

  // Display the brief state information of memory block and memory buf.
  void DumpDynamicMemPoolStateInfo();
  // Display the detailed debug information of memory block and memory buf.
  void DumpDynamicMemPoolDebugInfo();

  // The related interface of device memory real operation, needs override by device type.
  virtual size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) = 0;
  virtual bool FreeDeviceMem(const DeviceMemPtr &addr) = 0;
  virtual size_t free_mem_size() = 0;
  // Set mem pool block size
  virtual void SetMemPoolBlockSize(size_t available_device_mem_size);

 protected:
  const MemStatusManagerPtr &common_mem() const { return common_mem_; }
  const MemStatusManagerPtr &persistent_mem() const { return persistent_mem_; }
  // The real size by memory alloc aligned.
  virtual size_t AlignMemorySize(size_t size) const;
  // Calculate memory block required alloc size when adding the memory block.
  virtual size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem);

 private:
  // Find the idle memory buf by aligned size when memory alloc.
  DeviceMemPtr FindIdleMemBuf(size_t size, bool from_persistent_mem);
  // Add the memory block and memory buf when memory alloc not find the idle memory buf.
  DeviceMemPtr AddMemBlockAndMemBuf(size_t size, bool from_persistent_mem);
  // Judge whether need split the memory buf by alloc size and memory buf size.
  bool IsSplit(size_t tensor_size, size_t mem_buf_size) const;
  // Split the memory buf by alloc size.
  void SplitMemBuf(size_t size, const DynamicMemBufPtr &mem_buf, const MemStatusManagerPtr &mem_mng);
  // Find the memory block by device address.
  DynamicMemBlockPtr FindMemBlock(const DeviceMemPtr &device_addr, const MemStatusManagerPtr &mem_mgr) const;
  // The Comparator of memory block by device address, because memory blocks are arranged in order by device address.
  static bool CmpMemBlock(const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block);
  // Combine the memory buf when memory free, to avoid the memory fragmentation.
  void CombineMemBuf(const DynamicMemBlockPtr &mem_block, const DeviceMemPtr &device_addr,
                     const MemStatusManagerPtr &mem_mng);
  // Erase the idle memory buf by size and device address when idle memory buf is combined.
  void EraseIdleMemBuf(size_t size, const DeviceMemPtr &device_addr, const MemStatusManagerPtr &mem_mng) const;

#ifdef __APPLE__
  // There are some problems with using mutex on Mac, use spinlocks instead.
  SpinLock spin_lock_;
#else
  // Support multi-thread.
  std::mutex mutex_;
#endif
  MemStatusManagerPtr persistent_mem_{nullptr};
  MemStatusManagerPtr common_mem_{nullptr};
  // In the graph mode, the unit size set in the context will be modified through the FetchMemUnitSize function, so it
  // needs to be changed back after that
  size_t config_unit_size_{DYNAMIC_MEM_ALLOC_UNIT_SIZE};
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_
