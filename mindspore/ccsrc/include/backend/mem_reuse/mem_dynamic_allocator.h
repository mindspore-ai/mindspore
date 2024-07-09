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

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <string>
#include <tuple>

#include "utils/ms_utils.h"
#include "include/backend/visible.h"
#include "include/common/utils/stream_util.h"
#include "ir/device_event.h"
#ifdef __APPLE__
#include "mindrt/include/async/spinlock.h"
#endif

namespace mindspore {
namespace device {
// The status of memory buf.
enum class DynamicMemBufStatus : int { kMemBufIdle, kMemBufUsed, kMemBufEagerFree, kMemBufUsedByEvent };
// Memory allocator type is used to record the memory classification statistics information.
enum class AllocatorType : int { kWeight, kConstantValue, kKernelOutput, kGraphOutput, kWorkspace, kOther };
constexpr int kShiftOffset = 2;
constexpr int kAllocatorTypeNum = 6;
// Alloc memory aligned according to 512 bytes.
constexpr size_t kDynamicMemAlignSize = 512;
// The minimum unit size (1G) of memory block used for dynamic extend.
constexpr size_t kDynamicMemAllocUnitSize = 1024 << 20;

// The Comparator of device address from small to large.
using DeviceMemPtr = void(*);
struct DeviceAddrCmp {
  bool operator()(const DeviceMemPtr &addr1, const DeviceMemPtr &addr2) const { return addr1 < addr2; }
};

// The AllocatorDebugInfo wrapper which is the local thread for the dynamic memory pool.
class DynamicMemAllocatorDebugInfo;
// Memory buf is the smallest operation object of dynamic memory pool.
struct DynamicMemBuf;
using DynamicMemBufPtr = std::shared_ptr<DynamicMemBuf>;
// Multimap key is the tensor size, for finding the idle memory buf by tensor size.
using SizeMapMemBuf = std::multimap<size_t, DynamicMemBufPtr>;
// Map key is the device address, for finding the used memory buf in memory block by device address.
using DeviceAddrMapMemBuf = std::map<DeviceMemPtr, DynamicMemBufPtr, DeviceAddrCmp>;
// Memory block is composed of memory buf.
class DynamicMemBlock;
using DynamicMemBlockPtr = std::shared_ptr<DynamicMemBlock>;

struct MemStatusManager;
using MemStatusManagerPtr = std::shared_ptr<MemStatusManager>;

// pair has no hash method, need override it.
struct pair_hash {
  template <class L, class R>
  std::size_t operator()(const std::pair<L, R> &param) const {
    size_t hash = std::hash<L>{}(param.first);
    hash <<= (sizeof(size_t) << kShiftOffset);
    hash ^= std::hash<R>{}(param.second);
    return std::hash<size_t>{}(hash);
  }
};

// The main class of dynamic memory pool.
class BACKEND_EXPORT DynamicMemPoolBestFit {
 public:
  DynamicMemPoolBestFit()
      : persistent_mem_(std::make_shared<MemStatusManager>()), common_mem_(std::make_shared<MemStatusManager>()) {}
  virtual ~DynamicMemPoolBestFit();

  // The main program entry of memory alloc.
  DeviceMemPtr AllocTensorMem(size_t size, bool from_persistent_mem = false, bool need_recycle = false,
                              uint32_t stream_id = kDefaultStreamIndex);
  // The main program entry of continuous memory alloc.
  std::vector<DeviceMemPtr> AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                     uint32_t stream_id = kDefaultStreamIndex);
  // The main program entry of memory free.
  void FreeTensorMem(const DeviceMemPtr &device_addr);
  // The main program entry of part memorys free and part memorys keep.
  void FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs, const std::vector<DeviceMemPtr> &keep_addrs,
                          const std::vector<size_t> &keep_addr_sizes);

  // Release the real device memory.
  void ReleaseDeviceRes();

  // Get the minimum memory unit size using for dynamic extend.
  size_t MemAllocUnitSize(bool from_persistent_mem = false) const;
  // Set the minimum memory unit size using for dynamic extend.
  void SetMemAllocUintSize(size_t common_size, size_t persist_size = kDynamicMemAllocUnitSize);

  // Extract detailed block information
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> ExtractBlocksListInfo(
    const MemStatusManagerPtr &mem_mng) const;

  // The statistics information.
  size_t TotalMemStatistics() const;
  size_t TotalUsedMemStatistics() const;
  size_t TotalUsedByEventMemStatistics() const;
  size_t TotalIdleMemStatistics() const;
  size_t TotalEagerFreeMemStatistics() const;
  size_t UsedMemPeakStatistics() const;
  size_t MaxMemAllocatedStatistics() const;
  size_t MaxMemReservedStatistics() const;
  size_t ActualPeakStatistics() const;
  std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const;
  std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> CommonMemBlocksInfoStatistics()
    const;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> PersistentMemBlocksInfoStatistics()
    const;
  void ResetMaxMemReserved() const;
  void ResetMaxMemAllocated() const;

  // Display the brief state information of memory block and memory buf.
  void DumpDynamicMemPoolStateInfo();
  // Display the detailed debug information of memory block and memory buf.
  void DumpDynamicMemPoolDebugInfo();

  void DefragMemory();

  // The related interface of device memory real operation, needs override by device type.
  virtual size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) = 0;
  virtual bool FreeDeviceMem(const DeviceMemPtr &addr) = 0;
  virtual size_t free_mem_size() = 0;
  virtual uint64_t total_mem_size() const { return 0; }
  // Set mem pool block size
  virtual void SetMemPoolBlockSize(size_t available_device_mem_size);
  virtual size_t GetMaxUsedMemSize() const { return 0; }

  // Element in vector : memory_stream_id, address
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                   const DeviceEventPtr &event);
  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id);
  bool WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id);
  bool SyncAllEvents();
  virtual std::string GetMemoryPoolType() const { return "Other"; }
#ifdef WITH_BACKEND

 protected:
#endif
  const MemStatusManagerPtr &common_mem() const { return common_mem_; }
  const MemStatusManagerPtr &persistent_mem() const { return persistent_mem_; }
  void *GetMinUsingMemoryAddr() const;
  // The real size by memory alloc aligned.
  virtual size_t AlignMemorySize(size_t size) const;
  // Calculate memory block required alloc size when adding the memory block.
  virtual size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false);
  std::set<DeviceMemPtr> mem_bufs_;
  // The related interface of device memory eager free.
  virtual const bool IsEnableEagerFree() const { return false; }
  const bool IsEnableVmm() const { return enable_vmm_; }
  void SetEnableVmm(bool enable_vmm) { enable_vmm_ = enable_vmm; }
  virtual const bool SyncAllStreams() { return false; }
  virtual size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr *addr) { return 0; }
  virtual size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) { return 0; }
  virtual size_t MmapDeviceMem(size_t size, DeviceMemPtr addr) { return 0; }
  const size_t FreeIdleMemsByEagerFree();
#ifdef WITH_BACKEND

 private:
#endif
  // Find available memory buf from total pools by status, which contains idle and eager free.
  DeviceMemPtr FindAvailableMemBuf(size_t size, bool from_persistent_mem, uint32_t stream_id);
  // Find the target status memory buf from total pools by aligned size when memory alloc.
  DeviceMemPtr FindMemBufByStatus(size_t size, bool from_persistent_mem, DynamicMemBufStatus target_status,
                                  uint32_t stream_id);
  // Find the target status memory buf from specific pool by aligned size when memory alloc.
  DeviceMemPtr FindMemBufInSpecifiedMng(size_t size, bool from_persistent_mem, DynamicMemBufStatus target_status,
                                        uint32_t stream_id);

  // Add memory block and memory.
  DeviceMemPtr AddMemBlockAndMemBuf(size_t size, bool from_persistent_mem, bool need_recycle, uint32_t stream_id);
  // Add memory block and memory buf with eager free api.
  DeviceMemPtr AddMemBlockAndMemBufByEagerFree(size_t size, bool from_persistent_mem, uint32_t stream_id);
  // Add the memory block and memory buf when memory alloc not find the available memory buf.
  DeviceMemPtr CreateMemBlockAndMemBuf(size_t size, bool from_persistent_mem, DeviceMemPtr source_addr,
                                       size_t source_size, DynamicMemBufStatus mem_buf_status, uint32_t stream_id);

  // Judge whether need split the memory buf by alloc size and memory buf size.
  bool IsSplit(size_t tensor_size, size_t mem_buf_size) const;
  // Split the memory buf by alloc size.
  void SplitMemBuf(size_t size, const DynamicMemBufPtr &mem_buf, const MemStatusManagerPtr &mem_mng,
                   uint32_t stream_id);

  // Find the memory block by device address.
  DynamicMemBlockPtr FindMemBlock(const DeviceMemPtr &device_addr, const MemStatusManagerPtr &mem_mng) const;
  // The Comparator of memory block by device address, because memory blocks are arranged in order by device address.
  static bool CmpMemBlock(const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block);

  // Free memory inner with no lock, the caller need lock.
  void FreeTensorMemInner(const DeviceMemPtr &device_addr);
  // Pre combine mem buf, return false when mem buf can not combine.
  bool PreCombineMemBuf(const DynamicMemBufPtr &mem_buf, const MemStatusManagerPtr &mem_mng);
  // Combine the memory buf when memory free, to avoid the memory fragmentation.
  void CombineMemBuf(const DynamicMemBlockPtr &mem_block, const DeviceAddrMapMemBuf::iterator &iter,
                     const MemStatusManagerPtr &mem_mng, DynamicMemBufStatus origin_status,
                     DynamicMemBufStatus target_status);
  // Fetch the mem info by the strict addr.
  std::tuple<DynamicMemBlockPtr, DeviceAddrMapMemBuf::iterator, MemStatusManagerPtr> FindByStrictAddr(
    const DeviceMemPtr &device_addr) const;

  // Keep the part memorys by addr.
  void KeepTensorMemByAddr(const DeviceMemPtr &device_addr, size_t size);
  std::tuple<DynamicMemBlockPtr, DynamicMemBufPtr, MemStatusManagerPtr> FindByKeepAddr(
    const DeviceMemPtr &device_addr) const;
  DynamicMemBufPtr FindMemBufByKeepAddr(const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block) const;
  // Sync all events inner without lock.
  bool SyncAllEventsInner();

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
  size_t config_unit_size_{kDynamicMemAllocUnitSize};
  // Flag for eager free routine. This flag set to false when initializing, and set to true when triggering oom.
  bool is_trigger_eager_free_{false};

  // key : <user_stream_id, memory_stream_id>
  std::unordered_map<std::pair<uint32_t, uint32_t>, std::set<DynamicMemBufPtr>, pair_hash> stream_pair_addresses_;

  bool enable_vmm_{false};
  size_t eager_free_count_{0};
  size_t last_eager_free_count_{0};
};

// Recording information for debugging the memory allocator.
struct AllocatorDebugInfo {
  std::string name_{"Unknown"};
  AllocatorType type_{AllocatorType::kOther};
  int input_index_{-1};
  int output_index_{-1};
};

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

using TaskIdOnStreamEvent = std::pair<int64_t, DeviceEventPtr>;
struct DynamicMemBuf {
  DynamicMemBuf(DeviceMemPtr addr, DynamicMemBufStatus status, size_t size, uint32_t stream_id)
      : device_addr_(addr), status_(status), size_(size), stream_id_(stream_id) {}
  DynamicMemBuf(DeviceMemPtr addr, DynamicMemBufStatus status, size_t size, uint32_t stream_id,
                const std::string &allocator_name, AllocatorType allocator_type)
      : device_addr_(addr),
        status_(status),
        size_(size),
        stream_id_(stream_id),
        allocator_name_(allocator_name),
        allocator_type_{allocator_type} {}
  DynamicMemBuf(const DynamicMemBuf &) = delete;
  DynamicMemBuf &operator=(const DynamicMemBuf &) = delete;

  // Record event on mem buf.
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id, const DeviceEventPtr &event);

  // Release events on mem buf.
  bool WaitEvent(uint32_t task_id_on_stream, uint32_t user_stream_id);

  // Indidates if mem buf used by event, return true when no event bind on mem buf.
  bool IsEventNotUsed();

  // Sync all events that bound on mem buf.
  bool SyncAllEvents();

  DeviceMemPtr device_addr_;
  DynamicMemBufStatus status_;
  size_t size_;

  uint32_t stream_id_{0};

  // Debug info.
  std::string allocator_name_;
  AllocatorType allocator_type_{AllocatorType::kOther};

  // Parameter: user_stream_id, list of <task_id_on_stream, event>.
  std::shared_ptr<std::unordered_map<uint32_t, std::shared_ptr<std::list<TaskIdOnStreamEvent>>>> events_{nullptr};
};

class DynamicMemBlock {
 public:
  DynamicMemBlock() = delete;
  DynamicMemBlock(DeviceMemPtr addr_base, size_t size, const uint32_t stream_id)
      : device_addr_base_(addr_base), mem_block_size_(size), stream_id_(stream_id) {}
  ~DynamicMemBlock() { block_all_mem_buf_map_.clear(); }
  const DeviceMemPtr &device_addr() const { return device_addr_base_; }
  size_t size() const { return mem_block_size_; }
  void update_border_addr(DeviceMemPtr left_addr, DeviceMemPtr right_addr);
  size_t get_actual_peak();

#ifdef WITH_BACKEND

 private:
#endif
  friend class DynamicMemPoolBestFit;
  // MemStatusManager need dump block_all_mem_buf_map_ info, add friend class.
  friend class MemStatusManager;

  // The map of all memory buf in this memory block by device address.
  DeviceAddrMapMemBuf block_all_mem_buf_map_;

  DeviceMemPtr device_addr_base_{nullptr};

  // Max addr
  DeviceMemPtr max_addr_ = nullptr;
  // Min addr
  DeviceMemPtr min_addr_ = nullptr;

  size_t mem_block_size_{0};
  const uint32_t stream_id_;
};

struct DeviceState {
  // Update peak size.
  void UpdatePeakSize() {
    size_t total_used_size_ = total_used_mem_size_ + total_used_by_event_mem_size_;
    size_t temp_used_size_ = temp_total_used_mem_size_ + temp_total_used_by_event_mem_size_;
    used_mem_peak_size_ = std::max(used_mem_peak_size_, total_used_size_);
    if (total_used_size_ > temp_used_size_) {
      temp_used_mem_peak_size_ = std::max(temp_used_mem_peak_size_, total_used_size_ - temp_used_size_);
    }
  }

  // Memory allocated from device
  size_t total_mem_size_{0};
  // Memory in use
  size_t total_used_mem_size_{0};
  // Memory in use by event
  size_t total_used_by_event_mem_size_{0};
  // Memory in idle.
  size_t total_idle_mem_size_{0};
  // Memory in eager free.
  size_t total_eager_free_mem_size_{0};
  // Maximum peak memory usage
  size_t used_mem_peak_size_{0};
  // Recorded data for memory in use since reset maximum allocated memory
  size_t temp_total_used_mem_size_{0};
  // Recorded data for memory in use by event since reset maximum allocated memory
  size_t temp_total_used_by_event_mem_size_{0};
  // Recorded data for maximum peak memory usage since reset maximum allocated memory
  size_t temp_used_mem_peak_size_{0};
  // Temporary recorded data for memory reserved since reset maximum reserved memory
  size_t temp_total_mem_size_{0};
};

struct MemStatusManager {
  bool Empty() const { return mem_block_list_.empty(); }

  void AddMemBlock(const DynamicMemBlockPtr &mem_block, uint32_t stream_id);

  void DoAddMemBlock(const DynamicMemBlockPtr &mem_block, std::vector<DynamicMemBlockPtr> *mem_block_list);
  size_t CalActualPeak();

  SizeMapMemBuf &GetOrCreateMemBufMap(uint32_t stream_id, DynamicMemBufStatus status);

  void AddMemBuf(const DynamicMemBufPtr &mem_buf);

  void RemoveMemBuf(const DynamicMemBufPtr &mem_buf);

  void Clear() noexcept;

  const DeviceState DumpMemBlockDebugInfo(const std::string &mem_type);

  std::vector<uint32_t> GetStreamIds() const {
    std::vector<uint32_t> stream_ids;
    for (const auto &iter : mem_blocks_) {
      (void)stream_ids.emplace_back(iter.first);
    }
    return stream_ids;
  }

  size_t unit_size_{kDynamicMemAllocUnitSize};
  // Mem pool state
  DeviceState mps_;

  std::vector<DynamicMemBlockPtr> mem_block_list_;
  std::vector<DynamicMemBlockPtr> mem_block_insertion_order_;
  size_t total_block_size_ = 0;
  std::unordered_map<uint32_t, std::vector<DynamicMemBlockPtr>> mem_blocks_;
  std::unordered_map<std::pair<uint32_t, DynamicMemBufStatus>, SizeMapMemBuf, pair_hash> mem_bufs_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_
