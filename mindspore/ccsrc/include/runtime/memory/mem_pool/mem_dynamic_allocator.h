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

#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_MEM_DYNAMIC_ALLOCATOR_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_MEM_DYNAMIC_ALLOCATOR_H_

#include <algorithm>
#include <atomic>
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

#include "include/runtime/memory/mem_pool/dynamic_mem_pool.h"
#include "include/backend/visible.h"
#include "include/utils/stream_util.h"
#ifdef __APPLE__
#include "async/spinlock.h"
#endif

namespace mindspore {
class DeviceEvent;
using DeviceEventPtr = std::shared_ptr<DeviceEvent>;

namespace device {
struct DynamicMemBuf : public EventBase {
  DynamicMemBuf(DeviceMemPtr addr, DynamicMemBufStatus status, size_t size, uint32_t stream_id)
      : device_addr_(addr), status_(status), size_(size), stream_id_(stream_id) {}
  DynamicMemBuf(DeviceMemPtr addr, DynamicMemBufStatus status, size_t size, uint32_t stream_id,
                const std::string &mem_name, memory::mem_pool::MemType mem_type)
      : device_addr_(addr),
        status_(status),
        size_(size),
        stream_id_(stream_id),
        mem_name_(mem_name),
        mem_type_{mem_type} {}
  DynamicMemBuf(const DynamicMemBuf &) = delete;
  DynamicMemBuf &operator=(const DynamicMemBuf &) = delete;

  DeviceMemPtr device_addr_;
  DynamicMemBufStatus status_;
  size_t size_;

  uint32_t stream_id_{0};

  // Debug info.
  std::string mem_name_;
  memory::mem_pool::MemType mem_type_{memory::mem_pool::MemType::kOther};
};

class DynamicMemBlock {
 public:
  DynamicMemBlock(DeviceMemPtr addr_base, size_t size, const uint32_t stream_id)
      : device_addr_base_(addr_base), mem_block_size_(size), stream_id_(stream_id) {}
  DynamicMemBlock() = delete;
  DynamicMemBlock(const DynamicMemBlock &) = delete;
  DynamicMemBlock &operator=(const DynamicMemBlock &) = delete;

  ~DynamicMemBlock() { block_all_mem_buf_map_.clear(); }

  const DeviceMemPtr &device_addr() const { return device_addr_base_; }

  size_t size() const { return mem_block_size_; }

  void update_border_addr(DeviceMemPtr left_addr, DeviceMemPtr right_addr);

  size_t get_actual_peak();

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

struct BACKEND_EXPORT DeviceState {
  void UpdatePeakSize(const bool is_enable_vmm, size_t vmm_used_mem_size) {
    used_mem_peak_size_ = std::max(used_mem_peak_size_, total_used_mem_size_);
    iter_used_mem_peak_size_ = std::max(iter_used_mem_peak_size_, total_used_mem_size_);
    if (is_enable_vmm) {
      iter_total_mem_peak_size_ = std::max(iter_total_mem_peak_size_, vmm_used_mem_size);
    } else {
      iter_total_mem_peak_size_ = std::max(iter_total_mem_peak_size_, total_mem_size_);
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
  // Recorded data for maximum peak memory usage since reset maximum allocated memory
  size_t iter_used_mem_peak_size_{0};
  // Temporary recorded data for memory reserved since reset maximum reserved memory
  size_t iter_total_mem_peak_size_{0};
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

// Implement of best fit dynamic memory pool.
class BACKEND_EXPORT DynamicMemPoolBestFit : virtual public DynamicMemPool {
 public:
  DynamicMemPoolBestFit()
      : persistent_mem_(std::make_shared<MemStatusManager>()), common_mem_(std::make_shared<MemStatusManager>()) {}
  virtual ~DynamicMemPoolBestFit();

  void Initialize(size_t init_size, size_t increase_size, size_t max_size) override;

  // The main program entry of memory alloc.
  DeviceMemPtr AllocTensorMem(size_t size, bool from_persistent_mem = false, bool need_recycle = false,
                              uint32_t stream_id = kDefaultStreamIndex) override;
  // The main program entry of continuous memory alloc.
  std::vector<DeviceMemPtr> AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                     uint32_t stream_id = kDefaultStreamIndex) override;
  // The main program entry of memory free.
  void FreeTensorMem(const DeviceMemPtr &device_addr) override;
  // The main program entry of part memorys free and part memorys keep.
  void FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs, const std::vector<DeviceMemPtr> &keep_addrs,
                          const std::vector<size_t> &keep_addr_sizes) override;

  // Release the real device memory.
  void ReleaseDeviceRes() override;

  // Get the minimum memory unit size using for dynamic extend.
  size_t MemAllocUnitSize(bool from_persistent_mem = false) const;
  // Set the minimum memory unit size using for dynamic extend.
  void SetMemAllocUintSize(size_t common_size, size_t persist_size = kDynamicMemAllocUnitSize);

  // Extract detailed block information
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> ExtractBlocksListInfo(
    const MemStatusManagerPtr &mem_mng) const;

  // The statistics information.
  size_t TotalMemStatistics() const override;
  size_t TotalUsedMemStatistics() const override;
  size_t TotalUsedByEventMemStatistics() const override;
  size_t TotalIdleMemStatistics() const override;
  size_t TotalEagerFreeMemStatistics() const override;
  size_t UsedMemPeakStatistics() const override;
  size_t MaxMemAllocatedStatistics() const override;
  size_t MaxMemReservedStatistics() const override;
  size_t ActualPeakStatistics() const override;
  std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const override;
  std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> CommonMemBlocksInfoStatistics()
    const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> PersistentMemBlocksInfoStatistics()
    const override;
  void ResetMaxMemReserved() override;
  void ResetMaxMemAllocated() override;

  // Display the brief state information of memory block and memory buf.
  void DumpDynamicMemPoolStateInfo() override;
  // Display the detailed debug information of memory block and memory buf.
  void DumpDynamicMemPoolDebugInfo() override;

  void DefragMemory() override;

  void SetMemPoolBlockSize(size_t available_device_mem_size) override;

  // Element in vector : memory_stream_id, address
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                   const DeviceEventPtr &event) override;
  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) override;
  bool WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) override;
  bool SyncAllEvents() override;

  std::string GetMemoryPoolType() const override { return "Other"; }

  bool IsEnableTimeEvent() override { return enable_time_event_; }

  void SetEnableTimeEvent(bool enable_time_event) override { enable_time_event_ = enable_time_event; }

  virtual MemoryTimeEventPtr GenAllocateMemoryTimeEvent(const void *addr, size_t size, uint32_t stream_id,
                                                        bool from_persistent, bool is_persistent);

  virtual MemoryTimeEventPtr GenFreeMemoryTimeEvent(const void *addr);

  virtual void ReportMemoryTimeEvent(const MemoryTimeEventPtr &time_event) {}
#ifndef ENABLE_TEST

 protected:
#endif
  const MemStatusManagerPtr &common_mem() const { return common_mem_; }
  const MemStatusManagerPtr &persistent_mem() const { return persistent_mem_; }
  void *GetMinUsingMemoryAddr() const override;

  // Calculate memory block required alloc size when adding the memory block.
  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override;
  std::set<DeviceMemPtr> mem_bufs_;

  void WaitPipelineHelper();

  // The related interface of device memory eager free.
  const bool IsEnableEagerFree() const override { return false; }
  const bool IsEnableVmm() const override { return enable_vmm_; }
  void SetEnableVmm(bool enable_vmm) { enable_vmm_ = enable_vmm; }
  void WaitPipelineWithCallback();
  const std::pair<size_t, size_t> FreeIdleMemsByEagerFree() override;
#ifndef ENABLE_TEST

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
  std::atomic<bool> enable_time_event_{false};
  size_t increase_size_{kDynamicMemAllocUnitSize};
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIME_MEMORY_MEM_POOL_MEM_DYNAMIC_ALLOCATOR_H_
