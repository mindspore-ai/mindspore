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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_

#include <memory>
#include <map>
#include <vector>
#include <algorithm>
#include <utility>
#include <thread>
#include <mutex>

namespace mindspore {
namespace device {
using DeviceMemPtr = void(*);

// The status of memory buf.
enum DynamicMemBufStatus : int { kMemBufIdle, kMemBufUsed };

// Alloc memory aligned according to 512 bytes.
static const size_t DYNAMIC_MEM_ALIGN_SIZE = 512;

// The minimum unit size (1G) of memory block used for dynamic extend.
static const size_t DYNAMIC_MEM_ALLOC_UNIT_SIZE = 1024 << 20;

// The Comparator of device address from small to large.
struct DeviceAddrCmp {
  bool operator()(const DeviceMemPtr &addr1, const DeviceMemPtr &addr2) const { return addr1 < addr2; }
};

// Memory buf is the smallest operation object of dynamic memory pool.
struct DynamicMemBuf {
  DynamicMemBuf(DeviceMemPtr addr, DynamicMemBufStatus status, size_t size)
      : device_addr_(addr), status_(status), size_(size) {}
  DeviceMemPtr device_addr_;
  DynamicMemBufStatus status_;
  size_t size_;
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
  // The map of all memory buf in this memory block by device address.
  DeviceAddrMapMemBuf block_all_mem_buf_map_;

 private:
  DeviceMemPtr device_addr_base_{nullptr};
  size_t mem_block_size_{0};
};
using DynamicMemBlockPtr = std::shared_ptr<DynamicMemBlock>;

// The main class of dynamic memory pool.
class DynamicMemPoolBestFit {
 public:
  DynamicMemPoolBestFit() = default;
  virtual ~DynamicMemPoolBestFit();
  // The main program entry of memory alloc.
  DeviceMemPtr AllocTensorMem(size_t size);
  // The main program entry of continuous memory alloc.
  std::vector<DeviceMemPtr> AllocContinuousTensorMem(size_t total_size, std::vector<size_t> size_list);
  // The main program entry of memory free.
  void FreeTensorMem(const DeviceMemPtr &device_addr);
  // Release the real device memory.
  void ReleaseDeviceRes();
  // Display the information of memory block and memory buf.
  void DumpDynamicMemPoolInfo();
  // Get the map of global idle mem buf and size.
  SizeMapMemBuf global_idle_mem_buf_map() {
    std::lock_guard<std::mutex> locker(mutex_);
    return global_idle_mem_buf_map_;
  }

  // Get the related memory statistics information.
  size_t total_mem_statistics() const { return total_mem_statistics_; }
  size_t used_mem_statistics() const { return total_used_mem_statistics_; }
  size_t used_mem_peak_statistics() const { return used_mem_peak_statistics_; }

  // The related interface of device memory real operation, needs override by device type.
  virtual size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) = 0;
  virtual bool FreeDeviceMem(const DeviceMemPtr &addr) = 0;
  virtual size_t free_mem_size() = 0;
  virtual size_t total_mem_size() = 0;

 protected:
  // The real size by memory alloc aligned.
  virtual size_t AlignMemorySize(size_t size) const;
  // Get the minimum memory unit size using for dynamic extend.
  virtual size_t mem_alloc_unit_size() const { return DYNAMIC_MEM_ALLOC_UNIT_SIZE; }

 private:
  // Find the idle memory buf by aligned size when memory alloc.
  DeviceMemPtr FindIdleMemBuf(size_t size);
  // Add the memory block and memory buf when memory alloc not find the idle memory buf.
  DeviceMemPtr AddMemBlockAndMemBuf(size_t size);
  // Calculate memory block required alloc size when adding the memory block.
  size_t CalMemBlockAllocSize(size_t size);
  // Judge whether need divide the memory buf by alloc size and memory buf size.
  bool IsDivide(size_t tensor_size, size_t mem_buf_size) const;
  // Divide the memory buf by alloc size.
  void DivideMemBuf(size_t size, const DynamicMemBufPtr &mem_buf);
  // Find the memory block by device address.
  DynamicMemBlockPtr FindMemBlock(const DeviceMemPtr &device_addr);
  // The Comparator of memory block by device address, because memory blocks are arranged in order by device address.
  static bool CmpMemBlock(const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block);

  // Combine the memory buf when memory free, to avoid the memory fragmentation.
  void CombineMemBuf(const DynamicMemBlockPtr &mem_block, const DeviceMemPtr &device_addr);
  // Erase the idle memory buf by size and device address when idle memory buf is combined.
  void EraseIdleMemBuf(size_t size, const DeviceMemPtr &device_addr);

  // The global memory block list which is arranged in order by base device address of memory block.
  std::vector<DynamicMemBlockPtr> global_mem_block_list_;
  // The map of all idle memory buf by size.
  SizeMapMemBuf global_idle_mem_buf_map_;

  // The related memory statistics information.
  size_t total_mem_statistics_{0};
  size_t total_used_mem_statistics_{0};
  size_t used_mem_peak_statistics_{0};

  // Support multi-thread.
  std::mutex mutex_;
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_
