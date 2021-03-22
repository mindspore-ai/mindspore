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

#include "backend/optimizer/mem_reuse/mem_dynamic_allocator.h"
#include "utils/ms_utils.h"
#include "utils/convert_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
DynamicMemPoolBestFit::~DynamicMemPoolBestFit() {
  global_mem_block_list_.clear();
  global_idle_mem_buf_map_.clear();
}

DeviceMemPtr DynamicMemPoolBestFit::AllocTensorMem(size_t size) {
  size_t align_size = AlignMemorySize(size);
  std::lock_guard<std::mutex> locker(mutex_);
  // Find the idle memory buf by tensor size, if not find, then add new memory block and memory buf.
  DeviceMemPtr device_addr = FindIdleMemBuf(align_size);
  if (!device_addr) {
    device_addr = AddMemBlockAndMemBuf(align_size);
  }
  return device_addr;
}

std::vector<DeviceMemPtr> DynamicMemPoolBestFit::AllocContinuousTensorMem(size_t total_size,
                                                                          std::vector<size_t> size_list) {
  std::vector<DeviceMemPtr> device_addr_list;
  // Pre-alloc the one whole piece memory.
  auto device_addr = AllocTensorMem(total_size);
  if (!device_addr) {
    return device_addr_list;
  }
  std::lock_guard<std::mutex> locker(mutex_);
  // Remove the pre-alloc memory.
  auto mem_block = FindMemBlock(device_addr);
  MS_EXCEPTION_IF_NULL(mem_block);
  auto iter = mem_block->block_all_mem_buf_map_.find(device_addr);
  if (iter == mem_block->block_all_mem_buf_map_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the device address[" << device_addr << "].";
  }
  auto mem_buf = iter->second;
  MS_EXCEPTION_IF_NULL(mem_buf);
  auto rest_size = mem_buf->size_ - total_size;
  (void)mem_block->block_all_mem_buf_map_.erase(iter);
  // Split the pre-alloc memory into continuous memory by the size list.
  DynamicMemBufPtr continuous_mem_buf;
  auto buf_addr = device_addr;
  for (size_t i = 0; i < size_list.size(); i++) {
    continuous_mem_buf = std::make_shared<DynamicMemBuf>(buf_addr, kMemBufUsed, size_list[i]);
    (void)mem_block->block_all_mem_buf_map_.emplace(buf_addr, continuous_mem_buf);
    device_addr_list.emplace_back(buf_addr);
    buf_addr = AddressOffset(buf_addr, size_list[i]);
  }
  // Update the size of the last memory buf.
  continuous_mem_buf->size_ += rest_size;
  return device_addr_list;
}

size_t DynamicMemPoolBestFit::AlignMemorySize(size_t size) const {
  if (size == 0) {
    return DYNAMIC_MEM_ALIGN_SIZE;
  }
  return ((size + DYNAMIC_MEM_ALIGN_SIZE - 1) / DYNAMIC_MEM_ALIGN_SIZE) * DYNAMIC_MEM_ALIGN_SIZE;
}

DeviceMemPtr DynamicMemPoolBestFit::FindIdleMemBuf(size_t size) {
  auto iter = global_idle_mem_buf_map_.lower_bound(size);
  if (iter != global_idle_mem_buf_map_.end()) {
    auto mem_buf = iter->second;
    MS_EXCEPTION_IF_NULL(mem_buf);
    if (mem_buf->status_ != kMemBufIdle) {
      MS_LOG(EXCEPTION) << "Find the mem_buf is not idle, alloc_size[" << size << "] mem_buf_size[" << mem_buf->size_
                        << "] mem_buf_address[" << mem_buf->device_addr_ << "].";
    }
    mem_buf->status_ = kMemBufUsed;
    // Remove map of old idle memory buf
    (void)global_idle_mem_buf_map_.erase(iter);
    // Divide memory buf
    if (IsDivide(size, mem_buf->size_)) {
      DivideMemBuf(size, mem_buf);
    }
    // Memory statistics
    total_used_mem_statistics_ += mem_buf->size_;
    if (total_used_mem_statistics_ > used_mem_peak_statistics_) {
      used_mem_peak_statistics_ = total_used_mem_statistics_;
    }
    return mem_buf->device_addr_;
  }
  return nullptr;
}

DeviceMemPtr DynamicMemPoolBestFit::AddMemBlockAndMemBuf(size_t size) {
  size_t alloc_mem_size = CalMemBlockAllocSize(size);
  if (alloc_mem_size == 0) {
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
  auto mem_block = std::make_shared<DynamicMemBlock>(device_addr, real_alloc_size);
  MS_EXCEPTION_IF_NULL(mem_block);
  auto iter = std::upper_bound(global_mem_block_list_.begin(), global_mem_block_list_.end(), device_addr, CmpMemBlock);
  (void)global_mem_block_list_.insert(iter, mem_block);
  // Add new memory buf
  auto mem_buf = std::make_shared<DynamicMemBuf>(device_addr, kMemBufUsed, real_alloc_size);
  MS_EXCEPTION_IF_NULL(mem_buf);
  // Add map of new memory buf in the block
  (void)mem_block->block_all_mem_buf_map_.emplace(device_addr, mem_buf);
  // Divide memory buf
  if (IsDivide(size, mem_buf->size_)) {
    DivideMemBuf(size, mem_buf);
  }
  // Memory statistics
  total_mem_statistics_ += real_alloc_size;
  total_used_mem_statistics_ += mem_buf->size_;
  if (total_used_mem_statistics_ > used_mem_peak_statistics_) {
    used_mem_peak_statistics_ = total_used_mem_statistics_;
  }
  return mem_buf->device_addr_;
}

size_t DynamicMemPoolBestFit::CalMemBlockAllocSize(size_t size) {
  auto device_free_mem_size = free_mem_size();
  if (device_free_mem_size < size) {
    MS_LOG(WARNING) << "Memory not enough: current free memory size[" << device_free_mem_size
                    << "] is smaller than required size[" << size << "].";
    return 0;
  }
  auto alloc_mem_size = mem_alloc_unit_size();
  // Growing at twice of alloc size
  while (alloc_mem_size < size) {
    alloc_mem_size = alloc_mem_size * 2;
  }
  alloc_mem_size = std::min(alloc_mem_size, device_free_mem_size);
  return alloc_mem_size;
}

bool DynamicMemPoolBestFit::IsDivide(size_t tensor_size, size_t mem_buf_size) const {
  return mem_buf_size - tensor_size >= DYNAMIC_MEM_ALIGN_SIZE;
}

void DynamicMemPoolBestFit::DivideMemBuf(size_t size, const DynamicMemBufPtr &mem_buf) {
  MS_EXCEPTION_IF_NULL(mem_buf);
  auto mem_block = FindMemBlock(mem_buf->device_addr_);
  MS_EXCEPTION_IF_NULL(mem_block);
  // Divide new memory buf
  size_t newbuf_size = mem_buf->size_ - size;
  mem_buf->size_ = size;
  DeviceMemPtr newbuf_addr = AddressOffset(mem_buf->device_addr_, size);
  auto new_mem_buf = std::make_shared<DynamicMemBuf>(newbuf_addr, kMemBufIdle, newbuf_size);
  // Add map of new memory buf in the block
  (void)mem_block->block_all_mem_buf_map_.emplace(newbuf_addr, new_mem_buf);
  // Add map of new idle memory buf
  (void)global_idle_mem_buf_map_.emplace(newbuf_size, new_mem_buf);
}

bool DynamicMemPoolBestFit::CmpMemBlock(const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block) {
  MS_EXCEPTION_IF_NULL(device_addr);
  MS_EXCEPTION_IF_NULL(mem_block);
  return device_addr < mem_block->device_addr();
}

DynamicMemBlockPtr DynamicMemPoolBestFit::FindMemBlock(const DeviceMemPtr &device_addr) {
  MS_EXCEPTION_IF_NULL(device_addr);
  auto iter = std::upper_bound(global_mem_block_list_.begin(), global_mem_block_list_.end(), device_addr, CmpMemBlock);
  if (iter != global_mem_block_list_.begin()) {
    return *(--iter);
  }
  return nullptr;
}

void DynamicMemPoolBestFit::FreeTensorMem(const DeviceMemPtr &device_addr) {
  MS_EXCEPTION_IF_NULL(device_addr);
  std::lock_guard<std::mutex> locker(mutex_);
  auto mem_block = FindMemBlock(device_addr);
  if (mem_block == nullptr) {
    // May be destroy the memory pool first, then destroy the address, so this is normal case.
    MS_LOG(DEBUG) << "Can't find the mem_block of the device address[" << device_addr << "].";
    return;
  }
  CombineMemBuf(mem_block, device_addr);
}

void DynamicMemPoolBestFit::CombineMemBuf(const DynamicMemBlockPtr &mem_block, const DeviceMemPtr &device_addr) {
  MS_EXCEPTION_IF_NULL(mem_block);
  MS_EXCEPTION_IF_NULL(device_addr);
  auto iter = mem_block->block_all_mem_buf_map_.find(device_addr);
  if (iter == mem_block->block_all_mem_buf_map_.end()) {
    MS_LOG(EXCEPTION) << "Can't find the device address[" << device_addr << "].";
  }
  auto mem_buf = iter->second;
  MS_EXCEPTION_IF_NULL(mem_buf);
  if (mem_buf->status_ != kMemBufUsed) {
    MS_LOG(EXCEPTION) << "Find the mem_buf is not used, mem_buf_address[" << mem_buf->device_addr_ << "].";
  }
  mem_buf->status_ = kMemBufIdle;
  total_used_mem_statistics_ -= mem_buf->size_;
  // Combine backward(combine the next_mem_buf to mem_buf)
  auto next_iter = iter;
  (void)next_iter++;
  if (next_iter != mem_block->block_all_mem_buf_map_.end()) {
    auto next_mem_buf = next_iter->second;
    MS_EXCEPTION_IF_NULL(next_mem_buf);
    if (next_mem_buf->status_ == kMemBufIdle) {
      mem_buf->size_ += next_mem_buf->size_;
      EraseIdleMemBuf(next_mem_buf->size_, next_mem_buf->device_addr_);
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
    if (prev_mem_buf->status_ == kMemBufIdle) {
      EraseIdleMemBuf(prev_mem_buf->size_, prev_mem_buf->device_addr_);
      prev_mem_buf->size_ += mem_buf->size_;
      (void)mem_block->block_all_mem_buf_map_.erase(iter);
      forward_combine = true;
    }
  }
  // Add map of new idle memory
  if (forward_combine) {
    (void)global_idle_mem_buf_map_.emplace(prev_mem_buf->size_, prev_mem_buf);
  } else {
    (void)global_idle_mem_buf_map_.emplace(mem_buf->size_, mem_buf);
  }
}

void DynamicMemPoolBestFit::EraseIdleMemBuf(size_t size, const DeviceMemPtr &device_addr) {
  MS_EXCEPTION_IF_NULL(device_addr);
  auto iter = global_idle_mem_buf_map_.equal_range(size);
  while (iter.first != iter.second) {
    MS_EXCEPTION_IF_NULL(iter.first->second);
    // Remove map of the idle memory buf by size and device address
    if (iter.first->second->device_addr_ == device_addr) {
      (void)global_idle_mem_buf_map_.erase(iter.first);
      return;
    }
    (void)iter.first++;
  }
  MS_LOG(ERROR) << "Can't find the size[" << size << "] and device address[" << device_addr << "] in the idle mem_buf.";
}

void DynamicMemPoolBestFit::ReleaseDeviceRes() {
  std::lock_guard<std::mutex> locker(mutex_);
  MS_LOG(INFO) << "The dynamic memory pool total size is " << total_mem_statistics_ << ", total used size is "
               << total_used_mem_statistics_ << ", used peak size is " << used_mem_peak_statistics_ << ".";
  for (auto iter = global_mem_block_list_.begin(); iter != global_mem_block_list_.end(); ++iter) {
    auto device_addr = (*iter)->device_addr();
    if (device_addr != nullptr) {
      if (!FreeDeviceMem(device_addr)) {
        MS_LOG(EXCEPTION) << "Free device memory[" << device_addr << "] error.";
      }
    }
  }
}

void DynamicMemPoolBestFit::DumpDynamicMemPoolInfo() {
  std::lock_guard<std::mutex> locker(mutex_);
  MS_LOG(INFO) << "Start dump dynamic memory pool info.";
  DeviceAddrMapMemBuf mem_block_map;
  DynamicMemBufPtr mem_buf;
  size_t total_mem = 0;
  size_t total_used_mem = 0;
  size_t total_idle_mem1 = 0;
  size_t total_idle_mem2 = 0;
  // Dump the memory block info and memory buf info
  MS_LOG(INFO) << "Dump all mem_block info: counts[" << global_mem_block_list_.size() << "].";
  for (auto iter = global_mem_block_list_.begin(); iter != global_mem_block_list_.end(); ++iter) {
    total_mem += (*iter)->size();
    mem_block_map = (*iter)->block_all_mem_buf_map_;
    MS_LOG(INFO) << "MemBlock info: number[" << iter - global_mem_block_list_.begin() << "] mem_buf_counts["
                 << mem_block_map.size() << "] base_address[" << (*iter)->device_addr() << "] block_size["
                 << (*iter)->size() << "].";
    for (auto iter_mem_buf = mem_block_map.begin(); iter_mem_buf != mem_block_map.end(); ++iter_mem_buf) {
      mem_buf = iter_mem_buf->second;
      MS_EXCEPTION_IF_NULL(mem_buf);
      if (mem_buf->status_ == kMemBufIdle) {
        total_idle_mem1 += mem_buf->size_;
      } else {
        total_used_mem += mem_buf->size_;
      }
      MS_LOG(INFO) << "MemBuf info: address[" << mem_buf->device_addr_ << "] size[" << mem_buf->size_ << "] status["
                   << mem_buf->status_ << "].";
    }
  }
  // Dump all the idle memory buf info
  MS_LOG(INFO) << "Dump all idle mem_buf info: counts[" << global_idle_mem_buf_map_.size() << "].";
  for (auto iter_idle = global_idle_mem_buf_map_.begin(); iter_idle != global_idle_mem_buf_map_.end(); ++iter_idle) {
    mem_buf = iter_idle->second;
    MS_EXCEPTION_IF_NULL(mem_buf);
    total_idle_mem2 += mem_buf->size_;
    MS_LOG(INFO) << "Idle mem_buf info: size[" << mem_buf->size_ << "] address[" << mem_buf->device_addr_ << "] status["
                 << mem_buf->status_ << "].";
  }
  // Dump the memory statistical info
  MS_LOG(INFO) << "Total allocated memory[" << total_mem << "], used memory[" << total_used_mem << "], idle memory["
               << total_idle_mem1 << "].";
  if (total_idle_mem1 != total_idle_mem2) {
    MS_LOG(ERROR) << "Check error: the idle memory in the mem_block is not equal the global idle memory.";
  }
  if (total_mem != total_used_mem + total_idle_mem1) {
    MS_LOG(ERROR) << "Check error: the the total memory is not equal the sum of used memory and idle memory.";
  }
  MS_LOG(INFO) << "Finish dump dynamic memory pool info.";
}
}  // namespace device
}  // namespace mindspore
