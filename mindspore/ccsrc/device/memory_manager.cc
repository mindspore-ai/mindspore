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

#include "device/memory_manager.h"
#include "session/anf_runtime_algorithm.h"
#include "utils/context/ms_context.h"
using mindspore::memreuse::BestFitMemReuse;
using mindspore::memreuse::MemReuseUtilPtr;
namespace mindspore {
namespace device {
size_t MemoryManager::GetCommonAlignSize(size_t input_size) const {
  return (input_size + kMemAlignSize + 31) / kMemAlignSize * kMemAlignSize;
}

size_t MemoryManager::GetCommunicationAlignSize(size_t input_size) const {
  return (input_size + kMemAlignSize - 1) / kMemAlignSize * kMemAlignSize + 2 * kMemAlignSize;
}

void MemoryManager::MallocReusedDynamicMem(session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MemReuseUtilPtr mem_reuse_util_ptr = std::make_shared<memreuse::MemReuseUtil>();
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  // set all infos
  mem_reuse_util_ptr->SetAllInfo(graph);
  auto bestfit_mem_reuse = std::make_shared<BestFitMemReuse>();
  MS_EXCEPTION_IF_NULL(bestfit_mem_reuse);
  bestfit_mem_reuse->Reuse(mem_reuse_util_ptr.get());
  size_t total_allocated_size = bestfit_mem_reuse->GetAllocatedSize();
  MS_LOG(INFO) << "TotalReuseDynamicSize [" << total_allocated_size << "]";
  mem_reuse_util_ptr_ = mem_reuse_util_ptr;
  auto base_ptr = MallocDynamicMem(total_allocated_size, false);
  mem_reuse_util_ptr_->set_mem_base(base_ptr);
}

uint8_t *MemoryManager::MallocOutputMem(const AnfNodePtr &node, size_t index, int flag, size_t size) {
  MS_EXCEPTION_IF_NULL(node);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  uint8_t *ptr = nullptr;
  if (AnfAlgo::IsCommunicationOp(node)) {
    bool communication_mem = false;
    if (context_ptr->enable_hccl()) {
      communication_mem = true;
    }
    if (flag == kStaticMem) {
      ptr = MallocStaticMem(size, communication_mem);
    } else {
      ptr = MallocDynamicMem(size, communication_mem);
    }
    return ptr;
  }

  if (flag == kStaticMem) {
    ptr = MallocStaticMem(size, false);
  } else if (flag == kDynamicMem) {
    ptr = MallocDynamicMem(size, false);
  } else if (flag == kReuseDynamicMem) {
    ptr = mem_reuse_util_ptr_->GetNodeOutputPtr(node, index);
  }
  return ptr;
}

uint8_t *MemoryManager::MallocWorkSpaceMem(const AnfNodePtr &node, size_t index, int flag, size_t size) {
  if (flag == kReuseDynamicMem) {
    return mem_reuse_util_ptr_->GetNodeWorkSpacePtr(node, index);
  }
  return MallocDynamicMem(size, false);
}

uint8_t *MemoryManager::MallocMem(int flag, size_t size) {
  uint8_t *ptr = nullptr;
  if (flag == kStaticMem) {
    ptr = MallocStaticMem(size, false);
  } else if (flag == kDynamicMem) {
    ptr = MallocDynamicMem(size, false);
  }
  return ptr;
}

uint8_t *MemoryManager::MallocStaticMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }
  if (static_mem_offset_ < align_size) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  total_static_size_ += align_size;
  auto offset = static_mem_offset_ - align_size;
  if (dynamic_mem_offset_ > offset) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  static_mem_offset_ = offset;
  if (communication_mem) {
    return device_mem_base_ + offset + kMemAlignSize;
  } else {
    return device_mem_base_ + offset;
  }
}

uint8_t *MemoryManager::MallocDynamicMem(size_t size, bool communication_mem) {
  size_t align_size = 0;
  if (communication_mem) {
    align_size = GetCommunicationAlignSize(size);
  } else {
    align_size = GetCommonAlignSize(size);
  }
  uint64_t offset = dynamic_mem_offset_;
  auto new_offset = dynamic_mem_offset_ + align_size;
  if (new_offset > static_mem_offset_) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  total_dynamic_size_ += align_size;
  dynamic_mem_offset_ = new_offset;

  if (communication_mem) {
    return device_mem_base_ + offset + kMemAlignSize;
  } else {
    return device_mem_base_ + offset;
  }
}

void MemoryManager::MallocMemFromMemPool(const DeviceAddressPtr address, size_t size) {
  auto device_ptr = MallocMemFromMemPool(size);
  MS_EXCEPTION_IF_NULL(device_ptr);
  address->ptr_ = device_ptr;
  address->from_mem_pool_ = true;
}

void *MemoryManager::MallocMemFromMemPool(size_t size) {
  if (size == 0) {
    MS_LOG(ERROR) << "MallocMemFromMemPool size is 0.";
  }
  return nullptr;
}

void MemoryManager::FreeMemFromMemPool(const DeviceAddressPtr address) {
  MS_EXCEPTION_IF_NULL(address);
  MS_EXCEPTION_IF_NULL(address->ptr_);
  FreeMemFromMemPool(address->ptr_);
  address->ptr_ = nullptr;
}

void MemoryManager::FreeMemFromMemPool(void *device_ptr) {
  if (device_ptr == nullptr) {
    MS_LOG(ERROR) << "FreeMemFromMemPool device_ptr is null.";
  }
}

void MemoryManager::MallocContinuousMemFromMemPool(const DeviceAddressPtrList addr_list, size_t total_size,
                                                   std::vector<size_t> size_list) {
  auto device_ptr_list = MallocContinuousMemFromMemPool(total_size, size_list);
  if (addr_list.size() != device_ptr_list.size()) {
    MS_LOG(EXCEPTION) << "The size of device list is not equal  to the size of address list.";
  }
  for (size_t i = 0; i < addr_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(device_ptr_list[i]);
    MS_EXCEPTION_IF_NULL(addr_list[i]);
    addr_list[i]->ptr_ = device_ptr_list[i];
    addr_list[i]->from_mem_pool_ = true;
  }
}

std::vector<void *> MemoryManager::MallocContinuousMemFromMemPool(size_t total_size, std::vector<size_t> size_list) {
  if (total_size == 0) {
    MS_LOG(ERROR) << "MallocContinuousMemFromMemPool total_size is 0.";
  }
  std::vector<void *> device_ptr_list;
  device_ptr_list.emplace_back(nullptr);
  return device_ptr_list;
}
}  // namespace device
}  // namespace mindspore
