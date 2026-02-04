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

#include "kernel/ascend/internal/tiling_mem_mgr.h"

#include <algorithm>
#include "utils/ms_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "kernel/ascend/internal/internal_ascend_adapter.h"
#include "plugin/ascend/res_manager/mem_manager/ascend_memory_pool.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"
#include "acl/acl.h"

#define TMP_LOG(level) MS_LOG(level) << GetName() << ": "

namespace mindspore {
namespace kernel {
size_t TilingMemPool::GetAlignedSize(size_t size) { return (size + block_size_ - 1) & ~(block_size_ - 1); }

TilingMemPool::TilingMemPool(size_t block_size, size_t block_num) : block_size_(block_size), block_num_(block_num) {
  total_size_ = block_size * block_num;
  mem_slots_.emplace_back(Slot{0, total_size_});
  head_ = 0;
  tail_ = 1;
}

int TilingMemPool::Init() { return 0; }

void TilingMemPool::Rearrange() {
  auto CompareFunc = [this](size_t first, size_t second) {
    return mem_slots_[first].offset_ < mem_slots_[second].offset_;
  };

  if (head_ == tail_) {
    return;
  }

  TMP_LOG(INFO) << "Begin doing rearrange...";
  std::vector<size_t> indices;
  auto num = static_cast<int64_t>(head_ > tail_ ? tail_ + block_num_ - head_ : tail_ - head_);
  for (auto i = 0; i < num; i++) {
    indices.emplace_back(static_cast<size_t>((i + head_) % block_num_));
  }

  std::sort(indices.begin(), indices.end(), CompareFunc);
  std::vector<Slot> new_slots{mem_slots_[indices[0]]};
  size_t last_slot_idx = 0;
  for (auto i = 1; i < num; i++) {
    auto &last_slot = new_slots[last_slot_idx];
    auto &cur_slot = mem_slots_[indices[static_cast<size_t>(i)]];
    if (last_slot.offset_ + last_slot.length_ == cur_slot.offset_) {
      // can merge
      last_slot.length_ += cur_slot.length_;
    } else {
      new_slots.push_back(cur_slot);
      last_slot_idx++;
    }
  }

  mem_slots_ = std::move(new_slots);
  head_ = 0;
  tail_ = last_slot_idx + 1;
  TMP_LOG(INFO) << "Complete doing rearrange!!! New size of mem_slots_: " << mem_slots_.size()
                << ", new tail_: " << tail_;
  for (size_t i = 0; i < mem_slots_.size(); i++) {
    TMP_LOG(INFO) << "idx: " << i << ", offset: " << mem_slots_[i].offset_ << ", len: " << mem_slots_[i].length_;
  }
}

void *TilingMemPool::Malloc(size_t size) {
  auto aligned_size = GetAlignedSize(size);

  if (mem_base_ptr_ == nullptr) {
    mem_base_ptr_ = static_cast<int8_t *>(MallocInner(total_size_));
    TMP_LOG(INFO) << "Malloc base ptr: " << static_cast<const void *>(mem_base_ptr_) << ", size: " << total_size_;
    MS_EXCEPTION_IF_NULL(mem_base_ptr_);
  }

  if (head_ == tail_) {
    auto ret = MallocOneOffMem(aligned_size);
    TMP_LOG(INFO) << "Malloc one off memory because of empty slots, addr: " << ret << ", size: " << size
                  << ", aligned_size: " << aligned_size;
    return ret;
  }

  int8_t *ret_addr = nullptr;
  for (auto i = head_; i < tail_; i++) {
    auto &slot = mem_slots_[i];
    if (slot.length_ < aligned_size) {
      continue;
    }

    ret_addr = mem_base_ptr_ + slot.offset_;
    if (slot.length_ == aligned_size) {
      if (i == head_) {
        // the head slot is totally malloced, so move the head_ to next one
        head_ = RoundAdd(head_);
        break;
      } else if (i == tail_) {
        // the tail slot is totally malloced, so move the tail the previous one
        tail_ = RoundSub(tail_);
      } else {
        // the slot is in the middle of head and slot, move the head to this empty slot
        mem_slots_[i] = mem_slots_[head_];
        head_ = RoundAdd(head_);
      }
    } else {
      Slot new_slot{slot.offset_ + aligned_size, slot.length_ - aligned_size};
      mem_slots_[i] = new_slot;
    }
    break;
  }

  if (ret_addr == nullptr) {
    auto ret = MallocOneOffMem(aligned_size);
    TMP_LOG(INFO) << "Malloc one off memory because of not enough memory in slot, addr: " << ret << ", size: " << size
                  << ", aligned_size: " << aligned_size;
    return ret;
  }

  TMP_LOG(DEBUG) << "Malloc cached memory ret_addr: " << static_cast<const void *>(ret_addr) << ", size: " << size
                 << ", aligned_size: " << aligned_size << ", offset: " << ret_addr - mem_base_ptr_;
  return ret_addr;
}

void TilingMemPool::Free(void *addr, size_t size) {
  if (addr == nullptr || mem_base_ptr_ == nullptr || total_size_ == 0) {
    return;
  }
  if (IsOneOffMem(addr)) {
    TMP_LOG(INFO) << "Free directly for one off memory, addr: " << addr;
    FreeInner(addr);
    (void)one_off_mem_ptrs_.erase(addr);
    return;
  }

  auto offset = static_cast<size_t>(static_cast<int8_t *>(addr) - mem_base_ptr_);
  auto aligned_size = GetAlignedSize(size);
  bool merged = false;
  for (auto i = head_; i < tail_; i++) {
    auto &slot = mem_slots_[i];
    if (offset + aligned_size == slot.offset_) {
      slot.offset_ = offset;
      slot.length_ += aligned_size;
      merged = true;
      TMP_LOG(DEBUG) << "Merge slots: head_: " << head_ << ", tail_: " << tail_ << ", cur_idx: " << i
                     << ", new slot.offset_: " << slot.offset_ << ", new slot.length_: " << slot.length_;
      break;
    }
  }

  if (!merged) {
    if (tail_ == mem_slots_.size()) {
      mem_slots_.emplace_back(Slot{offset, aligned_size});
    } else {
      mem_slots_[tail_] = Slot{offset, aligned_size};
    }
    tail_ = RoundAdd(tail_);
    TMP_LOG(DEBUG) << "Create new slot, offset: " << offset << ", aligned_size: " << aligned_size
                   << ", new_tail_: " << tail_;
  }
}

void TilingMemPool::FreeMemPtrs() {
  if (mem_base_ptr_ != nullptr) {
    FreeInner(mem_base_ptr_);
  }

  for (const auto &ptr : one_off_mem_ptrs_) {
    FreeInner(ptr);
  }
}

TilingMemPoolDevice::TilingMemPoolDevice(size_t block_size, size_t block_num) : TilingMemPool(block_size, block_num) {
  SetName("DEVICE");
}

void *TilingMemPoolDevice::MallocInner(size_t size) {
  return device::ascend::AscendMemoryPool::GetInstance().AllocTensorMem(size);
}

void TilingMemPoolDevice::FreeInner(void *addr) {
  TMP_LOG(INFO) << "free addr: " << addr;
  device::ascend::AscendMemoryPool::GetInstance().FreeTensorMem(addr);
}

TilingMemPoolHost::TilingMemPoolHost(size_t block_size, size_t block_num) : TilingMemPool(block_size, block_num) {
  SetName("HOST");
}

void *TilingMemPoolHost::MallocInner(size_t size) { return malloc(size); }

void TilingMemPoolHost::FreeInner(void *addr) {
  TMP_LOG(INFO) << "free addr: " << addr;
  free(addr);
}

TilingMemMgr::TilingMemMgr() {
  auto context_ptr = mindspore::MsContext::GetInstance();
  uint32_t device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string device_name = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device::GetDeviceTypeByName(device_name), device_id});
}

void TilingMemMgr::CopyAsync(void *host_ptr, void *device_ptr, size_t size) {
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
  if (default_stream_ == nullptr) {
    auto default_stream_id = device_context_->device_res_manager_->DefaultStream();
    default_stream_ = device_context_->device_res_manager_->GetStream(default_stream_id);
  }
  auto ret = InternalAscendAdapter::AscendMemcpyAsync(device_ptr, size, host_ptr, size, ACL_MEMCPY_HOST_TO_DEVICE,
                                                      default_stream_);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Copy tiling data from host to device failed!";
  }
}

void TilingMemMgr::CopyAsyncD2H(void *host_ptr, void *device_ptr, size_t size) {
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
  if (default_stream_ == nullptr) {
    auto default_stream_id = device_context_->device_res_manager_->DefaultStream();
    default_stream_ = device_context_->device_res_manager_->GetStream(default_stream_id);
  }
  auto ret = InternalAscendAdapter::AscendMemcpyAsync(host_ptr, size, device_ptr, size, ACL_MEMCPY_DEVICE_TO_HOST,
                                                      default_stream_);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Copy tiling data from host to device failed!";
  }
}

}  // namespace kernel
}  // namespace mindspore
