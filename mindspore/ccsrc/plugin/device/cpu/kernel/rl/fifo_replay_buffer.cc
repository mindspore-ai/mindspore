/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/rl/fifo_replay_buffer.h"

#include <cstring>
#include <vector>
#include <algorithm>
#include <memory>
#include "kernel/kernel.h"
#include "plugin/device/cpu/hal/hardware/cpu_memory_pool.h"

namespace mindspore {
namespace kernel {
FIFOReplayBuffer::FIFOReplayBuffer(size_t capacity, const std::vector<size_t> &schema)
    : capacity_(capacity), head_(-1), size_(0), schema_(schema) {
  for (const auto &size : schema) {
    size_t alloc_size = size * capacity;
    if (alloc_size == 0) {
      MS_LOG(ERROR) << "Malloc size can not be 0.";
      return;
    }

    void *ptr = device::cpu::CPUMemoryPool::GetInstance().AllocTensorMem(alloc_size);
    MS_EXCEPTION_IF_NULL(ptr);
    AddressPtr item = std::make_shared<Address>(ptr, alloc_size);
    (void)buffer_.emplace_back(item);
  }
}

FIFOReplayBuffer::~FIFOReplayBuffer() {
  for (const auto &item : buffer_) {
    if (item->addr) {
      device::cpu::CPUMemoryPool::GetInstance().FreeTensorMem(item->addr);
      item->addr = nullptr;
    }
  }
}

bool FIFOReplayBuffer::Push(const std::vector<AddressPtr> &inputs) {
  if (inputs.size() != schema_.size()) {
    MS_LOG(EXCEPTION) << "Transition element num error. Expect " << schema_.size() << " , but got " << inputs.size();
  }

  // Head point to the latest item.
  head_ = head_ >= capacity_ ? 0 : head_ + 1;
  size_ = size_ >= capacity_ ? capacity_ : size_ + 1;

  return Emplace(head_, inputs);
}

bool FIFOReplayBuffer::Emplace(const size_t &pos, const std::vector<AddressPtr> &inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    void *offset = reinterpret_cast<uint8_t *>(buffer_[i]->addr) + pos * schema_[i];
    auto ret = memcpy_s(offset, buffer_[i]->size, inputs[i]->addr, inputs[i]->size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy_s() failed. Error code: " << ret;
    }
  }
  return true;
}

std::vector<AddressPtr> FIFOReplayBuffer::GetItem(size_t idx) {
  if (idx >= capacity_ || idx >= size_) {
    MS_LOG(EXCEPTION) << "Index " << idx << " out of range " << std::min(capacity_, size_);
  }

  std::vector<AddressPtr> ret;
  for (size_t i = 0; i < schema_.size(); i++) {
    void *offset = reinterpret_cast<uint8_t *>(buffer_[i]->addr) + schema_[i] * idx;
    ret.push_back(std::make_shared<Address>(offset, schema_[i]));
  }

  return ret;
}

std::vector<std::vector<AddressPtr>> FIFOReplayBuffer::GetItems(const std::vector<size_t> &indices) {
  std::vector<std::vector<AddressPtr>> ret;
  for (const auto &idx : indices) {
    auto item = GetItem(idx);
    (void)ret.emplace_back(item);
  }

  return ret;
}

const std::vector<AddressPtr> &FIFOReplayBuffer::GetAll() const { return buffer_; }
}  // namespace kernel
}  // namespace mindspore
