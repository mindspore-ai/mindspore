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

#include "replay_buffer/fifo_replay_buffer.h"

#include <vector>
#include <algorithm>
#include <memory>
#include "securec/include/securec.h"
#include "common/kernel_log.h"

namespace aicpu {
FIFOReplayBuffer::FIFOReplayBuffer(size_t capacity, const std::vector<size_t> &schema)
    : capacity_(capacity), head_(-1), size_(0), schema_(schema) {
  for (const auto &size : schema) {
    size_t alloc_size = size * capacity;
    if (alloc_size == 0) {
      AICPU_LOGW("Malloc size can not be 0.");
      return;
    }

    void *ptr = malloc(alloc_size);
    AddressPtr item = std::make_shared<Address>(ptr, alloc_size);
    (void)buffer_.emplace_back(item);
  }
}

FIFOReplayBuffer::~FIFOReplayBuffer() {
  for (const auto &item : buffer_) {
    free(item->addr);
    item->addr = nullptr;
  }
}

bool FIFOReplayBuffer::Push(const std::vector<AddressPtr> &inputs) {
  if (inputs.size() != schema_.size()) {
    AICPU_LOGE("Transition element num error. Expect %u, but got %u.", schema_.size(), inputs.size());
  }

  // Head point to the latest item.
  size_ = size_ >= capacity_ ? capacity_ : size_ + 1;
  head_ = head_ >= capacity_ ? 0 : head_ + 1;

  return Emplace(head_, inputs);
}

bool FIFOReplayBuffer::Emplace(const size_t &pos, const std::vector<AddressPtr> &inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    void *offset = reinterpret_cast<uint8_t *>(buffer_[i]->addr) + pos * schema_[i];
    auto ret = memcpy_s(offset, buffer_[i]->size, inputs[i]->addr, inputs[i]->size);
    if (ret != EOK) {
      AICPU_LOGE("memcpy_s() failed. Error code: %d.", ret);
    }
  }

  return true;
}

std::vector<AddressPtr> FIFOReplayBuffer::GetItem(size_t idx) {
  if (idx >= capacity_ || idx >= size_) {
    AICPU_LOGE("Idex: %u out of range %u.", idx, std::min(capacity_, size_));
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
}  // namespace aicpu
