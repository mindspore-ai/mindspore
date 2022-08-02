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

#include "plugin/device/cpu/kernel/rl/reservoir_replay_buffer.h"

#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
ReservoirReplayBuffer::ReservoirReplayBuffer(uint32_t seed, size_t capacity, const std::vector<size_t> &schema)
    : capacity_(capacity), schema_(schema) {
  generator_.seed(seed);
  fifo_replay_buffer_ = std::make_unique<FIFOReplayBuffer>(capacity, schema);
}

bool ReservoirReplayBuffer::Push(const std::vector<AddressPtr> &transition) {
  // The buffer is not full: Push the transition at end of buffer.
  if (total_ < capacity_) {
    auto ret = fifo_replay_buffer_->Push(transition);
    total_++;
    return ret;
  }

  // The buffer is full: Random discard this sample or replace the an old one.
  auto replace_threthold = static_cast<float>(capacity_) / static_cast<float>(total_);
  std::uniform_real_distribution<float> keep_dist(0, 1);
  auto prob = keep_dist(generator_);
  if (prob < replace_threthold) {
    total_++;
    std::uniform_int_distribution<size_t> pos_sampler(0, capacity_ - 1);
    return fifo_replay_buffer_->Emplace(pos_sampler(generator_), transition);
  }

  return true;
}

bool ReservoirReplayBuffer::Sample(const size_t &batch_size, const std::vector<AddressPtr> &output) {
  const size_t valid_size = std::min(capacity_, total_);
  std::uniform_int_distribution<size_t> pos_sampler(0, valid_size - 1);

  // [batch_size, transitions]
  for (size_t i = 0; i < batch_size; i++) {
    auto transition = fifo_replay_buffer_->GetItem(pos_sampler(generator_));
    for (size_t item_index = 0; item_index < schema_.size(); item_index++) {
      void *offset = reinterpret_cast<uint8_t *>(output[item_index]->addr) + schema_[item_index] * i;
      MS_EXCEPTION_IF_CHECK_FAIL(
        memcpy_s(offset, schema_[item_index], transition[item_index]->addr, transition[item_index]->size) == EOK,
        "memcpy_s() failed.");
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
