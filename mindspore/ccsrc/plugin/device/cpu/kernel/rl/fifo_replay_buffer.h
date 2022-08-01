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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FIFO_REPLAY_BUFFER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_FIFO_REPLAY_BUFFER_H_

#include <vector>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
// The FIFOReplayBuffer is container storing experiences.
// It lets the reinforcement learning agents remember and reuse experiences from the past.
// When the replay buffer is full, the oldest transition will be overridden.
class FIFOReplayBuffer {
 public:
  // Construct a fixed-length FIFO replay buffer.
  FIFOReplayBuffer(size_t capacity, const std::vector<size_t> &schema);

  ~FIFOReplayBuffer();

  // Push a transition to replay buffer. If the replay buffer is full, the oldest one will be overridden.
  bool Push(const std::vector<AddressPtr> &inputs);

  // Emplace a transition to replay buffer.
  bool Emplace(const size_t &pos, const std::vector<AddressPtr> &inputs);

  // Get a transition by the index.
  std::vector<AddressPtr> GetItem(size_t idx);

  // Get transitions by the indices.
  std::vector<std::vector<AddressPtr>> GetItems(const std::vector<size_t> &indices);

  // Get all transitions.
  const std::vector<AddressPtr> &GetAll() const;

  // Return the latest transition index. It returns -1 if the replay buffer is empty.
  size_t head() const { return head_; }

  // Return the valid transitions number.
  size_t size() const { return size_; }

 protected:
  size_t capacity_;
  size_t head_;
  size_t size_;
  std::vector<AddressPtr> buffer_;
  std::vector<size_t> schema_;
};
}  // namespace kernel
}  // namespace mindspore
#endif
