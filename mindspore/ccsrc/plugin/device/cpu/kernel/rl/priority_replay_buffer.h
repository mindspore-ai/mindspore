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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PRIORITY_REPLAY_BUFFER_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PRIORITY_REPLAY_BUFFER_H_

#include <vector>
#include <tuple>
#include <memory>
#include <limits>
#include <random>
#include "kernel/kernel.h"
#include "utils/log_adapter.h"
#include "plugin/device/cpu/kernel/rl/fifo_replay_buffer.h"
#include "plugin/device/cpu/kernel/rl/segment_tree.h"

namespace mindspore {
namespace kernel {
// Node value of PriorityTree. It contains sum and minimal priority.
struct PriorityItem {
  PriorityItem() : sum_priority(0), min_priority(std::numeric_limits<float>::max()) {}
  PriorityItem(float sum, float min) : sum_priority(sum), min_priority(min) {}

  float sum_priority;
  float min_priority;
};

// PriorityTree is tree which the value of node contains sum and minimal priority of its subnodes.
class PriorityTree : public SegmentTree<PriorityItem> {
 public:
  explicit PriorityTree(size_t capacity, const PriorityItem &init_value = PriorityItem());

  // Calculate sum and minimal priority of its subnodes.
  PriorityItem ReduceOp(const PriorityItem &lhs, const PriorityItem &rhs) override;

  // Find the minimal index greater than prefix_sum.
  size_t GetPrefixSumIdx(float prefix_sum) const;
};

// PriorityReplayBuffer is experience container used in Deep Q-Networks.
// The algorithm is proposed in `Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`.
// Same as the normal replay buffer, it lets the reinforcement learning agents remember and reuse experiences from the
// past. Besides, it replays important transitions more frequently and improve sample effciency.
class PriorityReplayBuffer {
 public:
  // Construct a fixed-length priority replay buffer.
  PriorityReplayBuffer(uint32_t seed, float alpha, size_t capacity, const std::vector<size_t> &schema);

  // Push an experience transition to the buffer which will be given the highest priority.
  bool Push(const std::vector<AddressPtr> &items);

  // Sample a batch transitions with indices and bias correction weights.
  std::tuple<std::vector<size_t>, std::vector<float>, std::vector<std::vector<AddressPtr>>> Sample(size_t batch_size,
                                                                                                   float beta);

  // Update experience transitions priorities.
  bool UpdatePriorities(const std::vector<size_t> &indices, const std::vector<float> &priorities);

 private:
  inline float Weight(float priority, float sum_priority, size_t size, float beta) const;

  float alpha_;
  size_t capacity_;
  float max_priority_;
  std::vector<size_t> schema_;
  std::default_random_engine random_engine_;
  std::uniform_real_distribution<float> dist_{0, 1};
  std::unique_ptr<FIFOReplayBuffer> fifo_replay_buffer_;
  std::unique_ptr<PriorityTree> priority_tree_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PRIORITY_REPLAY_BUFFER_H_
