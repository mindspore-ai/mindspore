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

#include "replay_buffer/priority_replay_buffer.h"

#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>
#include <limits>
#include "common/kernel_log.h"

namespace aicpu {
PriorityTree::PriorityTree(size_t capacity, const PriorityItem &init_value)
    : SegmentTree<PriorityItem>(capacity, init_value) {}

PriorityItem PriorityTree::ReduceOp(const PriorityItem &lhs, const PriorityItem &rhs) {
  return PriorityItem(lhs.sum_priority + rhs.sum_priority, std::min(lhs.min_priority, rhs.min_priority));
}

size_t PriorityTree::GetPrefixSumIdx(float prefix_sum) const {
  size_t idx = 1;
  while (idx < capacity_) {
    const auto &left_priority = buffer_[kNumSubnodes * idx].sum_priority;
    if (prefix_sum <= left_priority) {
      idx = kNumSubnodes * idx;
    } else {
      prefix_sum -= left_priority;
      idx = kNumSubnodes * idx + kRightOffset;
    }
  }

  return idx - capacity_;
}

PriorityReplayBuffer::PriorityReplayBuffer(uint32_t seed, float alpha, size_t capacity,
                                           const std::vector<size_t> &schema)
    : alpha_(alpha), capacity_(capacity), max_priority_(1.0), schema_(schema) {
  random_engine_.seed(seed);
  fifo_replay_buffer_ = std::make_unique<FIFOReplayBuffer>(capacity, schema);
  priority_tree_ = std::make_unique<PriorityTree>(capacity);
}

bool PriorityReplayBuffer::Push(const std::vector<AddressPtr> &items) {
  (void)fifo_replay_buffer_->Push(items);
  auto idx = fifo_replay_buffer_->head();

  // Set max priority for the newest item.
  float priority = static_cast<float>(pow(max_priority_, alpha_));
  priority_tree_->Insert(idx, {priority, priority});
  return true;
}

bool PriorityReplayBuffer::UpdatePriorities(const std::vector<size_t> &indices, const std::vector<float> &priorities) {
  if (indices.size() != priorities.size()) {
    return false;
  }

  for (size_t i = 0; i < indices.size(); i++) {
    float priority = static_cast<float>(pow(priorities[i], alpha_));
    if (priority <= 0.0f) {
      AICPU_LOGW("The sum priority is %f. It may leads to converge issue.");
      priority = std::numeric_limits<decltype(priority)>::epsilon();
    }
    priority_tree_->Insert(indices[i], {priority, priority});

    // Record max priority of transitions
    max_priority_ = std::max(max_priority_, priorities[i]);
  }

  return true;
}

std::tuple<std::vector<size_t>, std::vector<float>, std::vector<std::vector<AddressPtr>>> PriorityReplayBuffer::Sample(
  size_t batch_size, float beta) {
  if (batch_size == 0) {
    AICPU_LOGD("The batch size can not be zero.");
  }
  const PriorityItem &root = priority_tree_->Root();
  float sum_priority = root.sum_priority;
  float min_priority = root.min_priority;
  size_t size = fifo_replay_buffer_->size();
  float max_weight = Weight(min_priority, sum_priority, size, beta);
  float segment_len = root.sum_priority / batch_size;

  std::vector<size_t> indices;
  std::vector<float> weights;
  std::vector<std::vector<AddressPtr>> items;
  for (size_t i = 0; i < batch_size; i++) {
    float mass = (dist_(random_engine_) + i) * segment_len;
    size_t idx = priority_tree_->GetPrefixSumIdx(mass);

    (void)indices.emplace_back(idx);
    float priority = priority_tree_->GetByIndex(idx).sum_priority;

    if (max_weight <= 0.0f) {
      AICPU_LOGW("The sum priority is %f. It may leads to converge issue.");
      max_weight = std::numeric_limits<decltype(max_weight)>::epsilon();
    }
    (void)weights.emplace_back(Weight(priority, sum_priority, size, beta) / max_weight);
    (void)items.emplace_back(fifo_replay_buffer_->GetItem(idx));
  }

  return std::forward_as_tuple(indices, weights, items);
}

inline float PriorityReplayBuffer::Weight(float priority, float sum_priority, size_t size, float beta) const {
  if (sum_priority <= 0.0f) {
    AICPU_LOGW("The sum priority is %f. It may leads to converge issue.");
    sum_priority = std::numeric_limits<decltype(sum_priority)>::epsilon();
  }
  float sample_prob = priority / sum_priority;
  float weight = static_cast<float>(pow(sample_prob * size, -beta));
  return weight;
}
}  // namespace aicpu
