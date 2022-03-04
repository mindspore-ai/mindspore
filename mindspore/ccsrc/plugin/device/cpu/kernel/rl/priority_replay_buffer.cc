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

#include "plugin/device/cpu/kernel/rl/priority_replay_buffer.h"

#include <vector>
#include <tuple>
#include <memory>
#include <algorithm>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
constexpr float kMinPriority = 1e-7;

PriorityTree::PriorityTree(size_t capacity, const PriorityItem &init_value)
    : SegmentTree<PriorityItem>(capacity, init_value) {}

PriorityItem PriorityTree::ReduceOp(const PriorityItem &lhs, const PriorityItem &rhs) {
  return PriorityItem(lhs.sum_priority + rhs.sum_priority, std::min(lhs.min_priority, rhs.min_priority));
}

size_t PriorityTree::GetPrefixSumIdx(float prefix_sum) {
  size_t idx = 1;
  while (idx < capacity_) {
    if (prefix_sum <= buffer_[kNumSubnodes * idx].sum_priority) {
      idx = kNumSubnodes * idx;
    } else {
      prefix_sum -= buffer_[kRightOffset * idx].sum_priority;
      idx = kNumSubnodes * idx + kRightOffset;
    }
  }

  return idx - capacity_;
}

PriorityReplayBuffer::PriorityReplayBuffer(int seed, float alpha, float beta, size_t capacity,
                                           const std::vector<size_t> &schema)
    : alpha_(alpha), beta_(beta), capacity_(capacity), max_priority_(1.0), schema_(schema) {
  random_engine_.seed(seed);
  tree_ = std::make_unique<PriorityTree>(capacity);
}

bool PriorityReplayBuffer::Push(const std::vector<AddressPtr> &items) {
  // Set max priority for the newest item.
  tree_->Insert(0, {max_priority_, max_priority_});
  return true;
}

bool PriorityReplayBuffer::UpdatePriorities(const std::vector<size_t> &indices, const std::vector<float> &priorities) {
  if (indices.size() != priorities.size()) {
    return false;
  }

  for (size_t idx = 0; idx < indices.size(); idx++) {
    float priority = pow(priorities[idx], alpha_);
    if (priority <= 0.0f) {
      MS_LOG(WARNING) << "The priority is " << priority << ". It may lead to converge issue.";
      priority = kMinPriority;
    }
    tree_->Insert(idx, {priority, priority});

    // Record max priority of transitions
    max_priority_ = std::max(max_priority_, priority);
  }

  return true;
}

std::tuple<std::vector<size_t>, std::vector<float>, std::vector<std::vector<AddressPtr>>> PriorityReplayBuffer::Sample(
  size_t batch_size) {
  MS_EXCEPTION_IF_ZERO("batch size", batch_size);
  const PriorityItem &root = tree_->Root();
  float sum_priority = root.sum_priority;
  float min_priority = root.min_priority;
  float max_weight = Weight(min_priority, sum_priority, 0);
  float segment_len = root.sum_priority / batch_size;

  std::vector<size_t> indices;
  std::vector<float> weights;
  std::vector<std::vector<AddressPtr>> items;
  for (size_t i = 0; i < batch_size; i++) {
    float mass = (dist_(random_engine_) + i) * segment_len;
    size_t idx = tree_->GetPrefixSumIdx(mass);

    indices.emplace_back(idx);
    float priority = tree_->GetByIndex(idx).sum_priority;

    if (max_weight <= 0.0f) {
      MS_LOG(WARNING) << "The max priority is " << max_weight << ". It may leads to converge issue.";
      max_weight = kMinPriority;
    }
    weights.emplace_back(Weight(priority, sum_priority, 0) / max_weight);
  }

  return std::forward_as_tuple(indices, weights, items);
}

float PriorityReplayBuffer::Weight(float priority, float sum_priority, size_t size) {
  if (sum_priority <= 0.0f) {
    MS_LOG(WARNING) << "The sum priority is " << sum_priority << ". It may leads to converge issue.";
    sum_priority = kMinPriority;
  }
  float sample_prob = priority / sum_priority;
  float weight = pow(sample_prob * size, -beta_);
  return weight;
}
}  // namespace kernel
}  // namespace mindspore
