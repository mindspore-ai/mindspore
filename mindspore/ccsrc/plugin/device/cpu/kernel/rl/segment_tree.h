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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SEGMENT_TREE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SEGMENT_TREE_H_

#include <vector>
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
constexpr size_t kRootIndex = 1;
constexpr size_t kNumSubnodes = 2;
constexpr size_t kRightOffset = 1;

// SegmentTree is a binary tree use for storing intervals information.
// It allows querying stored intervals by given point with complex O(logN).
// It is constructed from a fixed-lengh array for performance.
// The intervals information are templated as type T. User could override
// the ReduceOp() method for generic perpose.
template <typename T>
class SegmentTree {
 public:
  // Construct fixed-length segment tree.
  SegmentTree(size_t capacity, const T &init_value) {
    size_t capacity_pow_two = 1;
    while (capacity_pow_two < capacity) {
      capacity_pow_two *= kNumSubnodes;
    }
    capacity_ = capacity_pow_two;
    buffer_.resize(capacity_ * kNumSubnodes, init_value);
  }

  virtual ~SegmentTree() = default;

  // Insert item to the segment tree.
  void Insert(size_t idx, const T &value) {
    if (idx >= capacity_) {
      MS_LOG(EXCEPTION) << "Index " << idx << " out of range " << capacity_;
    }

    // Update leaf node value.
    idx += capacity_;
    buffer_[idx] = value;

    // Update non-leaf node value.
    idx /= kNumSubnodes;
    while (idx >= kRootIndex) {
      buffer_[idx] = ReduceOp(buffer_[kNumSubnodes * idx], buffer_[kNumSubnodes * idx + kRightOffset]);
      idx /= kNumSubnodes;
    }
  }

  // Get root node information.
  const T &Root() { return buffer_[kRootIndex]; }

  // Get leaf node information with index.
  const T &GetByIndex(size_t idx) {
    if (idx >= capacity_) {
      MS_LOG(EXCEPTION) << "Index " << idx << " out of range " << capacity_;
    }
    return buffer_[idx + capacity_];
  }

  // Reduce method for non leaf node. Subclass should override it for general perpose.
  virtual T ReduceOp(const T &lhs, const T &rhs) = 0;

 protected:
  size_t capacity_;
  std::vector<T> buffer_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_SEGMENT_TREE_H_
