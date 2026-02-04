/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_MAX_SEGMENT_TREE_H_
#define MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_MAX_SEGMENT_TREE_H_

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include "utils/log_adapter.h"

namespace mindspore {
template <typename T>
class MaxSegmentTree {
 private:
  static constexpr size_t kTreeSizeMultiplier = 4;
  static constexpr T kInitialValue = 0;
  static constexpr size_t kChildMultiplier = 2;
  static constexpr size_t kLeftChildOffset = 1;
  static constexpr size_t kRightChildOffset = 2;

  struct Node {
    std::vector<T> values;
    std::vector<T> lazy;         // Lazy propagation array
    std::vector<bool> has_lazy;  // Lazy flag array

    explicit Node(size_t k) : values(k, kInitialValue), lazy(k, kInitialValue), has_lazy(k, false) {}

    void Merge(const Node &left, const Node &right, size_t idx) {
      if (idx >= values.size()) {
        MS_LOG(EXCEPTION) << "Invalid merge parameters";
      }
      values[idx] = std::max(left.values[idx], right.values[idx]);
    }

    // Push down lazy tag for specific index only
    void PushDown(Node *left_child, Node *right_child, size_t idx) {
      if (idx >= values.size()) {
        MS_LOG(EXCEPTION) << "Invalid push_down parameters";
      }

      if (has_lazy[idx]) {
        // Update child node values (only for current index)
        left_child->values[idx] = std::max(left_child->values[idx], lazy[idx]);
        right_child->values[idx] = std::max(right_child->values[idx], lazy[idx]);

        // Propagate lazy tags (only for current index)
        left_child->lazy[idx] = std::max(left_child->lazy[idx], lazy[idx]);
        right_child->lazy[idx] = std::max(right_child->lazy[idx], lazy[idx]);

        // Set lazy flags for child nodes
        left_child->has_lazy[idx] = true;
        right_child->has_lazy[idx] = true;

        // Clear lazy tag for current node
        has_lazy[idx] = false;
        lazy[idx] = kInitialValue;
      }
    }
  };

  std::vector<Node> tree;
  size_t n;
  size_t k;

  void Build(size_t node, size_t start, size_t end) {
    if (start == end) {
      for (size_t i = 0; i < k; ++i) {
        tree.at(node).values[i] = kInitialValue;
      }
      return;
    }

    size_t mid = start + (end - start) / 2;
    size_t left_child = kChildMultiplier * node + kLeftChildOffset;
    size_t right_child = kChildMultiplier * node + kRightChildOffset;

    Build(left_child, start, mid);
    Build(right_child, mid + 1, end);
    for (size_t i = 0; i < k; ++i) {
      tree.at(node).Merge(tree.at(left_child), tree.at(right_child), i);
    }
  }

  void UpdateRange(size_t node, size_t start, size_t end, size_t l, size_t r, size_t idx, T val) {
    // Intervals don't overlap, return directly
    if (start > r || end < l) return;

    // Current interval is completely covered by query interval
    if (l <= start && end <= r) {
      // Update current node value (max operation)
      tree.at(node).values[idx] = std::max(tree.at(node).values[idx], val);

      // If not a leaf node, set lazy tag
      if (start != end) {
        tree.at(node).lazy[idx] = std::max(tree.at(node).lazy[idx], val);
        tree.at(node).has_lazy[idx] = true;
      }
      return;
    }

    // Calculate mid point and child indices
    size_t mid = start + (end - start) / 2;
    size_t left_child = kChildMultiplier * node + kLeftChildOffset;
    size_t right_child = kChildMultiplier * node + kRightChildOffset;

    // Push down lazy tags (only for the current index)
    tree.at(node).PushDown(&tree.at(left_child), &tree.at(right_child), idx);

    // Recursively update sub-intervals
    UpdateRange(left_child, start, mid, l, r, idx, val);
    UpdateRange(right_child, mid + 1, end, l, r, idx, val);

    // Merge child node results
    tree.at(node).Merge(tree.at(left_child), tree.at(right_child), idx);
  }

  T QueryRange(size_t node, size_t start, size_t end, size_t l, size_t r, size_t idx) {
    // Intervals don't overlap, return negative infinity
    if (start > r || end < l) return std::numeric_limits<T>::lowest();

    // Current interval is completely covered by query interval
    if (l <= start && end <= r) {
      return tree.at(node).values[idx];
    }

    // Calculate mid point and child indices
    size_t mid = start + (end - start) / 2;
    size_t left_child = kChildMultiplier * node + kLeftChildOffset;
    size_t right_child = kChildMultiplier * node + kRightChildOffset;

    // Push down lazy tags (only for the current index)
    tree.at(node).PushDown(&tree.at(left_child), &tree.at(right_child), idx);

    // Recursively query sub-intervals
    T left_val = QueryRange(left_child, start, mid, l, r, idx);
    T right_val = QueryRange(right_child, mid + 1, end, l, r, idx);

    // Return the maximum value from sub-interval results
    return std::max(left_val, right_val);
  }

 public:
  MaxSegmentTree(size_t size, size_t elements_per_node) : n(size), k(elements_per_node) {
    MS_LOG(INFO) << "MaxSegmentTree constructor called with size: " << size
                 << " and elements_per_node: " << elements_per_node;
    tree.resize(kTreeSizeMultiplier * n, Node(k));
    Build(0, 0, n - 1);
  }

  void Update(size_t l, size_t r, size_t idx, T val) {
    if (l > r || l >= n || r >= n || idx >= k) {
      MS_LOG(EXCEPTION) << "Invalid update parameters";
    }
    UpdateRange(0, 0, n - 1, l, r, idx, val);
  }

  T Query(size_t l, size_t r, size_t idx) {
    if (l > r || l >= n || r >= n || idx >= k) {
      MS_LOG(EXCEPTION) << "Invalid query parameters";
    }
    return QueryRange(0, 0, n - 1, l, r, idx);
  }
};
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_MEMORY_MEM_POOL_MAX_SEGMENT_TREE_H_
