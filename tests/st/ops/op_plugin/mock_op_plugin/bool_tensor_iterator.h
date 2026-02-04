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

#ifndef MINDSPORE_TESTS_ST_OPS_OP_PLUGIN_MOCK_OP_PLUGIN_BOOL_TENSOR_ITERATOR_H_
#define MINDSPORE_TESTS_ST_OPS_OP_PLUGIN_MOCK_OP_PLUGIN_BOOL_TENSOR_ITERATOR_H_

#include <cstdint>
#include <vector>

// A simple row-major iterator over a strided boolean tensor.
class BoolTensorIterator {
 public:
  BoolTensorIterator(const bool *base_address, const std::vector<int64_t> &shape,
                     const std::vector<int64_t> &strides_in_elements, int64_t offset_in_elements)
      : base_(base_address),
        shape_(shape),
        strides_elems_(strides_in_elements),
        index_(shape.size(), 0),
        dims_(static_cast<int64_t>(shape.size())),
        total_elements_(ComputeNumElements(shape)),
        consumed_elements_(0),
        base_offset_elems_(offset_in_elements) {}

  bool has_next() const { return consumed_elements_ < total_elements_; }

  bool next() {
    // Caller must ensure has_next() == true
    const int64_t elem_offset = ComputeCurrentElementOffset();
    const bool value = *(base_ + elem_offset);
    AdvanceIndex();
    ++consumed_elements_;
    return value;
  }

  void reset() {
    std::fill(index_.begin(), index_.end(), 0);
    consumed_elements_ = 0;
  }

 private:
  static int64_t ComputeNumElements(const std::vector<int64_t> &shape) {
    if (shape.empty()) {
      return 1;  // scalar
    }
    int64_t num = 1;
    for (int64_t dim : shape) {
      num *= dim;
    }
    return num;
  }

  int64_t ComputeCurrentElementOffset() const {
    int64_t offset = base_offset_elems_;
    for (int64_t i = 0; i < dims_; ++i) {
      offset += index_[i] * strides_elems_[static_cast<size_t>(i)];
    }
    return offset;
  }

  void AdvanceIndex() {
    if (dims_ == 0) {
      return;  // scalar
    }
    for (int64_t d = dims_ - 1; d >= 0; --d) {
      ++index_[static_cast<size_t>(d)];
      if (index_[static_cast<size_t>(d)] < shape_[static_cast<size_t>(d)]) {
        break;
      }
      index_[static_cast<size_t>(d)] = 0;
    }
  }

  const bool *base_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_elems_;
  std::vector<int64_t> index_;
  int64_t dims_;
  int64_t total_elements_;
  int64_t consumed_elements_;
  int64_t base_offset_elems_;
};

#endif  // MINDSPORE_TESTS_ST_OPS_OP_PLUGIN_MOCK_OP_PLUGIN_BOOL_TENSOR_ITERATOR_H_
