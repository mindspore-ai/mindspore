/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef AICPU_UTILS_TENSOR_ITERATOR_H
#define AICPU_UTILS_TENSOR_ITERATOR_H

#include <vector>

class TensorIterator {
 public:
  TensorIterator() = delete;
  explicit TensorIterator(const std::vector<int64_t> &shape)
      : shape_(shape), coord_(shape.size(), 0), rank_(shape.size()) {}
  TensorIterator(const std::vector<int64_t> &shape, const std::vector<int64_t> &cur_coord)
      : shape_(shape), coord_(cur_coord), rank_(shape.size()) {}

  auto &begin() { return *this; }
  std::vector<int64_t> end() { return {}; }

  const std::vector<int64_t> &operator*() const { return coord_; }

  bool operator!=(const std::vector<int64_t> &rhs) const { return coord_ != rhs; }

  void operator++() { step(); }

  void step() { inc_at(rank_ - 1); }

 private:
  void inc_at(int dim) {
    if (dim < 0) {
      coord_.clear();
      return;
    }
    if (coord_[dim] < shape_[dim] - 1) {
      coord_[dim] += 1;
    } else {
      coord_[dim] = 0;
      inc_at(dim - 1);
    }
  }

  const std::vector<int64_t> shape_;
  std::vector<int64_t> coord_;
  const int rank_;
};
#endif  // AICPU_UTILS_TENSOR_ITERATOR_H
