/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
#include "cpu_kernel/utils/broadcast_iterator.h"

#include <algorithm>
#include <utility>

namespace aicpu {
BroadcastIterator::BroadcastIterator(std::vector<int64_t> &input_shape_a, std::vector<int64_t> &input_shape_b,
                                     std::vector<int64_t> &output_shape)
    : input_shape_a_(std::move(input_shape_a)),
      input_shape_b_(std::move(input_shape_b)),
      output_shape_(std::move(output_shape)) {
  output_dimension_ = output_shape_.size();  // Assign dimension to int for iterator
  BroadcastShape();
  // Allocate strides memory
  input_strides_a_.resize(output_dimension_);
  input_strides_b_.resize(output_dimension_);
  input_back_strides_a_.resize(output_dimension_);
  input_back_strides_b_.resize(output_dimension_);
  coordinates_.resize(output_dimension_);
  InitStrides();
}

void BroadcastIterator::SetPos(int64_t pos) {
  for (int i = output_dimension_ - 1; i >= 0 && pos != 0; --i) {
    coordinates_[i] = pos % output_shape_[i];
    input_pos_[0] += coordinates_[i] * input_strides_a_[i];
    input_pos_[1] += coordinates_[i] * input_strides_b_[i];
    pos /= output_shape_[i];
  }
}

void BroadcastIterator::GenNextPos() {
  // Calculate output next coordinate
  for (int i = output_dimension_ - 1; i >= 0; --i) {
    if (coordinates_[i] + 1 == output_shape_[i]) {
      coordinates_[i] = 0;
      input_pos_[0] -= input_back_strides_a_[i];
      input_pos_[1] -= input_back_strides_b_[i];
    } else {
      ++coordinates_[i];
      input_pos_[0] += input_strides_a_[i];
      input_pos_[1] += input_strides_b_[i];
      break;
    }
  }
}

void BroadcastIterator::BroadcastShape() {
  size_t input_dimension_a = input_shape_a_.size();
  if (input_dimension_a < output_dimension_) {
    input_shape_a_.insert(input_shape_a_.begin(), output_dimension_ - input_dimension_a, 1);
  }

  size_t input_dimension_b = input_shape_b_.size();
  if (input_dimension_b < output_dimension_) {
    input_shape_b_.insert(input_shape_b_.begin(), output_dimension_ - input_dimension_b, 1);
  }
}

void BroadcastIterator::InitStrides() {
  input_strides_a_[output_dimension_ - 1] = 1;
  input_strides_b_[output_dimension_ - 1] = 1;
  for (int i = output_dimension_ - 2; i >= 0; --i) {
    input_strides_a_[i] = input_shape_a_[i + 1] * input_strides_a_[i + 1];
    input_strides_b_[i] = input_shape_b_[i + 1] * input_strides_b_[i + 1];
    input_back_strides_a_[i + 1] = (input_shape_a_[i + 1] - 1) * input_strides_a_[i + 1];
    input_back_strides_b_[i + 1] = (input_shape_b_[i + 1] - 1) * input_strides_b_[i + 1];
  }

  // Update strides for broadcast
  // While the axis value is 1, the stride is 0
  (void)std::transform(input_strides_a_.begin(), input_strides_a_.end(), input_shape_a_.begin(),
                       input_strides_a_.begin(), [](const int64_t &a, const int64_t &b) { return (b == 1) ? 0 : a; });
  (void)std::transform(input_strides_b_.begin(), input_strides_b_.end(), input_shape_b_.begin(),
                       input_strides_b_.begin(), [](const int64_t &a, const int64_t &b) { return (b == 1) ? 0 : a; });
}

uint32_t GetBroadcastShape(const std::vector<int64_t> &x, const std::vector<int64_t> &y,
                           std::vector<int64_t> &broadcast_shape) {
  int64_t x_len = x.size();
  int64_t y_len = y.size();
  int64_t length = x_len < y_len ? x_len : y_len;
  std::vector<int64_t> broadcast_shape_back;
  for (int64_t i = -length; i < 0; ++i) {
    if (x[x_len + i] == 1) {
      broadcast_shape_back.push_back(y[y_len + i]);
    } else if (y[y_len + i] == 1) {
      broadcast_shape_back.push_back(x[x_len + i]);
    } else if (x[x_len + i] == y[y_len + i]) {
      broadcast_shape_back.push_back(x[x_len + i]);
    } else {
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  if (length == x_len) {
    for (int64_t i = 0; i < y_len - length; ++i) {
      broadcast_shape.push_back(y[i]);
    }
  } else {
    for (int64_t i = 0; i < x_len - length; ++i) {
      broadcast_shape.push_back(x[i]);
    }
  }
  for (int64_t i = 0; i < length; ++i) {
    broadcast_shape.push_back(broadcast_shape_back[i]);
  }
  return KERNEL_STATUS_OK;
}
}  // namespace aicpu
