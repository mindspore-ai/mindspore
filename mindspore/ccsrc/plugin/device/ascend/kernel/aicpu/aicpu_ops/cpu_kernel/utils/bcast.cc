/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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

#include "cpu_kernel/utils/bcast.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "mindspore/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_ops/common/kernel_log.h"
#include "cpu_kernel/common/status.h"

namespace {
const int64_t kNoBroadcastValue = 1;

enum class State { UNKNOWN, SAME, X_ONE, Y_ONE };
}  // namespace

namespace aicpu {
uint32_t Bcast::Init(const std::vector<int64_t> &x, const std::vector<int64_t> &y) {
  State prev = State::UNKNOWN;
  for (size_t i = 0; i < x.size(); ++i) {
    State curr = State::UNKNOWN;
    const int64_t x_i = x[i];
    const int64_t y_i = y[i];
    int64_t o_i = 0;
    int64_t bx_i = 0;
    int64_t by_i = 0;
    if (x_i == y_i) {
      // No broadcast
      o_i = x_i;
      bx_i = kNoBroadcastValue;
      by_i = kNoBroadcastValue;
      curr = State::SAME;
    } else if (x_i == kNoBroadcastValue) {
      // x broadcast to y on this dimension
      o_i = y_i;
      bx_i = y_i;
      by_i = kNoBroadcastValue;
      curr = State::X_ONE;
    } else if (y_i == kNoBroadcastValue) {
      // y broadcast to x on this dimension
      o_i = x_i;
      bx_i = kNoBroadcastValue;
      by_i = x_i;
      curr = State::Y_ONE;
    } else {
      valid_ = false;
      KERNEL_LOG_ERROR("Broadcast failed, x_shape[%zu]=%ld, y_shape[%zu]=%ld", i, x_i, i, y_i);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    shape_out_.emplace_back(o_i);
    if (curr == State::SAME && x_i == kNoBroadcastValue) {
      continue;
    } else if (prev == curr) {
      result_shape_.back() *= o_i;
      x_reshape_.back() *= x_i;
      x_bcast_.back() *= bx_i;
      y_reshape_.back() *= y_i;
      y_bcast_.back() *= by_i;
    } else {
      result_shape_.emplace_back(o_i);
      x_reshape_.emplace_back(x_i);
      x_bcast_.emplace_back(bx_i);
      y_reshape_.emplace_back(y_i);
      y_bcast_.emplace_back(by_i);
    }
    prev = curr;
  }
  return KERNEL_STATUS_OK;
}

Bcast::Bcast(std::vector<int64_t> &x_shape, std::vector<int64_t> &y_shape) : valid_(true) {
  if (x_shape == y_shape) {
    int64_t elements_num = 1;
    for (size_t i = 0; i < x_shape.size(); ++i) {
      elements_num *= x_shape[i];
      shape_out_.emplace_back(x_shape[i]);
    }
    x_reshape_.emplace_back(elements_num);
    y_reshape_.emplace_back(elements_num);
    result_shape_.emplace_back(elements_num);
    x_bcast_.emplace_back(kNoBroadcastValue);
    y_bcast_.emplace_back(kNoBroadcastValue);
  } else {
    std::vector<int64_t> x = x_shape;
    std::vector<int64_t> y = y_shape;
    std::reverse(x.begin(), x.end());
    std::reverse(y.begin(), y.end());
    if (x.size() > y.size()) {
      y.resize(x.size(), kNoBroadcastValue);
    } else {
      x.resize(y.size(), kNoBroadcastValue);
    }

    auto ret = Init(x, y);
    if (ret != KERNEL_STATUS_OK) {
      return;
    }

    if (result_shape_.empty()) {
      // when both x and y are scalar
      result_shape_.emplace_back(kNoBroadcastValue);
      x_reshape_.emplace_back(kNoBroadcastValue);
      x_bcast_.emplace_back(kNoBroadcastValue);
      y_reshape_.emplace_back(kNoBroadcastValue);
      y_bcast_.emplace_back(kNoBroadcastValue);
    }
    std::reverse(result_shape_.begin(), result_shape_.end());
    std::reverse(x_reshape_.begin(), x_reshape_.end());
    std::reverse(x_bcast_.begin(), x_bcast_.end());
    std::reverse(y_reshape_.begin(), y_reshape_.end());
    std::reverse(y_bcast_.begin(), y_bcast_.end());

    // generate strides, just for row major
    int32_t size = static_cast<int32_t>(result_shape_.size());
    x_input_strides_.resize(size, 0);
    y_input_strides_.resize(size, 0);
    x_output_strides_.resize(size, 0);
    y_output_strides_.resize(size, 0);
    x_input_strides_[size - 1] = 1;
    y_input_strides_[size - 1] = 1;
    x_output_strides_[size - 1] = 1;
    y_output_strides_[size - 1] = 1;
    for (int32_t i = size - 2; i >= 0; --i) {
      x_input_strides_[i] = x_input_strides_[i + 1] * x_reshape_[i + 1];
      y_input_strides_[i] = y_input_strides_[i + 1] * y_reshape_[i + 1];
      x_output_strides_[i] = x_output_strides_[i + 1] * result_shape_[i + 1];
      y_output_strides_[i] = y_output_strides_[i + 1] * result_shape_[i + 1];
    }
  }
}

int64_t Bcast::GetBroadcastXIndex(int64_t index) const {
  int64_t input_index = 0;
  const size_t num_dims = result_shape_.size();
  for (size_t i = 0; i < num_dims - 1; ++i) {
    const int64_t idx = index / x_output_strides_[i];
    if (x_bcast_[i] == kNoBroadcastValue) {
      input_index += idx * x_input_strides_[i];
    } else {
      if (x_reshape_[i] != kNoBroadcastValue) {
        input_index += (idx % x_reshape_[i]) * x_input_strides_[i];
      }
    }
    index -= idx * x_output_strides_[i];
  }
  if (x_bcast_[num_dims - 1] == kNoBroadcastValue) {
    input_index += index;
  } else {
    if (x_reshape_[num_dims - 1] != kNoBroadcastValue) {
      input_index += (index % x_reshape_[num_dims - 1]);
    }
  }
  return input_index;
}

int64_t Bcast::GetBroadcastYIndex(int64_t index) const {
  int64_t input_index = 0;
  const size_t num_dims = result_shape_.size();
  for (size_t i = 0; i < num_dims - 1; ++i) {
    const int64_t idx = index / y_output_strides_[i];
    if (y_bcast_[i] == kNoBroadcastValue) {
      input_index += idx * y_input_strides_[i];
    } else {
      if (y_reshape_[i] != kNoBroadcastValue) {
        input_index += (idx % y_reshape_[i]) * y_input_strides_[i];
      }
    }
    index -= idx * y_output_strides_[i];
  }
  if (y_bcast_[num_dims - 1] == kNoBroadcastValue) {
    input_index += index;
  } else {
    if (y_reshape_[num_dims - 1] != kNoBroadcastValue) {
      input_index += (index % y_reshape_[num_dims - 1]);
    }
  }
  return input_index;
}

uint32_t Bcast::GenerateBcastInfo(const BCalcInfo &calcInfo) {
  const std::vector<int64_t> &shape_x = calcInfo.input_0->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_y = calcInfo.input_1->GetTensorShape()->GetDimSizes();
  const std::vector<int64_t> &shape_out = calcInfo.output->GetTensorShape()->GetDimSizes();
  x_reshape_ = shape_x;
  y_reshape_ = shape_y;
  shape_out_ = shape_out;
  if (shape_x.empty() && shape_y.empty() && shape_out.empty()) {
    // Eigen support scalar
    return KERNEL_STATUS_OK;
  }

  // resize shape_x or shape_y to make size equal
  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());

  size_t dim_num_x = x_reshape_.size();
  size_t dim_num_y = y_reshape_.size();
  size_t max_size = dim_num_x > dim_num_y ? dim_num_x : dim_num_y;
  if (dim_num_x < dim_num_y) {
    x_reshape_.resize(max_size, kNoBroadcastValue);
  } else if (dim_num_x > dim_num_y) {
    y_reshape_.resize(max_size, kNoBroadcastValue);
  }
  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  // Check if shape match
  if (shape_out.size() != max_size) {
    KERNEL_LOG_ERROR("shape mismatch, max_dim_in=%zu, dim_out=%zu.", max_size, shape_out.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  for (size_t i = 0; i < max_size; i++) {
    if (shape_out_[i] != std::max(x_reshape_[i], y_reshape_[i])) {
      KERNEL_LOG_ERROR(
        "shape mismatch, dim_x[%zu]=%ld, dim_y[%zu]=%ld, "
        "dim_out[%zu]=%ld.",
        i, x_reshape_[i], i, y_reshape_[i], i, shape_out_[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  // generate broadcast info
  x_bcast_.resize(max_size, kNoBroadcastValue);
  y_bcast_.resize(max_size, kNoBroadcastValue);
  for (size_t i = 0; i < max_size; i++) {
    if (x_reshape_[i] == y_reshape_[i]) {
      continue;
    }
    if (x_reshape_[i] == kNoBroadcastValue) {
      x_bcast_[i] = y_reshape_[i];
    } else if (y_reshape_[i] == kNoBroadcastValue) {
      y_bcast_[i] = x_reshape_[i];
    } else {
      KERNEL_LOG_ERROR("Broadcast not support, dim_x[%zu]=%ld, dim_y[%zu]=%ld.", i, x_reshape_[i], i, y_reshape_[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  return KERNEL_STATUS_OK;
}

void Bcast::GetBcastVec(BCalcInfo &calcInfo) {
  calcInfo.reshape_0 = std::move(x_reshape_);
  calcInfo.reshape_1 = std::move(y_reshape_);
  calcInfo.shape_out = std::move(shape_out_);
  calcInfo.bcast_0 = std::move(x_bcast_);
  calcInfo.bcast_1 = std::move(y_bcast_);
}

void Bcast::BCastIndexes(std::vector<int64_t> &x_indexes, std::vector<int64_t> &y_indexes) {
  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  std::reverse(shape_out_.begin(), shape_out_.end());

  // Process 0-th dimension
  int64_t x_dim = 1;
  int64_t y_dim = 1;
  int64_t out_dim = 1;

  // If shape_out_ is not empty, get dim of shape vector
  if (!shape_out_.empty()) {
    x_dim = x_reshape_.at(0);
    y_dim = y_reshape_.at(0);
    out_dim = shape_out_.at(0);
  }

  int64_t x_bias = x_dim;
  int64_t y_bias = y_dim;

  for (int64_t i = 0; i < out_dim; i++) {
    x_indexes.push_back(x_dim == 1 ? 0 : i);
    y_indexes.push_back(y_dim == 1 ? 0 : i);
  }

  // Process the remaining dimensions
  for (size_t i = 1; i < shape_out_.size(); i++) {
    x_dim = x_reshape_.at(i);    // i-th dimension of x.
    y_dim = y_reshape_.at(i);    // i-th dimension of y.
    out_dim = shape_out_.at(i);  // i-th dimension of shape_out_.

    std::vector<int64_t>::size_type stride = x_indexes.size();
    for (int64_t j = 1; j < out_dim; j++) {
      for (std::vector<int64_t>::size_type k = 0; k < stride; k++) {
        x_indexes.push_back(x_indexes.at(k) + (x_dim == 1 ? 0 : (j * x_bias)));
        y_indexes.push_back(y_indexes.at(k) + (y_dim == 1 ? 0 : (j * y_bias)));
      }
    }
    x_bias *= x_dim;
    y_bias *= y_dim;
  }

  std::reverse(x_reshape_.begin(), x_reshape_.end());
  std::reverse(y_reshape_.begin(), y_reshape_.end());
  std::reverse(shape_out_.begin(), shape_out_.end());
}
}  // namespace aicpu
