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
#ifndef AICPU_UTILS_BROADCAST_ITERATOR_H
#define AICPU_UTILS_BROADCAST_ITERATOR_H

#include <array>
#include <cstdint>
#include <vector>

#include "cpu_kernel/common/status.h"

namespace aicpu {
class BroadcastIterator {
 public:
  BroadcastIterator(std::vector<int64_t> &input_shape_a, std::vector<int64_t> &input_shape_b,
                    std::vector<int64_t> &output_shape);
  virtual ~BroadcastIterator() = default;
  inline int64_t GetInputPosA() const { return input_pos_[0]; }
  inline int64_t GetInputPosB() const { return input_pos_[1]; }
  /**
   * @brief set broadcast start position
   * @param broadcast start position
   */
  void SetPos(int64_t pos);
  /**
   * @brief generate next position
   */
  void GenNextPos();

 private:
  void BroadcastShape();
  void InitStrides();

  std::vector<int64_t> coordinates_;
  std::vector<int64_t> input_shape_a_;
  std::vector<int64_t> input_shape_b_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> input_strides_a_;
  std::vector<int64_t> input_strides_b_;
  std::vector<int64_t> input_back_strides_a_;
  std::vector<int64_t> input_back_strides_b_;
  std::array<int64_t, 2> input_pos_ = {{0, 0}};
  size_t output_dimension_{0};
};

/**
 * @brief get broadcast shape
 * @param shape to broadcast
 * @return status
 */
uint32_t GetBroadcastShape(const std::vector<int64_t> &x, const std::vector<int64_t> &y,
                           std::vector<int64_t> &broadcast_shape);
}  // namespace aicpu
#endif
