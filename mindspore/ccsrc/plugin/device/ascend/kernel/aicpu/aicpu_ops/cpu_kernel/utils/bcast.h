/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#ifndef _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_
#define _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_

#include <vector>

#include "cpu_kernel/inc/cpu_context.h"

namespace aicpu {
// broadcast shape type
// 1. SAME_SHAPE : x and y have the same shape
// 2. X_ONE : x has only one element
// 3. Y_ONE : y has only one element
enum class BcastShapeType {
  SAME_SHAPE = 0,
  X_ONE_ELEMENT = 1,
  Y_ONE_ELEMENT = 2,
  DIFF_SHAPE = 3,
};

struct BCalcInfo {
  BCalcInfo() : input_0(nullptr), input_1(nullptr), output(nullptr) {}
  Tensor *input_0;
  Tensor *input_1;
  Tensor *output;
  std::vector<int64_t> reshape_0;
  std::vector<int64_t> reshape_1;
  std::vector<int64_t> shape_out;
  std::vector<int64_t> bcast_0;
  std::vector<int64_t> bcast_1;
  std::vector<int64_t> x_indexes;
  std::vector<int64_t> y_indexes;
};

class Bcast {
 public:
  Bcast() : valid_(true){};
  Bcast(std::vector<int64_t> &x_shape, std::vector<int64_t> &y_shape);
  ~Bcast() = default;

  uint32_t GenerateBcastInfo(const BCalcInfo &calcInfo);
  void GetBcastVec(BCalcInfo &calcInfo);
  void BCastIndexes(std::vector<int64_t> &x_indexes, std::vector<int64_t> &y_indexes);
  int64_t GetBroadcastXIndex(int64_t index) const;
  int64_t GetBroadcastYIndex(int64_t index) const;
  bool IsValid() const { return valid_; }
  const std::vector<int64_t> &x_reshape() const { return x_reshape_; }
  const std::vector<int64_t> &y_reshape() const { return y_reshape_; }
  const std::vector<int64_t> &result_shape() const { return result_shape_; }
  const std::vector<int64_t> &x_bcast() const { return x_bcast_; }
  const std::vector<int64_t> &y_bcast() const { return y_bcast_; }

 private:
  uint32_t Init(const std::vector<int64_t> &x, const std::vector<int64_t> &y);

  bool valid_;
  std::vector<int64_t> x_reshape_;
  std::vector<int64_t> y_reshape_;
  std::vector<int64_t> shape_out_;
  std::vector<int64_t> x_bcast_;
  std::vector<int64_t> y_bcast_;
  std::vector<int64_t> result_shape_;
  std::vector<int64_t> x_input_strides_;
  std::vector<int64_t> y_input_strides_;
  std::vector<int64_t> x_output_strides_;
  std::vector<int64_t> y_output_strides_;
};
}  // namespace aicpu
#endif  // _AICPU_AICPU_DEVICE_CPU_KERNELS_UTILS_BCAST_H_
