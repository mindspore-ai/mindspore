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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_OPS_UTILS_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_OPS_UTILS_H_

#include "include/api/cfg.h"
#include "src/expression/net.h"
#include "vector"

namespace mindspore {
namespace lite {
class BroadcastGradientArgs {
 public:
  BroadcastGradientArgs(const std::vector<int> &dim0, const std::vector<int> &dim1) : dim0_(dim0), dim1_(dim1) {}
  std::vector<std::vector<int>> operator()();

 private:
  static const int kInNum = 2;
  const std::vector<int> &dim0_;
  const std::vector<int> &dim1_;
};

class DynamicBroadcastGradientArgs {
 public:
  DynamicBroadcastGradientArgs(const std::vector<int> &dim0, const std::vector<int> &dim1) : dim0_(dim0), dim1_(dim1) {}
  std::vector<std::vector<int>> operator()();

 private:
  void AddElementToGradReduceIdx(std::vector<std::vector<int>> *grad_reduce_idx, std::vector<bool> current_is_one,
                                 bool none_is_one, const size_t largest_rank, size_t j);
  void UpdatePreIsOne(std::vector<bool> *prev_is_one, std::vector<bool> current_is_one);
  std::vector<std::vector<int>> GetGradientIndices(const std::vector<std::vector<int>> &reverse_shape,
                                                   const size_t largest_rank);
  std::vector<std::vector<int>> CalculateOutput(const std::vector<std::vector<int>> &x);
  std::vector<std::vector<int>> SetOutputValue(const std::vector<std::vector<int>> &grad_reduce_idx,
                                               const std::vector<std::vector<int>> &input_dim);
  static const int kInNum = 2;
  const std::vector<int> &dim0_;
  const std::vector<int> &dim1_;
};

class VectorDiv {
 public:
  VectorDiv() {}
  std::vector<int> operator()(const std::vector<int> &x, const std::vector<int> &d);
};

class ShapeReduce {
 public:
  ShapeReduce() {}
  std::vector<int> operator()(const std::vector<int> &x_shape, const std::vector<int> &axis);
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXPRESSION_OPS_UTILS_H_
