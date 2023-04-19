/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "frontend/expander/bprop/grad_ops/shape_calc_functors.h"
#include "frontend/expander/bprop/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"

namespace mindspore::expander::bprop {
ShapeArray SumGradShapeCalc::Calc(const ShapeArray &inputs) const {
  auto x_shape = inputs.at(0);
  auto axis_value = inputs.at(1);
  auto r_shape = ReduceShape(x_shape, axis_value);
  auto scaling = TupleDiv(x_shape, r_shape);
  return {r_shape, scaling};
}

std::vector<int64_t> SumGradShapeCalc::Infer(const ShapeArray &inputs, const HashSet<size_t> &) const {
  int64_t x_rank = IsDynamicRank(inputs.at(0)) ? -1 : static_cast<int64_t>(inputs.at(0).size());
  return {x_rank, x_rank};
}

ShapeArray SoftmaxShapeCalc::Calc(const ShapeArray &inputs) const { return {GetTransposeAxis(inputs.at(0), axis_)}; }

std::vector<int64_t> SoftmaxShapeCalc::Infer(const ShapeArray &inputs, const HashSet<size_t> &) const {
  int64_t x_rank = IsDynamicRank(inputs.at(0)) ? -1 : SizeToLong(inputs.at(0).size());
  return {x_rank};
}

ShapeArray BroadcastGradientArgsShapeCalc::Calc(const ShapeArray &inputs) const {
  auto shape_x = inputs.at(kIndex0);
  ShapeVector broadcast_shape_of_x;
  auto x_shape_num = shape_x.size() > shift_ ? (shape_x.size() - shift_) : 0;
  for (size_t i = 0; i < x_shape_num; ++i) {
    broadcast_shape_of_x.push_back(shape_x[i]);
  }
  auto shape_y = inputs.at(kIndex1);
  ShapeVector broadcast_shape_of_y;
  auto y_shape_num = shape_y.size() > shift_ ? (shape_y.size() - shift_) : 0;
  for (size_t i = 0; i < y_shape_num; ++i) {
    broadcast_shape_of_y.push_back(shape_y[i]);
  }
  auto broadcast_axis = bprop::BroadcastGradientArgs(broadcast_shape_of_x, broadcast_shape_of_y);
  return broadcast_axis;
}

std::vector<int64_t> BroadcastGradientArgsShapeCalc::Infer(const ShapeArray &, const HashSet<size_t> &) const {
  constexpr int64_t kShapeDimAny = -1;
  return {kShapeDimAny, kShapeDimAny};
}

REG_FUNCTOR(BroadcastGradientArgsShapeCalc);
REG_FUNCTOR(SoftmaxShapeCalc);
REG_FUNCTOR(SumGradShapeCalc);
}  // namespace mindspore::expander::bprop
