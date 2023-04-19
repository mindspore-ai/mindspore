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
#ifndef MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_GRAD_OPS_SHAPE_CALC_FUNCTORS_H_
#define MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_GRAD_OPS_SHAPE_CALC_FUNCTORS_H_

#include <vector>
#include "ir/functor.h"

namespace mindspore::expander::bprop {
class SumGradShapeCalc : public ShapeCalcFunctor {
 public:
  DECLARE_PURE_SHAPE_CALC(SumGradShapeCalc)
  ShapeArray Calc(const ShapeArray &inputs) const override;
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override;
};

class SoftmaxShapeCalc : public ShapeCalcFunctor {
 public:
  DECLARE_SHAPE_CALC(SoftmaxShapeCalc)
  explicit SoftmaxShapeCalc(int64_t axis) : ShapeCalcFunctor("SoftmaxShapeCalc"), axis_(axis) {}
  ValuePtr ToValue() const override { return MakeValue(axis_); }
  void FromValue(const ValuePtr &value) override { axis_ = GetValue<int64_t>(value); }
  ShapeArray Calc(const ShapeArray &inputs) const override;
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override;

 protected:
  int64_t axis_{0};
};

class BroadcastGradientArgsShapeCalc : public ShapeCalcFunctor {
 public:
  DECLARE_SHAPE_CALC(BroadcastGradientArgsShapeCalc)
  explicit BroadcastGradientArgsShapeCalc(size_t shift)
      : ShapeCalcFunctor("BroadcastGradientArgsShapeCalc"), shift_(shift) {}
  ValuePtr ToValue() const override { return MakeValue(shift_); }
  void FromValue(const ValuePtr &value) override { shift_ = GetValue<size_t>(value); }
  ShapeArray Calc(const ShapeArray &inputs) const override;
  std::vector<int64_t> Infer(const ShapeArray &, const HashSet<size_t> &) const override;

 protected:
  size_t shift_{0};
};
}  // namespace mindspore::expander::bprop
#endif  // MINDSPORE_CCSRC_FRONTEND_EXPANDER_BPROP_GRAD_OPS_SHAPE_CALC_FUNCTORS_H_
