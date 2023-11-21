/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "backend/common/expander/fallback/fallback_irbuilder.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace expander {
namespace {
bool IsLastAxis(const ShapeVector &shape, int64_t axis) {
  if (axis == -1) {
    return true;
  }
  if (IsDynamicRank(shape)) {
    return false;
  }
  auto rank = SizeToLong(shape.size());
  if (axis < 0) {
    axis += rank;
  }
  return (axis == (rank - 1));
}

std::vector<int64_t> GetTransposeAxis(const std::vector<int64_t> &x_shape, int64_t axis) {
  std::vector<int64_t> reverse_axis;
  if (x_shape.empty()) {
    return reverse_axis;
  }
  auto rk = static_cast<int64_t>(x_shape.size());
  if (axis < 0) {
    axis += rk;
  }
  reverse_axis.reserve(x_shape.size());
  for (int64_t i = 0; i < rk; ++i) {
    (void)reverse_axis.emplace_back(i);
  }
  reverse_axis[LongToSize(axis)] = rk - 1;
  reverse_axis[LongToSize(rk - 1)] = axis;
  return reverse_axis;
}
}  // namespace
REG_FALLBACK_BUILDER("SiLU").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto s = ib->Emit("Sigmoid", {ib->GetInput(kIndex0)});
  return {ib->Mul(input_x, s)};
});

REG_FALLBACK_BUILDER("SiLUGrad").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto sigmoid_input = ib->Emit("Sigmoid", {x});
  auto bc_dx = ib->Mul(x, dout);
  auto bc_dy = ib->Mul(sigmoid_input, dout);
  auto dx = ib->Emit("SigmoidGrad", {sigmoid_input, bc_dx});
  return {ib->Add(dx, bc_dy)};
});

REG_FALLBACK_BUILDER("Baddbmm").SetBody(BODYFUNC(ib) {
  // baddbmm equation: output = beta * input + alpha * matmul(batch1, batch2)
  auto input = ib->GetInput(kIndex0);
  auto batch1 = ib->GetInput(kIndex1);
  auto batch2 = ib->GetInput(kIndex2);
  auto beta = ib->GetInput(kIndex3);
  auto alpha = ib->GetInput(kIndex4);

  auto mm_output = ib->BatchMatMul(batch1, batch2);
  auto alpha_output = ib->Mul(mm_output, alpha);
  auto beta_output = ib->Mul(input, beta);
  return {ib->Add(beta_output, alpha_output)};
});

DEF_PURE_SHAPE_CALC(g_dense_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &x_shape = inputs.at(kIndex0);
    auto &w_shape = inputs.at(kIndex1);
    ShapeVector reshape_x_shape = {-1, x_shape.back()};
    ShapeVector reshape_w_shape = {-1, w_shape.back()};
    ShapeVector reshape_ret_shape;
    bool is_empty = std::any_of(x_shape.begin(), x_shape.end(), [](const int64_t shape) { return shape == 0; });
    if (is_empty) {
      reshape_x_shape[0] = 1;
      reshape_w_shape[0] = 1;
      return {reshape_x_shape, reshape_w_shape, reshape_ret_shape};
    }
    if (x_shape.size() != 1) {
      reshape_ret_shape = x_shape;
      reshape_ret_shape.back() = -1;
    }
    return {reshape_x_shape, reshape_w_shape, reshape_ret_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    constexpr const int64_t kRank2 = 2;
    int64_t ret_size = -1LL;
    if (!IsDynamicRank(inputs[0])) {
      if (inputs[0].size() == 1) {
        ret_size = 0;
      } else {
        ret_size = inputs[0].size();
      }
    }
    return {kRank2, kRank2, ret_size};
  });

REG_FALLBACK_BUILDER("Dense").SetBody(BODYFUNC(ib) {
  constexpr const size_t kRank2 = 2;
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  NodePtrList reshape_shapes;
  auto has_bias = ib->GetAttr<bool>("has_bias");
  auto x_shape = x->shape();
  auto w_shape = w->shape();
  bool is_empty_tensor = x_shape.size() == 1 && w_shape.size() == 1 && x_shape[0] == 0 && w_shape[0] == 0;
  if (is_empty_tensor) {
    if (has_bias) {
      return {ib->GetInput(kIndex2)};
    }
    return {ib->Tensor(0, x->dtype())};
  }
  bool is_dynamic_rank = IsDynamicRank(x_shape) || IsDynamicRank(w_shape);
  bool need_reshape = (is_dynamic_rank || x_shape.size() != kRank2 || w_shape.size() != kRank2);
  if (need_reshape) {
    reshape_shapes = ib->ShapeCalc(g_dense_shapecalc, {x, w});
    x = ib->Reshape(x, reshape_shapes[kIndex0]);
    w = ib->Reshape(w, reshape_shapes[kIndex1]);
  }
  auto ret = ib->MatMul(x, w, false, true);
  if (has_bias) {
    auto b = ib->GetInput(kIndex2);
    ret = ib->Add(ret, b);
  }
  if (need_reshape) {
    ret = ib->Reshape(ret, reshape_shapes[kIndex2]);
  }
  return {ret};
});

class SoftmaxShapeCalc : public ShapeCalcFunctor {
 public:
  SoftmaxShapeCalc() : ShapeCalcFunctor("ShapeCalc_Softmax") {}
  ~SoftmaxShapeCalc() override = default;
  MS_DECLARE_PARENT(SoftmaxShapeCalc, ShapeCalcFunctor)

  ValuePtr ToValue() const override { return nullptr; }
  void FromValue(const ValuePtr &value) override {}
  ShapeArray Calc(const ShapeArray &inputs) const override {
    // inputs: {dout_shape, dim}
    auto dim = inputs.at(1)[0];
    return {GetTransposeAxis(inputs.at(0), dim)};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    int64_t dout_rank = IsDynamicRank(inputs.at(0)) ? -1 : SizeToLong(inputs.at(0).size());
    return {dout_rank};
  }
};
REG_FUNCTOR("ShapeCalc_Softmax", SoftmaxShapeCalc);

REG_FALLBACK_BUILDER("SoftmaxBackward").SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dim = ib->GetInput(kIndex2);

  auto shp = out->shape();
  auto dim_value_ptr = dim->BuildValue();
  int64_t dim_value{0};
  bool success = false;
  if (!(dim_value_ptr->isa<ValueAny>() || dim_value_ptr->isa<None>())) {
    dim_value = GetValue<int64_t>(dim_value_ptr);
    success = true;
  }
  if (success && IsLastAxis(shp, dim_value)) {
    auto dx = ib->Mul(out, ib->Sub(dout, ib->ReduceSum(ib->Mul(out, dout), ShapeVector{-1}, true)));
    return {dx};
  }
  auto reverse_axis = (IsDynamicRank(shp) || !success)
                        ? ib->ShapeCalc(std::make_shared<SoftmaxShapeCalc>(), {dout, dim}, {1})[0]
                        : ib->Value(GetTransposeAxis(shp, dim_value));
  out = ib->Transpose(out, reverse_axis);
  dout = ib->Transpose(dout, reverse_axis);
  auto dx = ib->Mul(out, ib->Sub(dout, ib->ReduceSum(ib->Mul(out, dout), ShapeVector{-1}, true)));
  dx = ib->Transpose(dx, reverse_axis);
  return {dx};
});
}  // namespace expander
}  // namespace mindspore
