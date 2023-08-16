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

DEF_PURE_SHAPE_CALC(g_dense_shapecalc)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto &x_shape = inputs.at(kIndex0);
    ShapeVector reshape_x_shape = {-1, x_shape.back()};
    ShapeVector reshape_ret_shape = x_shape;
    reshape_ret_shape.back() = -1;
    return {reshape_x_shape, reshape_ret_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    constexpr const int64_t kRank2 = 2;
    return {kRank2, IsDynamicRank(inputs[0]) ? -1LL : static_cast<int64_t>(inputs[0].size())};
  });

REG_FALLBACK_BUILDER("Dense").SetBody(BODYFUNC(ib) {
  constexpr const size_t kRank2 = 2;
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  NodePtrList reshape_shapes;
  bool need_reshape = (IsDynamicRank(x->shape()) || x->shape().size() != kRank2);
  if (need_reshape) {
    reshape_shapes = ib->ShapeCalc(g_dense_shapecalc, {x});
    x = ib->Reshape(x, reshape_shapes[kIndex0]);
  }
  auto ret = ib->MatMul(x, w, false, true);
  auto has_bias = ib->GetAttr<bool>("has_bias");
  if (has_bias) {
    auto b = ib->GetInput(kIndex2);
    ret = ib->Emit("BiasAdd", {ret, b}, {{"format", MakeValue(kOpFormat_NCHW)}});
  }
  if (need_reshape) {
    ret = ib->Reshape(ret, reshape_shapes[kIndex1]);
  }
  return {ret};
});
}  // namespace expander
}  // namespace mindspore
