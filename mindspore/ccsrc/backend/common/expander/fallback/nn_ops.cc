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
}  // namespace expander
}  // namespace mindspore
