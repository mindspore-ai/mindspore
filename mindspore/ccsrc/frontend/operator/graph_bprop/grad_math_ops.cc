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

#include "frontend/operator/graph_bprop/bprop_meta_func_graph.h"
#include "frontend/operator/graph_bprop/utils.h"
#include "frontend/operator/graph_bprop/ops_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace graph_bprop {
FuncGraphPtr MatMulBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  // x, w, out, dout
  constexpr size_t expected_arg_size = 4;
  const auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);
  auto x = parameters[kIndex0];
  auto w = parameters[kIndex1];
  auto dout = parameters[kIndex3];

  auto x_origin_type = GetTensorDType(input_abs[0]);
  if (x_origin_type == kNumberTypeComplex64 || x_origin_type == kNumberTypeComplex128) {
    x = NewNode(fg, {Conj(), x});
    w = NewNode(fg, {Conj(), w});
  }

  auto ta = GetAttr<bool>(primal, "transpose_a");
  auto tb = GetAttr<bool>(primal, "transpose_b");
  auto mul1 = MatMul(fg, ta && tb, ta || !tb);
  auto mul2 = MatMul(fg, !ta || tb, ta && tb);
  AnfNodePtr dx;
  AnfNodePtr dw;
  if (ta) {
    dx = NewNode(fg, {mul1, w, dout});
  } else {
    dx = NewNode(fg, {mul1, dout, w});
  }
  if (tb) {
    dw = NewNode(fg, {mul2, dout, x});
  } else {
    dw = NewNode(fg, {mul2, x, dout});
  }
  fg->set_output(NewNode(fg, {MakeTuple(), dx, dw}));
  return fg;
}
REGISTER_PRIMITIVE_BPROP_IMPL(MatMul, prim::kPrimMatMul, MatMulBprop, 2);

FuncGraphPtr SubBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  // x, y, out, dout
  constexpr size_t expected_arg_size = 4;
  const auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);

  auto neg_dout = NewNode(fg, {Neg(), parameters[kIndex3]}, true);
  fg->set_output(BinopGradCommon(fg, parameters[kIndex0], parameters[kIndex1], parameters[kIndex3], neg_dout));
  return fg;
}
REGISTER_PRIMITIVE_BPROP_IMPL(Sub, prim::kPrimSub, SubBprop, 2);
}  // namespace graph_bprop
}  // namespace mindspore
