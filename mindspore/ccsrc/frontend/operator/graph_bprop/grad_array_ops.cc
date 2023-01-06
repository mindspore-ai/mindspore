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
namespace {
std::vector<int64_t> TransposePermPositive(const AbstractBasePtr &perm_abs) {
  auto perm = GetValue<std::vector<int64_t>>(perm_abs->BuildValue());
  std::vector<int64_t> res;
  for (auto &p : perm) {
    (void)res.emplace_back((p >= 0) ? p : (p + perm.size()));
  }
  return res;
}

AnfNodePtr DynTransposePermPositive(const FuncGraphPtr &fg, const AnfNodePtr &perm) {
  auto add = NewNode(fg, {Add(), perm, DynSize(fg, perm)});
  return NewNode(fg, {Mod(), add, DynSize(fg, perm)});
}
}  // namespace

FuncGraphPtr TransposeBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  // x, perm, out, dout
  constexpr size_t expected_arg_size = 4;
  const auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);
  auto perm = parameters[kIndex1];
  auto dout = parameters[kIndex3];

  bool is_mutable = ConvertToTensor(perm);
  if (is_mutable) {
    perm = DynTransposePermPositive(fg, perm);
    auto transpose = NewNode(fg, {Transpose(fg), dout, DynInvertPermutation(fg, perm)});
    auto zeros_like = ZerosLikeFunction(fg, perm);
    fg->set_output(NewNode(fg, {MakeTuple(), transpose, zeros_like}));
    return fg;
  }

  perm = NewValueNode(TransposePermPositive(input_abs[kIndex1]));
  auto invert_permutation = NewNode(fg, {InvertPermutation(fg), perm});
  auto transpose = NewNode(fg, {Transpose(fg), dout, invert_permutation});
  auto zeros_like = ZerosLikeFunction(fg, perm);
  fg->set_output(NewNode(fg, {MakeTuple(), transpose, zeros_like}));
  return fg;
}
REGISTER_PRIMITIVE_BPROP_IMPL(Transpose, prim::kPrimTranspose, TransposeBprop, 2);

FuncGraphPtr CastBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  constexpr size_t expected_arg_size = 4;
  auto fg = NewGraph(input_abs);
  // x, out, dout
  auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);
  auto x = parameters[kIndex0];
  auto t = parameters[kIndex1];
  auto dout = parameters[kIndex3];
  const auto zeros_like_node = ZerosLikeFunction(fg, t);
  const auto cast = Cast(fg);
  const auto dtype = DType();
  AnfNodePtr return_node;
  auto get_dtype = NewNode(fg, {dtype, x});
  if (input_abs[kIndex3]->isa<abstract::AbstractRowTensor>()) {
    auto row_tensor_values = NewNode(fg, {RowTensorGetValues(), dout});
    auto value = NewNode(fg, {cast, row_tensor_values, get_dtype});
    auto indices = NewNode(fg, {RowTensorGetIndices(), dout});
    auto dense_shape = NewNode(fg, {RowTensorGetDenseShape(), dout});
    return_node = NewNode(fg, {MakeRowTensor(), indices, value, dense_shape});
  } else {
    return_node = NewNode(fg, {cast, dout, get_dtype});
  }
  fg->set_output(NewNode(fg, {MakeTuple(), return_node, zeros_like_node}));
  return fg;
}
REGISTER_PRIMITIVE_BPROP_IMPL(Cast, prim::kPrimCast, CastBprop, 2);
}  // namespace graph_bprop
}  // namespace mindspore
