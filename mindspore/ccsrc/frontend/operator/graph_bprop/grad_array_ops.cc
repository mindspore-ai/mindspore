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

void RegArrayOps() { REGISTER_PRIMITIVE_BPROP_IMPL(Cast, CastBprop); }
}  // namespace graph_bprop
}  // namespace mindspore
