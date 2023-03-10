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
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace graph_bprop {
FuncGraphPtr ReluBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  // x, out, dout
  constexpr size_t expected_arg_size = 3;
  const auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);
  auto out = parameters[kIndex1];
  auto dout = parameters[kIndex2];

  auto dx = NewNode(fg, {ReluGrad(), dout, out});
  fg->set_output(NewNode(fg, {MakeTuple(), dx}));
  return fg;
}

FuncGraphPtr LayerNormBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  // x, gamma, beta, out, dout
  constexpr size_t expected_arg_size = 5;
  const auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);
  auto x = parameters[kIndex0];
  auto gamma = parameters[kIndex1];
  auto out = parameters[kIndex3];
  auto dout = parameters[kIndex4];

  auto begin_norm_axis = GetAndCheckAttr(primal, "begin_norm_axis");
  auto begin_params_axis = GetAndCheckAttr(primal, "begin_params_axis");
  auto layer_norm_grad_ops = LayerNormGrad(fg, begin_norm_axis, begin_params_axis);
  auto dout0 = TupleGetItem(fg, dout, SizeToLong(kIndex0));
  auto out2 = TupleGetItem(fg, out, SizeToLong(kIndex2));
  auto out1 = TupleGetItem(fg, out, SizeToLong(kIndex1));
  auto layer_norm_grad = NewNode(fg, {layer_norm_grad_ops, x, dout0, out2, out1, gamma});

  auto dx = TupleGetItem(fg, layer_norm_grad, SizeToLong(kIndex0));
  auto d_gamma = TupleGetItem(fg, layer_norm_grad, SizeToLong(kIndex1));
  auto d_beta = TupleGetItem(fg, layer_norm_grad, SizeToLong(kIndex2));
  fg->set_output(NewNode(fg, {MakeTuple(), dx, d_gamma, d_beta}));
  return fg;
}

FuncGraphPtr MaxPoolBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  // x, out, dout
  constexpr size_t expected_arg_size = 3;
  const auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);
  auto x = parameters[kIndex0];
  auto out = parameters[kIndex1];
  auto dout = parameters[kIndex2];

  auto dx = NewNode(fg, {MaxPoolGrad(fg, primal), x, out, dout});
  fg->set_output(NewNode(fg, {MakeTuple(), dx}));
  return fg;
}

FuncGraphPtr BatchNormBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  // x, scale, b, mean, variance, out, dout
  constexpr size_t expected_arg_size = 7;
  const auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);
  auto x = parameters[kIndex0];
  auto scale = parameters[kIndex1];
  auto mean = parameters[kIndex3];
  auto variance = parameters[kIndex4];
  auto out = parameters[kIndex5];
  auto dout = parameters[kIndex6];

  AnfNodePtr saved_mean;
  AnfNodePtr saved_variance;
  AnfNodePtr reserve;
  auto is_training = GetValue<bool>(GetAndCheckAttr(primal, "is_training"));
  if (is_training) {
    saved_mean = TupleGetItem(fg, out, SizeToLong(kIndex3));
    saved_variance = TupleGetItem(fg, out, SizeToLong(kIndex4));
    reserve = TupleGetItem(fg, out, SizeToLong(kIndex2));
  } else {
    saved_mean = mean;
    saved_variance = variance;
    reserve = TupleGetItem(fg, out, SizeToLong(kIndex2));
  }
  auto dout0 = TupleGetItem(fg, dout, SizeToLong(kIndex0));
  auto input_grad = NewNode(fg, {BatchNormGrad(fg, primal), dout0, x, scale, saved_mean, saved_variance, reserve});
  auto dx = TupleGetItem(fg, input_grad, SizeToLong(kIndex0));
  auto dscale = TupleGetItem(fg, input_grad, SizeToLong(kIndex1));
  auto dbias = TupleGetItem(fg, input_grad, SizeToLong(kIndex2));
  fg->set_output(
    NewNode(fg, {MakeTuple(), dx, dscale, dbias, ZerosLikeFunction(fg, mean), ZerosLikeFunction(fg, variance)}));
  return fg;
}

FuncGraphPtr BiasAddBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  auto format = GetAttr<std::string>(primal, "format");
  // x, out, dout
  constexpr size_t expected_arg_size = 4;
  const auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);
  auto dout = parameters[kIndex3];
  auto bais_add_grad = NewNode(fg, {BiasAddGrad(format), dout});
  fg->set_output(NewNode(fg, {MakeTuple(), dout, bais_add_grad}));
  return fg;
}

FuncGraphPtr GeLUBprop(const PrimitivePtr &primal, const AbstractBasePtrList &input_abs) {
  auto fg = NewGraph(input_abs);
  // x, out, dout
  constexpr size_t expected_arg_size = 3;
  const auto &parameters = fg->parameters();
  CheckArgSize(parameters, input_abs, primal, expected_arg_size);
  auto x = parameters[kIndex0];
  auto out = parameters[kIndex1];
  auto dout = parameters[kIndex2];
  auto dx = NewNode(fg, {GeLUGrad(), dout, x, out});
  fg->set_output(NewNode(fg, {MakeTuple(), dx}));
  return fg;
}

void RegNNOps() {
  REGISTER_PRIMITIVE_BPROP_IMPL(ReLU, ReluBprop);
  REGISTER_PRIMITIVE_BPROP_IMPL(LayerNorm, LayerNormBprop);
  REGISTER_PRIMITIVE_BPROP_IMPL(MaxPool, MaxPoolBprop);
  REGISTER_PRIMITIVE_BPROP_IMPL(BatchNorm, BatchNormBprop);
  REGISTER_PRIMITIVE_BPROP_IMPL(BiasAdd, BiasAddBprop);
  REGISTER_PRIMITIVE_BPROP_IMPL(GeLU, GeLUBprop);
}
}  // namespace graph_bprop
}  // namespace mindspore
