/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/expander/base/ir_builder.h"

namespace mindspore::graphkernel::expander {
REG_EXPANDER_FUNC("Tanh").SetBody(BODYFUNC(ib) {
  auto result = ib->Tanh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("Sinh").SetBody(BODYFUNC(ib) {
  auto result = ib->Sinh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("Cosh").SetBody(BODYFUNC(ib) {
  auto result = ib->Cosh(ib->input(kIndex0));
  return {result};
});

REG_EXPANDER_FUNC("LogicalXor").SetBody(BODYFUNC(ib) {
  const auto &input_x = ib->input(kIndex0);
  const auto &input_y = ib->input(kIndex1);

  auto result_b = ib->LogicalAnd(input_x, ib->LogicalNot(input_y));
  auto result_a = ib->LogicalAnd(input_y, ib->LogicalNot(input_x));
  return {ib->LogicalOr(result_a, result_b)};
});

NodePtrList FastGeluExpand(const DefaultIrBuilder *ib) {
  const auto &input_x = ib->input(kIndex0);
  const double val = 1.7020000219345093;
  auto const_0 = ib->Tensor(-val, input_x->GetDtype());
  auto const_1 = ib->Tensor(val / 2, input_x->GetDtype());
  auto const_2 = ib->Tensor(1, input_x->GetDtype());

  auto abs = ib->Abs(input_x);
  auto sub = ib->Sub(input_x, abs);
  auto exp_0 = ib->Exp(ib->Mul(const_1, sub));
  auto n = ib->Mul(input_x, exp_0);
  auto exp_1 = ib->Exp(ib->Mul(const_0, abs));
  auto d = ib->Add(exp_1, const_2);

  return {ib->Div(n, d)};
}

NodePtrList FastGeluGradExpand(const DefaultIrBuilder *ib) {
  const auto &input_x = ib->input(kIndex1);
  const auto &dout = ib->input(kIndex0);
  const double val = 1.7020000219345093;
  auto const_0 = ib->Tensor(val, input_x->GetDtype());
  auto const_1 = ib->Tensor(-val, input_x->GetDtype());
  auto const_2 = ib->Tensor(1, input_x->GetDtype());

  auto abs = ib->Abs(input_x);
  auto mul_1 = ib->Exp(ib->Mul(const_1, abs));
  auto mul_3 = ib->Mul(input_x, mul_1);
  mul_3 = ib->Mul(const_0, mul_3);
  mul_3 = ib->Add(mul_3, mul_1);

  auto sub = ib->Sub(input_x, abs);
  sub = ib->Exp(ib->Mul(sub, const_0));

  mul_3 = ib->Add(sub, mul_3);
  mul_1 = ib->Add(mul_1, const_2);
  mul_1 = ib->Mul(mul_1, mul_1);

  return {ib->Mul(ib->Div(mul_3, mul_1), dout)};
}

REG_EXPANDER_FUNC("FastGelu").SetBody(FastGeluExpand);
REG_EXPANDER_FUNC("FastGeLU").SetBody(FastGeluExpand);
REG_EXPANDER_FUNC("FastGeluGrad").SetBody(FastGeluGradExpand);
REG_EXPANDER_FUNC("FastGeLUGrad").SetBody(FastGeluGradExpand);
}  // namespace mindspore::graphkernel::expander
