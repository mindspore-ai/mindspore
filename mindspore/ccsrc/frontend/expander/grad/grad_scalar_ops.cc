/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/grad/grad_utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradScalarOps)
REG_BPROP_BUILDER("ScalarAdd").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i3);
  return {dout, dout};
});

REG_BPROP_BUILDER("ScalarSub").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i3);
  return {dout, ib->ScalarNeg(dout)};
});

REG_BPROP_BUILDER("ScalarMul").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return {ib->ScalarMul(y, dout), ib->ScalarMul(x, dout)};
});

REG_BPROP_BUILDER("ScalarDiv").SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i1);
  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto dx = ib->ScalarDiv(dout, y);
  return {dx, ib->ScalarNeg(ib->ScalarMul(dx, out))};
});

REG_BPROP_BUILDER("ScalarMod").SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  NodePtr dx = x->need_compute_grad_out() ? dout : ib->OutZeros(x);
  NodePtr dy = y->need_compute_grad_out()
                 ? ib->ScalarNeg(ib->ScalarMul(ib->ScalarDiv(dout, y), ib->ScalarFloorDiv(x, y)))
                 : ib->OutZeros(y);
  return {dx, dy};
});

REG_BPROP_BUILDER("ScalarFloorDiv").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ScalarEq").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ScalarLe").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ScalarLt").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ScalarGe").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ScalarGt").SetBody(ReturnZeros);
REG_BPROP_BUILDER("bit_and").SetBody(ReturnZeros);
REG_BPROP_BUILDER("bit_or").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ScalarBool").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ScalarMax").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ScalarMin").SetBody(ReturnZeros);
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
