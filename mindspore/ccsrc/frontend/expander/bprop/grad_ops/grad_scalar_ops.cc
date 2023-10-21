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
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/bprop/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradScalarOps)
REG_BPROP_BUILDER("ScalarAdd").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex3);
  return {dout, dout};
});

REG_BPROP_BUILDER("ScalarSub").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex3);
  return {dout, ib->ScalarNeg(dout)};
});

REG_BPROP_BUILDER("ScalarMul").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {ib->ScalarMul(y, dout), ib->ScalarMul(x, dout)};
});

REG_BPROP_BUILDER("ScalarDiv").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->ScalarDiv(dout, y);
  return {dx, ib->ScalarNeg(ib->ScalarMul(dx, out))};
});

REG_BPROP_BUILDER("ScalarMod").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->ScalarDiv(dout, y);
  return {dout, ib->ScalarNeg(ib->ScalarMul(dx, ib->ScalarFloordiv(x, y)))};
});

REG_BPROP_BUILDER("ScalarFloordiv").SetBody(ReturnZeros);
REG_BPROP_BUILDER("scalar_eq").SetBody(ReturnZeros);
REG_BPROP_BUILDER("scalar_le").SetBody(ReturnZeros);
REG_BPROP_BUILDER("scalar_lt").SetBody(ReturnZeros);
REG_BPROP_BUILDER("scalar_ge").SetBody(ReturnZeros);
REG_BPROP_BUILDER("scalar_gt").SetBody(ReturnZeros);
REG_BPROP_BUILDER("bit_and").SetBody(ReturnZeros);
REG_BPROP_BUILDER("bit_or").SetBody(ReturnZeros);
REG_BPROP_BUILDER("ScalarBool").SetBody(ReturnZeros);
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
