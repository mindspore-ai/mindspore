/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
REG_BPROP_BUILDERS_BEGIN(GradImplementationsOps)
REG_BPROP_BUILDER("Load").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &u_monad = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return {dout, ib->OutZeros(u_monad)};
});

REG_BPROP_BUILDER("UpdateState").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &u_monad = ib->GetInput(i0);
  const auto &dout = ib->GetInput(i3);
  return {ib->OutZeros(u_monad), dout};
});

REG_BPROP_BUILDER("Depend").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &param = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return {dout, ib->OutZeros(param)};
});

REG_BPROP_BUILDER("TensorMove").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i2);
  return {dout};
});

REG_BPROP_BUILDER("CopyToDevice").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto sync = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
  return {dout, ib->OutZeros(sync)};
});

REG_BPROP_BUILDER("CopyToHost").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto sync = ib->GetInput(i1);
  auto dout = ib->GetInput(i3);
  return {dout, ib->OutZeros(sync)};
});

REG_BPROP_BUILDER("Free").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(i0);
  auto sync = ib->GetInput(i1);
  auto dx = ib->Zeros(ib->Shape(x), ib->Value(static_cast<int64_t>(ib->GetDtypeId(x))));
  return {dx, ib->OutZeros(sync)};
});

REG_BPROP_BUILDER("SetData").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(i0);
  auto value = ib->GetInput(i1);
  auto dx = ib->Zeros(ib->Shape(input), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input))));
  return {dx, ib->OutZeros(value)};
});

REG_BPROP_BUILDER("GetData").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
