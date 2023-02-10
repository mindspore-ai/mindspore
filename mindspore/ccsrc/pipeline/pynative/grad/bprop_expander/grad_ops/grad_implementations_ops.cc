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

#include "pipeline/pynative/grad/bprop_expander/bprop_irbuilder.h"
#include "include/common/utils/utils.h"
#include "pipeline/pynative/grad/bprop_expander/grad_ops/common_utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradImplementationsOps)
REG_BPROP_BUILDER("Load").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto u_monad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {dout, ib->ZerosLike(u_monad)};
});

REG_BPROP_BUILDER("UpdateState").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto u_monad = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  return {ib->ZerosLike(u_monad), dout};
});

REG_BPROP_BUILDER("Depend").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto param = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {dout, ib->ZerosLike(param)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
