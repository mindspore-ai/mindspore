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
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "utils/ms_context.h"
#include "frontend/expander/grad/grad_utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradQuantOps)
REG_BPROP_BUILDER("BNTrainingReduce").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("MinMaxUpdatePerLayer").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("MinMaxUpdatePerChannel").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("WtsARQ").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &w_min = ib->GetInput(i1);
  const auto &w_max = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  return {dout, ib->OutZeros(w_min), ib->OutZeros(w_max)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
