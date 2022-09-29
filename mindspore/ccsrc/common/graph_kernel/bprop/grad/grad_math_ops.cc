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
#include "common/graph_kernel/bprop/bprop_irbuilder.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDER(kMatMulOpName).SetBody([](const BpropIRBuilder *builder) -> NodePtrList {
  auto ta = GetValue<bool>(builder->GetAttr("transpose_a"));
  auto tb = GetValue<bool>(builder->GetAttr("transpose_b"));
  auto x = builder->GetInput(kIndex0);
  auto w = builder->GetInput(kIndex1);
  auto dout = builder->GetInput(kIndex3);
  NodePtr dx;
  NodePtr dw;
  if (ta) {
    dx = builder->MatMul(w, dout, (ta && tb), (ta || (!tb)));
  } else {
    dx = builder->MatMul(dout, w, (ta && tb), (ta || (!tb)));
  }
  if (tb) {
    dw = builder->MatMul(dout, x, ((!ta) || tb), (ta && tb));
  } else {
    dw = builder->MatMul(x, dout, ((!ta) || tb), (ta && tb));
  }
  return {dx, dw};
});
}  // namespace mindspore::expander::bprop
