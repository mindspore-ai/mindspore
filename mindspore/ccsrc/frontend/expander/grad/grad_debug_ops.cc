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
#include "frontend/expander/grad/grad_utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradDebugOps)
REG_BPROP_BUILDER("ScalarSummary").SetUnusedInputs({i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &tag = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  return {tag, ib->OutZeros(x)};
});

REG_BPROP_BUILDER("TensorSummary").SetUnusedInputs({i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &tag = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  return {tag, ib->OutZeros(x)};
});

REG_BPROP_BUILDER("ImageSummary").SetUnusedInputs({i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &tag = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  return {tag, ib->OutZeros(x)};
});

REG_BPROP_BUILDER("HistogramSummary").SetUnusedInputs({i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &tag = ib->GetInput(i0);
  const auto &x = ib->GetInput(i1);
  return {tag, ib->OutZeros(x)};
});

REG_BPROP_BUILDER("VmapStackAssign").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &all_inputs = ib->GetInputs();
  NodePtrList gradients;
  std::transform(all_inputs.begin(), all_inputs.end() - i2, std::back_inserter(gradients),
                 [&ib](const NodePtr &node) { return ib->OutZeros(node); });
  return gradients;
});

REG_BPROP_BUILDER("VmapUnstackAssign").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &all_inputs = ib->GetInputs();
  NodePtrList gradients;
  std::transform(all_inputs.begin(), all_inputs.end() - i2, std::back_inserter(gradients),
                 [&ib](const NodePtr &node) { return ib->OutZeros(node); });
  return gradients;
});

REG_BPROP_BUILDER("JoinedStr").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &all_inputs = ib->GetInputs();
  return NodePtrList(all_inputs.begin(), all_inputs.end() - i2);
});

REG_BPROP_BUILDER("raise").FreeUselessValues_IO({}, {}).SetBody(BODYFUNC(ib) {
  const auto &all_inputs = ib->GetInputs();
  return NodePtrList(all_inputs.begin(), all_inputs.end() - i2);
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
