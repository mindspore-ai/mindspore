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
#include "include/common/utils/utils.h"
#include "frontend/expander/bprop/grad_ops/common_utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradOtherOps)
REG_BPROP_BUILDER("Assign").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {dout, ib->OutZeros(y)};
});

REG_BPROP_BUILDER("InvertPermutation").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IOU").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("SyncBatchNorm").SetUnusedInputs({i2, i3, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto scale = ib->GetInput(kIndex1);
  auto mean = ib->GetInput(kIndex3);
  auto variance = ib->GetInput(kIndex4);
  auto out = ib->GetInput(kIndex5);
  auto dout = ib->GetInput(kIndex6);
  auto saved_mean = ib->TupleGetItem(out, 3);
  auto saved_variance = ib->TupleGetItem(out, 4);
  out = ib->Emit(
    "SyncBatchNormGrad", {ib->TupleGetItem(dout, 0), x, scale, saved_mean, saved_variance},
    {{"epsilon", ib->GetAttr("epsilon")}, {"group", ib->GetAttr("group")}, {"device_num", ib->GetAttr("device_num")}});
  auto dx = ib->TupleGetItem(out, 0);
  auto dscale = ib->TupleGetItem(out, 1);
  auto dbias = ib->TupleGetItem(out, 2);
  return {dx, dscale, dbias, ib->OutZeros(mean), ib->OutZeros(variance)};
});

REG_BPROP_BUILDER("GpuConvertToDynamicShape").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("_DynamicLossScale").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto loss_scale = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto res = ib->Emit("Mul", {dout, loss_scale},
                      {{"split_overflow", MakeValue(true)}, {"layer_overflow", ib->GetAttr("layer")}});
  return {res, ib->OutZeros(loss_scale)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
