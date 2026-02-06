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
REG_BPROP_BUILDERS_BEGIN(GradOtherOps)
REG_BPROP_BUILDER("Assign").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &y = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  return {dout, ib->OutZeros(y)};
});

REG_BPROP_BUILDER("InplaceCopy").FreeUselessValues_IO({i0, i1, i2}, {}).SetBody(BODYFUNC(ib) {
  const auto &dst = ib->GetInput(i0);
  const auto &src = ib->GetInput(i1);
  const auto &non_blocking = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i4);
  NodePtr ds = nullptr;
  if (src->need_compute_grad_out()) {
    auto res = BinopGradCommon(ib, dst, src, nullptr, dout);
    ds = res[1];
  } else {
    ds = ib->OutZeros(src);
  }
  return {ib->OutZeros(dst), ds, ib->OutZeros(non_blocking)};
});

REG_BPROP_BUILDER("InplaceZero").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &input = ib->GetInput(i0);
  auto res = ib->Zeros(ib->Shape(input), ib->Value(static_cast<int64_t>(ib->GetDtypeId(input))));
  return {res};
});

REG_BPROP_BUILDER("InvertPermutation").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("RandExt").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("Randn").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("RandInt").SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("RandLikeExt").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("RandnLike").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("RandIntLike").SetUnusedInputs({i0, i1, i2, i3, i4, i5, i6}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("Generator").SetBody(ReturnZeros);
REG_BPROP_BUILDER("InplaceRandom").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IOU").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("SyncBatchNorm").FreeUselessValues_IO({i2, i3, i4}, {i0, i1, i2}).SetBody(BODYFUNC(ib) {
  const auto &x = ib->GetInput(i0);
  const auto &scale = ib->GetInput(i1);
  const auto &mean = ib->GetInput(i3);
  const auto &variance = ib->GetInput(i4);
  auto out = ib->GetInput(i5);
  const auto &dout = ib->GetInput(i6);
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
  const auto &dout = ib->GetInput(i2);
  return {dout};
});

REG_BPROP_BUILDER("RotaryPositionEmbedding").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  const auto &cos = ib->GetInput(i1);
  const auto &sin = ib->GetInput(i2);
  const auto &mode = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i5);
  auto grad_out = ib->RotaryPositionEmbeddingGrad(dout, cos, sin, ib->EmitValue(kNone), mode);
  auto dx = ib->TupleGetItem(grad_out, 0);
  return {dx, ib->OutZeros(cos), ib->OutZeros(sin), ib->OutZeros(mode)};
});

REG_BPROP_BUILDER("_DynamicLossScale").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  const auto &loss_scale = ib->GetInput(i1);
  const auto &dout = ib->GetInput(i3);
  auto res = ib->Emit("Mul", {dout, loss_scale},
                      {{"split_overflow", MakeValue(true)}, {"layer_overflow", ib->GetAttr("layer")}});
  return {res, ib->OutZeros(loss_scale)};
});

REG_BPROP_BUILDER("MoveTo").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  const auto &dout = ib->GetInput(i4);
  return {dout};
});

REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
