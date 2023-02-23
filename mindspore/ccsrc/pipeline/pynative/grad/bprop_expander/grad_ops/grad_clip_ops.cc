/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "pipeline/pynative/grad/bprop_expander/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradClipOps)
REG_BPROP_BUILDER("ClipByNorm").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto clip_norm = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto cast_x = ib->Cast(x, kFloat32);
  auto cast_clip_norm = ib->Cast(clip_norm, kFloat32);
  auto square_out = ib->Emit("Square", {cast_x});
  auto reduce_sum_axis = ib->EmitValue(ib->GetAttr("axis"));
  auto reduce_sum_out = ib->Emit("ReduceSum", {square_out, reduce_sum_axis}, {{"keep_dims", MakeValue(true)}});
  auto sqrt_out = ib->Emit("Sqrt", {reduce_sum_out});
  auto max_out = ib->Maximum(sqrt_out, cast_clip_norm);
  auto mul_out = ib->Mul(cast_x, cast_clip_norm);
  auto div_bc_x = ib->Div(dout, max_out);
  auto div_bc_y = ib->Emit("Neg", {ib->Mul(div_bc_x, out)});
  auto tmp_div_out = BinopGradCommon(ib, mul_out, max_out, div_bc_x, div_bc_y);
  auto div_dout_x = tmp_div_out[0];
  auto div_dout_y = tmp_div_out[1];
  auto mul_bc_x = ib->Mul(cast_clip_norm, div_dout_x);
  auto mul_bc_y = ib->Mul(cast_x, div_dout_x);
  auto tmp_mul_dout = BinopGradCommon(ib, cast_x, cast_clip_norm, mul_bc_x, mul_bc_y);
  auto mul_dout_x = tmp_mul_dout[0];
  auto mul_dout_y = tmp_mul_dout[1];
  auto tmp_max_dout = ib->Emit("MaximumGrad", {sqrt_out, cast_clip_norm, div_dout_y},
                               {{"grad_x", MakeValue(true)}, {"grad_y", MakeValue(true)}});
  auto max_dout_x = ib->TupleGetItem(tmp_max_dout, 0);
  auto max_dout_y = ib->TupleGetItem(tmp_max_dout, 1);
  auto sqrt_dout_x = ib->Emit("SqrtGrad", {sqrt_out, max_dout_x});
  auto reduce_sum_dout_x = SumGrad(ib, square_out, reduce_sum_axis, sqrt_dout_x);
  auto temp_out = ib->Mul(reduce_sum_dout_x, cast_x);
  auto square_dout_x = ib->Mul(ib->Tensor(2.0, ib->GetDtype(temp_out)), temp_out);
  auto x_dout = ib->Cast(ib->Add(mul_dout_x, square_dout_x), ib->GetDtype(x));
  auto clip_norm_dout = ib->Cast(ib->Add(mul_dout_y, max_dout_y), ib->GetDtype(clip_norm));
  return {x_dout, clip_norm_dout};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
