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
REG_BPROP_BUILDER("Assign").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {dout, ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("InvertPermutation").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("IOU").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("SyncBatchNorm").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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
  return {dx, dscale, dbias, ib->ZerosLike(mean), ib->ZerosLike(variance)};
});

REG_BPROP_BUILDER("GpuConvertToDynamicShape").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("_DynamicLossScale").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto loss_scale = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto res = ib->Emit("Mul", {dout, loss_scale},
                      {{"split_overflow", MakeValue(true)}, {"layer_overflow", ib->GetAttr("layer")}});
  return {res, ib->ZerosLike(loss_scale)};
});
}  // namespace mindspore::expander::bprop
