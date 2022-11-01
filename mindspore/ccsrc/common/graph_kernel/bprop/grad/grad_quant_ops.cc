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
#include "utils/ms_context.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDER("BNTrainingReduce").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("MinMaxUpdatePerLayer").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_min = ib->GetInput(kIndex1);
  auto x_max = ib->GetInput(kIndex2);
  return {ib->ZerosLike(x), ib->ZerosLike(x_min), ib->ZerosLike(x_max)};
});

REG_BPROP_BUILDER("MinMaxUpdatePerChannel").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_min = ib->GetInput(kIndex1);
  auto x_max = ib->GetInput(kIndex2);
  return {ib->ZerosLike(x), ib->ZerosLike(x_min), ib->ZerosLike(x_max)};
});

REG_BPROP_BUILDER("WtsARQ").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto w_min = ib->GetInput(kIndex1);
  auto w_max = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(w_min), ib->ZerosLike(w_max)};
});

REG_BPROP_BUILDER("FakeQuantPerLayer").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_min = ib->GetInput(kIndex1);
  auto x_max = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("FakeQuantPerLayerGrad", {dout, x, x_min, x_max},
                     {{"num_bits", ib->GetAttr("num_bits")},
                      {"quant_delay", ib->GetAttr("quant_delay")},
                      {"symmetric", MakeValue(false)},
                      {"narrow_range", MakeValue(false)},
                      {"ema", MakeValue(false)},
                      {"ema_decay", MakeValue<double>(0.999)},
                      {"training", MakeValue(true)}});
  return {dx, ib->ZerosLike(x_min), ib->ZerosLike(x_max)};
});

REG_BPROP_BUILDER("FakeQuantWithMinMaxVars").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_min = ib->GetInput(kIndex1);
  auto x_max = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("FakeQuantWithMinMaxVarsGradient", {dout, x, x_min, x_max},
                     {{"num_bits", ib->GetAttr("num_bits")}, {"narrow_range", ib->GetAttr("narrow_range")}});
  return {dx, ib->ZerosLike(x_min), ib->ZerosLike(x_max)};
});

REG_BPROP_BUILDER("FakeQuantWithMinMaxVarsPerChannel").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_min = ib->GetInput(kIndex1);
  auto x_max = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("FakeQuantWithMinMaxVarsPerChannelGradient", {dout, x, x_min, x_max},
                     {{"num_bits", ib->GetAttr("num_bits")}, {"narrow_range", ib->GetAttr("narrow_range")}});
  return {dx, ib->ZerosLike(x_min), ib->ZerosLike(x_max)};
});

REG_BPROP_BUILDER("FakeQuantPerChannel").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_min = ib->GetInput(kIndex1);
  auto x_max = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("FakeQuantPerChannelGrad", {dout, x, x_min, x_max},
                     {{"num_bits", ib->GetAttr("num_bits")},
                      {"quant_delay", ib->GetAttr("quant_delay")},
                      {"symmetric", ib->GetAttr("symmetric")},
                      {"narrow_range", ib->GetAttr("symmetric")},
                      {"channel_axis", ib->GetAttr("channel_axis")}});
  return {dx, ib->ZerosLike(x_min), ib->ZerosLike(x_max)};
});

REG_BPROP_BUILDER("BatchNormFold").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto mean = ib->GetInput(kIndex1);
  auto variance = ib->GetInput(kIndex2);
  auto global_step = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex5);
  auto dx = ib->Emit("BatchNormFoldGrad",
                     {ib->TupleGetItem(dout, 0), ib->TupleGetItem(dout, 1), x, ib->TupleGetItem(out, 0),
                      ib->TupleGetItem(out, 1), global_step},
                     {{"epsilon", ib->GetAttr("epsilon")},
                      {"is_training", ib->GetAttr("is_training")},
                      {"freeze_bn", ib->GetAttr("freeze_bn")}});
  return {dx, ib->ZerosLike(mean), ib->ZerosLike(variance), ib->ZerosLike(global_step)};
});

REG_BPROP_BUILDER("CorrectionMul").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto batch_std = ib->GetInput(kIndex1);
  auto running_std = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto channel_axis = ib->GetAttr("channel_axis");
  auto tmp = ib->Emit("CorrectionMulGrad", {dout, x, batch_std, running_std}, {{"channel_axis", channel_axis}});
  auto dx = ib->TupleGetItem(tmp, kIndex0);
  auto d_batch_std = ib->TupleGetItem(tmp, kIndex1);
  if (ib->GetTargetFromContext() == kAscendDevice) {
    d_batch_std = ib->Emit("CorrectionMulGradReduce", {d_batch_std}, {{"channel_axis", channel_axis}});
  }
  return {dx, d_batch_std, ib->ZerosLike(running_std)};
});

REG_BPROP_BUILDER("BatchNormFold2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto gamma = ib->GetInput(kIndex2);
  auto batch_std = ib->GetInput(kIndex3);
  auto batch_mean = ib->GetInput(kIndex4);
  auto running_std = ib->GetInput(kIndex5);
  auto running_mean = ib->GetInput(kIndex6);
  auto global_step = ib->GetInput(kIndex7);
  auto dout = ib->GetInput(kIndex9);
  auto tmp =
    ib->Emit("BatchNormFold2Grad", {dout, x, gamma, batch_std, batch_mean, running_std, running_mean, global_step},
             {{"freeze_bn", ib->GetAttr("freeze_bn")}});
  auto d_batch_std = ib->TupleGetItem(tmp, 0);
  auto d_batch_mean = ib->TupleGetItem(tmp, 1);
  auto d_beta = ib->TupleGetItem(tmp, 2);
  auto d_gamma = ib->TupleGetItem(tmp, 3);
  auto d_x = ib->TupleGetItem(tmp, 4);
  return {d_x,
          d_beta,
          d_gamma,
          d_batch_std,
          d_batch_mean,
          ib->ZerosLike(running_std),
          ib->ZerosLike(running_mean),
          ib->ZerosLike(global_step)};
});

REG_BPROP_BUILDER("BatchNormFoldD").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_sum = ib->GetInput(kIndex1);
  auto x_square_sum = ib->GetInput(kIndex2);
  auto mean = ib->GetInput(kIndex3);
  auto variance = ib->GetInput(kIndex4);
  auto out = ib->GetInput(kIndex5);
  auto dout = ib->GetInput(kIndex6);
  auto dx = ib->Emit(
    "BatchNormFoldGradD",
    {ib->TupleGetItem(dout, 1), ib->TupleGetItem(dout, 2), x, ib->TupleGetItem(out, 1), ib->TupleGetItem(out, 2)},
    {{"epsilon", ib->GetAttr("epsilon")},
     {"is_training", ib->GetAttr("is_training")},
     {"freeze_bn", ib->GetAttr("freeze_bn")}});
  return {dx, ib->ZerosLike(x_sum), ib->ZerosLike(x_square_sum), ib->ZerosLike(mean), ib->ZerosLike(variance)};
});

REG_BPROP_BUILDER("BatchNormFold2D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto gamma = ib->GetInput(kIndex2);
  auto batch_std = ib->GetInput(kIndex3);
  auto batch_mean = ib->GetInput(kIndex4);
  auto running_std = ib->GetInput(kIndex5);
  auto dout = ib->GetInput(kIndex7);
  auto tmp = ib->Emit("BatchNormFold2GradReduce", {dout, x}, {{"freeze_bn", ib->GetAttr("freeze_bn")}});
  auto dout_reduce = ib->TupleGetItem(tmp, 0);
  auto dout_x_reduce = ib->TupleGetItem(tmp, 1);
  tmp = ib->Emit("BatchNormFold2GradD", {dout, dout_reduce, dout_x_reduce, gamma, batch_std, batch_mean, running_std},
                 {{"freeze_bn", ib->GetAttr("freeze_bn")}});
  auto d_batch_std = ib->TupleGetItem(tmp, 0);
  auto d_batch_mean = ib->TupleGetItem(tmp, 1);
  auto d_gamma = ib->TupleGetItem(tmp, 2);
  auto d_x = ib->TupleGetItem(tmp, 3);
  return {d_x, dout_reduce, d_gamma, d_batch_std, d_batch_mean, ib->ZerosLike(running_std)};
});

REG_BPROP_BUILDER("ActsULQ").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto dout0 = ib->TupleGetItem(dout, kIndex0);
  auto out1 = ib->TupleGetItem(out, kIndex1);
  auto out2 = ib->TupleGetItem(out, kIndex2);
  auto out3 = ib->TupleGetItem(out, kIndex3);
  auto dx = ib->Emit("ActsULQInputGrad", {dout0, out1, out2});
  auto dx1 = ib->Emit("ActULQClampMinGrad", {dout0, out1, out3});
  auto dx2 = ib->Emit("ActULQClampMaxGrad", {dout0, out2, out3});
  return {dx, dx1, dx2};
});

REG_BPROP_BUILDER("FakeLearnedScaleQuantPerLayer").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_alpha = ib->GetInput(kIndex1);
  auto x_quant_max = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto tmp = ib->Emit("FakeLearnedScaleQuantPerLayerGrad", {dout, x, x_alpha, x_quant_max},
                      {{"quant_delay", ib->GetAttr("quant_delay")}, {"neg_trunc", ib->GetAttr("neg_trunc")}});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dalpha = ib->TupleGetItem(tmp, 1);
  return {dx, dalpha, ib->ZerosLike(x_quant_max)};
});

REG_BPROP_BUILDER("FakeLearnedScaleQuantPerChannel").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto x_alpha = ib->GetInput(kIndex1);
  auto x_quant_max = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto tmp = ib->Emit("FakeLearnedScaleQuantPerChannelGrad", {dout, x, x_alpha, x_quant_max},
                      {{"quant_delay", ib->GetAttr("quant_delay")},
                       {"neg_trunc", ib->GetAttr("neg_trunc")},
                       {"channel_axis", ib->GetAttr("channel_axis")}});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dalpha = ib->TupleGetItem(tmp, 1);
  return {dx, dalpha, ib->ZerosLike(x_quant_max)};
});
}  // namespace mindspore::expander::bprop
