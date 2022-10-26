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
REG_BPROP_BUILDER(kConv2DOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto x_shape = ib->GetShape(x);
  auto w_shape = ib->GetShape(w);
  auto dx = ib->Emit(kConv2DBackpropInputOpName, {dout, w, ib->Value<ShapeVector>(x_shape)},
                     {{"mode", ib->GetAttr("mode")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"stride", ib->GetAttr("stride")},
                      {"group", ib->GetAttr("group")},
                      {"format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"pad_list", ib->GetAttr("pad_list")}});
  auto dw = ib->Emit("Conv2DBackpropFilter", {dout, x, ib->Value<ShapeVector>(w_shape)},
                     {{"mode", ib->GetAttr("mode")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"stride", ib->GetAttr("stride")},
                      {"group", ib->GetAttr("group")},
                      {"format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"pad_list", ib->GetAttr("pad_list")}});
  return {dx, dw};
});

REG_BPROP_BUILDER(kMaxPoolOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(kMaxPoolGradOpName, {x, out, dout},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"format", ib->GetAttr("format")}});
  return {dx};
});

REG_BPROP_BUILDER(kBiasAddOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  return {dout, ib->Emit(kBiasAddGradOpName, {dout}, {{"format", ib->GetAttr("data_format")}})};
});

REG_BPROP_BUILDER(kReLUOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(kReluGradOpName, {dout, out});
  return {dx};
});

REG_BPROP_BUILDER(kTopKOpName).SetBody([](const BpropIRBuilder *builder) -> NodePtrList {
  auto input_x = builder->GetInput(kIndex0);
  auto out = builder->GetInput(kIndex2);
  auto dout = builder->GetInput(kIndex3);

  auto indices = builder->TupleGetItem(out, kIndex1);
  auto dout0 = builder->TupleGetItem(dout, kIndex0);

  auto in_shape = builder->GetShape(input_x);
  auto in_lastdim = in_shape.back();

  auto ind_shape = builder->GetShape(indices);
  auto ind_lastdim = ind_shape.back();

  auto ind_2d = builder->Reshape(indices, {-1, ind_lastdim});
  auto outerdim = builder->GetShape(ind_2d)[0];  // k

  // [0, outerdim, 2*outerdim, ..., (k-1)*outerdim]
  auto indices_dtype = builder->GetDtype(indices);
  std::vector<int64_t> range_flatten_index_vec(outerdim);
  for (int64_t i = 0; i < outerdim; i++) {
    range_flatten_index_vec[i] = i * in_lastdim;
  }
  auto range_flatten_index = builder->Tensor(range_flatten_index_vec);
  if (indices_dtype->type_id() != kNumberTypeInt64) {
    range_flatten_index = builder->Cast(range_flatten_index, indices_dtype);
  }

  auto ind = builder->Reshape(ind_2d + range_flatten_index, {-1, 1});
  auto in_shape_1d = ShapeVector(1, std::accumulate(in_shape.begin(), in_shape.end(), 1, std::multiplies<int64_t>()));
  auto out_grad = builder->Emit("ScatterNd", {ind, builder->Reshape(dout0, {-1}), builder->Tensor(in_shape_1d)});
  out_grad = builder->Reshape(out_grad, in_shape);

  auto grad_k = builder->ZerosLike(builder->GetInput(kIndex1));
  return {out_grad, grad_k};
});

REG_BPROP_BUILDER(kPReLUOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto res = ib->Emit("PReLUGrad", {dout, x, w});
  auto dx = ib->TupleGetItem(res, kIndex0);
  auto dw = ib->TupleGetItem(res, kIndex1);
  return {dx, dw};
});

REG_BPROP_BUILDER("SigmoidCrossEntropyWithLogits").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("SigmoidCrossEntropyWithLogitsGrad", {x, y, dout});
  return {dx, ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("Pad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto paddings = ib->GetAttr<std::vector<std::vector<int64_t>>>("paddings");
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  std::vector<int64_t> begin;
  for (const auto &item : paddings) {
    begin.push_back(item.at(0));
  }
  auto shp = ib->GetShape(x);
  auto dx = ib->Emit("Slice", {dout, ib->EmitValue(MakeValue(begin)), ib->EmitValue(MakeValue(shp))});
  return {dx};
});

REG_BPROP_BUILDER("ROIAlign").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto inputs = ib->GetInput(kIndex0);
  auto rois = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto inputs_shape = ib->GetShape(inputs);
  auto dx = ib->Emit("ROIAlignGrad", {dout, rois, ib->EmitValue(MakeValue(inputs_shape))},
                     {{"pooled_height", ib->GetAttr("pooled_height")},
                      {"pooled_width", ib->GetAttr("pooled_width")},
                      {"spatial_scale", ib->GetAttr("spatial_scale")},
                      {"sample_num", ib->GetAttr("sample_num")}});
  return {dx, ib->ZerosLike(rois)};
});

REG_BPROP_BUILDER("LRN").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("LRNGrad", {dout, x, out},
                     {{"depth_radius", ib->GetAttr("depth_radius")},
                      {"bias", ib->GetAttr("bias")},
                      {"alpha", ib->GetAttr("alpha")},
                      {"beta", ib->GetAttr("beta")}});
  return {dx};
});

REG_BPROP_BUILDER(kDropoutOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto mask = ib->TupleGetItem(out, 1);
  auto dy = ib->TupleGetItem(dout, 0);
  auto dx = ib->Emit(kDropoutGradOpName, {dy, mask}, {{"keep_prob", ib->GetAttr("keep_prob")}});
  return {dx};
});

REG_BPROP_BUILDER("BinaryCrossEntropy").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto weight = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("BinaryCrossEntropyGrad", {x, y, dout, weight}, {{"reduction", ib->GetAttr("reduction")}});
  return {dx, ib->ZerosLike(y), ib->ZerosLike(weight)};
});

REG_BPROP_BUILDER(kDropoutGradOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto mask = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dy = dout;
  auto dx = ib->Emit(kDropoutGradOpName, {dy, mask}, {{"keep_prob", ib->GetAttr("keep_prob")}});
  return {dx, ib->ZerosLike(mask)};
});

REG_BPROP_BUILDER("DeformableOffsets").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto offsets = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto out_grad = ib->Emit("DeformableOffsetsGrad", {dout, x, offsets},
                           {{"strides", ib->GetAttr("strides")},
                            {"pads", ib->GetAttr("pads")},
                            {"ksize", ib->GetAttr("ksize")},
                            {"dilations", ib->GetAttr("dilations")},
                            {"format", ib->GetAttr("format")},
                            {"deformable_groups", ib->GetAttr("deformable_groups")},
                            {"modulated", ib->GetAttr("modulated")}});
  return {ib->TupleGetItem(out_grad, 0), ib->TupleGetItem(out_grad, 1)};
});

REG_BPROP_BUILDER("LSTM").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_size = ib->GetAttr("input_size");
  auto hidden_size = ib->GetAttr("hidden_size");
  auto num_layers = ib->GetAttr("num_layers");
  auto has_bias = ib->GetAttr("has_bias");
  auto bidirectional = ib->GetAttr("bidirectional");
  auto dropout = ib->GetAttr("dropout");
  auto x = ib->GetInput(kIndex0);
  auto hx = ib->GetInput(kIndex1);
  auto cx = ib->GetInput(kIndex2);
  auto w = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex5);
  auto target = ib->GetTargetFromContext();
  if (target == "CPU") {
    auto y = ib->TupleGetItem(out, kIndex0);
    auto hy = ib->TupleGetItem(out, kIndex1);
    auto cy = ib->TupleGetItem(out, kIndex2);
    auto reserve = ib->TupleGetItem(out, kIndex3);
    auto dy = ib->TupleGetItem(dout, kIndex0);
    auto dhy = ib->TupleGetItem(dout, kIndex1);
    auto dcy = ib->TupleGetItem(dout, kIndex2);
    auto res = ib->Emit("LSTMGrad", {x, hx, cx, w, y, hy, cy, dy, dhy, dcy, reserve},
                        {{"input_size", input_size},
                         {"hidden_size", hidden_size},
                         {"num_layers", num_layers},
                         {"has_bias", has_bias},
                         {"bidirectional", bidirectional},
                         {"dropout", dropout}});
    auto dx = ib->TupleGetItem(res, kIndex0);
    auto dhx = ib->TupleGetItem(res, kIndex1);
    auto dcx = ib->TupleGetItem(res, kIndex2);
    auto dw = ib->TupleGetItem(res, kIndex3);
    return {dx, dhx, dcx, dw};
  }
  auto y = ib->TupleGetItem(out, kIndex0);
  auto reserve = ib->TupleGetItem(out, kIndex3);
  auto state = ib->TupleGetItem(out, kIndex4);
  auto dy = ib->TupleGetItem(dout, kIndex0);
  auto dhy = ib->TupleGetItem(dout, kIndex1);
  auto dcy = ib->TupleGetItem(dout, kIndex2);
  auto res1 = ib->Emit("LSTMGradData", {y, dy, dhy, dcy, w, hx, cx, reserve, state},
                       {{"input_size", input_size},
                        {"hidden_size", hidden_size},
                        {"num_layers", num_layers},
                        {"has_bias", has_bias},
                        {"bidirectional", bidirectional},
                        {"dropout", dropout}});
  auto dx = ib->TupleGetItem(res1, kIndex0);
  auto dhx = ib->TupleGetItem(res1, kIndex1);
  auto dcx = ib->TupleGetItem(res1, kIndex2);
  auto dw = ib->Emit("LSTMGradWeight", {ib->Emit("Depend", {x, dx}), hx, y, reserve, state},
                     {{"input_size", input_size},
                      {"hidden_size", hidden_size},
                      {"num_layers", num_layers},
                      {"has_bias", has_bias},
                      {"bidirectional", bidirectional},
                      {"dropout", dropout}});
  return {dx, dhx, dcx, dw};
});

REG_BPROP_BUILDER("CudnnGRU").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_size = ib->GetAttr("input_size");
  auto hidden_size = ib->GetAttr("hidden_size");
  auto num_layers = ib->GetAttr("num_layers");
  auto has_bias = ib->GetAttr("has_bias");
  auto bidirectional = ib->GetAttr("bidirectional");
  auto dropout = ib->GetAttr("dropout");
  auto x = ib->GetInput(kIndex0);
  auto hx = ib->GetInput(kIndex1);
  auto w = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto y = ib->TupleGetItem(out, kIndex0);
  auto reserve = ib->TupleGetItem(out, kIndex2);
  auto state = ib->TupleGetItem(out, kIndex3);
  auto dy = ib->TupleGetItem(dout, kIndex0);
  auto dhy = ib->TupleGetItem(dout, kIndex1);
  auto res1 = ib->Emit("GruGradData", {y, dy, dhy, w, hx, reserve, state},
                       {{"input_size", input_size},
                        {"hidden_size", hidden_size},
                        {"num_layers", num_layers},
                        {"has_bias", has_bias},
                        {"bidirectional", bidirectional},
                        {"dropout", dropout}});
  auto dx = ib->TupleGetItem(res1, kIndex0);
  auto dhx = ib->TupleGetItem(res1, kIndex1);
  auto dw = ib->Emit("GruGradWeight", {ib->Emit("Depend", {x, dx}), hx, y, reserve, state},
                     {{"input_size", input_size},
                      {"hidden_size", hidden_size},
                      {"num_layers", num_layers},
                      {"has_bias", has_bias},
                      {"bidirectional", bidirectional},
                      {"dropout", dropout}});
  return {dx, dhx, dw};
});

REG_BPROP_BUILDER("MirrorPad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto paddings = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("MirrorPadGrad", {dout, paddings}, {{kAttrMode, ib->GetAttr(kAttrMode)}});
  return {dx, ib->ZerosLike(paddings)};
});

REG_BPROP_BUILDER("LayerNorm").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto gamma = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto result = ib->Emit(
    "LayerNormGrad", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 2), ib->TupleGetItem(out, 1), gamma},
    {{"begin_norm_axis", ib->GetAttr("begin_norm_axis")}, {"begin_params_axis", ib->GetAttr("begin_params_axis")}});
  auto d_x = ib->TupleGetItem(result, 0);
  auto d_gamma = ib->TupleGetItem(result, 1);
  auto d_beta = ib->TupleGetItem(result, 2);
  return {d_x, d_gamma, d_beta};
});

REG_BPROP_BUILDER("LayerNormGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dy = ib->GetInput(kIndex1);
  auto variance = ib->GetInput(kIndex2);
  auto mean = ib->GetInput(kIndex3);
  auto gamma = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto result = ib->Emit(
    "LayerNormGradGrad",
    {x, dy, variance, mean, gamma, ib->TupleGetItem(dout, 0), ib->TupleGetItem(dout, 1), ib->TupleGetItem(dout, 2)},
    {{"begin_norm_axis", ib->GetAttr("begin_norm_axis")}, {"begin_params_axis", ib->GetAttr("begin_params_axis")}});
  auto d_x = ib->TupleGetItem(result, 0);
  auto d_dy = ib->TupleGetItem(result, 1);
  auto d_gamma = ib->TupleGetItem(result, 2);
  return {d_x, d_dy, ib->ZerosLike(variance), ib->ZerosLike(mean), d_gamma};
});

REG_BPROP_BUILDER("L2Normalize").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("L2NormalizeGrad", {x, out, dout}, {{"axis", ib->GetAttr("axis")}, {"epsilon", ib->GetAttr("epsilon")}});
  return {dx};
});

REG_BPROP_BUILDER("SoftmaxCrossEntropyWithLogits").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto labels = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto grad = ib->TupleGetItem(out, 1);
  grad = ib->Mul(grad, (ib->Emit("ExpandDims", {ib->TupleGetItem(dout, 0), ib->Tensor(-1)})));
  return {grad, ib->ZerosLike(labels)};
});

REG_BPROP_BUILDER("NLLLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto target = ib->GetInput(kIndex1);
  auto weight = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto total_weight = ib->TupleGetItem(out, 1);
  auto dout_x = ib->TupleGetItem(dout, 0);
  auto dx =
    ib->Emit("NLLLossGrad", {x, dout_x, target, weight, total_weight}, {{"reduction", ib->GetAttr("reduction")}});
  return {dx, ib->ZerosLike(target), ib->ZerosLike(weight)};
});

REG_BPROP_BUILDER("ResizeBilinear").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(
    "ResizeBilinearGrad", {dout, x},
    {{"align_corners", ib->GetAttr("align_corners")}, {"half_pixel_centers", ib->GetAttr("half_pixel_centers")}});
  return {dx};
});

REG_BPROP_BUILDER("OneHot").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex0);
  auto depth = ib->GetInput(kIndex1);
  auto on_value = ib->GetInput(kIndex2);
  auto off_value = ib->GetInput(kIndex3);
  return {ib->ZerosLike(indices), ib->ZerosLike(ib->Tensor(0, ib->GetDtype(depth))), ib->ZerosLike(on_value),
          ib->ZerosLike(off_value)};
});

REG_BPROP_BUILDER("SmoothL1Loss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto prediction = ib->GetInput(kIndex0);
  auto target = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("SmoothL1LossGrad", {prediction, target, dout},
                     {{"beta", ib->GetAttr("beta")}, {"reduction", ib->GetAttr("reduction")}});
  auto dy = ib->Emit("SmoothL1LossGrad", {target, prediction, dout},
                     {{"beta", ib->GetAttr("beta")}, {"reduction", ib->GetAttr("reduction")}});
  return {dx, dy};
});

REG_BPROP_BUILDER("L2Loss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(x, dout);
  return {dx};
});

REG_BPROP_BUILDER("RNNTLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto labels = ib->GetInput(kIndex1);
  auto act_lens = ib->GetInput(kIndex2);
  auto label_lens = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto grad = ib->TupleGetItem(out, 1);
  return {grad, ib->ZerosLike(labels), ib->ZerosLike(act_lens), ib->ZerosLike(label_lens)};
});

REG_BPROP_BUILDER("Conv3D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("Conv3DBackpropInput", {w, dout, ib->Value<ShapeVector>(ib->GetShape(x))},
                     {{"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"strides", ib->GetAttr("strides")},
                      {"dilations", ib->GetAttr("dilations")},
                      {"stride", ib->GetAttr("strides")},
                      {"dilation", ib->GetAttr("dilations")},
                      {"group", ib->GetAttr("groups")},
                      {"groups", ib->GetAttr("groups")},
                      {"format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"mode", ib->GetAttr("mode")}});
  auto dw = ib->Emit("Conv3DBackpropFilter", {x, dout, ib->Value<ShapeVector>(ib->GetShape(w))},
                     {{"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"strides", ib->GetAttr("strides")},
                      {"dilations", ib->GetAttr("dilations")},
                      {"stride", ib->GetAttr("strides")},
                      {"dilation", ib->GetAttr("dilations")},
                      {"group", ib->GetAttr("groups")},
                      {"groups", ib->GetAttr("groups")},
                      {"format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"mode", ib->GetAttr("mode")}});
  return {dx, dw};
});

REG_BPROP_BUILDER("Conv3DTranspose").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto strides = GetValue<std::vector<int64_t>>(ib->GetAttr("strides"));
  auto dilations = GetValue<std::vector<int64_t>>(ib->GetAttr("dilations"));
  std::vector<int64_t> stride = {strides.at(kIndex2), strides.at(kIndex3), strides.at(kIndex4)};
  std::vector<int64_t> dilation = {dilations.at(kIndex2), dilations.at(kIndex3), dilations.at(kIndex4)};
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("Conv3D", {dout, w},
                     {{"out_channel", ib->GetAttr("in_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"mode", ib->GetAttr("mode")},
                      {"pad_mode", MakeValue("pad")},
                      {"pad", ib->GetAttr("pad_list")},
                      {"strides", ib->GetAttr("strides")},
                      {"dilations", ib->GetAttr("dilations")},
                      {"stride", MakeValue(stride)},
                      {"dilation", MakeValue(dilation)},
                      {"group", ib->GetAttr("groups")},
                      {"groups", ib->GetAttr("groups")},
                      {"data_format", ib->GetAttr("format")}});
  auto dw = ib->Emit("Conv3DBackpropFilter", {dout, x, ib->Value<ShapeVector>(ib->GetShape(w))},
                     {{"out_channel", ib->GetAttr("in_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"mode", ib->GetAttr("mode")},
                      {"pad_mode", MakeValue("pad")},
                      {"pad", ib->GetAttr("pad_list")},
                      {"strides", ib->GetAttr("strides")},
                      {"dilations", ib->GetAttr("dilations")},
                      {"stride", ib->GetAttr("strides")},
                      {"dilation", ib->GetAttr("dilations")},
                      {"group", ib->GetAttr("groups")},
                      {"groups", ib->GetAttr("groups")},
                      {"data_format", ib->GetAttr("format")}});
  return {dx, dw};
});

REG_BPROP_BUILDER("MaxPoolWithArgmax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("MaxPoolGradWithArgmax", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1)},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"pad_mode", ib->GetAttr("pad_mode")}});
  return {dx};
});

REG_BPROP_BUILDER("MaxPoolGradGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto dx1 = ib->ZerosLike(x1);
  auto dx2 = ib->ZerosLike(x2);
  auto dgrad = ib->Emit("MaxPoolGrad", {x1, x2, dout},
                        {{"kernel_size", ib->GetAttr("kernel_size")},
                         {"strides", ib->GetAttr("strides")},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"format", MakeValue("NCHW")}});
  return {dx1, dx2, dgrad};
});
}  // namespace mindspore::expander::bprop
