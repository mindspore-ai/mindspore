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
static NodePtr GetMatrixDiagAssist(const BpropIRBuilder *ib, const ShapeVector &x_shape, TypePtr x_dtype) {
  auto eye = ib->Emit("Eye", {ib->EmitValue(MakeValue(x_shape.back())), ib->EmitValue(MakeValue(x_shape.back())),
                              ib->EmitValue(x_dtype)});
  auto base_eye = ib->Emit("Flatten", {eye});
  ShapeVector shape2(x_shape.begin(), x_shape.end() - 1);
  auto tile = ib->Emit("Tile", {base_eye, ib->EmitValue(MakeValue(shape2))});
  auto shape3 = x_shape;
  shape3.push_back(x_shape.back());
  return ib->Reshape(tile, shape3);
}

static NodePtr GetMatrixDiagPartAssist(const BpropIRBuilder *ib, const ShapeVector &x_shape, TypePtr x_dtype) {
  auto eye = ib->Emit("Eye", {ib->EmitValue(MakeValue(x_shape[x_shape.size() - kDim2])),
                              ib->EmitValue(MakeValue(x_shape.back())), ib->EmitValue(x_dtype)});
  auto base_eye = ib->Emit("Flatten", {eye});
  ShapeVector shape2(x_shape.begin(), x_shape.end() - kDim2);
  auto tile = ib->Emit("Tile", {base_eye, ib->EmitValue(MakeValue(shape2))});
  return ib->Reshape(tile, x_shape);
}

REG_BPROP_BUILDER("MatrixDiag").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape = ib->GetShape(dout);
  auto dtype = ib->GetDtype(dout);
  auto assist = GetMatrixDiagPartAssist(ib, shape, dtype);
  auto dx = ib->Emit("MatrixDiagPart", {dout, assist});
  return {dx, ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("MatrixDiagPart").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape = ib->GetShape(x);
  if (shape[shape.size() - kDim2] == shape.back()) {
    auto shape = ib->GetShape(dout);
    auto dtype = ib->GetDtype(dout);
    auto assist = GetMatrixDiagAssist(ib, shape, dtype);
    return {ib->Emit("MatrixDiag", {dout, assist}), ib->ZerosLike(y)};
  }
  auto assist1 = GetMatrixDiagPartAssist(ib, ib->GetShape(x), ib->GetDtype(x));
  return {ib->Emit("MatrixSetDiag", {ib->ZerosLike(x), dout, assist1}), ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("MatrixSetDiag").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto z = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto input_shape = ib->GetShape(x);
  auto diag_shape = ShapeVector(input_shape.begin(), input_shape.end() - kDim2);
  diag_shape.push_back(std::min(input_shape[input_shape.size() - kDim2], input_shape[input_shape.size() - 1]));
  auto grad_shape = ib->GetShape(dout);
  auto grad_dtype = ib->GetDtype(dout);
  auto assist = GetMatrixDiagPartAssist(ib, grad_shape, grad_dtype);
  auto dx =
    ib->Emit("MatrixSetDiag",
             {dout, ib->Emit("Zeros", {ib->EmitValue(MakeValue(diag_shape)), ib->EmitValue(grad_dtype)}), assist});
  auto dy = ib->Emit("MatrixDiagPart", {dout, assist});
  auto dz = ib->ZerosLike(z);
  return {dx, dy, dz};
});

REG_BPROP_BUILDER("DSDMatmul").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto w1_gm = ib->GetInput(kIndex0);
  auto w2_gm = ib->GetInput(kIndex1);
  auto v_gm = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto tmp = ib->Emit("DSDGrad", {w1_gm, w2_gm, v_gm, out, dout});
  auto d_w1_gm = ib->TupleGetItem(tmp, kIndex0);
  auto d_w2_gm = ib->TupleGetItem(tmp, kIndex1);
  auto d_v_gm = ib->TupleGetItem(tmp, kIndex2);
  return {d_w1_gm, d_w2_gm, d_v_gm};
});

REG_BPROP_BUILDER("MatmulDDS").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto q = ib->GetInput(kIndex0);
  auto k = ib->GetInput(kIndex1);
  auto local_mask = ib->GetInput(kIndex2);
  auto global_mask = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto d_out = ib->GetInput(kIndex5);
  auto lc = ib->TupleGetItem(out, kIndex0);
  auto gc = ib->TupleGetItem(out, kIndex1);
  auto d_lc = ib->TupleGetItem(out, kIndex0);
  auto d_gc = ib->TupleGetItem(out, kIndex1);
  auto tmp = ib->Emit("MatmulDDSGrad", {q, k, lc, gc, d_lc, d_gc});
  auto dq = ib->TupleGetItem(tmp, kIndex0);
  auto dk = ib->TupleGetItem(tmp, kIndex1);
  ShapeVector shape = {1, 0, 3, 2};
  dk = ib->Emit("Transpose", {dk, ib->EmitValue(MakeValue(shape))});
  return {dq, dk, ib->ZerosLike(local_mask), ib->ZerosLike(global_mask)};
});

REG_BPROP_BUILDER("PsROIPooling").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto pooled_height = GetValue<int64_t>(ib->GetAttr("pooled_height"));
  auto pooled_width = GetValue<int64_t>(ib->GetAttr("pooled_width"));
  auto spatial_scale = GetValue<float>(ib->GetAttr("spatial_scale"));
  auto out_dim = GetValue<int64_t>(ib->GetAttr("out_dim"));
  auto num_rois = GetValue<int64_t>(ib->GetAttr("num_rois"));
  auto inputs = ib->GetInput(kIndex0);
  auto rois = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto mapping_channel = ib->TupleGetItem(out, kIndex1);
  auto inputs_shape = ib->GetShape(inputs);
  auto batch_size = inputs_shape[kIndex0];
  auto channels = inputs_shape[kIndex1];
  auto height = inputs_shape[kIndex2];
  auto width = inputs_shape[kIndex3];
  auto dx = ib->Emit("PsROIPoolingGrad", {ib->TupleGetItem(dout, 0), rois, mapping_channel},
                     {{"batch_size", MakeValue(batch_size)},
                      {"channels", MakeValue(channels)},
                      {"height", MakeValue(height)},
                      {"width", MakeValue(width)},
                      {"num_rois", MakeValue(num_rois)},
                      {"pooled_height", MakeValue(pooled_height)},
                      {"pooled_width", MakeValue(pooled_width)},
                      {"spatial_scale", MakeValue(spatial_scale)},
                      {"out_dim", MakeValue(out_dim)}});
  return {dx, ib->ZerosLike(rois)};
});

REG_BPROP_BUILDER("ResizeBilinearV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit(
    "ResizeBilinearGrad", {dout, x},
    {{"align_corners", ib->GetAttr("align_corners")}, {"half_pixel_centers", ib->GetAttr("half_pixel_centers")}});
  return {dx, ib->ZerosLike(size)};
});
}  // namespace mindspore::expander::bprop
