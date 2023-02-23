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
#include <unordered_set>

#include "pipeline/pynative/grad/bprop_expander/bprop_irbuilder.h"
#include "include/common/utils/utils.h"
#include "pipeline/pynative/grad/bprop_expander/grad_ops/common_utils.h"

namespace mindspore::expander::bprop {
static NodePtr GetMatrixDiagAssist(const BpropIRBuilder *ib, const ShapeVector &x_shape, TypePtr x_dtype) {
  auto eye = ib->Emit("Eye", {ib->EmitValue(MakeValue(x_shape.back())), ib->EmitValue(MakeValue(x_shape.back())),
                              ib->EmitValue(x_dtype)});
  auto base_eye = ib->Reshape(eye, {-1});
  ShapeVector shape2(x_shape.begin(), x_shape.end() - 1);
  auto tile = ib->Tile(base_eye, shape2);
  auto shape3 = x_shape;
  shape3.push_back(x_shape.back());
  return ib->Reshape(tile, shape3);
}

static NodePtr GetMatrixDiagPartAssist(const BpropIRBuilder *ib, const ShapeVector &x_shape, TypePtr x_dtype) {
  auto eye = ib->Emit("Eye", {ib->EmitValue(MakeValue(x_shape[x_shape.size() - kDim2])),
                              ib->EmitValue(MakeValue(x_shape.back())), ib->EmitValue(x_dtype)});
  auto base_eye = ib->Reshape(eye, {-1});
  ShapeVector shape2(x_shape.begin(), x_shape.end() - kDim2);
  auto tile = ib->Tile(base_eye, shape2);
  return ib->Reshape(tile, x_shape);
}

REG_BPROP_BUILDERS_BEGIN(GradInnerOps)
REG_BPROP_BUILDER("MatrixDiag").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape = ib->GetShape(dout);
  auto dtype = ib->GetDtype(dout);
  auto assist = GetMatrixDiagPartAssist(ib, shape, dtype);
  auto dx = ib->Emit("MatrixDiagPart", {dout, assist});
  return {dx, ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("MatrixDiagPart").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
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

REG_BPROP_BUILDER("MatrixSetDiag").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
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

REG_BPROP_BUILDER("DSDMatmul").SetBody(BODYFUNC(ib) {
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

REG_BPROP_BUILDER("MatmulDDS").SetUnusedInputs({i2, i3, i5}).SetBody(BODYFUNC(ib) {
  auto q = ib->GetInput(kIndex0);
  auto k = ib->GetInput(kIndex1);
  auto local_mask = ib->GetInput(kIndex2);
  auto global_mask = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto lc = ib->TupleGetItem(out, kIndex0);
  auto gc = ib->TupleGetItem(out, kIndex1);
  auto d_lc = ib->TupleGetItem(out, kIndex0);
  auto d_gc = ib->TupleGetItem(out, kIndex1);
  auto tmp = ib->Emit("MatmulDDSGrad", {q, k, lc, gc, d_lc, d_gc});
  auto dq = ib->TupleGetItem(tmp, kIndex0);
  auto dk = ib->TupleGetItem(tmp, kIndex1);
  ShapeVector shape = {1, 0, 3, 2};
  dk = ib->Transpose(dk, shape);
  return {dq, dk, ib->ZerosLike(local_mask), ib->ZerosLike(global_mask)};
});

REG_BPROP_BUILDER("PsROIPooling").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
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

REG_BPROP_BUILDER("ResizeBilinearV2").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit(
    "ResizeBilinearGrad", {dout, x},
    {{"align_corners", ib->GetAttr("align_corners")}, {"half_pixel_centers", ib->GetAttr("half_pixel_centers")}});
  return {dx, ib->ZerosLike(size)};
});

REG_BPROP_BUILDER("ConvertToDynamic").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("_VirtualPipelineEnd").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("_VirtualPipelineEnd", {dout});
  return {dx};
});

REG_BPROP_BUILDER("FillV2").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto shape = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  auto dout_typeptr = ib->GetDtype(dout);
  auto dout_type = dout_typeptr->type_id();
  std::unordered_set<TypeId> type_list{TypeId::kNumberTypeInt8,   TypeId::kNumberTypeInt16,  TypeId::kNumberTypeInt32,
                                       TypeId::kNumberTypeInt64,  TypeId::kNumberTypeUInt8,  TypeId::kNumberTypeUInt16,
                                       TypeId::kNumberTypeUInt32, TypeId::kNumberTypeUInt64, TypeId::kNumberTypeFloat16,
                                       TypeId::kNumberTypeFloat64};

  if (type_list.count(dout_type) > 0) {
    dout = ib->Cast(dout, kFloat32);
  }
  auto dvalue = ib->ReduceSum(dout);
  return {ib->ZerosLike(shape), ib->Cast(dvalue, dout_typeptr)};
});

REG_BPROP_BUILDER("TensorCopySlices").SetUnusedInputs({i0, i5}).SetBody(BODYFUNC(ib) {
  auto update = ib->GetInput(kIndex1);
  auto begin = ib->GetInput(kIndex2);
  auto end = ib->GetInput(kIndex3);
  auto stride = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto x_grad = ib->Emit(kTensorCopySlicesOpName, {dout, ib->ZerosLike(update), begin, end, stride});
  constexpr int64_t begin_mask = 0;
  constexpr int64_t end_mask = 0;
  constexpr int64_t ellipsis_mask = 0;
  constexpr int64_t new_axis_mask = 0;
  constexpr int64_t shrink_axis_mask = 0;
  auto update_grad = ib->Emit(kStridedSliceOpName, {dout, begin, end, stride},
                              {{kAttrBeginMask, MakeValue(begin_mask)},
                               {kAttrEndMask, MakeValue(end_mask)},
                               {kAttrEllipsisMask, MakeValue(ellipsis_mask)},
                               {kAttrNewAxisMask, MakeValue(new_axis_mask)},
                               {kAttrShrinkAxisMask, MakeValue(shrink_axis_mask)}});
  return {x_grad, update_grad, ib->ZerosLike(begin), ib->ZerosLike(end), ib->ZerosLike(stride)};
});

REG_BPROP_BUILDER("Roll").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  std::vector<int64_t> shift = GetIntList(ib->GetAttr("shift"));
  std::transform(shift.begin(), shift.end(), shift.begin(), [](const int64_t &e) { return -e; });
  return {ib->Emit("Roll", {dout}, {{"axis", ib->GetAttr("axis")}, {"shift", MakeValue(shift)}})};
});

REG_BPROP_BUILDER("DynamicResizeNearestNeighbor").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto inputs = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shp = ib->GetShape(inputs);
  std::vector<int64_t> new_shp;
  for (size_t i = 2; i < shp.size(); i++) {
    new_shp.push_back(shp[i]);
  }
  return {
    ib->Emit("ResizeNearestNeighborGrad", {dout, ib->Value(shp)}, {{"align_corners", ib->GetAttr("align_corners")}}),
    ib->ZerosLike(size)};
});

REG_BPROP_BUILDER("ParallelResizeBilinear").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("ParallelResizeBilinearGrad", {dout, x, size},
                     {{"ori_image_size", ib->GetAttr("ori_image_size")},
                      {"src_start_w", ib->GetAttr("src_start_w")},
                      {"dst_start_w", ib->GetAttr("dst_start_w")},
                      {"align_corners", ib->GetAttr("align_corners")},
                      {"half_pixel_centers", MakeValue(false)}});
  return {dx, ib->ZerosLike(size)};
});

REG_BPROP_BUILDER("DynamicBroadcastTo").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto shp = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto x_shape = ib->Emit("TensorShape", {x});
  auto broadcast_shape = ib->Emit("TensorShape", {out});
  auto brod = ib->Emit("DynamicBroadcastGradientArgs", {broadcast_shape, x_shape});
  auto reduction_axes = ib->TupleGetItem(brod, 1);
  auto dout_dtype = dout->dtype();
  MS_EXCEPTION_IF_NULL(dout_dtype);
  auto dout_dtype_id = dout_dtype->type_id();
  bool need_cast =
    (dout_dtype_id == kNumberTypeInt16 || dout_dtype_id == kNumberTypeInt32 || dout_dtype_id == kNumberTypeInt64);
  if (need_cast) {
    dout = ib->Cast(dout, kFloat32);
  }
  auto reduced_grad =
    ib->Emit("ReduceSum", {dout, reduction_axes}, {{"keep_dims", MakeValue(true)}, {"skip_mode", MakeValue(true)}});
  if (need_cast) {
    reduced_grad = ib->Cast(reduced_grad, dout_dtype_id);
  }
  auto dx = ib->Reshape(reduced_grad, x_shape);
  return {dx, ib->ZerosLike(shp)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
