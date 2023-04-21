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

#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/bprop/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradInnerOps)
REG_BPROP_BUILDER("DSDMatmul.NotReady").SetBody(BODYFUNC(ib) {
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

REG_BPROP_BUILDER("MatmulDDS.NotReady").SetUnusedInputs({i2, i3, i5}).SetBody(BODYFUNC(ib) {
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

REG_BPROP_BUILDER("_VirtualPipelineEnd.NotReady").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
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
  std::vector<int64_t> axis{};
  for (int64_t i = 0; i < static_cast<int64_t>(dout->shape().size()); ++i) {
    axis.push_back(i);
  }
  auto dvalue = ib->ReduceSum(dout, axis);
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
  (void)std::transform(shift.begin(), shift.end(), shift.begin(), [](const int64_t &e) { return -e; });
  return {ib->Emit("Roll", {dout}, {{"axis", ib->GetAttr("axis")}, {"shift", MakeValue(shift)}})};
});

REG_BPROP_BUILDER("DynamicResizeNearestNeighbor").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto inputs = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(0);
    ShapeVector shape2(x_shape.begin() + 2, x_shape.end());
    return {shape2};
  };
  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &) -> ShapeVector {
    auto new_shape = inputs.at(0);
    auto rank = IsDynamicRank(new_shape) ? -1 : static_cast<int64_t>(new_shape.size()) - 2;
    return {rank};
  };
  auto res = ib->ShapeCalc({inputs}, shape_func, infer_func, {})[0];
  return {ib->Emit("ResizeNearestNeighborGrad", {dout, res}, {{"align_corners", ib->GetAttr("align_corners")}}),
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
  auto broadcast_axes = ib->BroadcastGradientArgs(out, x);
  auto reduction_axes = broadcast_axes[kIndex1];
  auto dout_dtype = dout->dtype();
  MS_EXCEPTION_IF_NULL(dout_dtype);
  auto dout_dtype_id = dout_dtype->type_id();
  bool need_cast =
    (dout_dtype_id == kNumberTypeInt16 || dout_dtype_id == kNumberTypeInt32 || dout_dtype_id == kNumberTypeInt64);
  if (need_cast) {
    dout = ib->Cast(dout, kFloat32);
  }
  auto reduced_grad = ib->ReduceSum(dout, reduction_axes, true, true);
  if (need_cast) {
    reduced_grad = ib->Cast(reduced_grad, dout_dtype_id);
  }
  auto dx = ib->Reshape(reduced_grad, ib->Shape(x));
  return {dx, ib->ZerosLike(shp)};
});

REG_BPROP_BUILDER("SiLU").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SiLUGrad", {dout, out});
  return {dx};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
