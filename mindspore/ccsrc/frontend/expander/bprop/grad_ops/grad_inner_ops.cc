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
#include "ops/array_op_name.h"
#include "ops/op_utils.h"

namespace mindspore::expander::bprop {

NodePtr GetMatrixDiagPartAssit(BpropBuilder *ib, const ShapeVector &x_shape, const TypePtr &x_dtype) {
  auto base_eye = ib->Emit(
    "Eye", {ib->Value(x_shape[x_shape.size() - i2]), ib->Value(x_shape[x_shape.size() - 1]), ib->EmitValue(x_dtype)});
  base_eye = ib->Reshape(base_eye, {-1});
  ShapeVector tile_shape(x_shape.begin(), x_shape.end() - i2);
  auto tile = ib->Tile(base_eye, tile_shape);
  auto assist = ib->Reshape(tile, x_shape);
  return assist;
}

NodePtr GetMatrixDiagAssit(BpropBuilder *ib, const ShapeVector &x_shape, const TypePtr &x_dtype) {
  auto base_eye = ib->Emit(
    "Eye", {ib->Value(x_shape[x_shape.size() - 1]), ib->Value(x_shape[x_shape.size() - 1]), ib->EmitValue(x_dtype)});
  base_eye = ib->Reshape(base_eye, {-1});
  ShapeVector tile_shape(x_shape.begin(), x_shape.end() - 1);
  auto tile = ib->Tile(base_eye, tile_shape);
  auto assist_shape(x_shape);
  assist_shape.push_back(x_shape.back());
  auto assist = ib->Reshape(tile, assist_shape);
  return assist;
}

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
  return {dq, dk, ib->OutZeros(local_mask), ib->OutZeros(global_mask)};
});

REG_BPROP_BUILDER("ResizeBilinearV2").SetUnusedInputs({i1, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto align_corners = ib->GetInput(kIndex2);
  auto half_pixel_centers = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto dx = ib->Emit("ResizeBilinearGrad", {dout, x, align_corners, half_pixel_centers});
  return {dx, ib->OutZeros(size), ib->OutZeros(align_corners), ib->OutZeros(half_pixel_centers)};
});

REG_BPROP_BUILDER("ConvertToDynamic").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
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
  return {ib->OutZeros(shape), ib->Cast(dvalue, dout_typeptr)};
});

REG_BPROP_BUILDER("FillScalar").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto size = ib->GetInput(kIndex0);
  auto type = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dvalue = ib->SumExt(dout, ib->EmitValue(kNone), ib->Value(false));
  return {ib->OutZeros(size), dvalue, ib->OutZeros(type)};
});

REG_BPROP_BUILDER("FillTensor").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto size = ib->GetInput(kIndex0);
  auto value = ib->GetInput(kIndex1);
  auto type = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto value_shape = value->shape();
  auto dvalue = ib->SumExt(dout, ib->EmitValue(kNone), ib->Value(false));
  if (IsDynamicRank(value_shape)) {
    auto v_shape = ib->Shape(value);
    auto input_dtype = ib->GetDtype(value)->type_id();
    auto value_dtype = ib->Value(static_cast<int64_t>(input_dtype));
    auto real_dvalue = ib->Reshape(dvalue, v_shape);
    auto dvalue_out = ib->Emit("FillTensor", {v_shape, real_dvalue, value_dtype});
    return {ib->OutZeros(size), dvalue_out, ib->OutZeros(type)};
  }
  if (!value->shape().empty()) {
    return {ib->OutZeros(size), ib->Reshape(dvalue, {1}), ib->OutZeros(type)};
  }
  return {ib->OutZeros(size), dvalue, ib->OutZeros(type)};
});

REG_BPROP_BUILDER("TensorCopySlices").SetUnusedInputs({i0, i5}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto update = ib->GetInput(kIndex1);
  auto begin = ib->GetInput(kIndex2);
  auto end = ib->GetInput(kIndex3);
  auto stride = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto x_grad = x->need_compute_grad_out()
                  ? ib->Emit(kTensorCopySlicesOpName, {dout, ib->ZerosLike(update), begin, end, stride})
                  : ib->OutZeros(x);
  auto update_grad =
    update->need_compute_grad_out() ? ib->StridedSlice(dout, begin, end, stride) : ib->OutZeros(update);
  return {x_grad, update_grad, ib->OutZeros(begin), ib->OutZeros(end), ib->OutZeros(stride)};
});

REG_BPROP_BUILDER("Roll").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto shift = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto is_dynamic_shifts = false;
  auto shift_value = shift->BuildValue();
  MS_EXCEPTION_IF_NULL(shift_value);
  auto shift_array_opt = ops::GetArrayValue<int64_t>(shift_value);
  if (!shift_array_opt.has_value() || shift_array_opt.value().HasUnknownValue()) {
    is_dynamic_shifts = True;
  }
  if (is_dynamic_shifts) {
    auto shifts_tensor = ib->Emit("TupleToTensor", {shift, ib->Value<int64_t>(kInt64->type_id())});
    auto neg_shifts = ib->Emit("Neg", {shifts_tensor});
    auto neg_shifts_tuple = ib->TensorToTuple(neg_shifts);
    return {ib->Emit("Roll", {dout, neg_shifts_tuple, axis}), ib->OutZeros(shift), ib->OutZeros(axis)};
  }
  auto shift_array = shift_array_opt.value();
  std::vector<int64_t> shift_vec = shift_array.ToVector();
  std::vector<int64_t> neg_shift(shift_vec.size());
  (void)std::transform(shift_vec.begin(), shift_vec.end(), neg_shift.begin(),
                       [](const int64_t &shift) { return shift * -1; });
  return {ib->Emit("Roll", {dout, ib->Value(neg_shift), axis}), ib->OutZeros(shift), ib->OutZeros(axis)};
});

DEF_PURE_SHAPE_CALC(g_dynamic_resize_nearest_neighbor)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(0);
    ShapeVector shape2(x_shape.begin() + 2, x_shape.end());
    return {shape2};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto new_shape = inputs.at(0);
    int64_t rank = IsDynamicRank(new_shape) ? -1 : SizeToLong(new_shape.size()) - 2;
    return {rank};
  });

REG_BPROP_BUILDER("DynamicResizeNearestNeighbor").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto inputs = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto res = ib->ShapeCalc(g_dynamic_resize_nearest_neighbor, {inputs})[0];
  return {ib->Emit("ResizeNearestNeighborGrad", {dout, res}, {{"align_corners", ib->GetAttr("align_corners")}}),
          ib->OutZeros(size)};
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
  return {dx, ib->OutZeros(size)};
});

REG_BPROP_BUILDER("DynamicBroadcastTo").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto shp = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto broadcast_axes = ib->BroadcastGradientArgs(out, x);
  auto reduction_axes = broadcast_axes[kIndex1];
  auto reduced_grad = ib->ReduceSum(dout, reduction_axes, true, true);
  auto dx = ib->Reshape(reduced_grad, ib->Shape(x));
  return {dx, ib->OutZeros(shp)};
});

REG_BPROP_BUILDER("SiLU").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SiLUGrad", {dout, x});
  return {dx};
});

REG_BPROP_BUILDER("SiLUGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto grad = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto sig = ib->Emit("Sigmoid", {y});
  auto mul0 = ib->Mul(grad, y);
  auto sig_grad1 = ib->Emit("SigmoidGrad", {sig, dout});
  NodePtr dy;
  if (y->need_compute_grad_out()) {
    auto mul1 = ib->Mul(grad, sig_grad1);
    auto mul2 = ib->Mul(mul0, dout);
    auto mul3 = ib->Mul(ib->Tensor(2, ib->GetDtype(sig)), sig);
    auto sub1 = ib->Sub(ib->Tensor(1, ib->GetDtype(mul3)), mul3);
    auto mul4 = ib->Mul(mul2, sub1);
    auto mul5 = ib->Mul(grad, dout);
    auto add1 = ib->Add(mul4, mul5);
    auto sig_grad2 = ib->Emit("SigmoidGrad", {sig, add1});
    dy = ib->Add(mul1, sig_grad2);
  } else {
    dy = ib->OutZeros(y);
  }

  NodePtr dgrad;
  if (grad->need_compute_grad_out()) {
    auto mul6 = ib->Mul(sig, dout);
    auto mul7 = ib->Mul(y, sig_grad1);
    dgrad = ib->Add(mul6, mul7);
  } else {
    dgrad = ib->OutZeros(grad);
  }
  return {dgrad, dy};
});

REG_BPROP_BUILDER("_VirtualPipelineEnd").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("_VirtualPipelineEnd", {dout});
  return {dx};
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

REG_BPROP_BUILDER("MatmulDDS").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  auto q = ib->GetInput(kIndex0);
  auto k = ib->GetInput(kIndex1);
  auto local_mask = ib->GetInput(kIndex2);
  auto global_mask = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto d_out = ib->GetInput(kIndex5);
  auto lc = ib->TupleGetItem(out, 0);
  auto gc = ib->TupleGetItem(out, 1);
  auto d_lc = ib->TupleGetItem(d_out, 0);
  auto d_gc = ib->TupleGetItem(d_out, 1);
  auto tmp = ib->Emit("MatmulDDSGrad", {q, k, lc, gc, d_lc, d_gc});
  auto dq = ib->TupleGetItem(tmp, 0);
  auto dk = ib->TupleGetItem(tmp, 1);
  dk = ib->Transpose(dk, {1, 0, 3, 2});
  return {dq, dk, ib->OutZeros(local_mask), ib->OutZeros(global_mask)};
});

REG_BPROP_BUILDER("ConvertToAdapterTensor").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("ConvertToMsTensor").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("MatrixDiag").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape = ib->GetShape(dout);
  if (IsDynamicRank(shape) || IsDynamicShape(shape)) {
    MS_LOG(EXCEPTION) << "MatrxiDiag bprop don't support dynamic rank or dynamic shape, because operation Eye need "
                         "constant values in shape";
  }
  auto dtype = ib->GetDtype(dout);
  auto assist = GetMatrixDiagPartAssit(ib, shape, dtype);
  auto dx = ib->Emit("MatrixDiagPart", {dout, assist});
  return {dx, ib->OutZeros(y)};
});

REG_BPROP_BUILDER("MatrixSetDiag").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto z = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto input_shape = ib->GetShape(x);
  auto grad_shape = GetValue<ShapeVector>(ib->Shape(dout)->BuildValue());
  if (IsDynamicRank(input_shape) || IsDynamicShape(input_shape) || IsDynamicRank(grad_shape) ||
      IsDynamicShape(grad_shape)) {
    MS_LOG(EXCEPTION) << "MatrxiSetDiag bprop don't support dynamic rank or dynamic shape, because operation Eye need "
                         "constant values in shape";
  }
  ShapeVector diag_shape(input_shape.begin(), input_shape.end() - 2);
  ShapeVector matrix_shape(input_shape.end() - 2, input_shape.end());
  diag_shape.push_back(std::min(matrix_shape[0], matrix_shape[1]));
  auto grad_dtype = ib->GetDtype(dout);
  auto assist = GetMatrixDiagPartAssit(ib, grad_shape, grad_dtype);
  auto dx =
    x->need_compute_grad_out()
      ? ib->Emit("MatrixSetDiag", {dout, ib->Emit("Zeros", {ib->Value(diag_shape), ib->EmitValue(grad_dtype)}), assist})
      : ib->OutZeros(x);
  auto dy = y->need_compute_grad_out() ? ib->Emit("MatrixDiagPart", {dout, assist}) : ib->OutZeros(y);
  auto dz = ib->OutZeros(z);
  return {dx, dy, dz};
});

REG_BPROP_BUILDER("MatrixDiagPart").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape = ib->GetShape(x);
  if (IsDynamicRank(shape) || IsDynamicShape(shape)) {
    MS_LOG(EXCEPTION) << "MatrxiDiagPart bprop don't support dynamic rank or dynamic shape, because operation Eye need "
                         "constant values in shape";
  }
  ShapeVector x_shape(shape.end() - 2, shape.end());
  if (x_shape[0] == x_shape[1]) {
    shape = ib->GetShape(dout);
    auto dtype = ib->GetDtype(dout);
    auto assist = GetMatrixDiagAssit(ib, shape, dtype);
    return {ib->Emit("MatrixDiag", {dout, assist}), ib->OutZeros(y)};
  }
  auto dtype = ib->GetDtype(x);
  auto assist = GetMatrixDiagPartAssit(ib, shape, dtype);
  return {ib->Emit("MatrixSetDiag", {ib->OutZeros(x), dout, assist}), ib->OutZeros(y)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
