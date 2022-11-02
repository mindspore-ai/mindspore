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
#include "common/graph_kernel/bprop/expander/common_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
namespace {
NodePtrList GatherDropNegatives(const BpropIRBuilder *ib, const NodePtr &params, const NodePtr &ids,
                                const NodePtr &zero_clipped_indices_param = nullptr,
                                const NodePtr &is_positive_param = nullptr) {
  NodePtr zero_clipped_indices = zero_clipped_indices_param;
  if (zero_clipped_indices_param == nullptr) {
    zero_clipped_indices = ib->Emit("Maximum", {ids, ib->ZerosLike(ids)});
  }
  auto gathered = ib->Emit("Gather", {params, zero_clipped_indices, ib->Tensor(0, kInt64)});

  NodePtr is_positive = is_positive_param;
  if (is_positive_param == nullptr) {
    is_positive = ib->Emit("GreaterEqual", {ids, ib->Tensor(0, ib->GetDtype(ids))});
    auto broadcastable_shape = ib->GetShape(is_positive);
    auto back_size = ib->GetShape(gathered).size() - ib->GetShape(is_positive).size();
    for (size_t i = 0; i < back_size; ++i) {
      broadcastable_shape.push_back(1);
    }
    is_positive = ib->Emit("Reshape", {is_positive, ib->Value<ShapeVector>(broadcastable_shape)});
    auto gathered_shape = ib->GetShape(gathered);
    is_positive = ib->Emit("LogicalAnd",
                           {is_positive, ib->Emit("Fill", {ib->EmitValue(kBool), ib->Value<ShapeVector>(gathered_shape),
                                                           ib->Tensor(1, kBool)})});
  }
  auto zero_slice = ib->ZerosLike(gathered);
  return {ib->Emit("Select", {is_positive, gathered, zero_slice}), zero_clipped_indices, is_positive};
}

NodePtrList UnsortedSegmentMinOrMaxGrad(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &segment_ids,
                                        const NodePtr &num_segments, const NodePtr &out, const NodePtr &dout) {
  auto temp_outs = GatherDropNegatives(ib, out, segment_ids, nullptr, nullptr);
  constexpr size_t out_size = 3;
  MS_EXCEPTION_IF_CHECK_FAIL(temp_outs.size() == out_size, "Outputs' size should be 3.");
  auto gathered_outputs = temp_outs[0];
  auto zero_clipped_indices = temp_outs[1];
  auto is_positive = temp_outs[2];

  auto tmp = ib->Emit("Equal", {x, gathered_outputs});
  auto is_selected = ib->Emit("LogicalAnd", {tmp, is_positive});
  auto num_selected = ib->Emit(
    "UnsortedSegmentSum", {ib->Emit("Cast", {is_selected, ib->Value(ib->GetDtype(dout))}), segment_ids, num_segments});
  auto weighted_grads = ib->Emit("RealDiv", {dout, num_selected});
  auto temp_outs_2 = GatherDropNegatives(ib, weighted_grads, nullptr, zero_clipped_indices, is_positive);
  MS_EXCEPTION_IF_CHECK_FAIL(temp_outs.size() > 0, "Outputs should not be empty.");
  auto gathered_grads = temp_outs_2[0];
  auto zeros = ib->ZerosLike(gathered_grads);
  return {ib->Emit("Select", {is_selected, gathered_grads, zeros}), ib->ZerosLike(segment_ids),
          ib->ZerosLike(num_segments)};
}
}  // namespace

REG_BPROP_BUILDER("GatherD").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto index = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("GatherDGradV2", {x, dim, index, dout});
  auto ddim = ib->Emit("ZerosLike", {dim});
  return {dx, ddim, ib->ZerosLike(index)};
});

REG_BPROP_BUILDER("GatherDGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dim = GetValue<int64_t>(ib->GetAttr("dim"));
  auto x_shp = GetValue<ShapeVector>(ib->GetAttr("shape"));
  auto index = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto index_shp = ib->GetShape(index);
  auto dim_before_axis = 1;
  for (int64_t i = 0; i < dim; ++i) {
    dim_before_axis *= x_shp[i];
  }
  auto dim_at_axis_index = index_shp[dim];
  auto dim_at_axis_output = x_shp[dim];
  auto dim_after_axis = 1;
  for (size_t i = dim + 1; i < x_shp.size(); ++i) {
    dim_after_axis *= x_shp[i];
  }
  auto element = (dim_before_axis * dim_at_axis_index) * dim_after_axis;
  auto index_type = ib->GetDtype(index);
  auto id = ib->Tensor(Range(element), index_type);
  auto i = ib->Emit("FloorDiv", {id, ib->Tensor((dim_at_axis_index * dim_after_axis), index_type)});
  auto k = ib->Emit("FloorMod", {id, ib->Tensor(dim_after_axis, index_type)});
  auto less = ib->Emit("Less", {index, ib->Tensor(0, index_type)});
  auto j = ib->Cast(less, index_type);
  auto j_read = ib->Add((ib->Mul(ib->Tensor(dim_at_axis_index, index_type), j)), index);
  auto j_read_reshape = ib->Reshape(j_read, {-1});
  auto i_after = ib->Mul(i, ib->Tensor(dim_at_axis_output * dim_after_axis, index_type));
  auto read_id = ib->Add((ib->Add(i_after, (ib->Mul(j_read_reshape, ib->Tensor(dim_after_axis, index_type))))), k);
  auto dout_reshape = ib->Reshape(dout, {-1});
  auto dx = ib->Emit("Gather", {dout_reshape, read_id, ib->Tensor(0)});
  dx = ib->Reshape(dx, ib->GetShape(x));
  return {ib->ZerosLike(index), dx};
});

REG_BPROP_BUILDER("GatherDGradV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dim = GetValue<int64_t>(ib->GetAttr("dim"));
  auto index = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto index_shp = ib->GetShape(index);
  auto x_shp = ib->GetShape(x);
  auto dim_before_axis = 1;
  for (int64_t i = 0; i < dim; ++i) {
    dim_before_axis *= x_shp[i];
  }
  auto dim_at_axis_index = index_shp[dim];
  auto dim_at_axis_output = x_shp[dim];
  auto dim_after_axis = 1;
  for (size_t i = dim + 1; i < x_shp.size(); ++i) {
    dim_after_axis *= x_shp[i];
  }
  auto element = (dim_before_axis * dim_at_axis_index) * dim_after_axis;
  ShapeVector ranges;
  for (int64_t i = 0; i < element; i += 1) {
    ranges.push_back(i);
  }
  auto index_type = ib->GetDtype(index);
  auto id = ib->Tensor(ranges, index_type);
  auto i = ib->RealDiv(id, ib->Tensor((dim_at_axis_index * dim_after_axis), index_type));
  auto k = ib->Emit("Mod", {id, ib->Tensor(dim_after_axis)});
  auto less = ib->Emit("Less", {index, ib->Tensor(0, index_type)});
  auto j = ib->Cast(less, ib->GetDtype(index));
  auto j_read = ib->Add((ib->Mul(ib->Tensor(dim_at_axis_index, index_type), j)), index);
  j_read = ib->Reshape(j_read, {-1});
  auto i_after = ib->Mul(i, ib->Tensor(dim_at_axis_output * dim_after_axis, index_type));
  auto read_id = ib->Add((ib->Add(i_after, (ib->Mul(j_read, ib->Tensor(dim_after_axis, index_type))))), k);
  dout = ib->Reshape(dout, {-1});
  auto dx = ib->Emit("Gather", {dout, read_id, 0});
  dx = ib->Reshape(dx, ib->GetShape(x));
  return {ib->ZerosLike(index), dx};
});

REG_BPROP_BUILDER("SparseGatherV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shp = ib->GetShape(x);
  auto axis_int = GetValue<int64_t>(axis->get<ValueNodePtr>()->value());
  if (axis_int == 0) {
    ShapeVector values_shape{ib->GetSize(indices)};
    if (x_shp.size() > 1) {
      values_shape.insert(values_shape.end(), x_shp.begin() + 1, x_shp.end());
    }
    auto values = ib->Reshape(dout, values_shape);
    auto indices_new = ib->Reshape(indices, {values_shape[0]});
    auto row_tensor = ib->MakeTuple({indices_new, values, ib->Value<ShapeVector>(x_shp)});
    return {row_tensor, ib->ZerosLike(indices), ib->ZerosLike(axis)};
  }
  auto out_shp = ib->GetShape(dout);
  auto ind_shp = ib->GetShape(indices);
  if (out_shp.size() == 0) {
    dout = ib->Emit("ExpandDims", {dout, ib->Value<int64_t>(-1)});
  }
  if (ind_shp.size() == 0) {
    indices = ib->Emit("ExpandDims", {indices, ib->Value<int64_t>(-1)});
  }
  out_shp = ib->GetShape(dout);
  ind_shp = ib->GetShape(indices);
  auto perm_1 = GenerateShapeIndex(out_shp, ind_shp, axis_int);
  auto values_transpose = ib->Emit("Transpose", {dout, ib->Value<ShapeVector>(perm_1)});
  auto params_grad = ib->Emit("UnsortedSegmentSum", {values_transpose, indices, ib->Value<int64_t>(x_shp[axis_int])});
  auto perm_2 = GenerateInverseIndex(x_shp, axis_int);
  params_grad = ib->Emit("Transpose", {params_grad, ib->Value<ShapeVector>(perm_2)});
  return {params_grad, ib->ZerosLike(indices), ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("Sort").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto axis = GetValue<int64_t>(ib->GetAttr("axis"));
  auto descending = GetValue<bool>(ib->GetAttr("descending"));
  auto input_x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(input_x);
  if (axis < 0) {
    axis += x_shape.size();
  }
  auto k = x_shape[axis];
  auto rank = x_shape.size();
  auto dvalue = ib->TupleGetItem(dout, 0);
  if (!descending) {
    input_x = ib->Emit("Neg", {input_x});
    dvalue = ib->Emit("Neg", {dvalue});
  }
  std::vector<int64_t> transposition;
  auto top_k_input = input_x;
  if ((static_cast<size_t>(axis + 1) != rank)) {
    transposition = GetTransposition(axis, rank);
    top_k_input = ib->Emit("Transpose", {input_x, ib->Value<ShapeVector>(transposition)});
  }
  auto tmp = ib->Emit("TopK", {top_k_input, ib->Value<int64_t>(k)});
  auto indices = ib->TupleGetItem(tmp, 1);
  auto ind_shape = ib->GetShape(indices);
  auto top_k_input_shape = ib->GetShape(top_k_input);
  auto in_lastdim = top_k_input_shape[top_k_input_shape.size() - 1];
  auto ind_lastdim = ind_shape[ind_shape.size() - 1];
  auto ind_2d = ib->Reshape(indices, {-1, ind_lastdim});
  auto outer_dim = ib->GetShape(ind_2d)[0];
  auto indices_dtype = ib->GetDtype(indices);
  auto range_flatten_index = ib->Tensor(Range(0, outer_dim * in_lastdim, in_lastdim), indices_dtype);
  range_flatten_index = ib->Emit("ExpandDims", {range_flatten_index, ib->Value<int64_t>(-1)});
  auto ind = ib->Reshape(ib->Add(ind_2d, range_flatten_index), {-1});
  auto x_size = 1;
  for (size_t i = 0; i < top_k_input_shape.size(); ++i) {
    x_size *= top_k_input_shape[i];
  }
  auto x_shape_1d = ib->Value<ShapeVector>({x_size});
  NodePtr dx = nullptr;
  if (!transposition.empty()) {
    auto invert_perm = ib->Value<ShapeVector>(InvertPermutation(transposition));
    dvalue = ib->Emit("Transpose", {dvalue, invert_perm});
    auto ind_expand = ib->Emit("ExpandDims", {ind, ib->Value<int64_t>(-1)});
    auto scatter = ib->Emit("ScatterNd", {ind_expand, ib->Reshape(dvalue, {-1}), x_shape_1d});
    auto out_grad = ib->Reshape(scatter, top_k_input_shape);
    dx = ib->Emit("Transpose", {out_grad, invert_perm});
  } else {
    auto ind_expand = ib->Emit("ExpandDims", {ind, ib->Value<int64_t>(-1)});
    auto scatter = ib->Emit("ScatterNd", {ind_expand, ib->Reshape(dvalue, {-1}), x_shape_1d});
    dx = ib->Reshape(scatter, top_k_input_shape);
  }
  if (!descending) {
    dx = ib->Emit("Neg", {dx});
  }
  return {dx};
});

REG_BPROP_BUILDER("Identity").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("Range").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto start = ib->GetInput(kIndex0);
  auto limit = ib->GetInput(kIndex1);
  auto delta = ib->GetInput(kIndex2);
  return {ib->ZerosLike(start), ib->ZerosLike(limit), ib->ZerosLike(delta)};
});

REG_BPROP_BUILDER("Pack").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto ret = ib->Emit("UnstackWithNum", {dout}, {{"num", ib->GetAttr("num")}, {"axis", ib->GetAttr("axis")}});
  return {ret};
});

REG_BPROP_BUILDER("Stack").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto ret = ib->Emit("UnstackWithNum", {dout}, {{"num", ib->GetAttr("num")}, {"axis", ib->GetAttr("axis")}});
  return {ret};
});

REG_BPROP_BUILDER("ReverseV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("ReverseV2", {dout}, {{"axis", ib->GetAttr("axis")}});
  return {dx};
});

REG_BPROP_BUILDER("Unstack").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  out = ib->Emit("Stack", {dout}, {{"axis", ib->GetAttr("axis")}});
  return {out};
});

REG_BPROP_BUILDER("StridedSlice").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto begin = ib->GetInput(kIndex1);
  auto end = ib->GetInput(kIndex2);
  auto strides = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto x_shape = ib->EmitValue(MakeValue(ib->GetShape(x)));
  auto dx = ib->Emit("StridedSliceGrad", {dout, x_shape, begin, end, strides},
                     {{"begin_mask", ib->GetAttr("begin_mask")},
                      {"end_mask", ib->GetAttr("end_mask")},
                      {"ellipsis_mask", ib->GetAttr("ellipsis_mask")},
                      {"new_axis_mask", ib->GetAttr("new_axis_mask")},
                      {"shrink_axis_mask", ib->GetAttr("shrink_axis_mask")}});
  auto dbegin = ib->ZerosLike(begin);
  auto dend = ib->ZerosLike(begin);
  auto dstrides = ib->ZerosLike(begin);
  return {dx, dbegin, dend, dstrides};
});

REG_BPROP_BUILDER("StridedSliceGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto shapex = ib->GetInput(kIndex1);
  auto begin = ib->GetInput(kIndex2);
  auto end = ib->GetInput(kIndex3);
  auto strides = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  return {ib->Emit("StridedSlice", {dout, begin, end, strides},
                   {{"begin_mask", ib->GetAttr("begin_mask")},
                    {"end_mask", ib->GetAttr("end_mask")},
                    {"ellipsis_mask", ib->GetAttr("ellipsis_mask")},
                    {"new_axis_mask", ib->GetAttr("new_axis_mask")},
                    {"shrink_axis_mask", ib->GetAttr("shrink_axis_mask")}}),
          ib->ZerosLike(shapex), ib->ZerosLike(begin), ib->ZerosLike(end), ib->ZerosLike(strides)};
});

REG_BPROP_BUILDER("Eye").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto n = ib->GetInput(kIndex0);
  auto m = ib->GetInput(kIndex1);
  auto t = ib->GetInput(kIndex2);
  return {ib->ZerosLike(n), ib->ZerosLike(m), t};
});

REG_BPROP_BUILDER("Select").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto cond = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto y = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {ib->ZerosLike(cond), ib->Emit("Select", {cond, dout, ib->ZerosLike(x)}),
          ib->Emit("Select", {cond, ib->ZerosLike(y), dout})};
});

REG_BPROP_BUILDER("OnesLike").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("ZerosLike").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("ResizeNearestNeighbor").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  ShapeVector new_shape;
  for (size_t i = 2; i < x_shape.size(); i++) {
    new_shape.push_back(x_shape[i]);
  }
  auto shape = ib->EmitValue(MakeValue(new_shape));
  auto out = ib->Emit("ResizeNearestNeighborGrad", {dout, shape}, {{"align_corners", ib->GetAttr("align_corners")}});
  return {out};
});

REG_BPROP_BUILDER("GatherNd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shp = ib->EmitValue(MakeValue(ib->GetShape(x)));
  return {ib->Emit("ScatterNd", {indices, dout, shp}), ib->ZerosLike(indices)};
});

REG_BPROP_BUILDER("ScatterNd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex0);
  auto shape = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {ib->ZerosLike(indices), ib->Emit("GatherNd", {dout, indices}), ib->ZerosLike(shape)};
});

REG_BPROP_BUILDER("ScatterNdUpdate").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Emit("GatherNd", {dout, indices})};
});

REG_BPROP_BUILDER("ScatterNonAliasingAdd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Emit("GatherNd", {dout, indices})};
});

REG_BPROP_BUILDER("TensorScatterUpdate").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_grad = ib->Emit("TensorScatterUpdate", {dout, indices, ib->ZerosLike(update)});
  auto update_grad = ib->Emit("GatherNd", {dout, indices});
  return {x_grad, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("Flatten").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Reshape(dout, ib->GetShape(x));
  return {dx};
});

REG_BPROP_BUILDER(kReshapeOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto shp = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shapex = ib->GetShape(x);
  return {ib->Reshape(dout, shapex), ib->ZerosLike(shp)};
});

REG_BPROP_BUILDER("NonZero").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("BatchMatMul").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto ta = GetValue<bool>(ib->GetAttr("transpose_a"));
  auto tb = GetValue<bool>(ib->GetAttr("transpose_b"));
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);

  NodePtr dx;
  if (ta) {
    dx =
      ib->Emit("BatchMatMul", {w, dout}, {{"transpose_a", MakeValue(ta && tb)}, {"transpose_b", MakeValue(ta || !tb)}});
  } else {
    dx =
      ib->Emit("BatchMatMul", {dout, w}, {{"transpose_a", MakeValue(ta && tb)}, {"transpose_b", MakeValue(ta || !tb)}});
  }

  NodePtr dw;
  if (tb) {
    dw = ib->Emit("BatchMatMul", {dout, x},
                  {{"transpose_a", MakeValue((!ta) || tb)}, {"transpose_b", MakeValue(ta && tb)}});
  } else {
    dw = ib->Emit("BatchMatMul", {x, dout},
                  {{"transpose_a", MakeValue((!ta) || tb)}, {"transpose_b", MakeValue(ta && tb)}});
  }

  return BinopGradCommonWithShift(ib, x, w, dx, dw, 2);
});

REG_BPROP_BUILDER("Argmax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Argmin").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Diag").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("DiagPart", {dout})};
});

REG_BPROP_BUILDER("DiagPart").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("Diag", {dout})};
});

REG_BPROP_BUILDER("SpaceToBatch").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("BatchToSpace", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"crops", ib->GetAttr("paddings")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchToSpace").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("SpaceToBatch", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"paddings", ib->GetAttr("crops")}});
  return {dx};
});

REG_BPROP_BUILDER("ReverseSequence").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto seq_lengths = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("ReverseSequence", {dout, seq_lengths},
                     {{"batch_dim", ib->GetAttr("batch_dim")}, {"seq_dim", ib->GetAttr("seq_dim")}});
  return {dx, ib->ZerosLike(seq_lengths)};
});

REG_BPROP_BUILDER("TensorScatterAdd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto update_grad = ib->Emit("GatherNd", {dout, indices});
  return {dout, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER(kConcatOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto axis = ib->GetAttr<int64_t>(kAttrAxis);
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto input_shapes = ib->GetShapes(x);
  if (input_shapes.empty()) {
    MS_EXCEPTION(ValueError) << "For 'ConcatOffset', 'x' can not be empty";
  }
  // axis
  auto rank = input_shapes[0].size();
  auto rank_i = SizeToLong(rank);
  if (rank == 0 || axis < -rank_i || axis >= rank_i) {
    MS_EXCEPTION(ValueError) << "For 'ConcatOffset', input shapes rank can not be 0 and 'axis' must be in range [-"
                             << rank_i << ", " << rank_i << "), but got " << axis;
  }
  if (axis < 0) {
    axis += rank_i;
  }
  auto axis_s = LongToSize(axis);
  // is_uniform
  bool is_uniform = true;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    if (input_shapes[i].size() != rank) {
      MS_EXCEPTION(ValueError) << "For 'ConcatOffset', input shapes [" << i
                               << "] and input shapes [0] must have same rank, but got: " << input_shapes[i].size()
                               << " vs " << rank;
    }
    if (input_shapes[i][axis_s] != input_shapes[0][axis_s]) {
      is_uniform = false;
    }
  }
  // use Split if is_uniform is true
  if (is_uniform) {
    auto input_nums = SizeToLong(input_shapes.size());
    auto dx = ib->Emit(kSplitOpName, {dout}, {{kAttrAxis, MakeValue(axis)}, {kAttrOutputNum, MakeValue(input_nums)}});
    return {dx};
  }
  // else use Slice
  NodePtrList res;
  int64_t sum_axis = 0;
  for (size_t i = 0; i < input_shapes.size(); ++i) {
    std::vector<int64_t> offset(rank, 0);
    offset[axis_s] = sum_axis;
    sum_axis += input_shapes[i][axis_s];
    auto slice_out = ib->Emit(kSliceOpName, {dout, ib->Value(offset), ib->Value(input_shapes[i])});
    res.push_back(slice_out);
  }
  return {ib->MakeTuple(res)};
});

REG_BPROP_BUILDER("Mvlgamma").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("MvlgammaGrad", {dout, x}, {{"p", ib->GetAttr("p")}});
  return {dx};
});

REG_BPROP_BUILDER("TensorScatterDiv").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto in_grad = ib->Emit("TensorScatterDiv", {dout, indices, update});
  auto gather_update = ib->Emit("GatherNd", {dout, indices});
  auto gather_x = ib->Emit("GatherNd", {x, indices});
  auto mul_result = ib->Mul(update, update);
  auto neg_result = ib->Emit("Neg", {mul_result});
  auto update_grad = ib->Mul(gather_update, (ib->Emit("Div", {gather_x, neg_result})));
  return {in_grad, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("TensorScatterSub").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto update_grad = ib->Emit("Neg", {ib->Emit("GatherNd", {dout, indices})});
  return {dout, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("TensorScatterMul").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto gather_update = ib->Emit("GatherNd", {dout, indices});
  auto gather_x = ib->Emit("GatherNd", {x, indices});
  auto dx = ib->Emit("TensorScatterMul", {dout, indices, update});
  auto d_update = ib->Mul(gather_x, gather_update);
  return {dx, ib->ZerosLike(indices), d_update};
});

NodePtrList TensorScatterPossibleReplacement(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto updates = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto x_indicators = ib->Cast(ib->Emit("Equal", {x, out}), kInt32);
  auto possibly_updated = ib->Emit("GatherNd", {out, indices});
  auto out_indicators = ib->Cast(ib->Emit("Equal", {updates, possibly_updated}), kInt32);
  auto input_shape = ib->GetShape(x);
  auto scattered_out_indicators = ib->Emit("ScatterNd", {indices, out_indicators, ib->Tensor(input_shape)});
  auto indicators = ib->Add(x_indicators, scattered_out_indicators);
  auto dx = ib->RealDiv((ib->Mul(dout, (ib->Cast(x_indicators, ib->GetDtype(dout))))),
                        (ib->Cast(indicators, ib->GetDtype(dout))));
  auto dupdates =
    ib->Mul((ib->Emit("GatherNd", {ib->RealDiv(dout, (ib->Cast(indicators, ib->GetDtype(dout)))), indices})),
            (ib->Cast(out_indicators, ib->GetDtype(dout))));
  return {ib->Cast(dx, ib->GetDtype(x)), ib->ZerosLike(indices), ib->Cast(dupdates, ib->GetDtype(updates))};
}

REG_BPROP_BUILDER("TensorScatterMax").SetBody(TensorScatterPossibleReplacement);
REG_BPROP_BUILDER("TensorScatterMin").SetBody(TensorScatterPossibleReplacement);

REG_BPROP_BUILDER("IndexFill").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto indices = ib->GetInput(kIndex2);
  auto value = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto zero_value = ib->ZerosLike(value);
  auto x_grad = ib->Emit("IndexFill", {dout, dim, indices, zero_value});
  NodePtr value_grad;
  if (ib->GetShape(x).empty()) {
    value_grad = dout;
  } else {
    auto tmp = ib->Emit("Gather", {dout, indices, dim});
    value_grad = ib->Emit("ReduceSum", {tmp, ib->Value(ShapeVector())}, {{"keep_dims", MakeValue(false)}});
  }
  return {x_grad, ib->ZerosLike(dim), ib->ZerosLike(indices), value_grad};
});

REG_BPROP_BUILDER("UnsortedSegmentSum").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {GatherDropNegatives(ib, dout, segment_ids, nullptr, nullptr)[0], ib->ZerosLike(segment_ids),
          ib->ZerosLike(num_segments)};
});

REG_BPROP_BUILDER("UnsortedSegmentMin").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  return UnsortedSegmentMinOrMaxGrad(ib, x, segment_ids, num_segments, out, dout);
});

REG_BPROP_BUILDER("UnsortedSegmentMax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  return UnsortedSegmentMinOrMaxGrad(ib, x, segment_ids, num_segments, out, dout);
});

REG_BPROP_BUILDER("UnsortedSegmentProd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);

  NodePtr is_zero = nullptr;
  auto x_dtype = ib->GetDtype(x);
  MS_EXCEPTION_IF_NULL(x_dtype);
  auto x_dtype_id = x_dtype->type_id();
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    is_zero = ib->Emit("Equal", {x, ib->Tensor(0, x_dtype)});
  } else {
    is_zero = ib->Emit("Equal", {ib->Cast(x, kFloat32), ib->Tensor(0, kFloat32)});
  }

  auto num_zero = ib->Emit("UnsortedSegmentSum", {ib->Cast(is_zero, kInt32), segment_ids, num_segments});
  auto grad = ib->Emit(
    "Select", {ib->Emit("Greater", {num_zero, ib->Tensor(1, ib->GetDtype(num_zero))}), ib->ZerosLike(dout), dout});
  NodePtr non_zero_data = nullptr;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    non_zero_data = ib->Emit("Select", {is_zero, ib->Emit("OnesLike", {x}), x});
  } else {
    auto temp_var = ib->Emit("OnesLike", {ib->Cast(x, kFloat32)});
    non_zero_data = ib->Emit("Select", {is_zero, ib->Cast(temp_var, x_dtype_id), x});
  }
  auto non_zero_prod = ib->Emit("UnsortedSegmentProd", {non_zero_data, segment_ids, num_segments});
  auto zero_clipped_indices = ib->Emit("Maximum", {segment_ids, ib->ZerosLike(segment_ids)});
  auto gathered_prod = ib->Emit("Gather", {out, zero_clipped_indices, ib->Tensor(0, kInt64)});
  auto gathered_non_zero_prod = ib->Emit("Gather", {non_zero_prod, zero_clipped_indices, ib->Tensor(0, kInt64)});

  NodePtr prod_divided_by_x = nullptr;
  if (x_dtype_id == kNumberTypeUInt32 || x_dtype_id == kNumberTypeUInt64) {
    prod_divided_by_x = ib->Emit("RealDiv", {ib->Cast(gathered_prod, kFloat32), ib->Cast(x, kFloat32)});
  } else {
    prod_divided_by_x = ib->Emit("RealDiv", {gathered_prod, x});
  }
  auto partial_derivative = ib->Emit(
    "Select", {is_zero, gathered_non_zero_prod, ib->Cast(prod_divided_by_x, ib->GetDtype(gathered_non_zero_prod))});

  auto temp_outs = GatherDropNegatives(ib, grad, segment_ids, zero_clipped_indices, nullptr);
  MS_EXCEPTION_IF_CHECK_FAIL(temp_outs.size() > 0, "Outputs should not be empty.");
  auto gathered_grad = temp_outs[0];
  NodePtr dx = nullptr;
  if (x_dtype_id == kNumberTypeUInt32 || x_dtype_id == kNumberTypeUInt64) {
    auto temp_dx = ib->Emit("Mul", {ib->Cast(gathered_grad, kFloat32), ib->Cast(partial_derivative, kFloat32)});
    dx = ib->Cast(temp_dx, x_dtype);
  } else {
    dx = ib->Emit("Mul", {gathered_grad, partial_derivative});
  }

  return {dx, ib->ZerosLike(segment_ids), ib->ZerosLike(num_segments)};
});

REG_BPROP_BUILDER("SpaceToBatchND").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("BatchToSpaceND", {dout},
                     {{"block_shape", ib->GetAttr("block_shape")}, {"crops", ib->GetAttr("paddings")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchToSpaceND").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SpaceToBatchND", {dout},
                     {{"block_shape", ib->GetAttr("block_shape")}, {"paddings", ib->GetAttr("crops")}});
  return {dx};
});

REG_BPROP_BUILDER("BroadcastTo").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto broadcast_shape = ib->GetAttr<ShapeVector>("shape");

  auto x_shape = ib->GetShape(x);
  auto dout_shape = ib->GetShape(dout);
  if (x_shape == dout_shape) {
    return {dout};
  }

  auto tuple_out = BroadcastGradientArgs(broadcast_shape, x_shape);
  MS_EXCEPTION_IF_CHECK_FAIL(!tuple_out.empty(), "BroadcastGradientArgs out should not be empty!");
  auto reduction_axes = tuple_out[kIndex1];
  auto reduced_grad = ib->Emit("ReduceSum", {dout, ib->Value(reduction_axes)}, {{"keep_dims", MakeValue(true)}});
  auto dx = ib->Reshape(reduced_grad, x_shape);
  return {dx};
});

REG_BPROP_BUILDER("SpaceToDepth").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {
    ib->Emit("DepthToSpace", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"format", ib->GetAttr("format")}})};
});

REG_BPROP_BUILDER("DepthToSpace").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("SpaceToDepth", {dout}, {{"block_size", ib->GetAttr("block_size")}})};
});

REG_BPROP_BUILDER("ScatterMax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Emit("Gather", {dout, indices, ib->Tensor(0, kInt64)})};
});

REG_BPROP_BUILDER("ScatterMin").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Emit("Gather", {dout, indices, ib->Tensor(0, kInt64)})};
});

REG_BPROP_BUILDER("ScatterUpdate").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Emit("Gather", {dout, indices, ib->Tensor(0, kInt64)})};
});

REG_BPROP_BUILDER("Fills").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto value = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(value)};
});

REG_BPROP_BUILDER("Cast").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto t = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype = ib->GetDtype(x);
  auto dx = ib->Cast(dout, x_dtype);
  return {dx, ib->ZerosLike(t)};
});

REG_BPROP_BUILDER("ExpandDims").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shapex = ib->GetShape(x);
  return {ib->Reshape(dout, shapex), ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("Squeeze").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shapex = ib->GetShape(x);
  return {ib->Reshape(dout, shapex)};
});

REG_BPROP_BUILDER("Padding").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shp = ib->GetShape(x);
  std::vector<int64_t> begin;
  (void)begin.insert(begin.end(), shp.size(), 0);
  auto dx = ib->Emit("Slice", {dout, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(shp)});
  return {dx};
});

REG_BPROP_BUILDER("Transpose").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto perm = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp_perm = GetTupleIntFromValueNode(perm);
  std::vector<int64_t> new_perm;
  (void)std::transform(tmp_perm.begin(), tmp_perm.end(), std::back_inserter(new_perm),
                       [&tmp_perm](const int64_t v) { return v >= 0 ? v : v + tmp_perm.size(); });
  auto res_perm = InvertPermutation(new_perm);
  return {ib->Emit("Transpose", {dout, ib->Value<ShapeVector>(res_perm)}), ib->ZerosLike(perm)};
});

REG_BPROP_BUILDER("Slice").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto begin = ib->GetInput(kIndex1);
  auto size = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("SliceGrad", {dout, x, begin, size});
  return {dx, ib->ZerosLike(begin), ib->ZerosLike(size)};
});

REG_BPROP_BUILDER("Split").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Concat", {dout}, {{"axis", ib->GetAttr("axis")}});
  return {dx};
});

REG_BPROP_BUILDER("Tile").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto input_multiples = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shapex = ib->GetShape(x);
  auto multiples = GetTupleIntFromValueNode(input_multiples);
  auto r_shape = TileShape(multiples, shapex);
  auto axis = Range(0, static_cast<int64_t>(r_shape.size()), 2);
  auto dout_reshaped = ib->Reshape(dout, r_shape);
  auto dout_dtype = ib->GetDtype(dout_reshaped)->type_id();
  NodePtr dx;
  if (dout_dtype == kNumberTypeInt16 || dout_dtype == kNumberTypeInt32 || dout_dtype == kNumberTypeInt64) {
    dout_reshaped = ib->Cast(dout_reshaped, kFloat32);
    dx = ib->Emit("ReduceSum", {dout_reshaped, ib->Value<ShapeVector>(axis)}, {{"keep_dims", MakeValue(false)}});
    dx = ib->Cast(dx, dout_dtype);
  } else {
    dx = ib->Emit("ReduceSum", {dout_reshaped, ib->Value<ShapeVector>(axis)}, {{"keep_dims", MakeValue(false)}});
  }
  dx = ib->Reshape(dx, shapex);
  return {dx, ib->ZerosLike(input_multiples)};
});

REG_BPROP_BUILDER("Gather").SetBody([](const BpropIRBuilder *ib) -> NodePtrList { return BinopGatherCommon(ib); });

REG_BPROP_BUILDER("GatherV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList { return BinopGatherCommon(ib); });

REG_BPROP_BUILDER("Fill").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dtype = ib->GetInput(kIndex0);
  auto dims = ib->GetInput(kIndex1);
  auto x = ib->GetInput(kIndex2);
  return {ib->ZerosLike(dtype), ib->ZerosLike(dims), ib->ZerosLike(x)};
});

const auto diag_max_length = 200000000;
REG_BPROP_BUILDER("MatrixDiagV3").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto k = ib->GetInput(kIndex1);
  auto num_rows = ib->GetInput(kIndex2);
  auto num_cols = ib->GetInput(kIndex3);
  auto padding_value = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto part = ib->Emit("MatrixDiagPartV3", {dout, k, ib->Tensor(0, ib->GetDtype(dout))},
                       {{"align", ib->GetAttr("align")}, {"max_length", MakeValue<int64_t>(diag_max_length)}});
  return {part, ib->ZerosLike(k), ib->ZerosLike(num_rows), ib->ZerosLike(num_cols), ib->ZerosLike(padding_value)};
});

REG_BPROP_BUILDER("MatrixDiagPartV3").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto align = ib->GetAttr("align");
  auto x = ib->GetInput(kIndex0);
  auto k = ib->GetInput(kIndex1);
  auto padding_value = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shape = ib->GetShape(x);
  auto row = x_shape[x_shape.size() - 2];
  auto col = x_shape[x_shape.size() - 1];
  auto diag = ib->Emit("MatrixDiagV3",
                       {dout, k, ib->Tensor(row, kInt32), ib->Tensor(col, kInt32), ib->Tensor(0, ib->GetDtype(dout))},
                       {{"align", align}});
  return {diag, ib->ZerosLike(k), ib->ZerosLike(padding_value)};
});

REG_BPROP_BUILDER("MatrixSetDiagV3").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto align = ib->GetAttr("align");
  auto diagonal = ib->GetInput(kIndex1);
  auto k = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto diagonal_cal = ib->Emit("MatrixDiagPartV3", {dout, k, ib->Tensor(0, ib->GetDtype(dout))},
                               {{"align", align}, {"max_length", MakeValue<int64_t>(diag_max_length)}});
  auto diagonal_shape = ib->GetShape(diagonal);
  auto x_cal = ib->Emit("MatrixSetDiagV3", {dout, ib->Tensor(0, ib->GetDtype(dout)), k},
                        {{"align", align}, {"max_length", MakeValue<int64_t>(diag_max_length)}});
  return {x_cal, diagonal_cal, ib->ZerosLike(k)};
});

REG_BPROP_BUILDER("LogNormalReverse").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_data = ib->GetInput(kIndex0);
  return {ib->ZerosLike(input_data)};
});

REG_BPROP_BUILDER("Shape").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Rank").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("DynamicShape").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("TensorShape").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("DType").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("StridedSliceV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto begin = ib->GetInput(kIndex1);
  auto end = ib->GetInput(kIndex2);
  auto strides = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto x_shape = ib->Tensor(ib->GetShape(x));
  auto dx = ib->Emit("StridedSliceV2Grad", {x_shape, begin, end, strides, dout},
                     {{"begin_mask", ib->GetAttr("begin_mask")},
                      {"end_mask", ib->GetAttr("end_mask")},
                      {"ellipsis_mask", ib->GetAttr("ellipsis_mask")},
                      {"new_axis_mask", ib->GetAttr("new_axis_mask")},
                      {"shrink_axis_mask", ib->GetAttr("shrink_axis_mask")}});
  return {dx, ib->ZerosLike(begin), ib->ZerosLike(end), ib->ZerosLike(strides)};
});

REG_BPROP_BUILDER("MaskedFill").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_data = ib->GetInput(kIndex0);
  auto mask = ib->GetInput(kIndex1);
  auto value = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  mask = ib->Cast(mask, kFloat32);
  auto dinput = ib->Mul(dout, ib->Sub((ib->Tensor(1, ib->GetDtype(mask))), mask));
  auto dvalue = ib->Mul(dout, mask);
  auto bout = BinopGradCommon(ib, input_data, mask, dinput, dvalue);
  dvalue = ib->ReduceSum(bout[1]);
  dinput = ib->Cast(bout[0], ib->GetDtype(input_data));
  if (value->isa<ValueNode>()) {
    dvalue = ib->ZerosLike(value);
  } else {
    dvalue = ib->Cast(dvalue, ib->GetDtype(value));
  }
  return {dinput, ib->ZerosLike(mask), dvalue};
});

REG_BPROP_BUILDER("Coalesce").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex4);
  return {dout};
});

REG_BPROP_BUILDER("ConjugateTranspose").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto perm = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp_perm = GetTupleIntFromValueNode(perm);
  std::vector<int64_t> new_perm;
  (void)std::transform(tmp_perm.begin(), tmp_perm.end(), std::back_inserter(new_perm),
                       [&tmp_perm](const int64_t v) { return v >= 0 ? v : v + tmp_perm.size(); });
  auto res_perm = InvertPermutation(new_perm);
  return {ib->Emit("ConjugateTranspose", {dout, ib->Value<ShapeVector>(res_perm)}), ib->ZerosLike(perm)};
});

REG_BPROP_BUILDER("Triu").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto diagonal = GetValue<int64_t>(ib->GetAttr("diagonal"));
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Triu", {dout}, {{"diagonal", MakeValue(diagonal)}});
  return {dx};
});

REG_BPROP_BUILDER("CheckNumerics").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("CheckNumerics", {dout})};
});

REG_BPROP_BUILDER("IdentityN").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("ResizeNearestNeighborV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto align_corners = GetValue<bool>(ib->GetAttr("align_corners"));
  auto half_pixel_centers = GetValue<bool>(ib->GetAttr("half_pixel_centers"));
  auto data_format = GetValue<std::string>(ib->GetAttr("format"));
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  auto x_shape = ib->GetShape(x);
  ShapeVector grad_in_size(x_shape.begin() + 1, x_shape.begin() + 3);
  if (data_format == "NCHW") {
    ShapeVector tmp(x_shape.begin() + 2, x_shape.begin() + 4);
    grad_in_size = tmp;
  }
  auto dx = ib->Emit("ResizeNearestNeighborV2Grad", {dout, ib->Tensor(grad_in_size, kInt32)},
                     {{"align_corners", MakeValue(align_corners)},
                      {"half_pixel_centers", MakeValue(half_pixel_centers)},
                      {"format", MakeValue(data_format)}});
  return {dx, ib->ZerosLike(ib->Value<ShapeVector>(grad_in_size))};
});

REG_BPROP_BUILDER("Tril").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto diagonal = GetValue<int64_t>(ib->GetAttr("diagonal"));
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Tril", {dout}, {{"diagonal", MakeValue(diagonal)}});
  return {dx};
});

REG_BPROP_BUILDER("SegmentSum").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto segment_ids = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dout_type = ib->GetDtype(dout);
  std::set<TypePtr> type_list = {kInt8, kInt16, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
  if (CheckType(dout_type, type_list)) {
    dout = ib->Cast(dout, kInt32);
  }
  if (dout_type->type_id() == TypeId::kNumberTypeFloat64) {
    dout = ib->Cast(dout, kFloat32);
  }
  return {ib->Cast(ib->Emit("Gather", {dout, segment_ids, ib->Tensor(0)}), dout_type), ib->ZerosLike(segment_ids)};
});

REG_BPROP_BUILDER("EmbeddingLookup").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto offset = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shp = ib->GetShape(x);
  auto offset_v = GetIntFromValueNode(offset);
  auto new_indices = ib->Sub(indices, ib->Tensor(offset_v, ib->GetDtype(indices)));
  auto indices_size = ib->GetSize(new_indices);
  ShapeVector new_indices_shape;
  ShapeVector x_shp_tail;
  ShapeVector actual_dout_shape;
  if (indices_size > 0) {
    new_indices_shape.push_back(indices_size);
    new_indices = ib->Reshape(new_indices, new_indices_shape);
  }
  int64_t x_rank = static_cast<int64_t>(x_shp.size());
  auto start1 = x_rank <= 1 ? x_shp.end() : x_shp.begin() + 1;
  (void)std::copy(start1, x_shp.end(), std::back_inserter(x_shp_tail));
  (void)std::copy(new_indices_shape.begin(), new_indices_shape.end(), std::back_inserter(actual_dout_shape));
  (void)std::copy(x_shp_tail.begin(), x_shp_tail.end(), std::back_inserter(actual_dout_shape));
  auto actual_dout = ib->Reshape(dout, actual_dout_shape);
  return {ib->MakeTuple({new_indices, actual_dout, ib->Value<ShapeVector>(x_shp)}), ib->ZerosLike(indices),
          ib->ZerosLike(offset)};
});

REG_BPROP_BUILDER("MaskedSelect").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto mask = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("MaskedSelectGrad", {x, mask, dout});
  return {dx, ib->ZerosLike(mask)};
});

REG_BPROP_BUILDER("SplitV").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto split_dim = GetValue<int64_t>(ib->GetAttr("split_dim"));
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Concat", {dout}, {{"axis", MakeValue(split_dim)}});
  return {dx};
});

REG_BPROP_BUILDER("Col2Im").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto ksizes = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
  auto dilations = GetValue<std::vector<int64_t>>(ib->GetAttr("dilation"));
  auto strides = GetValue<std::vector<int64_t>>(ib->GetAttr("stride"));
  auto pads = GetValue<std::vector<int64_t>>(ib->GetAttr("padding"));
  auto output_size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("Im2Col", {dout},
                     {{"ksizes", MakeValue(ksizes)},
                      {"dilations", MakeValue(dilations)},
                      {"strides", MakeValue(strides)},
                      {"padding_mode", MakeValue("CALCULATED")},
                      {"pads", MakeValue(pads)}});
  return {dx, ib->ZerosLike(output_size)};
});

REG_BPROP_BUILDER("ExtractVolumePatches").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto ksize = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
  auto ksize_d = ksize.at(2);
  auto ksize_h = ksize.at(3);
  auto ksize_w = ksize.at(4);
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  auto x_n = x_shape.at(0);
  auto x_c = x_shape.at(1);
  auto x_d = x_shape.at(2);
  auto x_h = x_shape.at(3);
  auto x_w = x_shape.at(4);
  auto x_indices_num = 1 + ((x_d * x_h) * x_w);
  auto x_idx = ib->Tensor(Range(1, x_indices_num), kFloat16);
  x_idx = ib->Reshape(x_idx, {1, 1, x_d, x_h, x_w});
  auto x_idx_patched = ib->Emit("ExtractVolumePatches", {x_idx},
                                {{"kernel_size", ib->GetAttr("kernel_size")},
                                 {"strides", ib->GetAttr("strides")},
                                 {"padding", ib->GetAttr("padding")}});
  x_idx_patched = ib->Emit("Transpose", {x_idx_patched, ib->Value<ShapeVector>({0, 2, 3, 4, 1})});
  x_idx_patched = ib->Cast(x_idx_patched, kInt32);
  auto out_shape = ib->GetShape(out);
  auto out_d = out_shape.at(2);
  auto out_h = out_shape.at(3);
  auto out_w = out_shape.at(4);
  auto out_indices_num = ((((out_d * out_h) * out_w) * ksize_d) * ksize_h) * ksize_w;
  auto out_idx = ib->Tensor(Range(0, out_indices_num), kInt32);
  out_idx = ib->Reshape(out_idx, {1, out_d, out_h, out_w, (ksize_d * ksize_h) * ksize_w});
  auto idx_tensor = ib->Emit("Concat",
                             {ib->MakeTuple({ib->Emit("ExpandDims", {x_idx_patched, ib->Value<int64_t>(-1)}),
                                             ib->Emit("ExpandDims", {out_idx, ib->Value<int64_t>(-1)})})},
                             {{"axis", MakeValue<int64_t>(-1)}});
  auto idx_map = ib->Reshape(idx_tensor, {-1, 2});
  std::vector<int64_t> sp_shape = {x_indices_num, out_indices_num};
  auto sp_mat_full = ib->Emit(
    "ScatterNd", {idx_map,
                  ib->Emit("Fill", {ib->EmitValue(ib->GetDtype(dout)), ib->Value<ShapeVector>({out_indices_num}),
                                    ib->Tensor(1, ib->GetDtype(x))}),
                  ib->Value<ShapeVector>(sp_shape)});
  auto sp_tensor = ib->Emit("Slice", {sp_mat_full, ib->Value<ShapeVector>({1, 0}),
                                      ib->Value<ShapeVector>({x_indices_num - 1, out_indices_num})});
  auto grad = ib->Emit("Transpose", {dout, ib->Value<ShapeVector>({0, 2, 3, 4, 1})});
  grad = ib->Reshape(grad, {x_n, out_d, out_h, out_w, ksize_d, ksize_h, ksize_w, x_c});
  auto grad_expended = ib->Emit("Transpose", {grad, ib->Value<ShapeVector>({1, 2, 3, 4, 5, 6, 0, 7})});
  auto grad_flat = ib->Reshape(grad_expended, {-1, x_n * x_c});
  auto jac = ib->MatMul(sp_tensor, grad_flat, false, false);
  auto dx = ib->Reshape(jac, {x_d, x_h, x_w, x_n, x_c});
  dx = ib->Emit("Transpose", {dx, ib->Value<ShapeVector>({3, 4, 0, 1, 2})});
  return {dx};
});

REG_BPROP_BUILDER("AffineGrid").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto align_corners = GetValue<bool>(ib->GetAttr("align_corners"));
  auto theta = ib->GetInput(kIndex0);
  auto output_size = GetAxisValue(ib->GetInput(kIndex1));
  auto dout = ib->GetInput(kIndex3);
  auto dtype = ib->GetDtype(theta);

  auto start = ib->Tensor(-1, dtype);
  auto stop = ib->Tensor(1, dtype);
  auto zero = ib->Tensor(0, dtype);
  auto perm1 = ib->Value<ShapeVector>({1, 0});
  auto perm2 = ib->Value<ShapeVector>({0, 2, 1});
  if (output_size.size() == 5) {
    const auto n_value = output_size[kIndex0];
    const auto d_value = output_size[kIndex2];
    const auto h_value = output_size[kIndex3];
    const auto w_value = output_size[kIndex4];
    auto vecx = (w_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(w_value)}) : zero;
    auto vecy = (h_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(h_value)}) : zero;
    auto vecz = (d_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(d_value)}) : zero;
    if (!align_corners) {
      vecx = (vecx * ib->Tensor(w_value - 1, dtype)) / ib->Tensor(w_value, dtype);
      vecy = (vecy * ib->Tensor(h_value - 1, dtype)) / ib->Tensor(h_value, dtype);
      vecz = (vecz * ib->Tensor(d_value - 1, dtype)) / ib->Tensor(d_value, dtype);
    }
    auto out = (h_value * d_value != 1) ? ib->Tile(vecx, {h_value * d_value, 1}) : vecx;
    auto one = ib->Reshape(out, {h_value * w_value * d_value, 1});
    out = (w_value == 1) ? ib->ExpandDims(vecy, 0) : ib->Tile(vecy, {w_value, 1});
    out = ib->Emit("Transpose", {out, perm1});
    if (d_value != 1) {
      out = ib->Tile(out, {d_value, 1});
    }
    auto two = ib->Reshape(out, {h_value * w_value * d_value, 1});
    out = (w_value * h_value != 1) ? ib->Tile(vecz, {w_value * h_value, 1}) : ib->ExpandDims(vecz, 0);
    out = ib->Emit("Transpose", {out, perm1});
    auto tre = ib->Reshape(out, {h_value * w_value * d_value, 1});
    auto fou = ib->Emit("OnesLike", {tre});
    auto output = ib->Concat({one, two, tre, fou}, 1);
    output = ib->Emit("Transpose", {output, perm1});
    if (n_value != 1) {
      output = ib->Tile(output, {n_value, 1});
    }
    output = ib->Reshape(output, {n_value, 4, h_value * w_value * d_value});
    dout = ib->Reshape(dout, {n_value, d_value * h_value * w_value, 3});
    auto dtheta = ib->BatchMatMul(output, dout);
    dtheta = ib->Emit("Transpose", {dtheta, perm2});
    return {dtheta, tre};
  }
  if (output_size.size() == 4) {
    auto x_shape = ib->GetShape(dout);
    const auto n_value = x_shape[kIndex0];
    const auto h_value = x_shape[kIndex1];
    const auto w_value = x_shape[kIndex2];
    auto vecx = (w_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(w_value)}) : zero;
    auto vecy = (h_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(h_value)}) : zero;
    if (!align_corners) {
      vecx = (vecx * ib->Tensor(w_value - 1, dtype)) / ib->Tensor(w_value, dtype);
      vecy = (vecy * ib->Tensor(h_value - 1, dtype)) / ib->Tensor(h_value, dtype);
    }
    auto out = (h_value != 1) ? ib->Tile(vecx, {h_value, 1}) : vecx;
    auto one = ib->Reshape(out, {h_value * w_value, 1});
    out = (w_value == 1) ? ib->ExpandDims(vecy, 0) : ib->Tile(vecy, {w_value, 1});
    out = ib->Emit("Transpose", {out, perm1});
    auto two = ib->Reshape(out, {h_value * w_value, 1});
    auto tre = ib->Emit("OnesLike", {two});
    auto output = ib->Concat({one, two, tre}, 1);
    output = ib->Emit("Transpose", {output, perm1});
    output = ib->Tile(output, {n_value, 1});
    output = ib->Reshape(output, {n_value, 3, h_value * w_value});
    dout = ib->Reshape(dout, {n_value, h_value * w_value, 2});
    auto dtheta = ib->BatchMatMul(output, dout);
    dtheta = ib->Emit("Transpose", {dtheta, perm2});
    return {dtheta, tre};
  }
  MS_LOG(EXCEPTION) << "For op[" << ib->name() << "], the length of output_size should be 4 or 5, but got "
                    << output_size.size();
  return {};
});

NodePtrList SegmentMinOrMaxGrad(const BpropIRBuilder *ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto output = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto input_x_type = ib->GetDtype(input_x);
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    input_x = ib->Cast(input_x, kFloat32);
    output = ib->Cast(output, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  auto zero_value = ib->Value<int64_t>(0);
  auto gathered_outputs = ib->Emit("Gather", {output, segment_ids, zero_value});
  auto is_selected = ib->Equal(input_x, gathered_outputs);
  const int64_t max_len = 1000000;
  auto num_selected =
    ib->Emit("SegmentSum", {ib->Cast(is_selected, kFloat32), segment_ids}, {{"max_length", MakeValue(max_len)}});
  auto weighted_grads = ib->Div(dout, num_selected);
  auto gathered_grads = ib->Emit("Gather", {weighted_grads, segment_ids, zero_value});
  auto dx = ib->Select(is_selected, gathered_grads, ib->ZerosLike(input_x));
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    dx = ib->Cast(dx, input_x_type);
  }
  return {dx, ib->ZerosLike(segment_ids)};
}
REG_BPROP_BUILDER("SegmentMax").SetBody(SegmentMinOrMaxGrad);
REG_BPROP_BUILDER("SegmentMin").SetBody(SegmentMinOrMaxGrad);

REG_BPROP_BUILDER("TensorScatterElements").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto axis = ib->GetAttr("axis");
  auto x_grad = ib->Emit("TensorScatterElements", {dout, indices, ib->ZerosLike(update)},
                         {{"axis", axis}, {"reduction", ib->GetAttr("reduction")}});
  auto update_grad = ib->Emit("GatherD", {dout, ib->EmitValue(axis), indices});
  return {x_grad, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("ScatterAddWithAxis").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto axis = ib->GetAttr("axis");
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto dout_shape = ib->GetShape(dout);
  auto index_shape = ib->GetShape(indices);
  NodePtr update_grad = nullptr;
  if (dout_shape != index_shape) {
    ShapeVector slice_list(dout_shape.size(), 0);
    std::vector<ShapeVector> pad_list;
    pad_list.reserve(dout_shape.size());
    for (size_t i = 0; i < dout_shape.size(); i++) {
      (void)pad_list.emplace_back(ShapeVector{0, dout_shape[i] - index_shape[i]});
    }
    auto out_index = ib->Emit("Pad", {indices}, {{"paddings", MakeValue(pad_list)}});
    auto out_gather = ib->Emit("GatherD", {dout, ib->EmitValue(axis), out_index});
    update_grad = ib->Emit("Slice", {out_gather, ib->Value(slice_list), ib->Value(index_shape)});
  } else {
    update_grad = ib->Emit("GatherD", {dout, ib->EmitValue(axis), indices});
  }
  return {dout, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("Expand").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  auto dout_shape = ib->GetShape(dout);
  if (dout_shape.empty()) {
    return {ib->ReduceSum(dout), ib->ZerosLike(dout)};
  }
  auto x_shape = ib->GetShape(x);
  auto leading_dims = dout_shape.size() - x_shape.size();
  auto reduce_dims = Range(SizeToLong(leading_dims));
  for (size_t j = leading_dims; j < dout_shape.size(); ++j) {
    if (x_shape[j - leading_dims] == 1 && dout_shape[j] != 1) {
      reduce_dims.push_back(j);
    }
  }
  if (!reduce_dims.empty()) {
    dout = ib->ReduceSum(dout, reduce_dims, true);
  }
  auto dx = leading_dims > 0 ? ib->Reshape(dout, x_shape) : dout;
  return {dx, ib->ZerosLike(dout)};
});

REG_BPROP_BUILDER("SegmentMean").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto input_x_type = ib->GetDtype(input_x);
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    input_x = ib->Cast(input_x, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  auto x_rank = ib->GetShape(input_x).size();
  auto ones_shape = ib->GetShape(segment_ids) + ShapeVector(x_rank - 1, 1LL);
  auto ones = ib->Emit("Fill", {ib->EmitValue(kFloat32), ib->Value(ones_shape), ib->Tensor(1, kFloat32)});
  const int64_t max_len = 1000000;
  auto scaled_grad = ib->Div(dout, ib->Emit("SegmentSum", {ones, segment_ids}, {{"max_length", MakeValue(max_len)}}));
  auto dx = ib->Emit("Gather", {scaled_grad, segment_ids, ib->Value<int64_t>(0)});
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    dx = ib->Cast(dx, input_x_type);
  }
  return {dx, ib->ZerosLike(segment_ids)};
});
}  // namespace mindspore::expander::bprop
