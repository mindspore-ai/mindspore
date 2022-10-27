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
}  // namespace mindspore::expander::bprop
