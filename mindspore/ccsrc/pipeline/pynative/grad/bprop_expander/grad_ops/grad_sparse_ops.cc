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

#include <tuple>
#include "pipeline/pynative/grad/bprop_expander/bprop_irbuilder.h"
#include "pipeline/pynative/grad/bprop_expander/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore::expander::bprop {
namespace {
std::tuple<NodePtr, NodePtr> PromoteTensor(const BpropIRBuilder *ib, const NodePtr &t1, const NodePtr &t2) {
  MS_EXCEPTION_IF_NULL(ib);
  MS_EXCEPTION_IF_NULL(t1);
  MS_EXCEPTION_IF_NULL(t2);
  auto t1_type = ib->GetDtype(t1);
  auto t2_type = ib->GetDtype(t2);
  MS_EXCEPTION_IF_NULL(t1_type);
  MS_EXCEPTION_IF_NULL(t2_type);
  auto t1_type_id = t1_type->type_id();
  auto t2_type_id = t2_type->type_id();
  auto dtype = PromoteBinaryDtype(t1_type_id, t2_type_id);
  NodePtr t1_new = t1;
  NodePtr t2_new = t2;
  if (t1_type_id != dtype) {
    t1_new = ib->Cast(t1, dtype);
  }
  if (t2_type_id != dtype) {
    t2_new = ib->Cast(t2, dtype);
  }
  return std::make_tuple(t1_new, t2_new);
}

NodePtr CsrMulDiv(const BpropIRBuilder *ib, const NodePtr &indptr, const NodePtr &indices, const NodePtr &values,
                  const NodePtr &shape, const NodePtr &y, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(y);
  NodePtr new_y = y;
  if (y->isa<ValueNode>()) {
    auto value_node = y->get<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto v = value_node->value();
    MS_EXCEPTION_IF_NULL(v);
    // isinstance(y, (int, float, bool))
    if (v->isa<Int64Imm>()) {
      new_y = ib->Tensor(GetValue<int64_t>(v));
    } else if (v->isa<FP32Imm>()) {
      new_y = ib->Tensor(GetValue<float>(v));
    } else if (v->isa<FP64Imm>()) {
      new_y = ib->Tensor(GetValue<double>(v));
    } else if (v->isa<BoolImm>()) {
      new_y = ib->Tensor(GetValue<bool>(v));
    }
  }
  if (ib->GetSize(new_y) == 1 && ib->GetShape(new_y).size() <= kDim2) {
    // y is scalar
    if (op_name == "CSRMul") {
      return ib->Reshape(ib->Mul(values, new_y), ib->GetShape(values));
    }
    return ib->Reshape(ib->RealDiv(values, new_y), ib->GetShape(values));
  }
  NodePtr new_values = nullptr;
  std::tie(new_values, new_y) = PromoteTensor(ib, values, y);
  return ib->Emit(op_name, {indptr, indices, new_values, shape, new_y});
}

ShapeVector ShapeSlice(const ShapeVector &sh, size_t start, size_t end) {
  ShapeVector res;
  auto real_end = sh.size() > end ? end : sh.size();
  for (size_t i = start; i < real_end; ++i) {
    res.push_back(sh[i]);
  }
  return res;
}

ShapeVector InferOutShape(const ShapeVector &sh1, const ShapeVector &sh2) {
  ShapeVector res;
  if (sh1.size() > sh2.size()) {
    for (size_t i = 0; i < sh1.size() - sh2.size(); ++i) {
      res.push_back(sh1[i]);
    }
  } else if (sh1.size() < sh2.size()) {
    for (size_t i = 0; i < sh2.size() - sh1.size(); ++i) {
      res.push_back(sh2[i]);
    }
  }
  auto n = sh1.size() > sh2.size() ? sh2.size() : sh1.size();
  for (size_t i = 0; i < n; ++i) {
    auto a = sh1[sh1.size() - n + i];
    auto b = sh2[sh2.size() - n + i];
    if (a == 1) {
      res.push_back(b);
    } else if (b == 1 || a == b) {
      res.push_back(a);
    } else {
      MS_EXCEPTION(ValueError) << "shape1 and shape2 can not broadcast: " << sh1 << " vs " << sh2;
    }
  }
  return res;
}

NodePtrList CommonSparseSegmentBprop(const BpropIRBuilder *ib, const std::string &grad_op, bool with_segments) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto segment_ids = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(with_segments ? kIndex5 : kIndex4);
  auto shape_x = ib->GetShape(x);
  auto output_dim0 = ib->Tensor(shape_x[0], kInt32);
  if (ib->GetDtypeId(indices) != kNumberTypeInt32) {
    indices = ib->Cast(indices, kInt32);
  }
  segment_ids = ib->Cast(segment_ids, kInt32);
  auto dx = ib->Emit(grad_op, {dout, indices, segment_ids, output_dim0});
  NodePtrList result = {dx, ib->ZerosLike(indices), ib->ZerosLike(segment_ids)};
  if (with_segments) {
    result.emplace_back(ib->ZerosLike(ib->GetInput(kIndex3)));
  }
  return result;
}

NodePtrList CommonSparseSegmentBpropDefault(const BpropIRBuilder *ib, bool with_segments) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto segment_ids = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(with_segments ? kIndex5 : kIndex4);
  auto shape_x = ib->GetShape(x);
  auto output_dim0 = ib->Cast(ib->Tensor(shape_x[0]), kInt32);
  segment_ids = ib->Cast(segment_ids, kInt32);
  auto input0 = ib->Emit("Gather", {dout, segment_ids, ib->Tensor(0, kInt64)});
  input0 = ib->Cast(input0, kFloat32);
  indices = ib->Cast(indices, kInt32);
  auto dx = ib->Emit("UnsortedSegmentSum", {input0, indices, output_dim0});
  dx = ib->Cast(dx, ib->GetDtype(dout));
  NodePtrList result = {dx, ib->ZerosLike(indices), ib->ZerosLike(segment_ids)};
  if (with_segments) {
    result.emplace_back(ib->ZerosLike(ib->GetInput(kIndex3)));
  }
  return result;
}

NodePtrList BpropSparseDenseCwiseCommon(const BpropIRBuilder *ib, const std::string &op_name) {
  auto x1_indices = ib->GetInput(kIndex0);
  auto x1_values = ib->GetInput(kIndex1);
  auto x1_shape = ib->GetInput(kIndex2);
  auto x2 = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto x2_shape = ib->GetShape(x2);
  auto scaling = ib->RealDiv(x1_shape, ib->Tensor(x2_shape));
  auto scaled_indices = ib->RealDiv(x1_indices, scaling);
  std::vector<int64_t> begin = {0, ib->GetSize(x1_shape) - SizeToLong(x2_shape.size())};
  std::vector<int64_t> size = {-1, -1};
  scaled_indices = ib->Cast(ib->Emit("Slice", {scaled_indices, ib->Value(begin), ib->Value(size)}), kInt64);
  auto dense_vals = ib->Emit("GatherNd", {x2, scaled_indices});
  NodePtr dx1 = nullptr;
  NodePtr dx2_val = nullptr;
  if (op_name == "SparseDenseCwiseMul") {
    dx1 = ib->Mul(dout, dense_vals);
    dx2_val = ib->Mul(dout, x1_values);
  } else {
    dx1 = ib->RealDiv(dout, dense_vals);
    auto dense_vals_2 = ib->Mul(dense_vals, dense_vals);
    auto w = ib->Neg(ib->RealDiv(x1_values, dense_vals_2));
    dx2_val = ib->Mul(dout, w);
  }
  auto dx2 = ib->Emit("SparseTensorDenseAdd", {scaled_indices, dx2_val, ib->Tensor(x2_shape), ib->ZerosLike(x2)});
  NodePtrList d_all = {ib->ZerosLike(x1_indices), dx1, ib->ZerosLike(x1_shape), dx2};
  return d_all;
}
}  // namespace
REG_BPROP_BUILDERS_BEGIN(GradSparseOps)
REG_BPROP_BUILDER("SparseToDense").SetUnusedInputs({i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex0);
  auto dense_shape = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {ib->ZerosLike(indices), ib->Emit("GatherNd", {dout, indices}), ib->ZerosLike(dense_shape)};
});

REG_BPROP_BUILDER("SparseToDenseV2").SetUnusedInputs({i1, i2, i3, i4}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex0);
  auto output_shape = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex5);
  auto sparse_values_grad = ib->Emit("GatherNd", {dout, indices});
  auto default_value_grad = ib->ReduceSum(dout) - ib->ReduceSum(sparse_values_grad);
  return {ib->ZerosLike(indices), ib->ZerosLike(output_shape), sparse_values_grad, default_value_grad};
});

REG_BPROP_BUILDER("SparseTensorDenseMatmul").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  auto adj_s = ib->GetAttr<bool>("adjoint_st");
  auto adj_d = ib->GetAttr<bool>("adjoint_dt");
  auto indices = ib->GetInput(kIndex0);
  auto values = ib->GetInput(kIndex1);
  auto dense_shape = ib->GetInput(kIndex2);
  auto dense = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto dense_grad = ib->Emit("SparseTensorDenseMatmul", {indices, values, dense_shape, dout},
                             {{"adjoint_st", MakeValue(!adj_s)}, {"adjoint_dt", MakeValue(false)}});
  ShapeVector perm = {1, 0};
  if (adj_d) {
    dense_grad = ib->Transpose(dense_grad, perm);
  }
  bool is_half = false;
  auto dense_type = ib->GetDtype(dense);
  MS_EXCEPTION_IF_NULL(dense_type);
  auto dense_type_id = dense_type->type_id();
  if (dense_type_id == kNumberTypeFloat16) {
    dense = ib->Cast(dense, kFloat32);
    dout = ib->Cast(dout, kFloat32);
    is_half = true;
  }
  constexpr int64_t axis = -1;
  constexpr int64_t output_num = 2;
  auto split_indices = ib->Emit(
    kSplitOpName, {indices},
    {{kAttrAxis, MakeValue(axis)}, {kAttrOutputNum, MakeValue(output_num)}, {"num_split", MakeValue(output_num)}});
  auto rows = ib->ReduceSum(ib->TupleGetItem(split_indices, kIndex0), {axis});
  auto cols = ib->ReduceSum(ib->TupleGetItem(split_indices, kIndex1), {axis});
  auto zero = ib->Value<int64_t>(0);
  NodePtr parts_a = nullptr;
  if (adj_s) {
    parts_a = ib->Emit("Gather", {dout, cols, zero});
  } else {
    parts_a = ib->Emit("Gather", {dout, rows, zero});
  }
  NodePtr tmp1 = adj_d ? ib->Transpose(dense, perm) : dense;
  NodePtr tmp2 = adj_s ? rows : cols;
  auto parts_b = ib->Emit("Gather", {tmp1, tmp2, zero});
  auto values_grad = ib->ReduceSum(parts_a * parts_b, {axis});
  if (is_half) {
    values_grad = ib->Cast(values_grad, kFloat16);
  }
  return {ib->ZerosLike(indices), values_grad, ib->ZerosLike(dense_shape), dense_grad};
});

REG_BPROP_BUILDER("SparseAdd").SetUnusedInputs({i1, i2, i4, i5, i6}).SetBody(BODYFUNC(ib) {
  auto x1_indices = ib->GetInput(kIndex0);
  auto x1_values = ib->GetInput(kIndex1);
  auto x1_shape = ib->GetInput(kIndex2);
  auto x2_indices = ib->GetInput(kIndex3);
  auto x2_values = ib->GetInput(kIndex4);
  auto x2_shape = ib->GetInput(kIndex5);
  auto thresh = ib->GetInput(kIndex6);
  auto out = ib->GetInput(kIndex7);
  auto dout = ib->GetInput(kIndex8);
  auto tmp = ib->Emit("SparseAddGrad", {ib->TupleGetItem(dout, 1), x1_indices, x2_indices, ib->TupleGetItem(out, 0)});
  auto dx1 = ib->TupleGetItem(tmp, 0);
  auto dx2 = ib->TupleGetItem(tmp, 1);
  auto ret0 = ib->ZerosLike(x1_indices);
  auto ret1 = ib->Reshape(dx1, ib->GetShape(x1_values));
  auto ret2 = ib->ZerosLike(x1_shape);
  auto ret3 = ib->ZerosLike(x2_indices);
  auto ret4 = ib->Reshape(dx2, ib->GetShape(x2_values));
  auto ret5 = ib->ZerosLike(x2_shape);
  auto ret6 = ib->ZerosLike(thresh);
  return {ret0, ret1, ret2, ret3, ret4, ret5, ret6};
});

REG_BPROP_BUILDER("CSRReduceSum").SetUnusedInputs({i2, i5}).SetBody(BODYFUNC(ib) {
  auto indptr = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto shape = ib->GetInput(kIndex3);
  auto axis = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto shape_vec = GetIntList(shape);
  auto output_shape_kept_dims = ReduceShape(shape_vec, GetIntList(axis));
  auto tile_scaling = TupleDiv(shape_vec, output_shape_kept_dims);
  auto values_grad_dense = ib->Tile(ib->Reshape(dout, output_shape_kept_dims), tile_scaling);
  auto values_grad = ib->Emit("CSRGather", {indptr, indices, values_grad_dense, shape});
  return {ib->ZerosLike(indptr), ib->ZerosLike(indices), values_grad, ib->ZerosLike(ib->Value<int64_t>(0)),
          ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("CSRMV").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  auto indptr = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto values = ib->GetInput(kIndex2);
  auto dense_shape = ib->GetInput(kIndex3);
  auto dense = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto indices_shape = ib->GetShape(indices);
  auto rows = ib->Emit("CSR2COO", {indptr, ib->Value(indices_shape.at(0))});
  auto idx_dtype = ib->GetDtype(rows);
  constexpr int64_t axis = -1;
  auto sort_res =
    ib->Emit("Sort", {ib->Cast(indices, kFloat32)}, {{kAttrAxis, MakeValue(axis)}, {"descending", MakeValue(false)}});
  auto rows_transposed = ib->Cast(ib->TupleGetItem(sort_res, 0), idx_dtype);
  auto cols_indexing = ib->TupleGetItem(sort_res, 1);
  auto zero = ib->Value<int64_t>(0);
  auto cols_transposed = ib->Emit("Gather", {rows, cols_indexing, zero});
  auto values_transposed = ib->Emit("Gather", {values, cols_indexing, zero});
  auto dense_shape_vec = GetIntList(dense_shape);
  auto indptr_transposed = ib->Emit("COO2CSR", {rows_transposed, ib->Value(dense_shape_vec.at(1))});
  NodePtr t1 = nullptr;
  NodePtr t2 = nullptr;
  std::tie(t1, t2) = PromoteTensor(ib, values_transposed, dout);
  ShapeVector sh{dense_shape_vec.at(1), dense_shape_vec.at(0)};
  auto dense_grad = ib->Emit("CSRMV", {indptr_transposed, cols_transposed, t1, ib->Value(sh), t2});
  auto parts_a = ib->Emit("Gather", {dout, rows, zero});
  auto parts_b = ib->Emit("Gather", {dense, indices, zero});
  auto values_grad = ib->ReduceSum(parts_a * parts_b, {1});
  return {ib->ZerosLike(indptr), ib->ZerosLike(indices), values_grad, ib->ZerosLike(zero), dense_grad};
});

REG_BPROP_BUILDER("CSRMul").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  auto indptr = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto values = ib->GetInput(kIndex2);
  auto shape = ib->GetInput(kIndex3);
  auto dense = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto csr_tensor_grad_value = CsrMulDiv(ib, indptr, indices, dout, shape, dense, "CSRMul");
  auto dense_grad_value = ib->Mul(dout, values);
  auto dense_shape = ib->GetShape(dense);
  if (dense_shape.size() == 1 || (dense_shape.size() > 1 && dense_shape[0] == 1)) {
    MS_EXCEPTION(ValueError)
      << "Backpropagation for CSRMul with broadcast for the first dimension is not supported! Use `*` instead";
  }
  NodePtr dense_grad = nullptr;
  if (dense_shape.size() > 1 && dense_shape[1] == 1) {
    dense_grad =
      ib->Emit("CSRReduceSum", {indptr, indices, dense_grad_value, shape, ib->Value(static_cast<int64_t>(1))});
  } else {
    auto row = ib->Emit("CSR2COO", {indptr, ib->Value(ib->GetShape(indices).at(0))});
    auto coo_idx = ib->Emit("Stack", {ib->MakeTuple({row, indices})}, {{"axis", MakeValue(static_cast<int64_t>(-1))}});
    dense_grad = ib->Emit("TensorScatterUpdate", {ib->ZerosLike(dense), coo_idx, dense_grad_value});
  }
  return {ib->ZerosLike(indptr), ib->ZerosLike(indices), csr_tensor_grad_value, ib->ZerosLike(ib->Value<int64_t>(0)),
          dense_grad};
});

REG_BPROP_BUILDER("CSRDiv").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto indptr = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto shape_node = ib->GetInput(kIndex3);
  auto shape = GetIntList(shape_node);
  auto dense = ib->GetInput(kIndex4);
  auto out = ib->GetInput(kIndex5);
  auto dout = ib->GetInput(kIndex6);
  auto dense_shape = ib->GetShape(dense);
  constexpr size_t batch_dim_csr_start = 2;
  int64_t batch_dim_dense_start_i =
    SizeToLong(dense_shape.size()) - SizeToLong(shape.size()) + SizeToLong(batch_dim_csr_start);
  if (batch_dim_dense_start_i < 0) {
    batch_dim_dense_start_i = 0;
  }
  auto batch_dim_dense_start = LongToSize(batch_dim_dense_start_i);
  ShapeVector shape1 = ShapeSlice(shape, 0, batch_dim_csr_start);
  ShapeVector shape2 = ShapeSlice(shape, batch_dim_csr_start, shape.size());
  ShapeVector shape3 = ShapeSlice(shape, batch_dim_dense_start, shape.size());
  ShapeVector dense_shape1 = ShapeSlice(dense_shape, 0, batch_dim_dense_start);
  auto feature_dim = InferOutShape(shape1, dense_shape1);
  auto shape_x = feature_dim + shape2;
  auto shape_y = feature_dim + shape3;
  auto tuple_out = BroadcastGradientArgs(shape_x, shape_y);
  auto csr_div_res = CsrMulDiv(ib, indptr, indices, dout, shape_node, dense, "CSRDiv");
  NodePtr csr_tensor_grad_value = csr_div_res;
  if (!tuple_out[0].empty()) {
    csr_tensor_grad_value = ib->ReduceSum(csr_div_res, tuple_out[0], true);
  }
  auto dense_grad_value = ib->Neg(ib->Mul(out, csr_tensor_grad_value));
  if (dense_shape.size() == 1 || (dense_shape.size() > 1 && dense_shape[0] == 1)) {
    MS_LOG(EXCEPTION)
      << "Backpropagation for CSRDiv with broadcast for the first dimension is not supported! Use `/` instead";
  }
  NodePtr dense_grad = nullptr;
  if (!tuple_out[1].empty()) {
    dense_grad = ib->ReduceSum(csr_tensor_grad_value, tuple_out[1], true);
  } else if (dense_shape.size() > 1 && dense_shape[1] == 1) {
    dense_grad =
      ib->Emit("CSRReduceSum", {indptr, indices, dense_grad_value, shape_node, ib->Value(static_cast<int64_t>(1))});
  } else {
    auto row = ib->Emit("CSR2COO", {indptr, ib->Value(ib->GetShape(indices).at(0))});
    auto coo_idx = ib->Emit("Stack", {ib->MakeTuple({row, indices})}, {{"axis", MakeValue(static_cast<int64_t>(-1))}});
    dense_grad = ib->Emit("TensorScatterUpdate", {ib->ZerosLike(dense), coo_idx, dense_grad_value});
  }
  return {ib->ZerosLike(indptr), ib->ZerosLike(indices), csr_tensor_grad_value, ib->ZerosLike(ib->Value<int64_t>(0)),
          dense_grad};
});

REG_BPROP_BUILDER("CSR2COO").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indptr = ib->GetInput(kIndex0);
  auto nnz = ib->GetInput(kIndex1);
  return {ib->ZerosLike(indptr), ib->ZerosLike(nnz)};
});

REG_BPROP_BUILDER("COO2CSR").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto row_indices = ib->GetInput(kIndex0);
  auto height = ib->GetInput(kIndex1);
  return {ib->ZerosLike(row_indices), ib->ZerosLike(height)};
});

REG_BPROP_BUILDER("MakeCOOTensor").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex4);
  auto dout_values = ib->TupleGetItem(dout, kIndex1);
  return {ib->ZerosLike(indices), dout_values};
});

REG_BPROP_BUILDER("COOTensorGetIndices").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto coo_tensor = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto coo_tensor_values = ib->TupleGetItem(coo_tensor, kIndex1);
  auto coo_tensor_shape = ib->TupleGetItem(coo_tensor, kIndex2);
  return {ib->MakeTuple({dout, ib->ZerosLike(coo_tensor_values), coo_tensor_shape})};
});

REG_BPROP_BUILDER("COOTensorGetValues").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto coo_tensor = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto coo_tensor_indices = ib->TupleGetItem(coo_tensor, kIndex0);
  auto coo_tensor_shape = ib->TupleGetItem(coo_tensor, kIndex2);
  return {ib->MakeTuple({ib->ZerosLike(coo_tensor_indices), dout, coo_tensor_shape})};
});

REG_BPROP_BUILDER("COOTensorGetDenseShape").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto coo_tensor = ib->GetInput(kIndex0);
  return {ib->ZerosLike(coo_tensor)};
});

REG_BPROP_BUILDER("MakeCSRTensor").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(BODYFUNC(ib) {
  auto indptr = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex5);
  auto dout_values = ib->TupleGetItem(dout, kIndex2);
  auto dout_shape = ib->TupleGetItem(dout, kIndex3);
  return {ib->ZerosLike(indptr), ib->ZerosLike(indices), dout_values, dout_shape};
});

REG_BPROP_BUILDER("CSRTensorGetIndptr").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto csr_tensor = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto csr_tensor_indices = ib->TupleGetItem(csr_tensor, kIndex1);
  auto csr_tensor_values = ib->TupleGetItem(csr_tensor, kIndex2);
  auto csr_tensor_shape = ib->TupleGetItem(csr_tensor, kIndex3);
  return {ib->MakeTuple({dout, ib->ZerosLike(csr_tensor_indices), ib->ZerosLike(csr_tensor_values), csr_tensor_shape})};
});

REG_BPROP_BUILDER("CSRTensorGetIndices").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto csr_tensor = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto csr_tensor_indptr = ib->TupleGetItem(csr_tensor, kIndex0);
  auto csr_tensor_values = ib->TupleGetItem(csr_tensor, kIndex2);
  auto csr_tensor_shape = ib->TupleGetItem(csr_tensor, kIndex3);
  return {ib->MakeTuple({ib->ZerosLike(csr_tensor_indptr), dout, ib->ZerosLike(csr_tensor_values), csr_tensor_shape})};
});

REG_BPROP_BUILDER("CSRTensorGetValues").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto csr_tensor = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto csr_tensor_indptr = ib->TupleGetItem(csr_tensor, kIndex0);
  auto csr_tensor_indices = ib->TupleGetItem(csr_tensor, kIndex1);
  auto csr_tensor_shape = ib->TupleGetItem(csr_tensor, kIndex3);
  return {ib->MakeTuple({ib->ZerosLike(csr_tensor_indptr), ib->ZerosLike(csr_tensor_indices), dout, csr_tensor_shape})};
});

REG_BPROP_BUILDER("CSRTensorGetDenseShape").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto csr_tensor = ib->GetInput(kIndex0);
  return {ib->ZerosLike(csr_tensor)};
});

REG_BPROP_BUILDER("CSRSparseMatrixToDense").SetUnusedInputs({i5}).SetBody(BODYFUNC(ib) {
  auto shape = ib->GetInput(kIndex0);
  auto batch = ib->GetInput(kIndex1);
  auto indptr = ib->GetInput(kIndex2);
  auto indices = ib->GetInput(kIndex3);
  auto values = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto tmp = ib->Emit("CSRSparseMatrixToSparseTensor", {shape, batch, indptr, indices, values});
  auto res = ib->Emit("DenseToCSRSparseMatrix", {dout, ib->TupleGetItem(tmp, kIndex0)});
  return {ib->TupleGetItem(res, kIndex0), ib->TupleGetItem(res, kIndex1), ib->TupleGetItem(res, kIndex2),
          ib->TupleGetItem(res, kIndex3), ib->TupleGetItem(res, kIndex4)};
});

REG_BPROP_BUILDER("DenseToCSRSparseMatrix").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto batch_ptr = ib->TupleGetItem(out, kIndex1);
  auto row_ptr = ib->TupleGetItem(out, kIndex2);
  auto col_ind = ib->TupleGetItem(out, kIndex3);
  auto dvalue = ib->TupleGetItem(dout, kIndex4);
  auto dense_shape = ib->GetShape(ib->GetInput(kIndex0));
  auto is_default_rank = (dense_shape.size() == kDim2);
  auto batch_size = is_default_rank ? 1 : dense_shape.at(kIndex0);
  auto num_rows = is_default_rank ? dense_shape.at(kIndex0) : dense_shape.at(kIndex1);
  auto num_cols = is_default_rank ? dense_shape.at(kIndex1) : dense_shape.at(kIndex2);
  auto indices_type = ib->GetDtypeId(indices);
  std::vector<int64_t> sh;
  if (is_default_rank) {
    sh = {num_rows, num_cols};
  } else {
    sh = {batch_size, num_rows, num_cols};
  }
  NodePtr shape = nullptr;
  if (indices_type == kNumberTypeInt32) {
    shape = ib->Tensor(sh, kInt32);
  } else if (indices_type != kNumberTypeInt64) {
    shape = ib->Tensor(sh);
  } else {
    MS_EXCEPTION(TypeError) << "For 'DenseToCSRSparseMatrix', 'indices' must be of type int32 or int64, but got: "
                            << TypeIdToString(indices_type);
  }
  return {ib->Emit("CSRSparseMatrixToDense", {shape, batch_ptr, row_ptr, col_ind, dvalue}), ib->ZerosLike(indices)};
});

REG_BPROP_BUILDER("SparseSoftmax").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex0);
  auto values = ib->GetInput(kIndex1);
  auto shape = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto default_values = ib->Tensor(0, ib->GetDtype(values));
  auto out_dout = ib->Mul(out, dout);
  constexpr int64_t max_length = 1000000;
  auto sp_product = ib->Emit("SparseToDenseV2", {indices, shape, out_dout, default_values},
                             {{"validate_indices", MakeValue(true)}, {"max_length", MakeValue<int64_t>(max_length)}});
  auto sum_reduced = ib->Neg(ib->ReduceSum(sp_product, {-1}, true));
  auto sp_sum = ib->Emit("SparseDenseCwiseAdd", {indices, dout, shape, sum_reduced});
  auto grad_x = ib->Mul(sp_sum, out);
  return {ib->ZerosLike(indices), grad_x, ib->ZerosLike(shape)};
});

REG_BPROP_BUILDER("SparseTensorToCSRSparseMatrix").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("CSRSparseMatrixToSparseTensor",
                     {ib->TupleGetItem(dout, kIndex0), ib->TupleGetItem(dout, kIndex1), ib->TupleGetItem(dout, kIndex2),
                      ib->TupleGetItem(dout, kIndex3), ib->TupleGetItem(dout, kIndex4)});
  return {ib->TupleGetItem(dx, kIndex0), ib->TupleGetItem(dx, kIndex1), ib->TupleGetItem(dx, kIndex2)};
});

REG_BPROP_BUILDER("CSRSparseMatrixToSparseTensor").SetUnusedInputs({i0, i1, i2, i3, i4, i5}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex6);
  auto dx = ib->Emit("SparseTensorToCSRSparseMatrix", {ib->TupleGetItem(dout, kIndex0), ib->TupleGetItem(dout, kIndex1),
                                                       ib->TupleGetItem(dout, kIndex2)});
  return {ib->TupleGetItem(dx, kIndex0), ib->TupleGetItem(dx, kIndex1), ib->TupleGetItem(dx, kIndex2),
          ib->TupleGetItem(dx, kIndex3), ib->TupleGetItem(dx, kIndex4)};
});

REG_BPROP_BUILDER("SparseSegmentSqrtN").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  return CommonSparseSegmentBprop(ib, "SparseSegmentSqrtNGrad", false);
});

REG_BPROP_BUILDER("SparseSegmentSqrtNWithNumSegments").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  return CommonSparseSegmentBprop(ib, "SparseSegmentSqrtNGrad", true);
});

REG_BPROP_BUILDER("SparseSegmentSum").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  if (ib->GetTargetFromContext() == kGPUDevice) {
    return CommonSparseSegmentBprop(ib, "SparseSegmentSumGrad", false);
  }
  return CommonSparseSegmentBpropDefault(ib, false);
});

REG_BPROP_BUILDER("SparseSegmentSumWithNumSegments").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  if (ib->GetTargetFromContext() == kGPUDevice) {
    return CommonSparseSegmentBprop(ib, "SparseSegmentSumGrad", true);
  }
  return CommonSparseSegmentBpropDefault(ib, true);
});

REG_BPROP_BUILDER("SparseTensorDenseAdd").SetUnusedInputs({i1, i2, i3, i4}).SetBody(BODYFUNC(ib) {
  auto x1_indices = ib->GetInput(kIndex0);
  auto x1_shape = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex5);
  return {ib->ZerosLike(x1_indices), ib->Emit("GatherNd", {dout, x1_indices}), ib->ZerosLike(x1_shape), dout};
});

REG_BPROP_BUILDER("SparseSegmentMeanWithNumSegments").SetUnusedInputs({i0, i3, i4}).SetBody(BODYFUNC(ib) {
  return CommonSparseSegmentBprop(ib, "SparseSegmentMeanGrad", true);
});

REG_BPROP_BUILDER("SparseReorder").SetUnusedInputs({i1, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex0);
  auto shape = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto num_entries = ib->GetShape(indices)[0];
  auto start = ib->Tensor(0, kInt32);
  auto limit = ib->Tensor(LongToInt(num_entries), kInt32);
  auto delta = ib->Tensor(1, kInt32);
  constexpr int64_t max_len = 1000000;
  auto entry_indices = ib->Emit("Range", {start, limit, delta}, {{"maxlen", MakeValue(max_len)}});
  auto output = ib->Emit("SparseReorder", {indices, entry_indices, shape});
  constexpr int64_t sort_axis = -1;
  constexpr int64_t gather_axis = 0;
  auto inverted_permutation = ib->Emit("Sort", {ib->Cast(ib->TupleGetItem(output, 1), kFloat32)},
                                       {{"axis", MakeValue(sort_axis)}, {"descending", MakeValue(false)}});
  auto res = {
    ib->ZerosLike(indices),
    ib->Emit("Gather", {ib->TupleGetItem(dout, 1), ib->TupleGetItem(inverted_permutation, 1), ib->Value(gather_axis)}),
    ib->ZerosLike(shape)};
  return res;
});

REG_BPROP_BUILDER("SparseDenseCwiseMul").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  return BpropSparseDenseCwiseCommon(ib, "SparseDenseCwiseMul");
});

REG_BPROP_BUILDER("SparseDenseCwiseDiv").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  return BpropSparseDenseCwiseCommon(ib, "SparseDenseCwiseDiv");
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
