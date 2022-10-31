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
#include "common/graph_kernel/bprop/bprop_irbuilder.h"
#include "common/graph_kernel/bprop/expander/common_utils.h"
#include "include/common/utils/utils.h"

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
};  // namespace

REG_BPROP_BUILDER("SparseToDense").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex0);
  auto dense_shape = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {ib->ZerosLike(indices), ib->Emit("GatherNd", {dout, indices}), ib->ZerosLike(dense_shape)};
});

REG_BPROP_BUILDER("SparseToDenseV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex0);
  auto output_shape = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex5);
  auto sparse_values_grad = ib->Emit("GatherNd", {dout, indices});
  auto default_value_grad = ib->ReduceSum(dout) - ib->ReduceSum(sparse_values_grad);
  return {ib->ZerosLike(indices), ib->ZerosLike(output_shape), sparse_values_grad, default_value_grad};
});

REG_BPROP_BUILDER("SparseTensorDenseMatmul").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto adj_s = ib->GetAttr<bool>("adjoint_st");
  auto adj_d = ib->GetAttr<bool>("adjoint_dt");
  auto indices = ib->GetInput(kIndex0);
  auto values = ib->GetInput(kIndex1);
  auto dense_shape = ib->GetInput(kIndex2);
  auto dense = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto dense_grad = ib->Emit("SparseTensorDenseMatmul", {indices, values, dense_shape, dout},
                             {{"adjoint_st", MakeValue(!adj_s)}, {"adjoint_dt", MakeValue(adj_d)}});
  std::vector<int64_t> perm_value{1, 0};
  auto perm = ib->Tensor(perm_value);
  if (adj_d) {
    dense_grad = ib->Emit("Transpose", {dense_grad, perm});
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
  auto split_indices =
    ib->Emit(kSplitOpName, {indices}, {{kAttrAxis, MakeValue(axis)}, {kAttrOutputNum, MakeValue(output_num)}});
  auto rows = ib->ReduceSum(ib->TupleGetItem(split_indices, kIndex0), {axis});
  auto cols = ib->ReduceSum(ib->TupleGetItem(split_indices, kIndex1), {axis});
  auto zero = ib->Value<int64_t>(0);
  NodePtr parts_a = nullptr;
  if (adj_s) {
    parts_a = ib->Emit("Gather", {dout, cols, zero});
  } else {
    parts_a = ib->Emit("Gather", {dout, rows, zero});
  }
  NodePtr tmp1 = adj_d ? ib->Emit("Transpose", {dense, perm}) : dense;
  NodePtr tmp2 = adj_s ? rows : cols;
  auto parts_b = ib->Emit("Gather", {tmp1, tmp2, zero});
  auto values_grad = ib->ReduceSum(parts_a * parts_b, {axis});
  if (is_half) {
    values_grad = ib->Cast(values_grad, kFloat16);
  }
  return {ib->ZerosLike(indices), values_grad, ib->ZerosLike(dense_shape), dense_grad};
});

REG_BPROP_BUILDER("SparseAdd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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

REG_BPROP_BUILDER("CSRReduceSum").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indptr = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto shape = ib->GetInput(kIndex3);
  auto axis = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto shape_vec = GetAxisValue(shape);
  auto output_shape_kept_dims = ReduceShape(shape_vec, GetAxisValue(axis));
  auto tile_scaling = TupleDiv(shape_vec, output_shape_kept_dims);
  auto values_grad_dense =
    ib->Emit("Tile", {ib->Reshape(dout, output_shape_kept_dims), ib->Value<ShapeVector>(tile_scaling)});
  auto values_grad = ib->Emit("CSRGather", {indptr, indices, values_grad_dense, shape});
  return {ib->ZerosLike(indptr), ib->ZerosLike(indices), values_grad, ib->ZerosLike(ib->Value<int64_t>(0)),
          ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("CSRMV").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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
  auto dense_shape_vec = GetAxisValue(dense_shape);
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

REG_BPROP_BUILDER("CSR2COO").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indptr = ib->GetInput(kIndex0);
  auto nnz = ib->GetInput(kIndex1);
  return {ib->ZerosLike(indptr), ib->ZerosLike(nnz)};
});

REG_BPROP_BUILDER("COO2CSR").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto row_indices = ib->GetInput(kIndex0);
  auto height = ib->GetInput(kIndex1);
  return {ib->ZerosLike(row_indices), ib->ZerosLike(height)};
});

REG_BPROP_BUILDER("SparseSoftmax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex0);
  auto values = ib->GetInput(kIndex1);
  auto shape = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto default_values = ib->Tensor(0, ib->GetDtype(values));
  auto out_dout = ib->Mul(out, dout);
  auto sp_product =
    ib->Emit("SparseToDenseV2", {indices, shape, out_dout, default_values}, {{"validate_indices", MakeValue(true)}});
  auto sum_reduced = ib->Neg(ib->ReduceSum(sp_product, {-1}, true));
  auto sp_sum = ib->Emit("SparseDenseCwiseAdd", {indices, dout, shape, sum_reduced});
  auto grad_x = ib->Mul(sp_sum, out);
  return {ib->ZerosLike(indices), grad_x, ib->ZerosLike(shape)};
});

REG_BPROP_BUILDER("SparseTensorToCSRSparseMatrix").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("CSRSparseMatrixToSparseTensor",
                     {ib->TupleGetItem(dout, kIndex0), ib->TupleGetItem(dout, kIndex1), ib->TupleGetItem(dout, kIndex2),
                      ib->TupleGetItem(dout, kIndex3), ib->TupleGetItem(dout, kIndex4)});
  return {ib->TupleGetItem(dx, kIndex0), ib->TupleGetItem(dx, kIndex1), ib->TupleGetItem(dx, kIndex2)};
});

REG_BPROP_BUILDER("CSRSparseMatrixToSparseTensor").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex6);
  auto dx = ib->Emit("SparseTensorToCSRSparseMatrix", {ib->TupleGetItem(dout, kIndex0), ib->TupleGetItem(dout, kIndex1),
                                                       ib->TupleGetItem(dout, kIndex2)});
  return {ib->TupleGetItem(dx, kIndex0), ib->TupleGetItem(dx, kIndex1), ib->TupleGetItem(dx, kIndex2),
          ib->TupleGetItem(dx, kIndex3), ib->TupleGetItem(dx, kIndex4)};
});

REG_BPROP_BUILDER("SparseToDenseV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex0);
  auto output_shape = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex5);
  auto sparse_values_grad = ib->Emit("GatherNd", {dout, indices});
  auto default_value_grad = ib->Sub(ib->ReduceSum(dout), ib->ReduceSum(sparse_values_grad));
  return {ib->ZerosLike(indices), ib->ZerosLike(output_shape), sparse_values_grad, default_value_grad};
});

NodePtrList CommonSparseSegmentBprop(const BpropIRBuilder *ib, const std::string &grad_op, bool with_segments) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto segment_ids = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(with_segments ? kIndex5 : kIndex4);
  auto shape_x = ib->GetShape(x);
  auto output_dim0 = ib->Tensor(shape_x[0], kInt32);
  if (ib->GetDtype(indices) != kInt32) {
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

REG_BPROP_BUILDER("SparseSegmentSqrtN").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  return CommonSparseSegmentBprop(ib, "SparseSegmentSqrtNGrad", false);
});

REG_BPROP_BUILDER("SparseSegmentSqrtNWithNumSegments").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  return CommonSparseSegmentBprop(ib, "SparseSegmentSqrtNGrad", true);
});

REG_BPROP_BUILDER("SparseSegmentSum").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  return CommonSparseSegmentBprop(ib, "SparseSegmentSumGrad", false);
});

REG_BPROP_BUILDER("SparseSegmentSumWithNumSegments").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  return CommonSparseSegmentBprop(ib, "SparseSegmentSumGrad", true);
});
}  // namespace mindspore::expander::bprop
