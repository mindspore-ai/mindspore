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
}  // namespace mindspore::expander::bprop
