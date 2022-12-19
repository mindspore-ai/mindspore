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

#include <map>
#include "frontend/operator/bprop/bprop_irbuilder.h"
#include "frontend/operator/bprop/grad/common_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore::expander::bprop {
NodePtr MatrixTranspose(const BpropIRBuilder *ib, const NodePtr &x) {
  auto shape = ib->GetShape(x);
  auto dim = shape.size();
  if (dim < kDim2) {
    MS_LOG_EXCEPTION << "To do MatrixTranspose for input a's ndim is not greater or equal to 2, which is invalid: "
                     << dim;
  }
  std::vector<int64_t> perm;
  for (int64_t i = 0; i < SizeToLong(dim); i++) {
    perm.push_back(i);
  }
  std::swap(perm[dim - kIndex2], perm[dim - kIndex1]);
  return ib->Transpose(x, perm);
}

NodePtr Adjoint(const BpropIRBuilder *ib, const NodePtr &x) { return MatrixTranspose(ib, ib->Emit("Conj", {x})); }

NodePtr MatrixDiag(const BpropIRBuilder *ib, const NodePtr &x) {
  auto shape = ib->GetShape(x);
  auto row = shape[shape.size() - 1];
  auto out = ib->Emit(
    "MatrixDiagV3",
    {x, ib->Tensor(0, kInt32), ib->Tensor(row, kInt32), ib->Tensor(row, kInt32), ib->Tensor(0, ib->GetDtype(x))},
    {{"align", MakeValue("RIGHT_LEFT")}});
  return out;
}

NodePtr DoMatMul(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y) {
  auto shape = ib->GetShape(x);
  if (shape.size() > kDim2) {
    return ib->BatchMatMul(x, y);
  }
  return ib->MatMul(x, y);
}

NodePtr SafeReciprocal(const BpropIRBuilder *ib, const NodePtr &x) {
  return ib->Mul(x, ib->Reciprocal(ib->Cast(ib->Add(ib->Square(x), ib->Tensor(1e-20, ib->GetDtype(x))), kFloat32)));
}

REG_BPROP_BUILDERS_BEGIN(GradLinalgOps)
REG_BPROP_BUILDER("Svd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto full_matrices = GetValue<bool>(ib->GetAttr("full_matrices"));
  auto compute_uv = GetValue<bool>(ib->GetAttr("compute_uv"));
  auto a = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  if (!compute_uv) {
    auto tmp = ib->Emit("Svd", {a}, {{"compute_uv", MakeValue(true)}, {"full_matrices", MakeValue(false)}});
    auto u = ib->TupleGetItem(tmp, 1);
    auto v = ib->TupleGetItem(tmp, 2);
    auto da = DoMatMul(
      ib, u, DoMatMul(ib, MatrixDiag(ib, ib->Cast(ib->TupleGetItem(dout, 0), ib->GetDtype(a))), Adjoint(ib, v)));
    return {da};
  }
  auto a_shape = ib->GetShape(a);
  if (a_shape.size() < 2) {
    MS_LOG_EXCEPTION << "For input a's ndim is not greater or equal to 2, which is invalid.";
  }

  auto m = a_shape[a_shape.size() - 2];
  auto n = a_shape[a_shape.size() - 1];
  auto s = ib->TupleGetItem(out, 0);
  auto u = ib->TupleGetItem(out, 1);
  auto v = ib->TupleGetItem(out, 2);
  auto ds = ib->TupleGetItem(dout, 0);
  auto du = ib->TupleGetItem(dout, 1);
  auto dv = ib->TupleGetItem(dout, 2);
  auto use_adjoint = false;
  if (m > n) {
    use_adjoint = true;
    std::swap(m, n);
    std::swap(u, v);
    std::swap(du, dv);
  }
  if (full_matrices && (std::abs(m - n) > 1)) {
    MS_LOG_EXCEPTION << "For 'Svd' gradient, not support for abs(m - n) > 1 with full_matrices is True.";
  }
  auto s_mat = MatrixDiag(ib, s);
  auto s2 = ib->Square(s);
  constexpr int64_t max_length = 200000000;
  auto f = ib->Emit("MatrixSetDiagV3",
                    {SafeReciprocal(ib, ib->Sub(ib->ExpandDims(s2, -2), ib->ExpandDims(s2, -1))), ib->ZerosLike(s),
                     ib->Tensor(0, kInt32)},
                    {{"align", MakeValue("RIGHT_LEFT")}, {"max_length", MakeValue(max_length)}});
  auto s_inv_mat = MatrixDiag(ib, SafeReciprocal(ib, s));
  std::map<int64_t, std::vector<int64_t>> slices;
  (void)slices.emplace(-1, std::vector<int64_t>{0, m});
  auto v1 = ib->StridedSlice(v, slices);
  auto dv1 = ib->StridedSlice(dv, slices);
  auto u_gu = DoMatMul(ib, Adjoint(ib, u), du);
  auto v_gv = DoMatMul(ib, Adjoint(ib, v1), dv1);
  auto f_u = ib->Mul(f, u_gu);
  auto f_v = ib->Mul(f, v_gv);
  auto ds_mat = MatrixDiag(ib, ib->Cast(ds, ib->GetDtype(a)));
  auto term1_nouv =
    ds_mat + DoMatMul(ib, ib->Add(f_u, Adjoint(ib, f_u)), s_mat) + DoMatMul(ib, s_mat, ib->Add(f_v, Adjoint(ib, f_v)));
  auto term1 = DoMatMul(ib, u, DoMatMul(ib, term1_nouv, Adjoint(ib, v1)));
  NodePtr da_before_transpose = nullptr;
  if (m == n) {
    da_before_transpose = term1;
  } else {
    auto gv1t = MatrixTranspose(ib, dv1);
    auto gv1t_v1 = DoMatMul(ib, gv1t, v1);
    auto term2_nous = gv1t - DoMatMul(ib, gv1t_v1, Adjoint(ib, v1));
    if (full_matrices) {
      std::map<int64_t, std::vector<int64_t>> slices_n;
      (void)slices_n.emplace(-1, std::vector<int64_t>{m, n});
      auto v2 = ib->StridedSlice(v, slices_n);
      auto d_v2 = ib->StridedSlice(dv, slices_n);
      auto v1t_gv2 = DoMatMul(ib, Adjoint(ib, v1), d_v2);
      term2_nous = term2_nous - DoMatMul(ib, v1t_gv2, Adjoint(ib, v2));
    }
    auto u_s_inv = DoMatMul(ib, u, s_inv_mat);
    auto term2 = DoMatMul(ib, u_s_inv, term2_nous);
    da_before_transpose = term1 + term2;
  }
  if (use_adjoint) {
    return {MatrixTranspose(ib, da_before_transpose)};
  } else {
    return {da_before_transpose};
  }
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
