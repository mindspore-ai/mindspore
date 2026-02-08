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
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "frontend/expander/grad/grad_utils.h"
#include "ops_utils/op_utils.h"
#include "include/op_enum.h"

namespace mindspore::expander::bprop {

NodePtr MatrixDiag(BpropBuilder *ib, const NodePtr &x) {
  auto shape = ib->GetShape(x);
  NodePtr row = nullptr;
  if (IsDynamic(shape)) {
    auto real_shape = ib->Shape(x);
    row = ib->Emit("ScalarToTensor", {ib->TupleGetItem(real_shape, ib->Value(static_cast<int64_t>(-1))),
                                      ib->Value<int64_t>(kInt32->type_id())});
  } else {
    row = ib->Tensor(shape[shape.size() - i1], kInt32);
  }
  auto out = ib->Emit("MatrixDiagV3", {x, ib->Tensor(0, kInt32), row, row, ib->Tensor(0, ib->GetDtype(x))},
                      {{"align", MakeValue("RIGHT_LEFT")}});
  return out;
}

NodePtr DoMatMul(BpropBuilder *ib, const NodePtr &x, const NodePtr &y) {
  auto shape = ib->GetShape(x);
  if (IsDynamicRank(shape)) {
    auto true_case = [&x, &y](Emitter *e) -> NodePtrList { return {e->BatchMatMul(x, y)}; };
    auto false_case = [&x, &y](Emitter *e) -> NodePtrList { return {e->MatMul(x, y)}; };
    auto rank = ib->Emit("Rank", {x});
    auto cond = ib->Emit("scalar_gt", {rank, ib->Value(static_cast<int64_t>(i2))});
    return ib->Conditional(cond, true_case, false_case);
  }
  if (shape.size() > i2) {
    return ib->BatchMatMul(x, y);
  }
  return ib->MatMul(x, y);
}

NodePtr SafeReciprocal(BpropBuilder *ib, const NodePtr &x) {
  return ib->Mul(x, ib->Reciprocal(ib->Cast(
                      ib->AddScalar(ib->Square(x), ib->ValueByType(1e-20, ib->GetDtype(x)), ib->Value(1)), kFloat32)));
}

/*
 * Matrix Symmetry Adjustment Operation.
 * Step1: ret = x + x^T
 * Step2: Multiply the elements on the diagonal of ret by 0.5.
 */
constexpr int64_t matrix_max_length = 200000000;
NodePtr Syminvadj(BpropBuilder *ib, const NodePtr &x) {
  auto ret = ib->Add(x, ib->TransposeExtView(x, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2)));

  // Extract the diagonal and multiply the value on the diagonal by 0.5.
  auto diag_half = ib->Emit(
    "Muls", {ib->Emit("Diagonal", {ret, ib->EmitValue(MakeValue<int64_t>(0)), ib->EmitValue(MakeValue<int64_t>(-2)),
                                   ib->EmitValue(MakeValue<int64_t>(-1))}),
             ib->Value<float>(0.5)});
  ret = ib->Emit("MatrixSetDiagV3", {ret, diag_half, ib->Tensor(0, kInt32)},
                 {{"align", MakeValue("RIGHT_LEFT")}, {"max_length", MakeValue(matrix_max_length)}});
  return ret;
}

NodePtr Syminvadj_dyn(Emitter *e, const NodePtr &x) {
  auto ret = e->Add(x, e->TransposeExtView(x, e->Value<int64_t>(-1), e->Value<int64_t>(-2)));
  auto diag_half =
    e->Muls(e->Emit("Diagonal", {ret, e->EmitValue(MakeValue<int64_t>(0)), e->EmitValue(MakeValue<int64_t>(-2)),
                                 e->EmitValue(MakeValue<int64_t>(-1))}),
            e->Value<float>(0.5));
  ret = e->Emit("MatrixSetDiagV3", {ret, diag_half, e->Tensor(0, kInt32)},
                {{"align", MakeValue("RIGHT_LEFT")}, {"max_length", MakeValue(matrix_max_length)}});

  return ret;
}

/* return tril(x - x^T) */
NodePtr TrilImInvAdjSkew(BpropBuilder *ib, const NodePtr &x) {
  auto tmp = ib->Sub(x, ib->TransposeExtView(x, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2)));
  auto out = ib->TrilExt(tmp, ib->EmitValue(MakeValue<int64_t>(0)));
  return out;
}

NodePtr TrilImInvAdjSkew_dyn(Emitter *e, const NodePtr &x) {
  auto tmp = e->Sub(x, e->TransposeExtView(x, e->Value<int64_t>(-1), e->Value<int64_t>(-2)));
  auto out = e->TrilExt(tmp, e->EmitValue(MakeValue<int64_t>(0)));
  return out;
}

REG_BPROP_BUILDERS_BEGIN(GradLinalgOps)
DEF_PURE_SHAPE_CALC(dynamic_calc_svd_m_n_idx)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto a_shape = inputs.at(i0);
    if (a_shape.size() < 2) {
      MS_LOG_EXCEPTION << "The rank of input `A` is " << std::to_string(a_shape.size())
                       << " less than 2, which is invalid.";
    }

    auto m = a_shape[a_shape.size() - 2];
    auto n = a_shape[a_shape.size() - 1];

    return {{m, n}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {2}; });

DEF_PURE_SHAPE_CALC(dynamic_calc_strided_slice_0_m)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto a_shape = inputs.at(i0);
    auto uv_shape = inputs.at(i1);

    auto m = a_shape[a_shape.size() - 2];
    auto n = a_shape[a_shape.size() - 1];
    if (m > n) {
      std::swap(m, n);
    }

    ShapeVector slice = {0, m};
    ShapeVector begin_strides(uv_shape.size(), 0);
    ShapeVector end_strides = uv_shape;
    ShapeVector step_strides(uv_shape.size(), 1);
    auto zero = MakeValue<int64_t>(0);
    auto dim = SizeToLong(uv_shape.size()) - 1;
    begin_strides[dim] = slice[i0];
    end_strides[dim] = slice[i1];
    if (end_strides[dim] == LLONG_MAX) {
      MS_EXCEPTION(ValueError) << "For StridedSlice, end_strides[" << dim << "] is too large.";
    }

    return {begin_strides, end_strides, step_strides};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto a_shape = inputs.at(i0);
    auto uv_shape = inputs.at(i1);
    if (IsDynamicRank(a_shape) || IsDynamicRank(uv_shape) || !unknown_inputs.empty()) {
      return {-1, -1, -1};
    }
    auto size = SizeToLong(uv_shape.size());
    return {size, size, size};
  });

DEF_PURE_SHAPE_CALC(dynamic_calc_strided_slice_m_n)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto a_shape = inputs.at(i0);
    auto uv_shape = inputs.at(i1);

    auto m = a_shape[a_shape.size() - 2];
    auto n = a_shape[a_shape.size() - 1];
    if (m > n) {
      std::swap(m, n);
    }

    ShapeVector slice = {m, n};
    ShapeVector begin_strides(uv_shape.size(), 0);
    ShapeVector end_strides = uv_shape;
    ShapeVector step_strides(uv_shape.size(), 1);
    auto zero = MakeValue<int64_t>(0);
    auto dim = SizeToLong(uv_shape.size()) - 1;
    begin_strides[dim] = slice[i0];
    end_strides[dim] = slice[i1];
    if (end_strides[dim] == LLONG_MAX) {
      MS_EXCEPTION(ValueError) << "For StridedSlice, end_strides[" << dim << "] is too large.";
    }

    return {begin_strides, end_strides, step_strides};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    auto a_shape = inputs.at(i0);
    auto uv_shape = inputs.at(i1);
    if (IsDynamicRank(a_shape) || IsDynamicRank(uv_shape) || !unknown_inputs.empty()) {
      return {-1, -1, -1};
    }
    auto size = SizeToLong(uv_shape.size());
    return {size, size, size};
  });

NodePtrList GetDynamicUv(BpropBuilder *ib, const NodePtr &uv, const NodePtr &m, const NodePtr &n) {
  auto u_true_case = [&uv](Emitter *e) -> NodePtrList { return {e->TupleGetItem(uv, i2)}; };
  auto v_true_case = [&uv](Emitter *e) -> NodePtrList { return {e->TupleGetItem(uv, i1)}; };
  auto u_false_case = [&uv](Emitter *e) -> NodePtrList { return {e->TupleGetItem(uv, i1)}; };
  auto v_false_case = [&uv](Emitter *e) -> NodePtrList { return {e->TupleGetItem(uv, i2)}; };
  auto cond = ib->Emit("scalar_gt", {m, n});
  auto u = ib->Conditional(cond, u_true_case, u_false_case);
  auto v = ib->Conditional(cond, v_true_case, v_false_case);
  return {u, v};
}

NodePtr ControlFlowMatMul(Emitter *e, const NodePtr &x, const NodePtr &y) {
  auto shape = x->shape();
  if (IsDynamicRank(shape)) {
    return e->BatchMatMul(x, y);
  }
  if (shape.size() > i2) {
    return e->BatchMatMul(x, y);
  }
  return e->MatMul(x, y);
}

NodePtr ControlFlowMatrixTranspose(Emitter *e, const NodePtr &x) {
  auto shape = x->shape();
  if (IsDynamicRank(shape)) {
    auto dim = e->Emit("Rank", {x});
    constexpr int64_t kMaxLen = 1000000;
    auto perm = e->Emit("Range", {e->Value<int64_t>(0LL), dim, e->Value<int64_t>(1LL), e->Value<int64_t>(kMaxLen)});
    auto part_1 =
      e->Emit("StridedSlice", {perm, e->Value<ShapeVector>(ShapeVector{0}), e->Value<ShapeVector>(ShapeVector{-2}),
                               e->Value<ShapeVector>(ShapeVector{1}), e->Value<int64_t>(0LL), e->Value<int64_t>(0LL),
                               e->Value<int64_t>(0LL), e->Value<int64_t>(0LL), e->Value<int64_t>(0LL)});
    auto part_2 =
      e->Emit("StridedSlice", {perm, e->Value<ShapeVector>(ShapeVector{-1}), e->Value<ShapeVector>(ShapeVector{0}),
                               e->Value<ShapeVector>(ShapeVector{1}), e->Value<int64_t>(0LL), e->Value<int64_t>(1LL),
                               e->Value<int64_t>(0LL), e->Value<int64_t>(0LL), e->Value<int64_t>(0LL)});
    auto part_3 =
      e->Emit("StridedSlice", {perm, e->Value<ShapeVector>(ShapeVector{-2}), e->Value<ShapeVector>(ShapeVector{-1}),
                               e->Value<ShapeVector>(ShapeVector{1}), e->Value<int64_t>(0LL), e->Value<int64_t>(0LL),
                               e->Value<int64_t>(0LL), e->Value<int64_t>(0LL), e->Value<int64_t>(0LL)});
    perm = e->Concat(e->MakeTuple({part_1, part_2, part_3}), e->Value<int64_t>(-1LL));
    return e->Transpose(x, e->TensorToTuple(perm));
  }
  auto dim = shape.size();
  if (dim < i2) {
    MS_LOG_EXCEPTION << "For MatrixTranspose, input's ndim " << dim << " is less or equal to 2, which is invalid";
  }
  std::vector<int64_t> perm(dim);
  for (size_t i = 0; i < dim; i++) {
    perm[i] = static_cast<int64_t>(i);
  }
  std::swap(perm[dim - i2], perm[dim - i1]);
  return e->Transpose(x, perm);
}

NodePtr ControlFlowAdjoint(Emitter *e, const NodePtr &x) { return ControlFlowMatrixTranspose(e, e->Conj(x)); }

NodePtr SvdBpropDynamic(BpropBuilder *ib, const NodePtr &a, const NodePtr &out, const NodePtr &dout,
                        bool full_matrices) {
  auto m_n_idx = ib->ShapeCalc(dynamic_calc_svd_m_n_idx, {a})[0];
  auto m = ib->TupleGetItem(m_n_idx, i0);
  auto n = ib->TupleGetItem(m_n_idx, i1);

  auto s = ib->TupleGetItem(out, i0);
  auto ds = ib->TupleGetItem(dout, i0);
  auto uv = GetDynamicUv(ib, out, m, n);
  auto duv = GetDynamicUv(ib, dout, m, n);
  auto u = uv[i0];
  auto v = uv[i1];
  auto du = duv[i0];
  auto dv = duv[i1];

  auto s_mat = MatrixDiag(ib, s);
  auto s2 = ib->Square(s);
  constexpr int64_t kMaxLength = 200000000;
  auto f = ib->Emit("MatrixSetDiagV3",
                    {SafeReciprocal(ib, ib->Sub(ib->ExpandDims(s2, -2), ib->ExpandDims(s2, -1))), ib->ZerosLike(s),
                     ib->Tensor(0, kInt32)},
                    {{"align", MakeValue("RIGHT_LEFT")}, {"max_length", MakeValue(kMaxLength)}});
  auto s_inv_mat = MatrixDiag(ib, SafeReciprocal(ib, s));

  auto strides_slice_v1 = ib->ShapeCalc(dynamic_calc_strided_slice_0_m, {a, v});
  auto v1 = ib->StridedSlice(v, strides_slice_v1[i0], strides_slice_v1[i1], strides_slice_v1[i2],
                             ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL),
                             ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL));
  auto strides_slice_dv1 = ib->ShapeCalc(dynamic_calc_strided_slice_0_m, {a, dv});
  auto dv1 = ib->StridedSlice(dv, strides_slice_dv1[i0], strides_slice_dv1[i1], strides_slice_dv1[i2],
                              ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL),
                              ib->Value<int64_t>(0LL), ib->Value<int64_t>(0LL));
  auto u_gu = DoMatMul(ib, Adjoint(ib, u), du);
  auto v_gv = DoMatMul(ib, Adjoint(ib, v1), dv1);
  auto f_u = ib->Mul(f, u_gu);
  auto f_v = ib->Mul(f, v_gv);
  auto ds_mat = MatrixDiag(ib, ib->Cast(ds, ib->GetDtype(a)));
  auto term1_nouv =
    ds_mat + DoMatMul(ib, ib->Add(f_u, Adjoint(ib, f_u)), s_mat) + DoMatMul(ib, s_mat, ib->Add(f_v, Adjoint(ib, f_v)));
  auto term1 = DoMatMul(ib, u, DoMatMul(ib, term1_nouv, Adjoint(ib, v1)));
  auto m_n_equal_true_branch = [&term1](const Emitter *e) -> NodePtrList { return {term1}; };
  auto m_n_equal_false_branch = [&](Emitter *e) -> NodePtrList {
    auto gv1t = ControlFlowMatrixTranspose(e, dv1);
    auto gv1t_v1 = ControlFlowMatMul(e, gv1t, v1);
    auto term2_nous = e->Sub(gv1t, ControlFlowMatMul(e, gv1t_v1, ControlFlowAdjoint(e, v1)));
    if (full_matrices) {
      auto strides_slice_v2 = e->ShapeCalc(dynamic_calc_strided_slice_m_n, {a, v});
      auto v2 = e->Emit("StridedSlice", {v, strides_slice_v2[i0], strides_slice_v2[i1], strides_slice_v2[i2],
                                         e->Value<int64_t>(0LL), e->Value<int64_t>(0LL), e->Value<int64_t>(0LL),
                                         e->Value<int64_t>(0LL), e->Value<int64_t>(0LL)});
      auto strides_slice_dv2 = e->ShapeCalc(dynamic_calc_strided_slice_m_n, {a, dv});
      auto dv2 = e->Emit("StridedSlice", {dv, strides_slice_dv2[i0], strides_slice_dv2[i1], strides_slice_dv2[i2],
                                          e->Value<int64_t>(0LL), e->Value<int64_t>(0LL), e->Value<int64_t>(0LL),
                                          e->Value<int64_t>(0LL), e->Value<int64_t>(0LL)});

      auto v1t_gv2 = ControlFlowMatMul(e, ControlFlowAdjoint(e, v1), dv2);
      term2_nous = e->Sub(term2_nous, ControlFlowMatMul(e, v1t_gv2, ControlFlowAdjoint(e, v2)));
    }
    auto u_s_inv = ControlFlowMatMul(e, u, s_inv_mat);
    auto term2 = ControlFlowMatMul(e, u_s_inv, term2_nous);
    return {e->Add(term1, term2)};
  };
  auto m_n_equal = ib->Emit("scalar_eq", {m, n});
  auto da_bef_trans = ib->Conditional(m_n_equal, m_n_equal_true_branch, m_n_equal_false_branch);

  auto use_adjoint_true_branch = [&](Emitter *e) -> NodePtrList {
    return {ControlFlowMatrixTranspose(e, da_bef_trans)};
  };
  auto use_adjoint_false_branch = [&da_bef_trans](const Emitter *e) -> NodePtrList { return {da_bef_trans}; };
  auto use_adjoint = ib->Emit("scalar_gt", {m, n});
  auto da = ib->Conditional(use_adjoint, use_adjoint_true_branch, use_adjoint_false_branch);

  return da;
}

NodePtr SvdBpropStatic(BpropBuilder *ib, const NodePtr &a, const ShapeVector &a_shape, const NodePtr &out,
                       const NodePtr &dout, bool full_matrices) {
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

  return use_adjoint ? MatrixTranspose(ib, da_before_transpose) : da_before_transpose;
}

REG_BPROP_BUILDER("Svd").SetBody(BODYFUNC(ib) {
  const auto &a = ib->GetInput(i0);
  const auto &full_matrices = ib->GetInput(i1);
  const auto &compute_uv = ib->GetInput(i2);
  const auto &out = ib->GetInput(i3);
  const auto &dout = ib->GetInput(i4);
  auto full_matrices_opt = GetScalarValue<bool>(full_matrices->BuildValue());
  auto compute_uv_opt = GetScalarValue<bool>(compute_uv->BuildValue());
  if (!full_matrices_opt.has_value() || !compute_uv_opt.has_value()) {
    MS_LOG_EXCEPTION << "For gradient calculation of Svd, 'full_matrices' or 'compute_uv' must have value.";
  }

  NodePtr da = nullptr;
  if (!compute_uv_opt.value()) {
    auto tmp = ib->Emit("Svd", {a, ib->Value<bool>(false), ib->Value<bool>(true)});
    auto u = ib->TupleGetItem(tmp, i1);
    auto v = ib->TupleGetItem(tmp, i2);
    da = DoMatMul(ib, u,
                  DoMatMul(ib, MatrixDiag(ib, ib->Cast(ib->TupleGetItem(dout, i0), ib->GetDtype(a))), Adjoint(ib, v)));
    return {da, ib->OutZeros(full_matrices), ib->OutZeros(compute_uv)};
  }

  auto a_shape = ib->GetShape(a);
  if (IsDynamic(a_shape)) {
    da = SvdBpropDynamic(ib, a, out, dout, full_matrices_opt.value());
  } else {
    da = SvdBpropStatic(ib, a, a_shape, out, dout, full_matrices_opt.value());
  }
  return {da, ib->OutZeros(full_matrices), ib->OutZeros(compute_uv)};
});

REG_BPROP_BUILDER("LstsqV2").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  const auto &a = ib->GetInput(i0);
  const auto &b = ib->GetInput(i1);
  const auto &driver = ib->GetInput(i2);
  const auto &grad_outs = ib->GetInput(i4);
  auto grad_solution = ib->TupleGetItem(grad_outs, 0);
  auto grads = ib->Emit("LstsqV2Grad", {grad_solution, a, b});
  auto grad_a = ib->TupleGetItem(grads, 0);
  auto grad_b = ib->TupleGetItem(grads, 1);
  return {grad_a, grad_b, ib->OutZeros(driver)};
});

DEF_PURE_SHAPE_CALC(dynamic_resize_r_shape)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto r_shape = inputs.at(i0);
    auto m_ = inputs.at(i1)[0];
    auto n_ = inputs.at(i2)[0];
    ShapeVector resize_shape(r_shape.begin(), r_shape.end());
    if (resize_shape[resize_shape.size() - 1] <= 0) {
      MS_LOG_EXCEPTION << "Error, R mat will be empty a tensor.";
    }
    resize_shape[resize_shape.size() - 1] = n_ - m_;

    return {resize_shape};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto new_shape = inputs.at(i0);
    int64_t rank = IsDynamicRank(new_shape) ? -1 : SizeToLong(new_shape.size());
    return {rank};
  });

DEF_PURE_SHAPE_CALC(dynamic_cal_m_n)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto a_shape = inputs.at(i0);
    if (a_shape.size() < 2) {
      MS_LOG_EXCEPTION << "The rank of input `A` is " << std::to_string(a_shape.size())
                       << " less than 2, which is invalid.";
    }
    auto m_ = a_shape[a_shape.size() - 2];
    auto n_ = a_shape[a_shape.size() - 1];

    return {{m_, n_}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> { return {2}; });

REG_BPROP_BUILDER("LinalgQr").SetBody(BODYFUNC(ib) {
  // A.shape = (*, M, N)
  const auto &a_mat = ib->GetInput(i0);
  auto dtype = ib->GetDtype(a_mat);
  auto dtype_id = ib->GetDtypeId(a_mat);
  if (dtype_id == kNumberTypeComplex64 || dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'LinalgQr', gradient not support for complex type currently.";
  }

  auto a_shape = ib->GetShape(a_mat);
  if (!IsDynamicRank(a_shape) && static_cast<int64_t>(a_shape.size()) < 2) {
    MS_LOG_EXCEPTION << "For gradient of 'LinalgQr' ops, the dimension of input must greater than or equal to 2.";
  }

  const auto &mode_node = ib->GetInput(i1);
  auto mode = GetScalarValue<int64_t>(mode_node->BuildValue());
  if (!mode.has_value()) {
    MS_EXCEPTION(ValueError) << "For gradient of 'LinalgQr', 'mode' should not be empty.";
  }

  int64_t mode_imm = mode.value();
  if (mode_imm == ops::LinalgQrMode::R) {
    MS_LOG_EXCEPTION << "In LinalgQr backward, 'r' mode is unsupported. Please use 'reduced' or 'complete'.";
  }

  const auto &out = ib->GetInput(i2);
  const auto &dout = ib->GetInput(i3);
  auto q_mat = ib->TupleGetItem(out, 0);
  auto r_mat = ib->TupleGetItem(out, 1);
  auto dq = ib->TupleGetItem(dout, 0);
  auto dr = ib->TupleGetItem(dout, 1);
  auto q_shape = ib->GetShape(q_mat);
  auto r_shape = ib->GetShape(r_mat);
  if (!IsDynamic(a_shape) && mode_imm == ops::LinalgQrMode::COMPLETE &&
      q_shape[q_shape.size() - 2] > r_shape[r_shape.size() - 1]) {
    MS_LOG_EXCEPTION << "The QR decomposition for A(*, M, N) is not differentiable when 'mode=complete' and dimension "
                        "'M > N'. Please input 'A' which 'M <= N'.";
  }

  // When the input `A` is a multi-dimensional tensor, only the last two dimensions
  // are transposed. Other dimensions are processed in batches.
  // Init dx based on dq and dr.
  NodePtr dx = nullptr;
  auto q_mat_T = ib->TransposeExtView(q_mat, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2));
  auto r_mat_T = ib->TransposeExtView(r_mat, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2));
  dx = ib->MatMulExt(dr, r_mat_T) - ib->MatMulExt(q_mat_T, dq);

  if (!IsDynamic(a_shape)) {
    auto m_ = a_shape[a_shape.size() - 2];
    auto n_ = a_shape[a_shape.size() - 1];

    if (m_ >= n_) {
      auto dx_triu = ib->Triu(dx, ib->EmitValue(MakeValue<int64_t>(0)));
      dx = ib->MatMulExt(q_mat, Syminvadj(ib, dx_triu));
      if (dq != nullptr) {
        dx = ib->Add(dx, dq);
      }
      // 'r_mat' is a upper triangular matrix, so 'r_mat_T' is the lower triangular.
      // 'TriangularSolve' is equivalent to solving `A*X=B`. In torch, calculate x * r^T = dx
      // So, here we should do convert: [x * r^T = dx] => [(r^T)^T * x^T = dx^T] => [r * x^T = dx^T]
      auto dx_T =
        ib->TupleGetItem(ib->TriangularSolve(ib->TransposeExtView(dx, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2)),
                                             r_mat, ib->Value(true), ib->Value(false), ib->Value(false)),
                         0);
      dx = ib->TransposeExtView(dx_T, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2));
    } else {
      dx = ib->MatMulExt(q_mat, TrilImInvAdjSkew(ib, ib->Neg(dx)));

      // Extract the `m_` columns of the r matrix.
      auto r_narrow = ib->Narrow(r_mat, ib->EmitValue(MakeValue<int64_t>(-1)), ib->EmitValue(MakeValue<int64_t>(0)),
                                 ib->EmitValue(MakeValue<int64_t>(m_)));

      // [x * r_narrow^T = dx] => [r_narrow * x^T = dx^T]
      auto dx_T =
        ib->TupleGetItem(ib->TriangularSolve(ib->TransposeExtView(dx, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2)),
                                             r_narrow, ib->Value(true), ib->Value(false), ib->Value(false)),
                         0);
      dx = ib->TransposeExtView(dx_T, ib->Value<int64_t>(-1), ib->Value<int64_t>(-2));

      // Step1. Modify the tensor shape of r_mat so that the size of the last dimension becomes `n - m`.
      // Step2. Concatenate dx along the last dimension and an all-zero tensor of the shape r_shape.
      auto r_reshape = r_shape;
      r_reshape[r_reshape.size() - 1] = n_ - m_;
      auto zero_tensor =
        ib->Zeros(ib->EmitValue(MakeValue(r_reshape)), ib->Value(static_cast<int64_t>(dtype->type_id())));
      dx = ib->Concat({dx, zero_tensor}, -1);

      if (dr != nullptr) {
        dx = dx + ib->MatMulExt(q_mat, dr);
      }
    }
  } else {
    auto real_m_n = ib->ShapeCalc(dynamic_cal_m_n, {a_mat})[0];
    auto m_ = ib->TupleGetItem(real_m_n, 0);
    auto n_ = ib->TupleGetItem(real_m_n, 1);

    auto true_case = [&](Emitter *e) -> NodePtrList {
      NodePtr ret = nullptr;
      auto dx_triu = e->Triu(dx, e->EmitValue(MakeValue<int64_t>(0)));
      ret = e->MatMulExt(q_mat, Syminvadj_dyn(e, dx_triu));
      if (dq != nullptr) {
        ret = e->Add(ret, dq);
      }

      auto dx_T =
        e->TupleGetItem(e->TriangularSolve(e->TransposeExtView(ret, e->Value<int64_t>(-1), e->Value<int64_t>(-2)),
                                           r_mat, e->Value(true), e->Value(false), e->Value(false)),
                        0);
      ret = e->TransposeExtView(dx_T, e->Value<int64_t>(-1), e->Value<int64_t>(-2));

      return {ret};
    };

    auto false_case = [&](Emitter *e) -> NodePtrList {
      NodePtr ret = nullptr;
      ret = e->MatMulExt(q_mat, TrilImInvAdjSkew_dyn(e, e->Mul(e->Tensor(-1, dtype), dx)));
      auto r_narrow = e->Narrow(r_mat, e->EmitValue(MakeValue<int64_t>(-1)), e->EmitValue(MakeValue<int64_t>(0)), m_);
      auto dx_T =
        e->TupleGetItem(e->TriangularSolve(e->TransposeExtView(ret, e->Value<int64_t>(-1), e->Value<int64_t>(-2)),
                                           r_narrow, e->Value(true), e->Value(false), e->Value(false)),
                        0);
      ret = e->TransposeExtView(dx_T, e->Value<int64_t>(-1), e->Value<int64_t>(-2));
      auto r_reshape = e->ShapeCalc(dynamic_resize_r_shape, {r_mat, m_, n_}, {1, 2})[0];
      auto zero_tensor = e->Zeros(r_reshape, e->Value(static_cast<int64_t>(dtype_id)));
      ret = e->Concat({ret, zero_tensor}, -1);
      if (dr != nullptr) {
        ret = ret + e->MatMulExt(q_mat, dr);
      }

      return {ret};
    };

    auto cond = ib->Emit("scalar_ge", {m_, n_});
    dx = ib->Conditional(cond, true_case, false_case);
  }

  return {dx, ib->OutZeros(mode_node)};
});

REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
