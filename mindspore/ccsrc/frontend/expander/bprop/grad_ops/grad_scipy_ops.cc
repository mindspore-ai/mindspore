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

#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "include/common/utils/utils.h"
#include "frontend/expander/bprop/grad_ops/common_utils.h"

namespace mindspore::expander::bprop {
REG_BPROP_BUILDERS_BEGIN(GradScipyOps)
REG_BPROP_BUILDER("SolveTriangular").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto reverse_perm = [](const ShapeVector &shape) -> ShapeVector {
    ShapeVector perm;
    for (int64_t i = SizeToLong(shape.size()) - 1; i >= 0; --i) {
      perm.push_back(i);
    }
    return perm;
  };
  auto lower = ib->GetAttr("lower");
  auto unit_diagonal = ib->GetAttr("unit_diagonal");
  auto a = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto trans = GetValue<std::string>(ib->GetAttr("trans"));
  std::string bp_trans = trans == "T" || trans == "C" ? "N" : "T";
  auto grad_b = ib->Emit("SolveTriangular", {a, dout},
                         {{"lower", lower}, {"unit_diagonal", unit_diagonal}, {"trans", MakeValue(bp_trans)}});
  auto a_shape = ib->GetShape(a);
  auto row_size = a_shape[a_shape.size() - 2];
  auto grad_b_align = ib->Reshape(grad_b, {row_size, -1});
  auto x_align = ib->Reshape(out, {row_size, -1});
  NodePtr grad_a;
  if (bp_trans.compare("T") == 0) {
    auto conj = ib->Conj(x_align);
    grad_a = ib->MatMul(grad_b_align, ib->Transpose(conj, reverse_perm(ib->GetShape(conj))));
  } else {
    auto conj = ib->Conj(grad_b_align);
    grad_a = ib->MatMul(x_align, ib->Transpose(conj, reverse_perm(ib->GetShape(conj))));
  }
  int is_lower = static_cast<int>(GetValue<bool>(lower));
  grad_a = ib->Neg(ib->Emit("MatrixBandPart", {grad_a, ib->Value(-is_lower), ib->Value(is_lower - 1)}));
  if (GetValue<bool>(unit_diagonal)) {
    auto fill = ib->Emit("Fill", {ib->EmitValue(ib->GetDtype(grad_a)), ib->Value<ShapeVector>(ShapeVector(1, row_size)),
                                  ib->Tensor(0, ib->GetDtype(grad_a))});
    grad_a =
      ib->MatrixSetDiagV3(grad_a, fill, ib->Fill(int64_t(0), {2}, TypeId::kNumberTypeInt32), MakeValue("RIGHT_LEFT"));
  }
  return {grad_a, grad_b};
});

REG_BPROP_BUILDER("Eigh").SetBody(BODYFUNC(ib) {
  auto is_compute_v = GetValue<bool>(ib->GetAttr("compute_eigenvectors"));
  auto is_lower = GetValue<bool>(ib->GetAttr("lower"));
  auto lower = static_cast<int64_t>(is_lower);
  auto a = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);

  // helper functions
  auto Adjoint = [](BpropIRBuilder *ib, const NodePtr &x) -> NodePtr {
    auto conj = ib->Conj(x);
    auto shape = ib->GetShape(conj);
    ShapeVector perm;
    for (int64_t i = SizeToLong(shape.size()) - 1; i >= 0; --i) {
      perm.push_back(i);
    }
    return ib->Transpose(conj, perm);
  };

  auto EyeTensor = [](BpropIRBuilder *ib, int m, int n) -> NodePtr {
    ShapeVector eye_shape{m, n};
    std::vector<int32_t> eyes_value;
    for (auto i = 0; i < m; ++i) {
      for (auto j = 0; j < n; ++j) {
        if (i == j) {
          (void)eyes_value.emplace_back(1);
        } else {
          (void)eyes_value.emplace_back(0);
        }
      }
    }
    auto eye_tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt32, eye_shape, &eyes_value[0], kNumberTypeInt32);
    return ib->Value(eye_tensor);
  };

  // constants in the computation
  auto zero_tensor = ib->Fill(int64_t(0), {2}, TypeId::kNumberTypeInt32);
  auto kValueNeg1 = ib->Value<int64_t>(-1);
  auto kValueNeg2 = ib->Value<int64_t>(-2);

  // final grad to compute
  NodePtr grad_a;

  if (!is_compute_v) {
    // _, v equal eigh(a)
    auto v = ib->TupleGetItem(ib->Emit("Eigh", {a}, {{"compute_eigenvectors", MakeValue(true)}}), 1);
    // grad_a is _matmul(v * F.expand_dims(dout, -2), _adjoint(v))
    grad_a = ib->MatMul(ib->Mul(v, ib->Emit("ExpandDims", {dout, kValueNeg2})), Adjoint(ib, v), false, false);

  } else {
    //  vh equal _adjoint(out[1])
    auto vh = Adjoint(ib, ib->TupleGetItem(out, 1));

    //  vh_gv equal _matmul(vh, dout[1])
    auto vh_gv = ib->MatMul(vh, ib->TupleGetItem(dout, 1), false, false);

    auto out_0 = ib->TupleGetItem(out, 0);
    // diff_inv equal diff / (diff * diff + epsilon)
    // f equal matrix_set_diag(diff_inv, F.zeros_like(w))
    auto diff = ib->Sub(ib->Emit("ExpandDims", {out_0, kValueNeg2}), ib->Emit("ExpandDims", {out_0, kValueNeg1}));
    auto diff_inv = ib->RealDiv(diff, ib->Add(ib->Mul(diff, diff), ib->Tensor(1e-20, ib->GetDtype(diff))));

    auto f = ib->MatrixSetDiagV3(diff_inv, ib->ZerosLike(out_0), zero_tensor, MakeValue("RIGHT_LEFT"));

    // _diag(a) equal F.expand_dims(a, -2) * F.eye(F.shape(a)[-1], F.shape(a)[-1], a.dtype)
    // compute F.eye(F.shape(a)[-1], F.shape(a)[-1], a.dtype)
    auto dout_0 = ib->TupleGetItem(dout, 0);
    auto dout_shape = ib->GetShape(dout_0);
    auto dout_0_size = LongToInt(dout_shape[dout_shape.size() - 1]);
    auto eye_tensor_node = EyeTensor(ib, dout_0_size, dout_0_size);

    // compute the product
    auto diag_dout_0 =
      ib->Mul(ib->Emit("ExpandDims", {dout_0, kValueNeg2}), ib->Cast(eye_tensor_node, ib->GetDtype(dout_0)));

    //  mid_part equal _diag(dout[0]) + f * vh_gv
    auto mid_part = ib->Add(diag_dout_0, ib->Mul(f, vh_gv));

    //  grad_a equal _matmul(out[1], _matmul(mid_part, vh))
    grad_a = ib->MatMul(ib->TupleGetItem(out, 1), ib->MatMul(mid_part, vh, false, false), false, false);
  }
  //  grad_a equal grad_a + _adjoint(grad_a)
  grad_a = ib->Add(grad_a, Adjoint(ib, grad_a));
  grad_a = ib->Emit("MatrixBandPart", {grad_a, ib->Value<int64_t>(-lower), ib->Value<int64_t>(lower - 1)});

  // grad_a.diagonal(0, -2, -1)
  NodePtr grad_a_diagonal;
  auto grad_a_shape = ib->GetShape(grad_a);
  auto eye_node_for_diag =
    EyeTensor(ib, LongToInt(grad_a_shape[grad_a_shape.size() - kDim2]), LongToInt(grad_a_shape.back()));
  auto eye_tensor_broadcast = ib->Emit("BroadcastTo", {eye_node_for_diag}, {{"shape", MakeValue(grad_a_shape)}});

  auto prod = ib->Mul(grad_a, ib->Cast(eye_tensor_broadcast, ib->GetDtype(grad_a)));
  auto res = ib->ReduceSum(ib->Cast(prod, kFloat32), {-1}, false);

  // create the begin and size arg for slice
  std::vector<int64_t> begin(grad_a_shape.size() - 1, 0);
  std::vector<int64_t> size_vec;
  for (int idx = 0; idx < static_cast<int>(grad_a_shape.size()) - 2; idx++) {
    size_vec.push_back(grad_a_shape[IntToSize(idx)]);
  }
  size_vec.push_back(std::min(grad_a_shape[grad_a_shape.size() - kDim2], grad_a_shape.back()));

  grad_a_diagonal =
    ib->Cast(ib->Slice(res, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(size_vec)), ib->GetDtype(grad_a));

  //  middle_diag equal 0.5 * grad_a.diagonal(0, -2, -1)
  auto middle_diag = ib->Mul(ib->Tensor(0.5, ib->GetDtype(grad_a_diagonal)), grad_a_diagonal);
  grad_a = ib->MatrixSetDiagV3(grad_a, middle_diag, zero_tensor, MakeValue("RIGHT_LEFT"));
  return {grad_a};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
