/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#include "ir/functor.h"
#include "ops/math_ops.h"
#include "utils/ms_context.h"
#include "ops/op_utils.h"

namespace mindspore::expander::bprop {
NodePtrList AddnGradFunc(BpropBuilder *ib) {
  auto dout = ib->GetInput(kIndex2);
  auto x_abs = ib->GetInput(kIndex0)->abstract();
  auto x_len = x_abs->cast<abstract::AbstractSequencePtr>()->elements().size();
  NodePtrList result(x_len, dout);
  if (x_abs->isa<abstract::AbstractList>()) {
    return {ib->MakeList(result)};
  }
  return {ib->MakeTuple(result)};
}

NodePtrList IgammaBpropExpanderDyn(BpropBuilder *ib) {
  auto a = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto sa = ib->Shape(a);
  auto sx = ib->Shape(x);
  auto rax = ib->BroadcastGradientArgs(sa, sx);
  auto ra = rax[0];
  auto rx = rax[1];
  auto partial_a = ib->Emit("IgammaGradA", {a, x});
  auto lgamma = LGamma(ib, a);
  auto partial_x = ib->Exp(
    ib->Sub((ib->Add((ib->Neg(x)), (ib->Mul((ib->Sub(a, (ib->Tensor(1, ib->GetDtype(a))))), (ib->Log(x)))))), lgamma));
  auto r1 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_a, dout), ra, false, true), sa);
  auto r2 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_x, dout), rx, false, true), sx);
  return {r1, r2};
}

NodePtrList IgammaBpropExpander(BpropBuilder *ib) {
  auto a = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto sa = ib->GetShape(a);
  auto sx = ib->GetShape(x);
  if (IsDynamic(sa) || IsDynamic(sx)) {
    return IgammaBpropExpanderDyn(ib);
  }
  auto rax = BroadcastGradientArgsInferValue(sa, sx);
  auto ra = rax[0];
  auto rx = rax[1];
  auto partial_a = ib->Emit("IgammaGradA", {a, x});
  auto lgamma = ib->Emit("Lgamma", {a});
  auto partial_x = ib->Exp(
    ib->Sub((ib->Add((ib->Neg(x)), (ib->Mul((ib->Sub(a, (ib->Tensor(1, ib->GetDtype(a))))), (ib->Log(x)))))), lgamma));
  auto dout = ib->GetInput(kIndex3);
  NodePtr r1 = nullptr;
  NodePtr r2 = nullptr;
  if (!ra.empty()) {
    r1 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_a, dout), ra), sa);
  } else {
    r1 = ib->Reshape(ib->Mul(partial_a, dout), sa);
  }
  if (!rx.empty()) {
    r2 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_x, dout), rx), sx);
  } else {
    r2 = ib->Reshape(ib->Mul(partial_x, dout), sx);
  }
  return {r1, r2};
}

NodePtrList MinimumMaximumGrad(BpropBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dout,
                               bool is_minimum) {
  auto half_ratio = ib->Emit("FillV2", {ib->Shape(dout), ib->Tensor(2, ib->GetDtype(dout))});
  auto half_dout = ib->Div(dout, half_ratio);
  NodePtr equal_mask = ib->Equal(x, y);
  auto zeros = ib->Emit("FillV2", {ib->Shape(dout), ib->Tensor(0, ib->GetDtype(dout))});
  NodePtr is_less = ib->Less(x, y);
  NodePtr is_greater = ib->Greater(x, y);
  NodePtr grad_x = nullptr;
  NodePtr grad_y = nullptr;
  if (x->need_compute_grad_out()) {
    grad_x = ib->Select(equal_mask, half_dout, dout);
    if (is_minimum) {
      grad_x = ib->Select(is_greater, zeros, grad_x);
    } else {
      grad_x = ib->Select(is_less, zeros, grad_x);
    }
  }
  if (y->need_compute_grad_out()) {
    grad_y = ib->Select(equal_mask, half_dout, dout);
    if (is_minimum) {
      grad_y = ib->Select(is_less, zeros, grad_y);
    } else {
      grad_y = ib->Select(is_greater, zeros, grad_y);
    }
  }
  return BinopGradCommon(ib, x, y, grad_x, grad_y);
}

ShapeArray MatrixDeterminantShapeFunc(const ShapeArray &inputs) {
  auto new_shape = inputs.at(0);
  new_shape.push_back(1);
  new_shape.push_back(1);
  return {new_shape};
}

ShapeVector MatrixDeterminantInferFunc(const ShapeArray &inputs, const HashSet<size_t> &) {
  auto new_shape = inputs.at(0);
  return {IsDynamicRank(new_shape) ? -1 : SizeToLong(new_shape.size()) + 2};
}

NodePtrList BpropAddcCommon(BpropBuilder *ib, const std::string &op_name, const std::unordered_set<TypeId> &type_list) {
  auto input_data = ib->GetInput(kIndex0);
  auto x1 = ib->GetInput(kIndex1);
  auto x2 = ib->GetInput(kIndex2);
  auto value = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto dinput_data = dout;
  auto dout_typeptr = ib->GetDtype(dout);
  bool need_cast = type_list.count(dout_typeptr->type_id()) > 0;
  if (need_cast) {
    input_data = ib->Cast(input_data, kFloat32);
    x1 = ib->Cast(x1, kFloat32);
    x2 = ib->Cast(x2, kFloat32);
    value = ib->Cast(value, kFloat32);
    if (op_name == "Addcdiv") {
      dinput_data = ib->Cast(dinput_data, kFloat32);
    }
  }
  NodePtr inner_out = nullptr;
  NodePtr dx1 = nullptr;
  NodePtr dx2 = nullptr;
  NodePtr dvalue = nullptr;
  if (op_name == "Addcdiv") {
    constexpr int64_t const_val = -2;
    inner_out = ib->Add((ib->Mul(value, ib->Div(x1, x2))), input_data);
    dx2 =
      ib->Neg(ib->Mul(ib->Mul(ib->Mul(x1, value), ib->Pow(x2, ib->Tensor(const_val, ib->GetDtype(x2)))), dinput_data));
    dx1 = ib->Mul(dinput_data, ib->Div(value, x2));
    dvalue = ib->Mul(dinput_data, ib->Div(x1, x2));
  } else {
    dx1 = ib->Mul(dout, ib->Mul(value, x2));
    dx2 = ib->Mul(dout, ib->Mul(value, x1));
    inner_out = ib->Add((ib->Mul((ib->Mul(x1, x2)), value)), input_data);
    dvalue = ib->Mul(dout, ib->Mul(x1, x2));
  }
  auto tmp_dinput_data = BinopGradCommon(ib, inner_out, input_data, dout, dinput_data);
  dinput_data = tmp_dinput_data[1];
  auto tmp_dx1 = BinopGradCommon(ib, inner_out, x1, dout, dx1);
  dx1 = tmp_dx1[1];
  auto tmp_dx2 = BinopGradCommon(ib, inner_out, x2, dout, dx2);
  dx2 = tmp_dx2[1];
  auto tmp_dvalue = BinopGradCommon(ib, inner_out, value, dout, dvalue);
  dvalue = tmp_dvalue[1];
  if (need_cast) {
    dinput_data = ib->Cast(dinput_data, dout_typeptr);
    dx1 = ib->Cast(dx1, dout_typeptr);
    dx2 = ib->Cast(dx2, dout_typeptr);
    dvalue = ib->Cast(dvalue, dout_typeptr);
  }
  return {dinput_data, dx1, dx2, dvalue};
}

class ReduceStdShapeCalc : public ShapeCalcFunctor {
 public:
  // cppcheck-suppress unknownMacro
  DECLARE_SHAPE_CALC("ShapeCalc_ReduceStd", ReduceStdShapeCalc)
  explicit ReduceStdShapeCalc(const std::vector<int64_t> &axis)
      : ShapeCalcFunctor("ShapeCalc_ReduceStd"), axis_(axis) {}
  ValuePtr ToValue() const override { return MakeValue(axis_); }
  void FromValue(const ValuePtr &value) override { axis_ = GetValue<std::vector<int64_t>>(value); }
  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto new_axis = axis_;
    auto x_shape = inputs.at(0);
    if (new_axis.empty() && !x_shape.empty()) {
      for (int64_t i = 0; i < SizeToLong(x_shape.size()); i++) {
        new_axis.push_back(i);
      }
    }
    (void)std::transform(new_axis.begin(), new_axis.end(), new_axis.begin(), [&x_shape](const int64_t &c) {
      if (c < 0) {
        return c + SizeToLong(x_shape.size());
      }
      return c;
    });
    for (size_t i = 1; i < new_axis.size(); ++i) {
      for (size_t j = 0; j < new_axis.size() - i; ++j) {
        if (new_axis[j] > (new_axis[j + 1])) {
          std::swap(new_axis[j], new_axis[j + 1]);
        }
      }
    }
    // input_x:[2,3,4,5]  new_axis:   [0, 2]
    // reduce: [3,5]      reshape:[1,3,1,5]
    auto reshape = x_shape;
    for (auto &i : new_axis) {
      reshape[LongToSize(i)] = 1;
    }
    int64_t num = 1;
    for (const auto &i : new_axis) {
      num *= x_shape[LongToSize(i)];
    }
    return {reshape, {num - 1}, {num}};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    auto shape_x = inputs.at(0);
    auto rank = IsDynamicRank(shape_x) ? -1 : SizeToLong(shape_x.size());
    return {rank, 1, 1};
  }

 protected:
  std::vector<int64_t> axis_;
};
REG_FUNCTOR("ShapeCalc_ReduceStd", ReduceStdShapeCalc);

NodePtrList FminFmaxGrad(BpropBuilder *ib, bool if_fmin) {
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x1_dtype = ib->GetDtype(x1);
  auto x2_dtype = ib->GetDtype(x2);
  x1 = ib->Cast(x1, kFloat32);
  x2 = ib->Cast(x2, kFloat32);
  dout = ib->Cast(dout, kFloat32);
  auto x1_nan = ib->Emit("IsNan", {x1});
  auto x2_nan = ib->Emit("IsNan", {x2});
  NodePtr b1 = nullptr;
  NodePtr b2 = nullptr;
  if (if_fmin) {
    b1 = ib->LogicalOr(ib->LessEqual(x1, x2), x2_nan);
    b2 = ib->LogicalOr(ib->Less(x2, x1), ib->LogicalAnd(x1_nan, ib->LogicalNot(x2_nan)));
  } else {
    b1 = ib->LogicalOr(ib->LogicalAnd(ib->GreaterEqual(x1, x2), ib->LogicalNot(x1_nan)), x2_nan);
    b2 = ib->LogicalOr(ib->LogicalAnd(ib->Greater(x2, x1), ib->LogicalNot(x2_nan)),
                       ib->LogicalAnd(x1_nan, ib->LogicalNot(x2_nan)));
  }
  auto rx1 = ib->Emit("MaskedFill", {x1, b1, ib->Tensor(1.0, kFloat32)});
  rx1 = ib->Emit("MaskedFill", {rx1, ib->LogicalNot(b1), ib->Tensor(0.0, kFloat32)});
  auto rx2 = ib->Emit("MaskedFill", {x2, b2, ib->Tensor(1.0, kFloat32)});
  rx2 = ib->Emit("MaskedFill", {rx2, ib->LogicalNot(b2), ib->Tensor(0.0, kFloat32)});
  auto rrx1 = ib->Mul(rx1, dout);
  auto rrx2 = ib->Mul(rx2, dout);
  auto shape_of_x1 = ib->Shape(x1);
  auto shape_of_x2 = ib->Shape(x2);
  auto x1_dim = ib->GetRank(x1);
  auto x2_dim = ib->GetRank(x2);
  NodePtr sum_r1;
  NodePtr sum_r2;
  if (x1_dim == 0 && x2_dim != 0) {
    sum_r1 = ib->ReduceSum(rrx1);
    sum_r2 = rrx2;
  } else if (x1_dim == 0 && x2_dim == 0) {
    sum_r1 = ib->ReduceSum(rrx1);
    sum_r2 = ib->ReduceSum(rrx2);
  } else if (x1_dim != 0 && x2_dim == 0) {
    sum_r2 = ib->ReduceSum(rrx2);
    sum_r1 = rrx1;
  } else {
    auto tmp = ib->BroadcastGradientArgs(x1, x2);
    auto rx = tmp[0];
    auto ry = tmp[1];
    sum_r1 = ib->ReduceSum(rrx1, rx, false, true);
    sum_r2 = ib->ReduceSum(rrx2, ry, false, true);
  }
  auto brrx1 = ib->Reshape(sum_r1, shape_of_x1);
  auto brrx2 = ib->Reshape(sum_r2, shape_of_x2);
  brrx1 = ib->Cast(brrx1, x1_dtype);
  brrx2 = ib->Cast(brrx2, x2_dtype);
  return {brrx1, brrx2};
}

NodePtrList FFTGradCommon(BpropBuilder *ib, const std::string &op_name) {
  auto x = ib->GetInput(kIndex0);
  auto n = ib->GetInput(kIndex1);
  auto dim = ib->GetInput(kIndex2);
  auto norm = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex5);

  constexpr int64_t norm_backward = 0;
  constexpr int64_t norm_forward = 1;
  constexpr int64_t norm_ortho = 2;

  // step1：Get the inputs needed to solve for the gradient.
  auto norm_type = norm->abstract()->BuildType();
  int64_t grad_norm_value = norm_forward;
  if (!norm_type->isa<TypeNone>()) {
    auto norm_value = GetValue<int64_t>(norm->BuildValue());
    switch (norm_value) {
      case norm_backward:
        grad_norm_value = norm_forward;
        break;
      case norm_forward:
        grad_norm_value = norm_backward;
        break;
      case norm_ortho:
        grad_norm_value = norm_ortho;
        break;
      default:
        break;
    }
  }

  // step2：Get the gradient.
  auto grad_dout = ib->Emit(op_name, {dout, n, dim, ib->Value(grad_norm_value)});

  // step3：If given, the gradient will be zero-padded or trimmed to this length.
  auto n_type = n->abstract()->BuildType();
  if (!n_type->isa<TypeNone>()) {
    auto input_shape = ib->GetShape(x);
    int64_t dim_value = CheckRange(GetIntValue(dim), SizeToLong(input_shape.size()));

    auto input_shape_node = ib->Shape(x);
    auto output_shape_node = ib->Shape(dout);
    auto input_dim_shape = ib->TupleGetItem(input_shape_node, dim_value);
    ShapeVector begin;
    ShapeVector strides;
    for (size_t i = 0; i < input_shape.size(); i++) {
      (void)begin.emplace_back(0LL);
      (void)strides.emplace_back(1LL);
    }
    auto slice_dout =
      ib->StridedSlice(grad_dout, ib->Value<ShapeVector>(begin), input_shape_node, ib->Value<ShapeVector>(strides));

    auto slicegrad_dout = ib->Emit(
      "StridedSliceGrad",
      {grad_dout, input_shape_node, ib->Value<ShapeVector>(begin), output_shape_node, ib->Value<ShapeVector>(strides)},
      {{kAttrBeginMask, MakeValue<int64_t>(0)},
       {kAttrEndMask, MakeValue<int64_t>(0)},
       {kAttrEllipsisMask, MakeValue<int64_t>(0)},
       {kAttrNewAxisMask, MakeValue<int64_t>(0)},
       {kAttrShrinkAxisMask, MakeValue<int64_t>(0)}});

    int64_t n_value = GetIntValue(n);
    int64_t dim_shape = GetIntValue(input_dim_shape);
    if (dim_shape < n_value) {
      grad_dout = slice_dout;
    } else {
      grad_dout = slicegrad_dout;
    }
  }

  // step4：Return gradient results.
  return {grad_dout, ib->OutZeros(n), ib->OutZeros(dim), ib->OutZeros(norm)};
}

inline NodePtr GradDiagonal(Emitter *ib, const NodePtr &dout, const NodePtr &dx_trans_shape,
                            std::tuple<int64_t, int64_t, int64_t, size_t> int_tuple, const TypePtr &x_dtype) {
  auto [offset, dim1, dim2, x_dim] = int_tuple;
  auto value = ib->Tensor(0, x_dtype);
  auto dx = ib->Emit("FillV2", {dx_trans_shape, value});
  auto k = ib->Tensor(offset, kInt32);
  constexpr int64_t max_length = 200000000;
  dx = ib->Emit("MatrixSetDiagV3", {dx, dout, k},
                {{"align", MakeValue("LEFT_RIGHT")}, {"max_length", MakeValue(max_length)}});
  int64_t dim = 0;
  ShapeVector perm(x_dim, 0);
  for (size_t i = 0; i < x_dim; ++i) {
    if (i == static_cast<size_t>(dim1)) {
      perm[i] = static_cast<int64_t>(x_dim - i2);
    } else if (i == static_cast<size_t>(dim2)) {
      perm[i] = static_cast<int64_t>(x_dim - i1);
    } else {
      perm[i] = dim;
      dim++;
    }
  }
  dx = ib->Transpose(dx, perm);
  return dx;
}

class DiagonalShapeCalc : public ShapeCalcFunctor {
 public:
  // cppcheck-suppress unknownMacro
  DECLARE_SHAPE_CALC("ShapeCalc_Diagonal", DiagonalShapeCalc)
  explicit DiagonalShapeCalc(int64_t dim1, int64_t dim2)
      : ShapeCalcFunctor("ShapeCalc_Diagonal"), dim1_(dim1), dim2_(dim2) {}
  ValuePtr ToValue() const override {
    auto values = {MakeValue(dim1_), MakeValue(dim2_)};
    return std::make_shared<ValueTuple>(values);
  }
  void FromValue(const ValuePtr &value) override {
    auto values = value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(values);
    if (values->value().size() != i2) {
      MS_LOG(EXCEPTION) << "DiagonalShapeCalc's value size should be 2, but got " << values->value().size();
    }
    dim1_ = GetValue<int64_t>(values->value()[0]);
    dim2_ = GetValue<int64_t>(values->value()[1]);
  }
  ShapeArray Calc(const ShapeArray &inputs) const override {
    auto x_shape = inputs.at(i0);
    auto out_shape = inputs.at(i1);
    ShapeVector dx_trans_shape(out_shape.begin(), out_shape.end() - i1);
    dx_trans_shape.push_back(x_shape[static_cast<size_t>(dim1_)]);
    dx_trans_shape.push_back(x_shape[static_cast<size_t>(dim2_)]);
    return {dx_trans_shape};
  }
  std::vector<int64_t> Infer(const ShapeArray &inputs, const HashSet<size_t> &) const override {
    auto out_shape = inputs.at(1);
    auto rank = IsDynamicRank(out_shape) ? -1 : SizeToLong(out_shape.size() + 1);
    return {rank};
  }

 protected:
  int64_t dim1_;
  int64_t dim2_;
};
REG_FUNCTOR("ShapeCalc_Diagonal", DiagonalShapeCalc);

REG_BPROP_BUILDERS_BEGIN(GradMathOps)
REG_BPROP_BUILDER("MatMul").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto ta = ib->GetAttr<bool>("transpose_a");
  auto tb = ib->GetAttr<bool>("transpose_b");
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto x_type = ib->GetDtype(x);
  auto w_type = ib->GetDtype(w);
  auto dout = ib->GetInput(kIndex3);
  NodePtr dx;
  NodePtr dw;

  if (((*x_type) == (*kComplex64) && (*w_type) == (*kComplex64)) ||
      ((*x_type) == (*kComplex128) && (*w_type) == (*kComplex128))) {
    // complex need conjoint transform
    if (x->need_compute_grad_out()) {
      if (ta) {
        dx = ib->MatMul(w, (ib->Emit("Conj", {dout})), (ta && tb), (ta || (!tb)));
        dx = ib->Emit("Conj", {dx});
      } else {
        dx = ib->MatMul((ib->Emit("Conj", {dout})), w, (ta && tb), (ta || (!tb)));
        dx = ib->Emit("Conj", {dx});
      }
    } else {
      dx = ib->OutZeros(x);
    }
    if (w->need_compute_grad_out()) {
      if (tb) {
        dw = ib->MatMul((ib->Emit("Conj", {dout})), x, ((!ta) || tb), (ta && tb));
        dw = ib->Emit("Conj", {dw});
      } else {
        dw = ib->MatMul((ib->Emit("Conj", {x})), dout, ((!ta) || tb), (ta && tb));
      }
    } else {
      dw = ib->OutZeros(w);
    }
    return {dx, dw};
  }

  if ((*x_type) == (*kComplex64) || (*x_type) == (*kComplex128) || (*w_type) == (*kComplex64) ||
      (*w_type) == (*kComplex128)) {
    // only support complex64 * complex64 and complex128 * complex128, others throw exception
    MS_EXCEPTION(TypeError) << "For 'MatMul', gradient not support x_type " << x_type << " * w_type " << w_type;
  }
  if (x->need_compute_grad_out()) {
    if (ta) {
      dx = ib->MatMul(w, dout, (ta && tb), (ta || (!tb)));
    } else {
      dx = ib->MatMul(dout, w, (ta && tb), (ta || (!tb)));
    }
  } else {
    dx = ib->OutZeros(x);
  }
  if (w->need_compute_grad_out()) {
    if (tb) {
      dw = ib->MatMul(dout, x, ((!ta) || tb), (ta && tb));
    } else {
      dw = ib->MatMul(x, dout, ((!ta) || tb), (ta && tb));
    }
  } else {
    dw = ib->OutZeros(w);
  }
  return {dx, dw};
});

REG_BPROP_BUILDER("Add").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr dx = nullptr;
  NodePtr dy = nullptr;
  if (x->need_compute_grad_out()) {
    dx = dout;
  }
  if (y->need_compute_grad_out()) {
    dy = dout;
  }
  return BinopGradCommon(ib, x, y, dx, dy);
});

REG_BPROP_BUILDER("AddExt").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);

  auto alpha = ib->GetInput(kIndex2);
  auto alpha_tensor = ib->ScalarToTensor(alpha, ib->GetDtype(x));
  auto alpha_value = alpha->BuildValue();
  if (!alpha_value->ContainsValueAny()) {
    MS_EXCEPTION_IF_NULL(alpha_value);
    auto imm_int64 = alpha_value->cast_ptr<Int64Imm>();
    if (imm_int64 != nullptr && imm_int64->value() == 1) {
      return BinopGradCommon(ib, x, y, dout, dout);
    }

    auto imm_float = alpha_value->cast_ptr<FP32Imm>();
    if (imm_float == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Invalid alpha type " << alpha_value->type_name()
                                 << " , alpha type should be Int64 or Float32";
    }
    auto alpha_scalar_float = imm_float->value();
    if (alpha_scalar_float == 1.0) {
      return BinopGradCommon(ib, x, y, dout, dout);
    }
  }
  std::vector<NodePtr> ret = BinopGradCommon(ib, x, y, dout, dout);
  ret.emplace_back(alpha);

  if (ib->GetDtypeId(x) == kNumberTypeComplex64) {
    alpha_tensor = ib->Cast(alpha_tensor, kNumberTypeComplex64);
  } else if (ib->GetDtypeId(x) == kNumberTypeComplex128) {
    alpha_tensor = ib->Cast(alpha_tensor, kNumberTypeComplex128);
  } else if (ib->GetDtypeId(x) == kNumberTypeBFloat16) {
    alpha_tensor = ib->Cast(alpha_tensor, kNumberTypeBFloat16);
  }

  ret[1] = ib->Mul(ret[1], alpha_tensor);
  return ret;
});

REG_BPROP_BUILDER("SubExt").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);

  auto alpha = ib->GetInput(kIndex2);
  auto alpha_tensor = ib->ScalarToTensor(alpha, ib->GetDtype(x));

  auto alpha_value = alpha->BuildValue();
  if (!alpha_value->ContainsValueAny()) {
    MS_EXCEPTION_IF_NULL(alpha_value);
    auto imm_int64 = alpha_value->cast_ptr<Int64Imm>();
    if (imm_int64 != nullptr && imm_int64->value() == 1) {
      return BinopGradCommon(ib, x, y, dout, ib->Emit(kNegOpName, {dout}));
    }

    auto imm_float = alpha_value->cast_ptr<FP32Imm>();
    if (imm_float == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Invalid alpha type " << alpha_value->type_name()
                                 << " , alpha type should be Int64 or Float32";
    }
    auto alpha_scalar_float = imm_float->value();
    if (alpha_scalar_float == 1.0) {
      return BinopGradCommon(ib, x, y, dout, ib->Emit(kNegOpName, {dout}));
    }
  }
  std::vector<NodePtr> ret = BinopGradCommon(ib, x, y, dout, ib->Emit(kNegOpName, {dout}));
  ret.emplace_back(alpha);

  if (ib->GetDtypeId(x) == kNumberTypeComplex64) {
    alpha_tensor = ib->Cast(alpha_tensor, kNumberTypeComplex64);
  } else if (ib->GetDtypeId(x) == kNumberTypeComplex128) {
    alpha_tensor = ib->Cast(alpha_tensor, kNumberTypeComplex128);
  } else if (ib->GetDtypeId(x) == kNumberTypeBFloat16) {
    alpha_tensor = ib->Cast(alpha_tensor, kNumberTypeBFloat16);
  }

  ret[1] = ib->Mul(ret[1], alpha_tensor);
  return ret;
});

REG_BPROP_BUILDER("Mul").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Mul', gradient not support for complex type currently.";
  }
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = ib->Mul(y, dout);
  }
  if (y->need_compute_grad_out()) {
    bc_dy = ib->Mul(x, dout);
  }
  return BinopGradCommon(ib, x, y, bc_dx, bc_dy);
});

REG_BPROP_BUILDER("Sub").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr dx = nullptr;
  NodePtr dy = nullptr;
  if (x->need_compute_grad_out()) {
    dx = dout;
  }
  if (y->need_compute_grad_out()) {
    dy = ib->Emit(kNegOpName, {dout});
  }
  return BinopGradCommon(ib, x, y, dx, dy);
});

REG_BPROP_BUILDER("Div").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  auto x_dtype_id = ib->GetDtypeId(x);
  bc_dx = ib->Div(dout, y);
  if (y->need_compute_grad_out()) {
    bc_dy = -(bc_dx * out);
  }
  auto result = BinopGradCommon(ib, x, y, bc_dx, bc_dy);
  bool is_complex = (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128);
  if (is_complex) {
    result[kIndex0] = ib->Conj(result[kIndex0]);
    result[kIndex1] = y->need_compute_grad_out() ? ib->Conj(result[kIndex1]) : ib->OutZeros(y);
  }
  return result;
});

REG_BPROP_BUILDER("BitwiseAnd").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseOr").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("BitwiseXor").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("InplaceSub").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("InplaceAdd").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("InplaceUpdate").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);
REG_BPROP_BUILDER("InplaceUpdateV2").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Less").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("LessEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("LogicalNot").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("LogicalAnd").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("LogicalOr").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("AssignAdd").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("AssignSub").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Sin").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(dout, (ib->Emit("Cos", {x})));
  return {dx};
});

REG_BPROP_BUILDER("Asin").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AsinGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("AsinGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr d2x;
  if (x->need_compute_grad_out()) {
    auto one = ib->Tensor(1, ib->GetDtype(x));
    auto minus_one_p5 = ib->Tensor(-1.5, ib->GetDtype(x));
    d2x = ib->Mul((ib->Mul((ib->Mul(dout, grad)), x)), (ib->Pow(ib->Sub(one, (ib->Mul(x, x))), minus_one_p5)));
  } else {
    d2x = ib->OutZeros(x);
  }
  auto ddy = grad->need_compute_grad_out() ? ib->Emit("AsinGrad", {x, dout}) : ib->OutZeros(grad);
  return {d2x, ddy};
});

REG_BPROP_BUILDER("Asinh").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AsinhGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("AsinhGrad").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto y = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  NodePtr dy;
  if (y->need_compute_grad_out()) {
    auto minus_one = ib->Tensor(-1.0, ib->GetDtype(out));
    dy = ib->Mul((ib->Mul((ib->Mul(dout, out)), minus_one)), (ib->Emit("Tanh", {y})));
  } else {
    dy = ib->OutZeros(y);
  }

  auto dgrad = grad->need_compute_grad_out() ? ib->Emit("AsinhGrad", {y, dout}) : ib->OutZeros(grad);
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Sinh").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto conj_x = ib->Conj(x);
  auto dx = ib->Mul((ib->Emit("Cosh", {conj_x})), dout);
  return {dx};
});

REG_BPROP_BUILDER("Cos").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(dout, (ib->Emit("Neg", {ib->Emit("Sin", {x})})));
  return {dx};
});

REG_BPROP_BUILDER("ACos").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("ACosGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("ACosGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr d2x;
  if (x->need_compute_grad_out()) {
    auto one = ib->Tensor(1, ib->GetDtype(x));
    auto minus_one_p5 = ib->Tensor(-1.5, ib->GetDtype(x));
    d2x = ib->Mul((ib->Mul((ib->Mul((ib->Emit("Neg", {dout})), grad)), x)),
                  (ib->Pow(ib->Sub(one, (ib->Mul(x, x))), minus_one_p5)));
  } else {
    d2x = ib->OutZeros(x);
  }
  auto ddy = grad->need_compute_grad_out() ? ib->Emit("ACosGrad", {x, dout}) : ib->OutZeros(grad);
  return {d2x, ddy};
});

REG_BPROP_BUILDER("Acosh").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AcoshGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("AcoshGrad").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto y = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dy =
    y->need_compute_grad_out()
      ? ib->RealDiv((ib->Mul((ib->Mul(dout, out)), ib->Tensor(-1.0, ib->GetDtype(out)))), (ib->Emit("Tanh", {y})))
      : ib->OutZeros(y);
  auto dgrad = grad->need_compute_grad_out() ? ib->Emit("AcoshGrad", {y, dout}) : ib->OutZeros(grad);
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Cosh").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Cosh', gradient not support for complex type currently.";
  } else {
    dx = ib->Mul((ib->Emit("Sinh", {x})), dout);
  }
  return {dx};
});

REG_BPROP_BUILDER("Abs").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AbsGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("Conj").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Conj(dout);
  return {dx};
});

REG_BPROP_BUILDER("ScalarCast").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto t = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  if (x->abstract()->isa<abstract::AbstractTensor>()) {
    auto dx = ib->Emit("ScalarToTensor", {dout, ib->Value<int64_t>(ib->GetDtype(x)->type_id())});
    return {dx, ib->OutZeros(t)};
  }
  auto dx = ib->Emit("ScalarCast", {dout, ib->Value<int64_t>(ib->GetDtypeId(x))});
  return {dx, ib->OutZeros(t)};
});

REG_BPROP_BUILDER("Sign").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Round").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Atan2").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp = ib->RealDiv(dout, (ib->Add((ib->Emit("Square", {x})), (ib->Emit("Square", {y})))));
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = ib->Mul(tmp, y);
  }
  if (y->need_compute_grad_out()) {
    bc_dy = ib->Mul(tmp, (ib->Emit("Neg", {x})));
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("BesselI0e").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(dout, (ib->Sub((ib->Emit("BesselI1e", {x})), (ib->Mul((ib->Emit("Sign", {x})), out)))));
  return {dx};
});

REG_BPROP_BUILDER("Atan").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AtanGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("AtanGrad").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dgrad = ib->Emit("AtanGrad", {x, dout});
  auto dx = x->need_compute_grad_out() ? ib->Mul((ib->Mul((ib->Mul(out, dgrad)), ib->Tensor(-2.0, ib->GetDtype(x)))), x)
                                       : ib->OutZeros(x);
  return {dx, dgrad};
});

REG_BPROP_BUILDER("Log1p").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_1p = ib->Add(x, ib->Tensor(1, ib->GetDtype(x)));
  auto g = ib->Emit("Reciprocal", {x_1p});
  auto dx = ib->Mul(g, dout);
  return {dx};
});

REG_BPROP_BUILDER("Erf").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto half_root_pi =
    ib->Cast(ib->RealDiv(ib->Tensor(2, ib->GetDtype(x)), (ib->Sqrt(ib->Tensor(pi, ib->GetDtype(x))))), ib->GetDtype(x));
  auto x_square = ib->Emit("Square", {x});
  auto dx = ib->Mul((ib->Mul(dout, half_root_pi)), (ib->Exp(ib->Emit("Neg", {x_square}))));
  return {dx};
});

REG_BPROP_BUILDER("Erfc").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto half_root_pi =
    ib->Cast(ib->RealDiv(ib->Tensor(2, ib->GetDtype(x)), (ib->Sqrt(ib->Tensor(pi, ib->GetDtype(x))))), ib->GetDtype(x));
  auto x_square = ib->Emit("Square", {x});
  auto dx = ib->Mul(dout, (ib->Mul((ib->Emit("Neg", {half_root_pi})), (ib->Exp(ib->Emit("Neg", {x_square}))))));
  return {dx};
});

REG_BPROP_BUILDER("Pow").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto power = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Pow', gradient not support for complex type currently.";
  }
  NodePtr dx = nullptr;
  NodePtr grad_power = nullptr;
  if (x->need_compute_grad_out()) {
    dx = ib->Mul((ib->Mul(power, (ib->Pow(x, ib->Sub(power, ib->Tensor(1.0, ib->GetDtype(x))))))), dout);
  }
  if (power->need_compute_grad_out()) {
    x = ib->Select(ib->Less(x, ib->Tensor(0, ib->GetDtype(x))), ib->Fill(1.0, ib->Shape(x), ib->GetDtype(x)->type_id()),
                   x);
    grad_power = ib->Mul((ib->Mul(out, (ib->Log(x)))), dout);
  }
  return {BinopGradCommon(ib, x, power, dx, grad_power)};
});

REG_BPROP_BUILDER("Exp").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto g = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(g, dout);
  return {dx};
});

REG_BPROP_BUILDER("Expm1").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto g = ib->Exp(x);
  auto dx = ib->Mul(g, dout);
  return {dx};
});

REG_BPROP_BUILDER("Minimum").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return MinimumMaximumGrad(ib, x, y, dout, true);
});

REG_BPROP_BUILDER("Maximum").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return MinimumMaximumGrad(ib, x, y, dout, false);
});

REG_BPROP_BUILDER("CumSum").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  auto axis = ib->GetInput(kIndex1);
  auto exclusive = ib->GetInput(kIndex2);
  auto reverse = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  return {ib->CumSum(dout, axis, GetValue<bool>(exclusive->BuildValue()), !GetValue<bool>(reverse->BuildValue())),
          ib->OutZeros(axis), ib->OutZeros(exclusive), ib->OutZeros(reverse)};
});

REG_BPROP_BUILDER("MulNoNan").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_shape = ib->Shape(x);
  auto y_shape = ib->Shape(y);
  auto dx = ib->MulNoNan(dout, y);
  auto dy = ib->MulNoNan(x, dout);
  auto bc_axis = ib->BroadcastGradientArgs(x, y);
  auto broadcast_x = bc_axis[kIndex0];
  auto broadcast_y = bc_axis[kIndex1];
  dx = x->need_compute_grad_out() ? ib->Reshape(ib->ReduceSum(dx, broadcast_x, false, true), x_shape) : ib->OutZeros(x);
  dy = y->need_compute_grad_out() ? ib->Reshape(ib->ReduceSum(dy, broadcast_y, false, true), y_shape) : ib->OutZeros(y);
  return {dx, dy};
});

REG_BPROP_BUILDER("BesselI0").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_i1 = ib->Emit("BesselI1", {x});
  auto dx = ib->Mul(dout, bessel_i1);
  return {dx};
});

REG_BPROP_BUILDER("BesselI1").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_i0 = ib->Emit("BesselI0", {x});
  auto zero = ib->ZerosLike(x);
  auto one = ib->Fill(1.0, ib->Shape(x), ib->GetDtype(x)->type_id());
  auto dout_dx = ib->Select(ib->Equal(x, zero), one, ib->Sub(bessel_i0, (ib->Div(out, x))));
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselJ0").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_j1 = ib->Emit("BesselJ1", {x});
  auto dx = ib->Mul(ib->Emit("Neg", {dout}), bessel_j1);
  return {dx};
});

REG_BPROP_BUILDER("BesselJ1").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_j0 = ib->Emit("BesselJ0", {x});
  auto zero = ib->ZerosLike(x);
  auto zero_p5 = ib->Fill(0.5, ib->Shape(x), ib->GetDtype(x)->type_id());
  auto dout_dx = ib->Select(ib->Equal(x, zero), zero_p5, ib->Sub(bessel_j0, (ib->Div(out, x))));
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselK0").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_k1 = ib->Emit("BesselK1", {x});
  auto dx = ib->Mul(ib->Emit("Neg", {dout}), bessel_k1);
  return {dx};
});

REG_BPROP_BUILDER("BesselK1").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_k0 = ib->Emit("BesselK0", {x});
  auto dout_dx = ib->Emit("Neg", {ib->Add(bessel_k0, (ib->Div(out, x)))});
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselK0e").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_k1e = ib->Emit("BesselK1e", {x});
  auto dx = ib->Mul(dout, (ib->Sub(out, bessel_k1e)));
  return {dx};
});

REG_BPROP_BUILDER("BesselK1e").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_k0e = ib->Emit("BesselK0e", {x});
  auto one = ib->Tensor(1, ib->GetDtype(x));
  auto dout_dx = ib->Sub((ib->Mul(out, (ib->Sub(one, (ib->Emit("Reciprocal", {x})))))), bessel_k0e);
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselY0").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_y1 = ib->Emit("BesselY1", {x});
  auto dx = ib->Mul(ib->Emit("Neg", {dout}), bessel_y1);
  return {dx};
});

REG_BPROP_BUILDER("BesselY1").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_y0 = ib->Emit("BesselY0", {x});
  auto dout_dx = ib->Sub(bessel_y0, (ib->Div(out, x)));
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("AddN").SetUnusedInputs({i0, i1}).SetBody(AddnGradFunc);

REG_BPROP_BUILDER("AccumulateNV2").SetUnusedInputs({i0, i1}).SetBody(AddnGradFunc);

REG_BPROP_BUILDER("Tan").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Tan', gradient not support for complex type currently.";
  } else {
    auto cosx = ib->Emit("Cos", {x});
    auto secx2 = ib->Square(ib->Reciprocal(cosx));
    dx = secx2 * dout;
  }
  return {dx};
});

REG_BPROP_BUILDER("BesselI1e").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_dtype = ib->GetDtype(x);
  auto zeros = ib->ZerosLike(x);
  auto eps = GetEps(ib, x_dtype);
  auto x_is_valid = ib->Less(eps, ib->Abs(x));
  auto x_safe = ib->Select(x_is_valid, x, eps + zeros);
  auto besselI0e = ib->Emit(prim::kPrimBesselI0e->name(), {x_safe});
  auto tmp = besselI0e - out * (ib->Sign(x_safe) + ib->Reciprocal(x_safe));
  auto dx = ib->Select(x_is_valid, tmp, ib->Tensor(0.5, x_dtype) + zeros) * dout;
  return {dx};
});

REG_BPROP_BUILDER("Atanh").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_dtype = ib->GetDtype(x);
  auto x_dtype_id = x_dtype->type_id();
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Atanh', gradient not support for complex type currently.";
  } else {
    auto one = ib->Tensor(1, x_dtype);
    auto tmp = one - ib->Pow(x, ib->Tensor(2, x_dtype));
    dx = ib->Div(one, tmp) * dout;
  }
  return {dx};
});

REG_BPROP_BUILDER("Inv").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(prim::kPrimInvGrad->name(), {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("LinSpace").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IndexAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto axis = ib->EmitValue(ib->GetAttr(kAttrAxis));
  auto dy = ib->Gather(dout, indices, axis);
  return {dout, ib->OutZeros(indices), dy};
});

REG_BPROP_BUILDER("Logit").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto eps = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("LogitGrad", {dout, x, eps});
  return {dx, ib->OutZeros(eps)};
});

DEF_PURE_SHAPE_CALC(g_cdist)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto dout_shape = inputs.at(0);
    auto dout_dim = dout_shape.size();
    ShapeVector perm;
    for (uint64_t i = 0; i < dout_dim - 2; ++i) {
      perm.push_back(i);
    }
    perm.push_back(dout_dim - 1);
    perm.push_back(dout_dim - 2);
    return {perm};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto dout_shape = inputs.at(0);
    return {IsDynamicRank(dout_shape) ? -1 : static_cast<int64_t>(dout_shape.size())};
  });
REG_BPROP_BUILDER("Cdist").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto input_y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto res = ib->ShapeCalc(g_cdist, {dout})[0];
  auto dout_transpose = ib->Transpose(dout, res);
  auto out_transpose = ib->Transpose(out, res);
  auto dx = input_x->need_compute_grad_out()
              ? ib->Emit("CdistGrad", {dout, input_x, input_y, out}, {{"p", ib->GetAttr("p")}})
              : ib->OutZeros(input_x);
  auto dy = input_y->need_compute_grad_out()
              ? ib->Emit("CdistGrad", {dout_transpose, input_y, input_x, out_transpose}, {{"p", ib->GetAttr("p")}})
              : ib->OutZeros(input_y);
  return {dx, dy};
});

REG_BPROP_BUILDER("LuUnpack").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  auto lu_data = ib->GetInput(kIndex0);
  auto lu_pivots = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp = ib->Emit(
    "LuUnpackGrad", {ib->TupleGetItem(dout, 1), ib->TupleGetItem(dout, 2), lu_data},
    {{"L_grad_flag", MakeValue(true)}, {"U_grad_flag", MakeValue(true)}, {"cust_aicpu", MakeValue("LuUnpackGrad")}});
  auto dl = ib->TupleGetItem(tmp, 0);
  auto du = ib->TupleGetItem(tmp, 1);
  auto lu_data_grad = ib->Add(dl, du);
  return {lu_data_grad, ib->OutZeros(lu_pivots)};
});

REG_BPROP_BUILDER("Sinc").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto product = ib->Mul(ib->Tensor(pi, ib->GetDtype(x)), x);
  auto reciprocal = ib->RealDiv(
    (ib->Sub((ib->Mul(product, (ib->Emit("Cos", {product})))), (ib->Emit("Sin", {product})))), (ib->Mul(product, x)));
  TypeId rec_type = ib->GetDtypeId(reciprocal);
  if (rec_type == kNumberTypeComplex64 || rec_type == kNumberTypeComplex128) {
    reciprocal = ib->Conj(reciprocal);
  }
  auto dx = ib->Mul(reciprocal, dout);
  return {dx};
});

REG_BPROP_BUILDER("CumProd").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto x_shape = ib->GetShape(x);
  auto num_elements =
    std::accumulate(x_shape.begin(), x_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  auto axis = ib->GetInput(kIndex1);
  auto axis_value_ptr = axis->BuildValue();
  auto axis_opt = mindspore::ops::GetScalarValue<int64_t>(axis_value_ptr);
  int64_t axis_value;
  if (axis_opt.has_value()) {
    axis_value = axis_opt.value();
  } else {
    MS_LOG_EXCEPTION << "For CumProd, got an invalid 'axis'.";
  }
  auto exclusive = ib->GetInput(kIndex2);
  auto reverse = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex5);
  constexpr const int64_t One = 1;
  // to par with standards when dim is 1 or element num of input is no greater than 1.
  if (!IsDynamic(x_shape) && (num_elements <= One || x_shape[axis_value] == One)) {
    return {dout, ib->OutZeros(axis), ib->OutZeros(exclusive), ib->OutZeros(reverse)};
  }
  auto prod = ib->Emit("CumProd", {x, axis, exclusive, reverse});
  out = ib->CumSum(ib->Mul(prod, dout), axis, GetValue<bool>(exclusive->BuildValue()),
                   !GetValue<bool>(reverse->BuildValue()));
  return {ib->RealDiv(out, x), ib->OutZeros(axis), ib->OutZeros(exclusive), ib->OutZeros(reverse)};
});

REG_BPROP_BUILDER("IsFinite").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IsNan").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IsInf").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ReduceAll").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ReduceAny").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ApproximateEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Equal").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NotEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Greater").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("GreaterEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("MatrixInverse").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto out_shape = ib->GetShape(out);
  auto dx = out;
  if (out_shape.size() == 2) {
    dx = ib->MatMul(dout, dx, false, true);
    dx = ib->MatMul(out, dx, true, false);
  } else if (out_shape.size() > 2 || IsDynamicRank(out_shape)) {
    dx = ib->BatchMatMul(dout, dx, false, true);
    dx = ib->BatchMatMul(out, dx, true, false);
  }
  return {-dx};
});

REG_BPROP_BUILDER("Neg").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {-dout};
});

REG_BPROP_BUILDER("RealDiv").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'RealDiv', gradient not support for complex type currently.";
  }
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto bc_dx = ib->RealDiv(dout, y);
  NodePtr bc_dy = nullptr;
  if (y->need_compute_grad_out()) {
    bc_dy = -(bc_dx * out);
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("DivNoNan").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto bc_dx = ib->DivNoNan(dout, y);
  NodePtr bc_dy = nullptr;
  if (y->need_compute_grad_out()) {
    bc_dy = -(bc_dx * out);
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Xdivy").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype = ib->GetDtype(x);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    auto not_zero_x = ib->Cast(ib->NotEqual(x, ib->Tensor(0.0, x_dtype)), x_dtype);
    bc_dx = (ib->Xdivy(not_zero_x, y)) * dout;
  }
  if (y->need_compute_grad_out()) {
    bc_dy = (ib->Xdivy(-x, ib->Emit("Square", {y}))) * dout;
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("FloorDiv").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("FloorMod").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = ib->Cast(dout, ib->GetDtype(x));
  }
  if (y->need_compute_grad_out()) {
    bc_dy = (-dout) * (ib->FloorDiv(x, y));
    bc_dy = ib->Cast(bc_dy, ib->GetDtype(y));
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("TruncateDiv").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("TruncateMod").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = dout;
  }
  if (y->need_compute_grad_out()) {
    bc_dy = (-dout) * (ib->Emit("TruncateDiv", {x, y}));
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Mod").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = ib->Cast(dout, ib->GetDtype(x));
  }
  if (y->need_compute_grad_out()) {
    bc_dy = (-dout) * (ib->FloorDiv(x, y));
    bc_dy = ib->Cast(bc_dy, ib->GetDtype(y));
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Xlogy").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype = ib->GetDtype(x);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    auto not_zero_x = ib->Cast(ib->NotEqual(x, ib->Tensor(0.0, x_dtype)), x_dtype);
    bc_dx = ib->Xlogy(not_zero_x, y) * dout;
  }
  if (y->need_compute_grad_out()) {
    bc_dy = ib->Xdivy(x, y) * dout;
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Sqrt").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SqrtGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("SqrtGrad").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto y = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto gy = ib->RealDiv(dout, y);
  auto dy = y->need_compute_grad_out() ? (-gy) * out : ib->OutZeros(y);
  NodePtr dgrad;
  if (grad->need_compute_grad_out()) {
    auto gy_dtype = ib->GetDtype(gy);
    dgrad = ib->Tensor(0.5, gy_dtype) * gy;
  } else {
    dgrad = ib->OutZeros(grad);
  }
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Rsqrt").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("RsqrtGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("RsqrtGrad").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto y = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr dy;
  if (y->need_compute_grad_out()) {
    auto grad_dtype = ib->GetDtype(grad);
    dy = ib->Tensor(-1.5, grad_dtype) * grad * y * y * dout;
  } else {
    dy = ib->OutZeros(y);
  }
  auto dgrad = grad->need_compute_grad_out() ? ib->Emit("RsqrtGrad", {y, dout}) : ib->OutZeros(grad);
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Reciprocal").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("ReciprocalGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("Log").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("Div", {dout, x})};
});

REG_BPROP_BUILDER("Floor").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Ceil").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Square").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = dout * x * ib->Tensor(2.0, ib->GetDtype(x));
  return {dx};
});

REG_BPROP_BUILDER("SquaredDifference").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = dout * (x - y) * ib->Tensor(2.0, ib->GetDtype(x));
  return {BinopGradCommon(ib, x, y, dx, -dx)};
});

REG_BPROP_BUILDER("SquareSumAll").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr dx;
  if (x->need_compute_grad_out()) {
    auto dout_0 = ib->TupleGetItem(dout, kIndex0);
    dx = dout_0 * x * ib->Tensor(2.0, ib->GetDtype(x));
  } else {
    dx = ib->OutZeros(x);
  }
  NodePtr dy;
  if (y->need_compute_grad_out()) {
    auto dout_1 = ib->TupleGetItem(dout, kIndex1);
    dy = dout_1 * y * ib->Tensor(2.0, ib->GetDtype(y));
  } else {
    dy = ib->OutZeros(y);
  }
  return {dx, dy};
});

REG_BPROP_BUILDER("Hypot").SetBody(BODYFUNC(ib) {
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto x1_f32 = ib->Cast(x1, kFloat32);
  auto x2_f32 = ib->Cast(x2, kFloat32);
  auto out_f32 = ib->Cast(out, kFloat32);
  auto dout_f32 = ib->Cast(dout, kFloat32);
  NodePtr dx1 = nullptr;
  NodePtr dx2 = nullptr;
  if (x1->need_compute_grad_out()) {
    dx1 = ib->Mul(ib->Div(x1_f32, out_f32), dout_f32);
  }
  if (x2->need_compute_grad_out()) {
    dx2 = ib->Mul(ib->Div(x2_f32, out_f32), dout_f32);
  }
  auto tmp = BinopGradCommon(ib, x1_f32, x2_f32, dx1, dx2);
  auto result_dx1 = x1->need_compute_grad_out() ? ib->Cast(tmp[0], ib->GetDtype(x1)) : ib->OutZeros(x1);
  auto result_dx2 = x2->need_compute_grad_out() ? ib->Cast(tmp[1], ib->GetDtype(x2)) : ib->OutZeros(x2);
  return {result_dx1, result_dx2};
});

REG_BPROP_BUILDER("Trunc").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Ger").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto input_y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  ShapeVector axis = {1};
  NodePtr dx;
  if (input_x->need_compute_grad_out()) {
    auto m1 = ib->ExpandDims(input_y, 1);
    dx = ib->Squeeze(ib->MatMul(dout, m1, false, false), MakeValue(axis));
  } else {
    dx = ib->OutZeros(input_x);
  }
  NodePtr dy;
  if (input_y->need_compute_grad_out()) {
    auto m2 = ib->ExpandDims(input_x, 1);
    ShapeVector perm = {1, 0};
    auto transpose = ib->Transpose(dout, perm);
    dy = ib->Squeeze(ib->MatMul(transpose, m2, false, false), MakeValue(axis));
  } else {
    dy = ib->OutZeros(input_y);
  }
  return {dx, dy};
});

REG_BPROP_BUILDER("Cross").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto input1 = ib->GetInput(kIndex0);
  auto input2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dinput1 = input1->need_compute_grad_out() ? ib->Emit("Cross", {input2, dout}, {{"dim", ib->GetAttr("dim")}})
                                                 : ib->OutZeros(input1);
  auto dinput2 = input2->need_compute_grad_out() ? ib->Emit("Cross", {dout, input1}, {{"dim", ib->GetAttr("dim")}})
                                                 : ib->OutZeros(input2);
  return {dinput1, dinput2};
});

REG_BPROP_BUILDER("Median").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Cast(ib->Emit("MedianGrad", {ib->TupleGetItem(dout, 0), x, ib->TupleGetItem(out, 0), ib->TupleGetItem(out, 1)},
                      {{"global_median", ib->GetAttr("global_median")},
                       {"axis", ib->GetAttr("axis")},
                       {"keep_dims", ib->GetAttr("keep_dims")}}),
             ib->GetDtype(x));
  return {dx};
});

REG_BPROP_BUILDER("Trace").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shape = ib->Shape(x, true);
  auto dx = ib->Emit("TraceGrad", {dout, shape});
  return {dx};
});

REG_BPROP_BUILDER("Erfinv").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto out_type = ib->GetDtype(dout);
  auto sqrt = ib->Sqrt(ib->Tensor(pi, out_type));
  auto root_pi_over_two = ib->RealDiv(sqrt, ib->Tensor(2, ib->GetDtype(sqrt)));
  auto out_square = ib->Emit("Square", {out});
  auto dx = ib->Mul((ib->Mul(dout, root_pi_over_two)), (ib->Exp(out_square)));
  return {dx};
});

REG_BPROP_BUILDER("Bernoulli").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ReduceSum").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto skip_mode = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto dx =
    SumGrad(ib, x, axis, dout, GetValue<bool>(keep_dims->BuildValue()), GetValue<bool>(skip_mode->BuildValue()));
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims), ib->OutZeros(skip_mode)};
});

DEF_PURE_SHAPE_CALC(g_reduce_prod)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_shape = inputs.at(0);
    auto axis = inputs.at(1);
    auto output_shape_kept_dims = ReduceShape(input_shape, axis);
    auto tile_scaling = TupleDiv(input_shape, output_shape_kept_dims);
    auto [pack_shape, perm] = SplitShapeIndex(input_shape, axis);
    return {output_shape_kept_dims, tile_scaling, pack_shape, perm, InvertPermutation(perm)};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    if (!unknown_inputs.empty()) {
      return {-1, -1, 2, -1, -1};
    }
    auto size = SizeToLong(inputs.at(0).size());
    return {size, size, 2, size, size};
  });

REG_BPROP_BUILDER("ReduceProd").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceProd', gradient not support for complex type currently.";
  }
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  if (ib->GetRank(x) == 0) {
    return {dout, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
  }
  auto res = ib->ShapeCalc(g_reduce_prod, {x, axis}, {1});
  auto keep_dims_value = GetValue<bool>(keep_dims->BuildValue());
  auto grad = keep_dims_value ? dout : ib->Reshape(dout, res[0]);
  grad = ib->Tile(grad, res[1]);

  auto permuted = ib->Transpose(x, res[3]);
  auto permuted_shape = ib->Shape(permuted);
  auto reshaped = ib->Reshape(permuted, res[2]);
  auto left = ib->CumProd(reshaped, ib->Value<int64_t>(0), true, false);
  auto right = ib->CumProd(reshaped, ib->Value<int64_t>(0), true, true);
  auto y = ib->Reshape(ib->Mul(left, right), permuted_shape);
  auto out = ib->Mul(ib->Transpose(y, res[4]), grad);
  auto dx = ib->Reshape(out, ib->Shape(x));
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("ReduceMax").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceMax', gradient not support for complex type currently.";
  } else {
    auto dx = MinOrMaxGrad(ib, x, axis, keep_dims, out, dout);
    return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
  }
});

REG_BPROP_BUILDER("ReduceMin").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceMin', gradient not support for complex type currently.";
  } else {
    auto dx = MinOrMaxGrad(ib, x, axis, keep_dims, out, dout);
    return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
  }
});

REG_BPROP_BUILDER("ReduceMean").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceMean', gradient not support for complex type currently.";
  }
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto grad = SumGrad(ib, x, axis, dout, GetValue<bool>(keep_dims->BuildValue()));
  NodePtr div_shape_node;
  if (IsDynamic(ib->GetShape(x)) || IsDynamic(ib->GetShape(out))) {
    auto shape_out_sz = ib->DynSize(out, kFloat32);
    auto div_shape = ib->DynSize(x, kFloat32) / shape_out_sz;
    div_shape_node = ib->Cast(div_shape, ib->GetDtype(grad));
  } else {
    auto shape_out_sz = ib->GetSize(out);
    if (shape_out_sz == 0) {
      MS_EXCEPTION(ValueError) << "out shape size can not be 0";
    }
    auto div_shape = ib->GetSize(x) / shape_out_sz;
    div_shape_node = ib->Tensor(div_shape, ib->GetDtype(grad));
  }
  auto dx = ib->RealDiv(grad, div_shape_node);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("ArgMaxWithValue").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, true);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("ArgMinWithValue").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto keep_dims = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, false);
  return {dx, ib->OutZeros(axis), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("ComplexAbs").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  return {ib->DivNoNan(ib->Mul(ib->Complex(dout, ib->ZerosLike(dout)), x), ib->Complex(out, ib->ZerosLike(out)))};
});

REG_BPROP_BUILDER("Real").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto zero = ib->ZerosLike(dout);
  return {ib->Complex(dout, zero)};
});

REG_BPROP_BUILDER("Imag").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto zero = ib->ZerosLike(dout);
  return {ib->Complex(zero, dout)};
});

REG_BPROP_BUILDER("Betainc").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto input_a = ib->GetInput(kIndex0);
  auto input_b = ib->GetInput(kIndex1);
  auto input_x = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto sx = ib->Shape(input_x);
  auto log_beta =
    ib->Emit("Lgamma", {input_a}) + ib->Emit("Lgamma", {input_b}) - ib->Emit("Lgamma", {ib->Add(input_a, input_b)});
  auto partial_x = ib->Exp(ib->Sub(
    (ib->Add(
      (ib->Mul((ib->Sub(input_b, ib->Tensor(1, ib->GetDtype(input_b)))), (ib->Emit("Log1p", {ib->Neg(input_x)})))),
      (ib->Xlogy(ib->Sub(input_a, ib->Tensor(1, ib->GetDtype(input_b))), input_x)))),
    log_beta));
  return {ib->OutZeros(input_a), ib->OutZeros(input_b), ib->Reshape(ib->Mul(partial_x, dout), sx)};
});

DEF_PURE_SHAPE_CALC(g_matrix_determinant).SetCalc(MatrixDeterminantShapeFunc).SetInfer(MatrixDeterminantInferFunc);
REG_BPROP_BUILDER("LogMatrixDeterminant").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_adj_inv = ib->Emit("MatrixInverse", {x}, {{"adjoint", MakeValue(true)}});
  auto res = ib->ShapeCalc(g_matrix_determinant, {ib->TupleGetItem(out, 1)})[0];
  auto multipliers = ib->Reshape(ib->TupleGetItem(dout, 1), res);
  auto dx = ib->Mul(multipliers, x_adj_inv);
  return {dx};
});

REG_BPROP_BUILDER("MatrixDeterminant").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_adj_inv = ib->Emit("MatrixInverse", {x}, {{"adjoint", MakeValue(true)}});
  auto res = ib->ShapeCalc(g_matrix_determinant, {out})[0];
  auto multipliers = ib->Reshape(ib->Mul(dout, out), res);
  auto dx = ib->Mul(multipliers, x_adj_inv);
  return {dx};
});

REG_BPROP_BUILDER("MatrixPower").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto n = GetValue<int64_t>(ib->GetAttr("n"));
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  dout = ib->Cast(dout, kFloat32);
  x = ib->Cast(x, kFloat32);
  auto power = n;
  auto dx = ib->ZerosLike(x);
  auto EmitBmmPowerGrad = [&ib, &dx, &dout](int64_t repeat, const NodePtr &a, const NodePtr &b) {
    for (int64_t i = 0; i < repeat; ++i) {
      dx = ib->Add(dx, (ib->BatchMatMul(b, ib->Emit("MatrixPower", {a}, {{"n", MakeValue<int64_t>(repeat - 1 - i)}}),
                                        false, true)));
      dout = ib->BatchMatMul(a, b, true, false);
    }
  };
  if (power < 0) {
    auto x_inv = ib->Emit("MatrixPower", {x}, {{"n", MakeValue<int64_t>(-1)}});
    EmitBmmPowerGrad(-power, x_inv, dout);
    dx = ib->BatchMatMul(dx, x_inv, false, true);
    dx = ib->BatchMatMul(x_inv, dx, true, false);
    dx = ib->Emit("Neg", {dx});
  } else {
    EmitBmmPowerGrad(power, x, dout);
  }
  dx = ib->Cast(dx, ib->GetDtype(out));
  return {dx};
});

REG_BPROP_BUILDER("MatrixSolve").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto adjoint = GetValue<bool>(ib->GetAttr("adjoint"));
  auto input_a = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  if (ib->GetDtypeId(out) == kNumberTypeFloat64) {
    out = ib->Cast(out, kFloat32);
  }
  auto grad_b = ib->Emit("MatrixSolve", {input_a, dout}, {{"adjoint", MakeValue(!adjoint)}});
  if (ib->GetDtypeId(grad_b) == kNumberTypeFloat64) {
    grad_b = ib->Cast(grad_b, kFloat32);
  }
  auto a_shape = ib->GetShape(input_a);
  auto matrix_rank = a_shape.size();
  auto EmitBmmGrad = [&ib](const NodePtr &a, const NodePtr &b) -> NodePtr {
    auto grad_a = ib->BatchMatMul(a, b, false, true);
    grad_a = ib->Emit("Neg", {grad_a});
    return grad_a;
  };
  auto EmitMatmulGrad = [&ib](const NodePtr &a, const NodePtr &b) -> NodePtr {
    auto grad_a = ib->MatMul(a, b, false, true);
    grad_a = ib->Emit("Neg", {grad_a});
    return grad_a;
  };
  if (adjoint) {
    if (matrix_rank > 2) {
      return {EmitBmmGrad(out, grad_b), grad_b};
    } else {
      return {EmitMatmulGrad(out, grad_b), grad_b};
    }
  } else {
    if (matrix_rank > 2) {
      return {EmitBmmGrad(grad_b, out), grad_b};
    } else {
      return {EmitMatmulGrad(grad_b, out), grad_b};
    }
  }
});

DEF_PURE_SHAPE_CALC(g_matrix_exp)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto shape_x = inputs.at(0);
    auto x_len = shape_x.size();
    auto input_perm = Range(SizeToLong(x_len));
    input_perm[x_len - kDim2] = SizeToLong(x_len - kDim1);
    input_perm[x_len - kDim1] = SizeToLong(x_len - kDim2);
    auto n = shape_x[x_len - kDim1];
    ShapeVector begins(x_len, 0);
    begins[x_len - kDim1] = n;
    auto sizes = shape_x;
    sizes[x_len - kDim1] = n;
    sizes[x_len - kDim2] = n;
    return {input_perm, begins, sizes};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    auto shape_x = inputs.at(0);
    auto rank = IsDynamicRank(shape_x) ? -1 : static_cast<int64_t>(shape_x.size());
    return {rank, rank, rank};
  });

REG_BPROP_BUILDER("MatrixExp").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto res = ib->ShapeCalc(g_matrix_exp, {x});
  auto input_perm = res[0];
  auto begins = res[1];
  auto sizes = res[2];
  auto x_transpose = ib->Transpose(x, input_perm);
  auto zero_matrix = ib->ZerosLike(x);
  zero_matrix = ib->Cast(zero_matrix, ib->GetDtype(dout));
  auto meta_grad_up = ib->Concat({x_transpose, dout}, -1);
  auto meta_grad_down = ib->Concat({zero_matrix, x_transpose}, -1);
  auto meta_grad = ib->Concat({meta_grad_up, meta_grad_down}, -2);
  meta_grad = ib->Emit("MatrixExp", {meta_grad});
  return {ib->Slice(meta_grad, begins, sizes)};
});

REG_BPROP_BUILDER("Complex").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = x->need_compute_grad_out() ? ib->Real(dout) : ib->OutZeros(x);
  auto dy = y->need_compute_grad_out() ? ib->Imag(dout) : ib->OutZeros(y);
  return {dx, dy};
});

REG_BPROP_BUILDER("CholeskyInverse").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto upper = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  ShapeVector input_perm = {1, 0};
  NodePtr dx;
  auto DealWithUpper = [&upper, &dx, &ib, &input_x](const NodePtr &common_term) {
    if (ib->Equal(upper, ib->Value<bool>(true))) {
      dx = ib->Emit("Neg", {ib->MatMul(input_x, common_term, false, false)});
    } else {
      dx = ib->Emit("Neg", {ib->MatMul(common_term, input_x, false, false)});
    }
  };
  if ((ib->GetDtypeId(dout)) == kNumberTypeFloat64) {
    input_x = ib->Cast(input_x, kFloat32);
    out = ib->Cast(out, kFloat32);
    dout = ib->Cast(dout, kFloat32);
    auto common_term = ib->Add(dout, ib->Transpose(dout, input_perm));
    common_term = ib->Cast(common_term, kFloat32);
    common_term = ib->MatMul(out, ib->MatMul(common_term, out, false, false), false, false);
    DealWithUpper(common_term);
    dx = ib->Cast(dx, kFloat64);
    return {dx, ib->OutZeros(upper)};
  }
  auto common_term = ib->Add(dout, ib->Transpose(dout, input_perm));
  common_term = ib->MatMul(out, ib->MatMul(common_term, out, false, false), false, false);
  DealWithUpper(common_term);
  return {dx, ib->OutZeros(upper)};
});

REG_BPROP_BUILDER("CholeskySolve").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto upper = GetValue<bool>(ib->GetAttr("upper"));
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto flag = 0;
  auto shape_x1 = ib->GetShape(x1);
  auto len_x1 = shape_x1.size();
  if ((ib->GetDtypeId(dout)) == kNumberTypeFloat64) {
    flag = 1;
    x2 = ib->Cast(x2, kFloat32);
    out = ib->Cast(out, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  ShapeVector input_perm = {1, 0};
  NodePtr dx2;
  auto DealWithUpper2D = [&upper, &dx2, &ib, &x2](const NodePtr &common_term) {
    if (upper) {
      dx2 = ib->Emit("Neg", {ib->MatMul(x2, common_term, false, false)});
    } else {
      dx2 = ib->Emit("Neg", {ib->MatMul(common_term, x2, false, false)});
    }
  };
  auto DealWithUpperND = [&upper, &dx2, &ib, &x2](const NodePtr &common_term) {
    if (upper) {
      dx2 = ib->Emit("Neg", {ib->BatchMatMul(x2, common_term, false, false)});
    } else {
      dx2 = ib->Emit("Neg", {ib->BatchMatMul(common_term, x2, false, false)});
    }
  };
  auto dx1 = ib->Emit("CholeskySolve", {dout, x2}, {{"upper", ib->GetAttr("upper")}});
  if (len_x1 == 2) {
    auto common_term = ib->MatMul(dx1, ib->Transpose(out, input_perm), false, false);
    common_term = ib->Add(common_term, (ib->Transpose(common_term, input_perm)));
    DealWithUpper2D(common_term);
  } else {
    auto x2_dim_size = static_cast<int64_t>(ib->GetShape(x2).size());
    auto target_order = Range(x2_dim_size - 2);
    target_order.push_back(x2_dim_size - 1);
    target_order.push_back(x2_dim_size - 2);
    auto common_term = ib->BatchMatMul(dx1, ib->Transpose(out, target_order), false, false);
    common_term = ib->Add(common_term, (ib->Transpose(common_term, target_order)));
    DealWithUpperND(common_term);
  }
  if (flag == 1) {
    dx1 = ib->Cast(dx1, kFloat64);
    dx2 = ib->Cast(dx2, kFloat64);
  }
  return {dx1, dx2};
});

REG_BPROP_BUILDER("NextAfter").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dout_type = ib->GetDtype(dout);
  auto x1_type = ib->GetDtype(x1);
  if (x1_type->type_id() == kNumberTypeFloat64) {
    x1 = ib->Cast(x1, kFloat32);
  }
  if (ib->GetDtypeId(x2) == kNumberTypeFloat64) {
    x2 = ib->Cast(x2, kFloat32);
  }
  if (dout_type->type_id() == kNumberTypeFloat64) {
    dout = ib->Cast(dout, kFloat32);
  }
  auto s_x1 = ib->Shape(x1);
  auto partial_x1 = ib->Fill(1.0, s_x1, x1_type->type_id());
  auto s_x2 = ib->Shape(x2);
  auto partial_x2 = ib->ZerosLike(x2);
  auto dx1 = ib->Reshape(ib->Mul(partial_x1, dout), s_x1);
  auto dx2 = ib->Reshape(ib->Mul(partial_x2, dout), s_x2);
  return {ib->Cast(dx1, dout_type), ib->Cast(dx2, dout_type)};
});

REG_BPROP_BUILDER("Lerp").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto start = ib->GetInput(kIndex0);
  auto end = ib->GetInput(kIndex1);
  auto weight = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  dout = ib->Cast(dout, kFloat32);
  auto dout_type = ib->GetDtype(dout);
  NodePtr sub_w, mul_w;
  if (weight->input_type() == InputType::kConstant) {
    auto v = weight->BuildValue();
    MS_EXCEPTION_IF_NULL(v);
    auto val = GetValue<float>(v);
    sub_w = ib->Tensor(1.0 - val, dout_type);
    mul_w = ib->Tensor(val, dout_type);
  } else if (weight->input_type() == InputType::kParameter || weight->input_type() == InputType::kInput) {
    auto v = weight->BuildValue();
    MS_EXCEPTION_IF_NULL(v);
    if (v->isa<Scalar>()) {
      auto val = GetValue<float>(v);
      sub_w = ib->Tensor(1.0 - val, dout_type);
      mul_w = ib->Tensor(val, dout_type);
    } else {
      sub_w = ib->Sub(ib->Tensor(1.0, ib->GetDtype(weight)), weight);
      mul_w = weight;
    }
  } else {
    sub_w = ib->Sub(ib->Tensor(1.0, ib->GetDtype(weight)), weight);
    mul_w = weight;
  }
  auto dstart = ib->Mul(dout, sub_w);
  auto dend = ib->Mul(dout, mul_w);
  auto dweight = ib->Mul(dout, ib->Sub(end, start));
  auto tmp = BinopGradCommon(ib, start, end, dstart, dend);
  dstart = tmp[0];
  dend = tmp[1];
  if (weight->input_type() == InputType::kConstant) {
    dweight = ib->OutZeros(weight);
  } else {
    auto tmp2 = BinopGradCommon(ib, start, weight, dstart, dweight);
    dweight = tmp2[1];
    if (ib->GetDtypeId(dweight) != ib->GetDtypeId(weight)) {
      dweight = ib->Cast(dweight, ib->GetDtype(weight));
    }
  }
  dstart = ib->Cast(dstart, ib->GetDtype(start));
  dend = ib->Cast(dend, ib->GetDtype(end));
  return {dstart, dend, dweight};
});

REG_BPROP_BUILDER("TridiagonalMatMul").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  auto LeftShift = [](BpropBuilder *ib, const NodePtr &x) {
    auto x_shape = ib->GetShape(x);
    std::vector<std::vector<int64_t>> paddings;
    auto rank = x_shape.size();
    for (size_t i = 0; i < rank - 2; i++) {
      (void)paddings.emplace_back(ShapeVector{0LL, 0LL});
    }
    (void)paddings.emplace_back(ShapeVector{0LL, 1LL});
    (void)paddings.emplace_back(ShapeVector{0LL, 0LL});
    ShapeVector begin, end, strides;
    for (size_t i = 0; i < rank; i++) {
      (void)begin.emplace_back(0LL);
      (void)end.emplace_back(x_shape[i]);
      (void)strides.emplace_back(1LL);
    }
    begin[rank - 2] = 1LL;
    return ib->Emit("Pad",
                    {ib->StridedSlice(x, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(end),
                                      ib->Value<ShapeVector>(strides))},
                    {{"paddings", MakeValue(paddings)}});
  };
  auto RightShift = [](BpropBuilder *ib, const NodePtr &x) {
    auto x_shape = ib->GetShape(x);
    std::vector<std::vector<int64_t>> paddings;
    auto rank = x_shape.size();
    for (size_t i = 0; i < rank - 2; i++) {
      (void)paddings.emplace_back(ShapeVector{0LL, 0LL});
    }
    (void)paddings.emplace_back(ShapeVector{1LL, 0LL});
    (void)paddings.emplace_back(ShapeVector{0LL, 0LL});
    ShapeVector begin, end, strides;
    for (size_t i = 0; i < rank; i++) {
      (void)begin.emplace_back(0LL);
      (void)end.emplace_back(x_shape[i]);
      (void)strides.emplace_back(1LL);
    }
    end[rank - 2] = -1;
    return ib->Emit("Pad",
                    {ib->StridedSlice(x, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(end),
                                      ib->Value<ShapeVector>(strides))},
                    {{"paddings", MakeValue(paddings)}});
  };
  auto MatrixTranspose = [](BpropBuilder *ib, const NodePtr &x) {
    auto x_shape = ib->GetShape(x);
    auto rank = x_shape.size();
    ShapeVector perm;
    if (rank > IntToSize(2)) {
      for (size_t i = 0; i < rank - 2; i++) {
        (void)perm.emplace_back(SizeToLong(i));
      }
    }
    (void)perm.emplace_back(rank - 1);
    (void)perm.emplace_back(rank - 2);
    return ib->Transpose(x, perm);
  };
  auto superdiag = ib->GetInput(kIndex0);
  auto maindiag = ib->GetInput(kIndex1);
  auto subdiag = ib->GetInput(kIndex2);
  auto rhs = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto superdiag_type = ib->GetDtype(superdiag);
  auto superdiag_conj = MatrixTranspose(ib, superdiag);
  auto maindiag_conj = MatrixTranspose(ib, maindiag);
  auto subdiag_conj = MatrixTranspose(ib, subdiag);
  auto rhs_conj = rhs;
  if ((*superdiag_type) == (*kComplex64) || (*superdiag_type) == (*kComplex128)) {
    superdiag_conj = ib->Conj(superdiag_conj);
    maindiag_conj = ib->Conj(maindiag_conj);
    subdiag_conj = ib->Conj(subdiag_conj);
    rhs_conj = ib->Conj(rhs);
  }
  auto superdiag_grad = ib->ReduceSum(LeftShift(ib, rhs_conj) * dout, ShapeVector{-1LL});
  auto maindiag_grad = ib->ReduceSum(rhs_conj * dout, ShapeVector{-1LL});
  auto subdiag_grad = ib->ReduceSum(RightShift(ib, rhs_conj) * dout, ShapeVector{-1LL});
  auto rhs_grad = RightShift(ib, superdiag_conj * dout) + maindiag_conj * dout + LeftShift(ib, subdiag_conj * dout);
  superdiag_grad = ib->ExpandDims(superdiag_grad, -2);
  maindiag_grad = ib->ExpandDims(maindiag_grad, -2);
  subdiag_grad = ib->ExpandDims(subdiag_grad, -2);
  return {superdiag_grad, maindiag_grad, subdiag_grad, rhs_grad};
});

REG_BPROP_BUILDER("AddV2").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr bc_dx = nullptr;
  NodePtr bc_dy = nullptr;
  if (x->need_compute_grad_out()) {
    bc_dx = dout;
  }
  if (y->need_compute_grad_out()) {
    bc_dy = dout;
  }
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("Addcdiv").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  std::unordered_set<TypeId> type_list{TypeId::kNumberTypeInt64, TypeId::kNumberTypeFloat16,
                                       TypeId::kNumberTypeFloat64};
  return BpropAddcCommon(ib, "Addcdiv", type_list);
});

REG_BPROP_BUILDER("Addcmul").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  std::unordered_set<TypeId> type_list{TypeId::kNumberTypeInt8,    TypeId::kNumberTypeInt16,
                                       TypeId::kNumberTypeInt64,   TypeId::kNumberTypeUInt8,
                                       TypeId::kNumberTypeFloat16, TypeId::kNumberTypeFloat64};
  return BpropAddcCommon(ib, "Addcmul", type_list);
});

REG_BPROP_BUILDER("LpNorm").SetBody(BODYFUNC(ib) {
  auto p = GetValue<int64_t>(ib->GetAttr("p"));
  auto keep_dims = GetValue<bool>(ib->GetAttr("keep_dims"));
  auto axis = GetIntList(ib->GetAttr("axis"));
  auto input_x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto input_x_shape = ib->GetShape(input_x);
  if ((!keep_dims) && (!input_x_shape.empty())) {
    for (const auto &i : axis) {
      dout = ib->ExpandDims(dout, i);
      out = ib->ExpandDims(out, i);
    }
  }
  if (p == 0) {
    return {ib->OutZeros(input_x)};
  }
  if (p == 1) {
    return {ib->Mul(dout, (ib->Sign(input_x)))};
  }
  if (p == 2) {
    auto input_scaled = input_x;
    auto scale_v = ib->RealDiv(dout, out);
    return {ib->Mul(input_scaled, scale_v)};
  } else {
    auto input_x_abs = ib->Emit("Abs", {input_x});
    auto input_scaled = ib->Mul(ib->Pow(input_x_abs, ib->Tensor(p - 2, ib->GetDtype(input_x_abs))), input_x);
    auto scale_v = ib->RealDiv(dout, ib->Pow(out, ib->Tensor(p - 1, ib->GetDtype(out))));
    auto equal_zero = ib->Equal(input_scaled, ib->Tensor(0, ib->GetDtype(input_scaled)));
    return {ib->Select(equal_zero, ib->Fill(0.0, ib->Shape(input_scaled), ib->GetDtype(input_scaled)->type_id()),
                       ib->Mul(input_scaled, scale_v))};
  }
});

REG_BPROP_BUILDER("Renorm").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto p = static_cast<int64_t>(GetValue<float>(ib->GetAttr("p")));
  float ext = 1e-07;
  auto dim = GetIntList(ib->GetAttr("dim"))[0];
  auto max_norm = GetValue<float>(ib->GetAttr("maxnorm"));
  auto input_x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shape = ib->GetShape(input_x);
  int64_t new_dim = dim >= 0 ? dim : (SizeToLong(shape.size()) + dim);
  std::vector<int64_t> dims;
  for (int64_t i = 0; i < SizeToLong(shape.size()); i++) {
    if (i != new_dim) {
      dims.push_back(i);
    }
  }
  auto norm = ib->Emit("LpNorm", {input_x},
                       {{"keep_dims", MakeValue(true)},
                        {"axis", MakeValue(dims)},
                        {"p", MakeValue<int64_t>(p)},
                        {"epsilon", MakeValue<float>(1e-12)}});
  norm = ib->BroadcastTo(norm, input_x);
  auto grad_out = ib->Mul(input_x, dout);
  grad_out = ib->ReduceSum(grad_out, dims, true);
  NodePtr norm_bp = nullptr;
  if (p == 1) {
    auto sig = ib->Sign(input_x);
    norm_bp = ib->Mul(sig, grad_out);
  } else if (p == 2) {
    auto m = ib->Mul(input_x, (ib->RealDiv(grad_out, norm)));
    norm_bp = ib->Emit("MaskedFill",
                       {m, ib->Equal(norm, (ib->Tensor(0.0, ib->GetDtype(norm)))), ib->Tensor(0.0, ib->GetDtype(m))});
  } else {
    auto abs_ = ib->Emit("Abs", {input_x});
    auto input_scaled = ib->Mul(input_x, ib->Pow(abs_, ib->Tensor(p - 2)));
    auto pow_ = ib->Pow(norm, ib->Tensor(p - 1));
    auto scale_v = ib->RealDiv(grad_out, pow_);
    scale_v = ib->Emit("MaskedFill", {scale_v, ib->Equal(norm, (ib->Tensor(0.0, ib->GetDtype(norm)))),
                                      ib->Tensor(0.0, ib->GetDtype(scale_v))});
    norm_bp = ib->Mul(input_scaled, scale_v);
  }

  auto v = ib->Add(norm, ib->Tensor(ext, ib->GetDtype(norm)));
  auto inv_norm = ib->Reciprocal(v);
  auto grad_norm = ib->Mul(ib->Mul(ib->Tensor(max_norm, ib->GetDtype(inv_norm)), inv_norm),
                           ib->Sub(dout, (ib->Mul(inv_norm, norm_bp))));
  auto q = ib->Greater(norm, ib->Tensor(max_norm, ib->GetDtype(norm)));
  return {ib->Select(q, grad_norm, dout)};
});

DEF_PURE_SHAPE_CALC(g_reduce_std)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(0);
    auto new_axis = inputs.at(1);
    if (new_axis.empty() && !x_shape.empty()) {
      for (int64_t i = 0; i < SizeToLong(x_shape.size()); i++) {
        new_axis.push_back(i);
      }
    }
    (void)std::transform(new_axis.begin(), new_axis.end(), new_axis.begin(), [&x_shape](const int64_t &c) {
      if (c < 0) {
        return c + SizeToLong(x_shape.size());
      }
      return c;
    });
    for (size_t i = 1; i < new_axis.size(); ++i) {
      for (size_t j = 0; j < new_axis.size() - i; ++j) {
        if (new_axis[j] > (new_axis[j + 1])) {
          std::swap(new_axis[j], new_axis[j + 1]);
        }
      }
    }
    // input_x:[2,3,4,5]  new_axis:   [0, 2]
    // reduce: [3,5]      reshape:[1,3,1,5]
    auto reshape = x_shape;
    for (auto &i : new_axis) {
      reshape[LongToSize(i)] = 1;
    }
    int64_t num = 1;
    for (const auto &i : new_axis) {
      num *= x_shape[LongToSize(i)];
    }
    return {reshape, {num - 1}, {num}};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> ShapeVector {
    auto shape_x = inputs.at(0);
    auto rank = IsDynamicRank(shape_x) ? -1 : SizeToLong(shape_x.size());
    return {rank, 1, 1};
  });

REG_BPROP_BUILDER("ReduceStd").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto unbiased = ib->GetInput(kIndex2);
  auto keep_dims = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex5);
  auto std_d = ib->TupleGetItem(dout, 0);
  auto std = ib->TupleGetItem(out, 0);
  auto mean_d = ib->TupleGetItem(dout, 1);
  auto mean = ib->TupleGetItem(out, 1);
  auto res = ib->ShapeCalc(g_reduce_std, {x, axis}, {1});
  res[1] = ib->SequenceToTensor(res[1]);
  res[2] = ib->SequenceToTensor(res[2]);

  auto keep_dims_value = keep_dims->BuildValue();
  auto keep_dims_opt = ops::GetScalarValue<bool>(keep_dims_value);
  if (keep_dims_opt.has_value()) {
    if (!keep_dims_opt.value() && !ib->GetShape(x).empty()) {
      std_d = ib->Reshape(std_d, res[0]);
      std = ib->Reshape(std, res[0]);
      mean_d = ib->Reshape(mean_d, res[0]);
      mean = ib->Reshape(mean, res[0]);
    }
  } else {
    auto true_branch = [&res, &std_d, &std, &mean_d, &mean](Emitter *e) -> NodePtrList {
      auto std_d_r = e->Reshape(std_d, res[0]);
      auto std_r = e->Reshape(std, res[0]);
      auto mean_d_r = e->Reshape(mean_d, res[0]);
      auto mean_r = e->Reshape(mean, res[0]);
      return {std_d_r, std_r, mean_d_r, mean_r};
    };
    auto false_branch = [&std_d, &std, &mean_d, &mean](const Emitter *e) -> NodePtrList {
      return {std_d, std, mean_d, mean};
    };
    auto keep_dims_t =
      ib->Emit("ScalarToTensor", {ib->Equal(keep_dims, ib->Value<bool>(false)), ib->Value<int64_t>(kBool->type_id())});
    auto cond = ib->LogicalAnd(keep_dims_t, ib->Tensor(!(ib->GetShape(x).empty()), kBool));
    auto cond_block = ib->Conditional(cond, true_branch, false_branch);
    std_d = ib->TupleGetItem(cond_block, 0);
    std = ib->TupleGetItem(cond_block, 1);
    mean_d = ib->TupleGetItem(cond_block, 2);
    mean = ib->TupleGetItem(cond_block, 3);
  }

  auto dx = ib->Sub(x, mean);
  dx = ib->Mul(dx, std_d);
  dx = ib->Div(dx, std);

  auto unbiased_value = unbiased->BuildValue();
  auto unbiased_opt = ops::GetScalarValue<bool>(unbiased_value);
  if (unbiased_opt.has_value()) {
    if (unbiased_opt.value()) {
      dx = ib->Div(dx, ib->Cast(res[1], ib->GetDtype(dx)));
    } else {
      dx = ib->Div(dx, ib->Cast(res[2], ib->GetDtype(dx)));
    }
  } else {
    auto unbiased_true_branch = [&dx, &res](Emitter *e) -> NodePtrList {
      return {e->Div(dx, e->Cast(res[1], dx->dtype()))};
    };
    auto unbiased_false_branch = [&dx, &res](Emitter *e) -> NodePtrList {
      return {e->Div(dx, e->Cast(res[2], dx->dtype()))};
    };
    auto unbiased_cond = ib->Equal(unbiased, ib->Value<bool>(true));
    dx = ib->Conditional(unbiased_cond, unbiased_true_branch, unbiased_false_branch);
  }
  auto temp = ib->Div(mean_d, ib->Cast(res[2], ib->GetDtype(mean_d)));
  dx = ib->Add(dx, temp);
  return {dx, ib->OutZeros(axis), ib->OutZeros(unbiased), ib->OutZeros(keep_dims)};
});

REG_BPROP_BUILDER("CumulativeLogsumexp").SetBody(BODYFUNC(ib) {
  // this dsl has pression error
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  bool reverse = GetValue<bool>(ib->GetAttr("reverse"));
  NodePtr dtype_min = nullptr;
  if ((ib->GetDtype(x))->type_id() == TypeId::kNumberTypeFloat16) {
    dtype_min = ib->Fill(-65500e+0, ib->Shape(dout), TypeId::kNumberTypeFloat16);
  } else {
    if ((ib->GetDtype(x))->type_id() == TypeId::kNumberTypeFloat32) {
      dtype_min = ib->Fill(-3.4028235e+38, ib->Shape(dout), TypeId::kNumberTypeFloat32);
    } else {
      dtype_min = ib->Fill(-1.7976931348623157e+308, ib->Shape(dout), TypeId::kNumberTypeFloat64);
    }
  }
  auto log_grad_positive = ib->Select(ib->Greater(dout, ib->Tensor(0, ib->GetDtype(dout))), ib->Log(dout), dtype_min);
  auto log_grad_negative =
    ib->Select(ib->Less(dout, ib->Tensor(0, ib->GetDtype(dout))), ib->Log(ib->Neg(dout)), dtype_min);
  auto output_pos =
    ib->Exp(ib->Add((ib->Emit("CumulativeLogsumexp", {ib->Sub(log_grad_positive, out), axis},
                              {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", MakeValue(!reverse)}})),
                    x));
  auto output_neg =
    ib->Exp(ib->Add((ib->Emit("CumulativeLogsumexp", {ib->Sub(log_grad_negative, out), axis},
                              {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", MakeValue(!reverse)}})),
                    x));
  return {ib->Sub(output_pos, output_neg), ib->OutZeros(x)};
});

REG_BPROP_BUILDER("NPUAllocFloatStatus").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) { return {}; });

REG_BPROP_BUILDER("NPUGetFloatStatus").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("NPUClearFloatStatus").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Igamma").SetUnusedInputs({i2}).SetBody(IgammaBpropExpander);

REG_BPROP_BUILDER("Igammac").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto r_dx = IgammaBpropExpander(ib);
  r_dx[0] = ib->Neg(r_dx[0]);
  r_dx[1] = ib->Neg(r_dx[1]);
  return r_dx;
});

REG_BPROP_BUILDER("Einsum").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("EinsumGrad", {x, dout}, {{"equation", ib->GetAttr("equation")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchMatMul").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto ta = GetValue<bool>(ib->GetAttr("transpose_a"));
  auto tb = GetValue<bool>(ib->GetAttr("transpose_b"));
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);

  NodePtr dx = nullptr;
  if (x->need_compute_grad_out()) {
    if (ta) {
      dx = ib->BatchMatMul(w, dout, tb, true);
    } else {
      dx = ib->BatchMatMul(dout, w, false, !tb);
    }
  }
  NodePtr dw = nullptr;
  if (w->need_compute_grad_out()) {
    if (tb) {
      dw = ib->BatchMatMul(dout, x, true, ta);
    } else {
      dw = ib->BatchMatMul(x, dout, !ta, false);
    }
  }
  return BinopGradCommon(ib, x, w, dx, dw, 2);
});

REG_BPROP_BUILDER("Eps").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("EuclideanNorm").SetBody(BODYFUNC(ib) {
  auto keep_dims = GetValue<bool>(ib->GetAttr("keep_dims"));
  auto x = ib->GetInput(kIndex0);
  auto axes = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto scale_v = ib->RealDiv(dout, out);
  if ((!keep_dims) && (ib->GetShape(x).size() > 0)) {
    scale_v = ib->Emit("ExpandDims", {scale_v, axes});
  }
  return {ib->Mul(x, scale_v), ib->OutZeros(axes)};
});

REG_BPROP_BUILDER("MatrixTriangularSolve").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto adjoint_a = GetValue<bool>(ib->GetAttr("adjoint"));
  auto lower_a = GetValue<bool>(ib->GetAttr("lower"));
  auto matrix = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto grad_rhs = ib->Emit("MatrixTriangularSolve", {matrix, dout},
                           {{"lower", MakeValue(lower_a)}, {"adjoint", MakeValue(!adjoint_a)}});
  NodePtr grad_rhs_temp;
  NodePtr out_temp;
  if (((ib->GetDtype(matrix)) == kComplex64) || ((ib->GetDtype(matrix)) == kComplex128)) {
    grad_rhs_temp = Adjoint(ib, grad_rhs);
    out_temp = Adjoint(ib, out);
  } else {
    grad_rhs_temp = MatrixTranspose(ib, grad_rhs);
    out_temp = MatrixTranspose(ib, out);
  }
  NodePtr grad_matrix;
  auto m_rank = ib->GetShape(matrix).size();
  auto NegMatMul = [&ib, &grad_matrix, &m_rank](const NodePtr &a, const NodePtr &b) {
    if (m_rank == 2) {
      grad_matrix = ib->MatMul(a, b, false, false);
    } else {
      grad_matrix = ib->BatchMatMul(a, b, false, false);
    }
    grad_matrix = ib->Neg(grad_matrix);
  };
  if (adjoint_a) {
    NegMatMul(out, grad_rhs_temp);
  } else {
    NegMatMul(grad_rhs, out_temp);
  }
  auto BandPart = [&ib](const NodePtr &matrix, int lower, int upper) -> NodePtr {
    if (((ib->GetDtype(matrix)) == kComplex64) || ((ib->GetDtype(matrix)) == kComplex128)) {
      auto grad_matrix_real = ib->Emit("MatrixBandPart", {ib->Real(matrix), ib->Value(lower), ib->Value(upper)});
      auto grad_matrix_imag = ib->Emit("MatrixBandPart", {ib->Imag(matrix), ib->Value(lower), ib->Value(upper)});
      return ib->Emit("Complex", {grad_matrix_real, grad_matrix_imag});
    } else {
      return ib->Emit("MatrixBandPart", {matrix, ib->Value(lower), ib->Value(upper)});
    }
  };
  if (lower_a) {
    grad_matrix = BandPart(grad_matrix, -1, 0);
  } else {
    grad_matrix = BandPart(grad_matrix, 0, -1);
  }
  return {grad_matrix, grad_rhs};
});

REG_BPROP_BUILDER("NanToNum").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto nan = ib->GetInput(kIndex1);
  auto posinf = ib->GetInput(kIndex2);
  auto neginf = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto dx = ib->Mul(dout, (ib->Emit("IsFinite", {x})));
  return {dx, ib->OutZeros(nan), ib->OutZeros(posinf), ib->OutZeros(neginf)};
});

REG_BPROP_BUILDER("Fmin").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) { return FminFmaxGrad(ib, true); });

REG_BPROP_BUILDER("Fmax").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) { return FminFmaxGrad(ib, false); });

REG_BPROP_BUILDER("Angle").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto re = ib->Real(x);
  auto im = ib->Imag(x);
  re = ib->Complex(im, re);
  auto z = ib->Reciprocal(re);
  auto zero = ib->ZerosLike(dout);
  auto complex_dout = ib->Complex(dout, zero);
  return {ib->Neg(ib->Mul(complex_dout, z))};
});

REG_BPROP_BUILDER("Lgamma").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(dout, ib->Emit(kDigammaOpName, {x}));
  return {dx};
});

REG_BPROP_BUILDER("Digamma").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto a = ib->Tensor(1, kInt64);
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(dout, ib->Emit(kPolygammaOpName, {a, x}));
  return {dx};
});

REG_BPROP_BUILDER("Polygamma").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto a = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto one = ib->Tensor(1);
  a = ib->Add(a, one);
  NodePtr dx;
  if (ib->GetDtypeId(x) == kNumberTypeFloat16) {
    x = ib->Cast(x, kNumberTypeFloat64);
    dx = ib->Mul(dout, ib->Emit(kPolygammaOpName, {a, x}));
    dx = ib->Cast(dx, kNumberTypeFloat16);
  } else {
    dx = ib->Mul(dout, ib->Emit(kPolygammaOpName, {a, x}));
  }
  return {ib->OutZeros(a), dx};
});

REG_BPROP_BUILDER("Cholesky").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto upper_input = ib->GetInput(kIndex1);
  auto upper_input_value = upper_input->BuildValue();
  if (upper_input_value->ContainsValueAny()) {
    MS_EXCEPTION(ValueError) << "Input `upper` does not support variable in GRAPH_MODE currently.";
  }
  auto upper = GetValue<bool>(upper_input_value);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  if (upper) {
    out = MatrixTranspose(ib, out);
    dout = MatrixTranspose(ib, dout);
  }
  auto dx = ib->Emit("CholeskyGrad", {out, dout});
  return {dx, ib->OutZeros(upper_input)};
});

REG_BPROP_BUILDER("InplaceIndexAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->OutZeros(indices), ib->Gather(dout, indices, ib->EmitValue(ib->GetAttr("axis")))};
});

REG_BPROP_BUILDER("Zeta").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto q = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dq =
    ib->Mul((ib->Mul((ib->Neg(x)), (ib->Emit("Zeta", {ib->Add(x, (ib->Tensor(1, ib->GetDtype(x)))), q})))), dout);
  return {ib->OutZeros(x), dq};
});

REG_BPROP_BUILDER("Baddbmm").SetUnusedInputs({}).SetBody(BODYFUNC(ib) {
  auto input = ib->GetInput(kIndex0);
  auto batch1 = ib->GetInput(kIndex1);
  auto batch2 = ib->GetInput(kIndex2);
  auto beta = ib->GetInput(kIndex3);
  auto alpha = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  NodePtr input_grad;
  if (input->need_compute_grad_out()) {
    auto beta_tensor = ib->ScalarToTensor(beta);
    input_grad = ib->Mul(beta_tensor, dout);
    auto dout_shape = ib->GetShape(dout);
    auto input_shape = ib->GetShape(input);
    if (input_shape != dout_shape) {
      auto bc_axis = ib->BroadcastGradientArgs(input, dout);
      input_grad = ib->Reshape(ib->ReduceSum(input_grad, bc_axis[0], false, true), input_shape);
    }
  } else {
    input_grad = ib->OutZeros(input);
  }
  auto alpha_tensor = ib->ScalarToTensor(alpha);
  auto batch1_grad = batch1->need_compute_grad_out() ? ib->Mul(alpha_tensor, ib->BatchMatMul(dout, batch2, false, true))
                                                     : ib->OutZeros(batch1);
  auto batch2_grad = batch2->need_compute_grad_out() ? ib->Mul(alpha_tensor, ib->BatchMatMul(batch1, dout, true, false))
                                                     : ib->OutZeros(batch2);
  return {input_grad, batch1_grad, batch2_grad, ib->OutZeros(beta), ib->OutZeros(alpha)};
});

REG_BPROP_BUILDER("Diagonal").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto offset_node = ib->GetInput(kIndex1);
  auto dim1_node = ib->GetInput(kIndex2);
  auto dim2_node = ib->GetInput(kIndex3);
  auto offset = GetValue<int64_t>(offset_node->BuildValue());
  auto dim1 = GetValue<int64_t>(dim1_node->BuildValue());
  auto dim2 = GetValue<int64_t>(dim2_node->BuildValue());
  auto out = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex5);
  auto x_shape = ib->GetShape(x);
  if (IsDynamicRank(x_shape)) {
    MS_LOG_EXCEPTION << "Diagonal doesn't support dynamic rank now, because it need rank of x to do transpose";
  }
  auto x_dtype = ib->GetDtype(x);
  auto x_dim = ib->GetRank(x);
  if (dim1 < 0) {
    dim1 += x_dim;
  }
  if (dim2 < 0) {
    dim2 += x_dim;
  }
  auto true_case = [offset, dim1, dim2, &x, &out, &dout, &x_shape, &x_dtype, &offset_node, &dim1_node, &dim2_node,
                    x_dim](Emitter *ib) -> NodePtrList {
    auto dx_trans_shape = ib->ShapeCalc(std::make_shared<DiagonalShapeCalc>(dim1, dim2), {x, out})[0];
    auto grad_diagonal = GradDiagonal(ib, dout, dx_trans_shape, {offset, dim1, dim2, x_dim}, x_dtype);
    return {grad_diagonal, ib->ZerosLike(offset_node), ib->ZerosLike(dim1_node), ib->ZerosLike(dim2_node)};
  };
  auto false_case = [&x, &x_dtype, &offset_node, &dim1_node, &dim2_node](Emitter *ib) -> NodePtrList {
    return {ib->ZerosLike(x), ib->ZerosLike(offset_node), ib->ZerosLike(dim1_node), ib->ZerosLike(dim2_node)};
  };
  if (IsDynamic(ib->GetShape(out))) {
    auto out_size = ib->Emit("Size", {out});
    auto cond = ib->Emit("scalar_eq", {out_size, ib->Value<int64_t>(0)});
    auto dx = ib->Conditional(cond, false_case, true_case);
    return {dx, ib->OutZeros(offset_node), ib->OutZeros(dim1_node), ib->OutZeros(dim2_node)};
  }
  if (ib->GetSize(out) > 0) {
    return true_case(ib);
  } else {
    return false_case(ib);
  }
});

REG_BPROP_BUILDER("Polar").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto input1 = ib->GetInput(kIndex0);
  auto angle = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto grad_conj = ib->Emit("Conj", {dout});
  NodePtr grad_abs =
    input1->need_compute_grad_out() ? ib->Real(ib->Mul(grad_conj, ib->Sign(out))) : ib->OutZeros(input1);
  NodePtr grad_angle;
  if (angle->need_compute_grad_out()) {
    auto zeros = ib->ZerosLike(dout);
    zeros = ib->Cast(zeros, ib->GetDtype(input1));
    auto ones = ib->OnesLike(dout);
    ones = ib->Cast(ones, ib->GetDtype(input1));
    auto i = ib->Complex(zeros, ones);
    auto result_mul_1_j = ib->Mul(out, i);
    grad_angle = ib->Real(ib->Mul(grad_conj, result_mul_1_j));
  } else {
    grad_angle = ib->OutZeros(angle);
  }
  return {grad_abs, grad_angle};
});

REG_BPROP_BUILDER("TridiagonalSolve").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto diagonals = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  constexpr int64_t kLast2 = -2;
  constexpr int64_t k2 = 2;
  auto diag1 = ib->StridedSlice(diagonals, {{kLast2, {1}}});
  auto diag_shape = ib->GetShape(diagonals);
  ShapeVector zeros1_shape(diag_shape.begin(), diag_shape.end() - i2);
  zeros1_shape.push_back(1);
  auto zeros1 = ib->Emit("Zeros", {ib->Value<ShapeVector>(zeros1_shape), ib->EmitValue(ib->GetDtype(diagonals))});
  auto superdiag1 = ib->Concat({ib->StridedSlice(diagonals, {{kLast2, {k2}}, {-1, {1, LLONG_MAX}}}), zeros1}, -1);
  auto subdiag1 = ib->Concat({zeros1, ib->StridedSlice(diagonals, {{kLast2, {0}}, {-1, {0, -1}}})}, -1);
  auto diags_transposed = ib->Stack({superdiag1, diag1, subdiag1}, kLast2);
  auto grad_rhs = ib->Emit("TridiagonalSolve", {diags_transposed, dout}, {{"partial_pivoting", MakeValue<bool>(true)}});
  NodePtr grad_diags;
  if (diagonals->need_compute_grad_out()) {
    auto diag2 = ib->ReduceSum(ib->Mul(grad_rhs, out), {-1});
    ShapeVector zeros2_shape = ib->GetShape(grad_rhs);
    if (zeros2_shape.size() > i1) {
      zeros2_shape[zeros2_shape.size() - i2] = 1;
    }
    auto zeros2 = ib->Emit("Zeros", {ib->Value<ShapeVector>(zeros2_shape), ib->EmitValue(ib->GetDtype(grad_rhs))});
    auto superdiag2 = ib->ReduceSum(
      ib->Mul(grad_rhs, ib->Concat({ib->StridedSlice(out, {{kLast2, {1, LLONG_MAX}}}), zeros2}, -k2)), {-1});
    auto subdiag2 =
      ib->ReduceSum(ib->Mul(grad_rhs, ib->Concat({zeros2, ib->StridedSlice(out, {{kLast2, {0, -1}}})}, -k2)), {-1});
    auto a = ib->Stack({superdiag2, diag2, subdiag2}, kLast2);
    grad_diags = ib->Sub(ib->Tensor(0, ib->GetDtype(a)), a);
  } else {
    grad_diags = ib->OutZeros(diagonals);
  }
  return {grad_diags, grad_rhs};
});

REG_BPROP_BUILDER("FFT").SetBody(BODYFUNC(ib) { return FFTGradCommon(ib, "IFFT"); });

REG_BPROP_BUILDER("IFFT").SetBody(BODYFUNC(ib) { return FFTGradCommon(ib, "FFT"); });

REG_BPROP_BUILDER("FFTShift").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto dim = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {ib->Emit("IFFTShift", {dout, dim}), ib->OutZeros(dim)};
});

REG_BPROP_BUILDER("IFFTShift").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto dim = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {ib->Emit("FFTShift", {dout, dim}), ib->OutZeros(dim)};
});

DEF_PURE_SHAPE_CALC(g_correlate)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    constexpr int64_t input_num = 4;
    if (inputs.size() != input_num) {
      MS_LOG_EXCEPTION << "ShapeCalc[g_correlate] expect 4 inputs, but got " << inputs.size() << "inputs";
    }
    auto a_size = inputs.at(kIndex0)[0];
    auto v_size = inputs.at(kIndex1)[0];
    auto dout_size = inputs.at(kIndex2)[0];
    auto mode_value = inputs.at(kIndex3)[0];

    std::vector<int64_t> size_arr;
    size_arr.emplace_back(a_size + v_size - 1);
    std::vector<int64_t> begin_arr;
    begin_arr.emplace_back((a_size + v_size - dout_size) / 2);
    std::vector<int64_t> end_arr;
    end_arr.emplace_back((a_size + v_size - dout_size) / 2 + dout_size);

    constexpr int64_t same_mode = 1;
    if (mode_value == same_mode && a_size >= v_size && v_size % 2 == 0) {
      begin_arr[0] = begin_arr[0] - 1;
      end_arr[0] = end_arr[0] - 1;
    }
    return {size_arr, begin_arr, end_arr};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &) -> std::vector<int64_t> {
    return {1, 1, 1};
  });

REG_BPROP_BUILDER("Correlate").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto a = ib->GetInput(kIndex0);
  auto v = ib->GetInput(kIndex1);
  auto mode = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);

  // step1: if dtype of a/v is not float or complex, cast a, v dtyte as to dout dtype
  static const std::vector<TypeId> complex_or_float = {
    kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeComplex64, kNumberTypeComplex128,
  };
  auto a_type = ib->GetDtypeId(a);
  bool is_complex_or_float = std::any_of(complex_or_float.begin(), complex_or_float.end(),
                                         [&a_type](const TypeId &type_id) { return a_type == type_id; });
  if (!is_complex_or_float) {
    a = ib->Cast(a, ib->GetDtype(dout));
    v = ib->Cast(v, ib->GetDtype(dout));
  }

  // step2: pad dout to (a_size + v_size - 1)
  auto dout_padded = dout;
  int64_t mode_value = GetValue<int64_t>(mode->BuildValue());
  constexpr int64_t full_mode = 3;
  if (mode_value != full_mode) {
    // calculate StridedSliceGrad paragram [size] [begin] [end]
    auto param_array = ib->ShapeCalc(g_correlate, {a, v, dout, mode}, {3});
    dout_padded =
      ib->Emit("StridedSliceGrad",
               {dout, param_array[0], param_array[1], param_array[2], ib->Value<ShapeVector>(ShapeVector{1LL})},
               {{"begin_mask", MakeValue<int64_t>(0LL)},
                {"end_mask", MakeValue<int64_t>(0LL)},
                {"ellipsis_mask", MakeValue<int64_t>(0LL)},
                {"new_axis_mask", MakeValue<int64_t>(0LL)},
                {"shrink_axis_mask", MakeValue<int64_t>(0LL)}});
  }

  // step3: calculate da, dv by convolution1d, reverse, conj
  auto a_conj = a;
  if (ib->GetDtypeId(a) == kNumberTypeComplex64 || ib->GetDtypeId(a) == kNumberTypeComplex128) {
    a_conj = ib->Emit("Conj", {a});
  }
  auto v_r = ib->Emit("ReverseV2", {v, ib->Value<ShapeVector>(ShapeVector{0LL})});
  auto dv_r = ib->Emit("Correlate", {dout_padded, a_conj, ib->Value<int64_t>(2LL)});
  auto da = ib->Emit("Correlate", {dout_padded, v_r, ib->Value<int64_t>(2LL)});
  auto dv_conj = ib->Emit("ReverseV2", {dv_r, ib->Value<ShapeVector>(ShapeVector{0LL})});
  auto dv = dv_conj;
  if (ib->GetDtypeId(a) == kNumberTypeComplex64 || ib->GetDtypeId(a) == kNumberTypeComplex128) {
    dv = ib->Emit("Conj", {dv_conj});
  }

  return {da, dv, ib->OutZeros(mode)};
});

REG_BPROP_BUILDER("DCT").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto type = ib->GetInput(kIndex1);
  auto type_value = GetValue<int64_t>(type->BuildValue());
  auto n = ib->GetInput(kIndex2);
  auto n_value = GetValue<int64_t>(n->BuildValue());
  auto axis = ib->GetInput(kIndex3);
  auto axis_value = GetValue<int64_t>(axis->BuildValue());
  auto norm = ib->GetInput(kIndex4);
  auto norm_value = GetValue<int64_t>(norm->BuildValue());
  auto forward = ib->GetInput(kIndex5);
  auto forward_value = GetValue<bool>(forward->BuildValue());
  auto grad = ib->GetInput(kIndex6);
  auto out = ib->GetInput(kIndex7);
  auto dout = ib->GetInput(kIndex8);

  auto input_shape_vec = ib->GetShape(x);

  int64_t grad_norm_value = 2;
  if (norm_value == 0) {
    grad_norm_value = 1;
  } else if (norm_value == 1) {
    grad_norm_value = 0;
  }
  bool grad_forward_value = !forward_value;

  auto temp_dout = ib->Emit("DCT", {dout, ib->Value(type_value), n, axis, ib->Value(grad_norm_value),
                                    ib->Value(grad_forward_value), ib->Value(true)});
  auto grad_dout = temp_dout;
  if (forward_value && norm_value != 2) {
    auto ones = ib->OnesLike(dout);
    grad_dout = temp_dout + ones;
  }

  auto output_shape_vec = ib->GetShape(out);
  auto x_rank = static_cast<int64_t>(input_shape_vec.size());

  axis_value = axis_value < 0 ? axis_value + x_rank : axis_value;
  n_value = n_value > 0 ? n_value : input_shape_vec[axis_value];

  ShapeVector begin;
  ShapeVector end;
  ShapeVector strides;
  for (int64_t i = 0; i < x_rank; i++) {
    (void)begin.emplace_back(0LL);
    (void)end.emplace_back(output_shape_vec[i]);
    (void)strides.emplace_back(1LL);
  }
  bool need_slice = false;
  bool need_pad = false;
  if (input_shape_vec[axis_value] < n_value) {
    end[axis_value] = input_shape_vec[axis_value];
    need_slice = true;
  } else if (input_shape_vec[axis_value] > n_value) {
    need_pad = true;
  }

  // at most one of need_pad or need_slice is true
  if (need_pad) {
    NodePtr input_shape_node = ib->EmitValue(MakeValue(input_shape_vec));
    grad_dout = ib->Emit("StridedSliceGrad",
                         {grad_dout, input_shape_node, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(end),
                          ib->Value<ShapeVector>(strides)},
                         {{kAttrBeginMask, MakeValue<int64_t>(0)},
                          {kAttrEndMask, MakeValue<int64_t>(0)},
                          {kAttrEllipsisMask, MakeValue<int64_t>(0)},
                          {kAttrNewAxisMask, MakeValue<int64_t>(0)},
                          {kAttrShrinkAxisMask, MakeValue<int64_t>(0)}});
  } else if (need_slice) {
    grad_dout = ib->StridedSlice(grad_dout, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(end),
                                 ib->Value<ShapeVector>(strides));
  }

  return {grad_dout,          ib->OutZeros(type),    ib->OutZeros(n),   ib->OutZeros(axis),
          ib->OutZeros(norm), ib->OutZeros(forward), ib->OutZeros(grad)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
