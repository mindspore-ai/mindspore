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
#include "include/common/utils/utils.h"
#include "frontend/expander/bprop/grad_ops/common_utils.h"
#include "utils/ms_context.h"
#include "ir/functor.h"
#include "ops/math_ops.h"

namespace mindspore::expander::bprop {
NodePtrList AddnGradFunc(BpropIRBuilder *ib) {
  auto dout = ib->GetInput(kIndex2);
  auto x_abs = ib->GetInput(kIndex0)->abstract();
  auto x_len = x_abs->cast<abstract::AbstractSequencePtr>()->elements().size();
  NodePtrList result(x_len, dout);
  if (x_abs->isa<abstract::AbstractList>()) {
    return {ib->MakeList(result)};
  }
  return {ib->MakeTuple(result)};
}

NodePtrList IgammaBpropExpanderDyn(BpropIRBuilder *ib) {
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

NodePtrList IgammaBpropExpander(BpropIRBuilder *ib) {
  auto a = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto sa = ib->GetShape(a);
  auto sx = ib->GetShape(x);
  if (IsDynamic(sa) || IsDynamic(sx)) {
    return IgammaBpropExpanderDyn(ib);
  }
  auto rax = BroadcastGradientArgs(sa, sx);
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

NodePtrList MinimumMaximumGrad(BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dout,
                               bool is_minimum) {
  auto zeros = ib->Emit("FillV2", {ib->Shape(dout), ib->Tensor(0, ib->GetDtype(dout))});
  NodePtr x_mask;
  if (is_minimum) {
    x_mask = ib->LessEqual(x, y);
  } else {
    x_mask = ib->GreaterEqual(x, y);
  }

  auto grad_x = ib->Select(x_mask, dout, zeros);
  auto grad_y = ib->Select(x_mask, zeros, dout);

  auto half_ratio = ib->Emit("FillV2", {ib->Shape(dout), ib->Tensor(2, ib->GetDtype(dout))});
  auto half_dout = ib->Div(dout, half_ratio);
  NodePtr equal_mask = ib->Equal(x, y);
  grad_x = ib->Select(equal_mask, half_dout, grad_x);
  grad_y = ib->Select(equal_mask, half_dout, grad_y);

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

NodePtrList BpropAddcCommon(BpropIRBuilder *ib, const std::string &op_name,
                            const std::unordered_set<TypeId> &type_list) {
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

NodePtrList FminFmaxGrad(BpropIRBuilder *ib, bool if_fmin) {
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
    if (ta) {
      dx = ib->MatMul(w, (ib->Emit("Conj", {dout})), (ta && tb), (ta || (!tb)));
      dx = ib->Emit("Conj", {dx});
    } else {
      dx = ib->MatMul((ib->Emit("Conj", {dout})), w, (ta && tb), (ta || (!tb)));
      dx = ib->Emit("Conj", {dx});
    }
    if (tb) {
      dw = ib->MatMul((ib->Emit("Conj", {dout})), x, ((!ta) || tb), (ta && tb));
      dw = ib->Emit("Conj", {dw});
    } else {
      dw = ib->MatMul((ib->Emit("Conj", {x})), dout, ((!ta) || tb), (ta && tb));
    }
    return {dx, dw};
  }

  if ((*x_type) == (*kComplex64) || (*x_type) == (*kComplex128) || (*w_type) == (*kComplex64) ||
      (*w_type) == (*kComplex128)) {
    // only support complex64 * complex64 and complex128 * complex128, others throw exception
    MS_EXCEPTION(TypeError) << "For 'MatMul', gradient not support x_type " << x_type << " * w_type " << w_type;
  }

  if (ta) {
    dx = ib->MatMul(w, dout, (ta && tb), (ta || (!tb)));
  } else {
    dx = ib->MatMul(dout, w, (ta && tb), (ta || (!tb)));
  }
  if (tb) {
    dw = ib->MatMul(dout, x, ((!ta) || tb), (ta && tb));
  } else {
    dw = ib->MatMul(x, dout, ((!ta) || tb), (ta && tb));
  }
  return {dx, dw};
});

REG_BPROP_BUILDER("Add").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return BinopGradCommon(ib, x, y, dout, dout);
});

REG_BPROP_BUILDER("Mul").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'Mul', gradient not support for complex type currently.";
  }
  auto bc_dx = ib->Mul(y, dout);
  auto bc_dy = ib->Mul(x, dout);
  return BinopGradCommon(ib, x, y, bc_dx, bc_dy);
});

REG_BPROP_BUILDER("Sub").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return BinopGradCommon(ib, x, y, dout, ib->Emit(kNegOpName, {dout}));
});

REG_BPROP_BUILDER("Div").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = ib->Emit(kDivOpName, {dout, y});
  auto bc_y = -(bc_x * out);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    auto result = BinopGradCommon(ib, x, y, bc_x, bc_y);
    return {ib->Conj(result[0]), ib->Conj(result[1])};
  }
  return BinopGradCommon(ib, x, y, bc_x, bc_y);
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
  auto one = ib->Tensor(1, ib->GetDtype(x));
  auto minus_one_p5 = ib->Tensor(-1.5, ib->GetDtype(x));
  auto d2x = ib->Mul((ib->Mul((ib->Mul(dout, grad)), x)), (ib->Pow(ib->Sub(one, (ib->Mul(x, x))), minus_one_p5)));
  auto ddy = ib->Emit("AsinGrad", {x, dout});
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
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto minus_one = ib->Tensor(-1.0, ib->GetDtype(out));
  auto dy = ib->Mul((ib->Mul((ib->Mul(dout, out)), minus_one)), (ib->Emit("Tanh", {y})));
  auto dgrad = ib->Emit("AsinhGrad", {y, dout});
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
  auto one = ib->Tensor(1, ib->GetDtype(x));
  auto minus_one_p5 = ib->Tensor(-1.5, ib->GetDtype(x));
  auto d2x = ib->Mul((ib->Mul((ib->Mul((ib->Emit("Neg", {dout})), grad)), x)),
                     (ib->Pow(ib->Sub(one, (ib->Mul(x, x))), minus_one_p5)));
  auto ddy = ib->Emit("ACosGrad", {x, dout});
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
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dy = ib->RealDiv((ib->Mul((ib->Mul(dout, out)), ib->Tensor(-1.0, ib->GetDtype(out)))), (ib->Emit("Tanh", {y})));
  auto dgrad = ib->Emit("AcoshGrad", {y, dout});
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
    auto dx = ib->Emit("ScalarToTensor", {dout, ib->Value(ib->GetDtype(x))});
    return {dx, ib->OutZeros(t)};
  }
  auto dx = ib->Emit("ScalarCast", {dout, ib->Value(ib->GetDtype(x))});
  return {dx, ib->OutZeros(t)};
});

REG_BPROP_BUILDER("Sign").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Round").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Atan2").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp = ib->RealDiv(dout, (ib->Add((ib->Emit("Square", {x})), (ib->Emit("Square", {y})))));
  auto bc_dx = ib->Mul(tmp, y);
  auto bc_dy = ib->Mul(tmp, (ib->Emit("Neg", {x})));
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
  auto dx = ib->Mul((ib->Mul((ib->Mul(out, dgrad)), ib->Tensor(-2.0, ib->GetDtype(x)))), x);
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
  auto bc_dx = ib->Mul((ib->Mul(power, (ib->Pow(x, ib->Sub(power, ib->Tensor(1.0, ib->GetDtype(x))))))), dout);
  x =
    ib->Select(ib->Less(x, ib->Tensor(0, ib->GetDtype(x))), ib->Fill(1.0, ib->Shape(x), ib->GetDtype(x)->type_id()), x);
  auto bc_dpower = ib->Mul((ib->Mul(out, (ib->Log(x)))), dout);
  return {BinopGradCommon(ib, x, power, bc_dx, bc_dpower)};
});

REG_BPROP_BUILDER("Exp").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto g = ib->Exp(x);
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

REG_BPROP_BUILDER("CumSum").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto reverse = GetValue<bool>(ib->GetAttr("reverse"));
  return {ib->CumSum(dout, axis, ib->GetAttr("exclusive"), MakeValue(!reverse)), ib->OutZeros(axis)};
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
  dx = ib->Reshape(ib->ReduceSum(dx, broadcast_x, false, true), x_shape);
  dy = ib->Reshape(ib->ReduceSum(dy, broadcast_y, false, true), y_shape);
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

REG_BPROP_BUILDER("Logit").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("LogitGrad", {dout, x}, {{"eps", ib->GetAttr("eps")}});
  return {dx};
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
  auto dx = ib->Emit("CdistGrad", {dout, input_x, input_y, out}, {{"p", ib->GetAttr("p")}});
  auto dy = ib->Emit("CdistGrad", {dout_transpose, input_y, input_x, out_transpose}, {{"p", ib->GetAttr("p")}});
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
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto reverse = GetValue<bool>(ib->GetAttr("reverse"));
  auto prod = ib->CumProd(x, axis, ib->GetAttr("exclusive"), MakeValue(reverse));
  out = ib->CumSum(ib->Mul(prod, dout), axis, ib->GetAttr("exclusive"), MakeValue(!reverse));
  return {ib->RealDiv(out, x), ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("IsFinite").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IsNan").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("IsInf").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ReduceAll").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("ReduceAny").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

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
  auto bc_x = ib->RealDiv(dout, y);
  auto bc_y = -(bc_x * out);
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("DivNoNan").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = ib->DivNoNan(dout, y);
  auto bc_y = -(bc_x * out);
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("Xdivy").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype = ib->GetDtype(x);
  auto not_zero_x = ib->Cast(ib->NotEqual(x, ib->Tensor(0.0, x_dtype)), x_dtype);
  auto bc_x = (ib->Xdivy(not_zero_x, y)) * dout;
  auto bc_y = (ib->Xdivy(-x, ib->Emit("Square", {y}))) * dout;
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("FloorDiv").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("FloorMod").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = dout;
  auto bc_y = (-dout) * (ib->FloorDiv(x, y));
  bc_x = ib->Cast(bc_x, ib->GetDtype(x));
  bc_y = ib->Cast(bc_y, ib->GetDtype(y));
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("TruncateDiv").SetUnusedInputs({i0, i1, i2, i3}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("TruncateMod").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = dout;
  auto bc_y = (-dout) * (ib->Emit("TruncateDiv", {x, y}));
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("Mod").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = dout;
  auto bc_y = (-dout) * (ib->FloorDiv(x, y));
  bc_x = ib->Cast(bc_x, ib->GetDtype(x));
  bc_y = ib->Cast(bc_y, ib->GetDtype(y));
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("Xlogy").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype = ib->GetDtype(x);
  auto not_zero_x = ib->Cast(ib->NotEqual(x, ib->Tensor(0.0, x_dtype)), x_dtype);
  auto bc_x = ib->Xlogy(not_zero_x, y) * dout;
  auto bc_y = ib->Xdivy(x, y) * dout;
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("Sqrt").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SqrtGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("SqrtGrad").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto y = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto gy = ib->RealDiv(dout, y);
  auto dy = (-gy) * out;
  auto gy_dtype = ib->GetDtype(gy);
  auto dgrad = ib->Tensor(0.5, gy_dtype) * gy;
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
  auto grad_dtype = ib->GetDtype(grad);
  auto dy = ib->Tensor(-1.5, grad_dtype) * grad * y * y * dout;
  auto dgrad = ib->Emit("RsqrtGrad", {y, dout});
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
  auto g = ib->Emit("Reciprocal", {x});
  auto dx = g * dout;
  return {dx};
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
  auto dout_0 = ib->TupleGetItem(dout, kIndex0);
  auto dout_1 = ib->TupleGetItem(dout, kIndex1);
  auto dx = dout_0 * x * ib->Tensor(2.0, ib->GetDtype(x));
  auto dy = dout_1 * y * ib->Tensor(2.0, ib->GetDtype(y));
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
  auto dx1 = ib->Mul(ib->Div(x1_f32, out_f32), dout_f32);
  auto dx2 = ib->Mul(ib->Div(x2_f32, out_f32), dout_f32);
  auto tmp = BinopGradCommon(ib, x1_f32, x2_f32, dx1, dx2);
  auto result_dx1 = ib->Cast(tmp[0], ib->GetDtype(x1));
  auto result_dx2 = ib->Cast(tmp[1], ib->GetDtype(x2));
  return {result_dx1, result_dx2};
});

REG_BPROP_BUILDER("Trunc").SetUnusedInputs({i0, i1, i2}).SetBody(ReturnZeros);

REG_BPROP_BUILDER("Ger").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto input_y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto m1 = ib->ExpandDims(input_y, 1);
  auto m2 = ib->ExpandDims(input_x, 1);
  ShapeVector axis = {1};
  auto dx = ib->Squeeze(ib->MatMul(dout, m1, false, false), MakeValue(axis));
  ShapeVector perm = {1, 0};
  auto transpose = ib->Transpose(dout, perm);
  auto dy = ib->Squeeze(ib->MatMul(transpose, m2, false, false), MakeValue(axis));
  return {dx, dy};
});

REG_BPROP_BUILDER("Cross").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto input1 = ib->GetInput(kIndex0);
  auto input2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {ib->Emit("Cross", {input2, dout}, {{"dim", ib->GetAttr("dim")}}),
          ib->Emit("Cross", {dout, input1}, {{"dim", ib->GetAttr("dim")}})};
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

REG_BPROP_BUILDER("ReduceSum").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = SumGrad(ib, x, axis, dout, ib->GetAttr<bool>("keep_dims"));
  return {dx, ib->OutZeros(axis)};
});

DEF_PURE_SHAPE_CALC(g_reduce_prod)
  .SetCalc([](const ShapeArray &inputs) -> ShapeArray {
    auto input_shape = inputs.at(0);
    auto axis = inputs.at(1);
    auto output_shape_kept_dims = ReduceShape(input_shape, axis);
    auto [pack_shape, perm] = SplitShapeIndex(input_shape, axis);
    return {output_shape_kept_dims, pack_shape, perm, InvertPermutation(perm)};
  })
  .SetInfer([](const ShapeArray &inputs, const HashSet<size_t> &unknown_inputs) -> std::vector<int64_t> {
    if (!unknown_inputs.empty()) {
      return {-1, 2, -1, -1};
    }
    auto size = SizeToLong(inputs.at(0).size());
    return {size, 2, size, size};
  });
REG_BPROP_BUILDER("ReduceProd").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceProd', gradient not support for complex type currently.";
  }
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  if (ib->GetRank(x) == 0) {
    return {dout, ib->OutZeros(axis)};
  }
  auto res = ib->ShapeCalc(g_reduce_prod, {x, axis}, {1});
  auto grad = ib->GetAttr<bool>("keep_dims") ? dout : ib->Reshape(dout, res[0]);
  grad = ib->BroadcastTo(grad, x);

  auto permuted = ib->Transpose(x, res[2]);
  auto permuted_shape = ib->Shape(permuted);
  auto reshaped = ib->Reshape(permuted, res[1]);
  auto left = ib->CumProd(reshaped, ib->Value<int64_t>(0), true, false);
  auto right = ib->CumProd(reshaped, ib->Value<int64_t>(0), true, true);
  auto y = ib->Reshape(ib->Mul(left, right), permuted_shape);
  auto out = ib->Mul(ib->Transpose(y, res[3]), grad);
  auto dx = ib->Reshape(out, ib->Shape(x));
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("ReduceMax").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceMax', gradient not support for complex type currently.";
  } else {
    auto dx = MinOrMaxGrad(ib, x, axis, out, dout);
    return {dx, ib->OutZeros(axis)};
  }
});

REG_BPROP_BUILDER("ReduceMin").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceMin', gradient not support for complex type currently.";
  } else {
    auto dx = MinOrMaxGrad(ib, x, axis, out, dout);
    return {dx, ib->OutZeros(axis)};
  }
});

REG_BPROP_BUILDER("ReduceMean").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceMean', gradient not support for complex type currently.";
  }
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto grad = SumGrad(ib, x, axis, dout, ib->GetAttr<bool>("keep_dims"));
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
  return {dx, ib->OutZeros(axis)};
});

REG_BPROP_BUILDER("ArgMaxWithValue").SetBody(BODYFUNC(ib) {
  auto axis = GetValue<int64_t>(ib->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(ib->GetAttr("keep_dims"));
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, true);
  return {dx};
});

REG_BPROP_BUILDER("ArgMinWithValue").SetBody(BODYFUNC(ib) {
  auto axis = GetValue<int64_t>(ib->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(ib->GetAttr("keep_dims"));
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, false);
  return {dx};
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
  auto meta_grad_up = ib->Emit("Concat", {ib->MakeTuple({x_transpose, dout})}, {{"axis", MakeValue<int64_t>(-1)}});
  auto meta_grad_down =
    ib->Emit("Concat", {ib->MakeTuple({zero_matrix, x_transpose})}, {{"axis", MakeValue<int64_t>(-1)}});
  auto meta_grad =
    ib->Emit("Concat", {ib->MakeTuple({meta_grad_up, meta_grad_down})}, {{"axis", MakeValue<int64_t>(-2)}});
  meta_grad = ib->Emit("MatrixExp", {meta_grad});
  return {ib->Slice(meta_grad, begins, sizes)};
});

REG_BPROP_BUILDER("Complex").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Real(dout);
  auto dy = ib->Imag(dout);
  return {dx, dy};
});

REG_BPROP_BUILDER("CholeskyInverse").SetBody(BODYFUNC(ib) {
  auto upper = GetValue<bool>(ib->GetAttr("upper"));
  auto input_x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  ShapeVector input_perm = {1, 0};
  NodePtr dx;
  auto DealWithUpper = [&upper, &dx, &ib, &input_x](const NodePtr &common_term) {
    if (upper) {
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
    return {dx};
  }
  auto common_term = ib->Add(dout, ib->Transpose(dout, input_perm));
  common_term = ib->MatMul(out, ib->MatMul(common_term, out, false, false), false, false);
  DealWithUpper(common_term);
  return {dx};
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
  if (weight->isa<ValueNode>()) {
    auto val_weight = weight->get<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(val_weight);
    auto v = val_weight->value();
    MS_EXCEPTION_IF_NULL(v);
    auto val = GetValue<float>(v);
    sub_w = ib->Tensor(1.0 - val, dout_type);
    mul_w = ib->Tensor(val, dout_type);
  } else if (weight->isa<Parameter>()) {
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
  if (weight->isa<ValueNode>()) {
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
  auto LeftShift = [](BpropIRBuilder *ib, NodePtr x) {
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
    return ib->Emit(
      "Pad",
      {ib->Emit("StridedSlice",
                {x, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(end), ib->Value<ShapeVector>(strides)},
                {{"begin_mask", MakeValue<int64_t>(0LL)},
                 {"end_mask", MakeValue<int64_t>(0LL)},
                 {"ellipsis_mask", MakeValue<int64_t>(0LL)},
                 {"new_axis_mask", MakeValue<int64_t>(0LL)},
                 {"shrink_axis_mask", MakeValue<int64_t>(0LL)}})},
      {{"paddings", MakeValue(paddings)}});
  };
  auto RightShift = [](BpropIRBuilder *ib, NodePtr x) {
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
    return ib->Emit(
      "Pad",
      {ib->Emit("StridedSlice",
                {x, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(end), ib->Value<ShapeVector>(strides)},
                {{"begin_mask", MakeValue<int64_t>(0)},
                 {"end_mask", MakeValue<int64_t>(0)},
                 {"ellipsis_mask", MakeValue<int64_t>(0)},
                 {"new_axis_mask", MakeValue<int64_t>(0)},
                 {"shrink_axis_mask", MakeValue<int64_t>(0)}})},
      {{"paddings", MakeValue(paddings)}});
  };
  auto MatrixTranspose = [](BpropIRBuilder *ib, const NodePtr &x) {
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
  return {BinopGradCommon(ib, x, y, dout, dout)};
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
  if (IsDynamic(shape)) {
    norm = ib->Emit("DynamicBroadcastTo", {norm, ib->Shape(input_x)});
  } else {
    norm = ib->Emit("BroadcastTo", {norm}, {{"shape", MakeValue(shape)}});
  }
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

REG_BPROP_BUILDER("ReduceStd").SetBody(BODYFUNC(ib) {
  auto axis = GetIntList(ib->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(ib->GetAttr("keep_dims"));
  auto unbiased = GetValue<bool>(ib->GetAttr("unbiased"));
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto std_d = ib->TupleGetItem(dout, 0);
  auto std = ib->TupleGetItem(out, 0);
  auto mean_d = ib->TupleGetItem(dout, 1);
  auto mean = ib->TupleGetItem(out, 1);
  auto res = ib->ShapeCalc(std::make_shared<ReduceStdShapeCalc>(axis), {x});
  res[1] = ib->SequenceToTensor(res[1]);
  res[2] = ib->SequenceToTensor(res[2]);
  if (!keep_dims && !ib->GetShape(x).empty()) {
    std_d = ib->Reshape(std_d, res[0]);
    std = ib->Reshape(std, res[0]);
    mean_d = ib->Reshape(mean_d, res[0]);
    mean = ib->Reshape(mean, res[0]);
  }
  auto dx = ib->Sub(x, mean);
  dx = ib->Mul(dx, std_d);
  dx = ib->Div(dx, std);
  if (unbiased) {
    dx = ib->Div(dx, ib->Cast(res[1], ib->GetDtype(dx)));
  } else {
    dx = ib->Div(dx, ib->Cast(res[2], ib->GetDtype(dx)));
  }
  auto temp = ib->Div(mean_d, ib->Cast(res[2], ib->GetDtype(mean_d)));
  dx = ib->Add(dx, temp);
  return {dx};
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

  NodePtr dx;
  if (ta) {
    dx = ib->BatchMatMul(w, dout, tb, true);
  } else {
    dx = ib->BatchMatMul(dout, w, false, !tb);
  }

  NodePtr dw;
  if (tb) {
    dw = ib->BatchMatMul(dout, x, true, ta);
  } else {
    dw = ib->BatchMatMul(x, dout, !ta, false);
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
    auto grad_rhs_temp = Adjoint(ib, grad_rhs);
    auto out_temp = Adjoint(ib, out);
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

REG_BPROP_BUILDER("NanToNum").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(dout, (ib->Emit("IsFinite", {x})));
  return {dx};
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
  auto dx = ib->Mul(dout, ib->Emit(kPolygammaOpName, {a, x}));
  return {ib->OutZeros(a), dx};
});

REG_BPROP_BUILDER("Cholesky").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto upper = ib->GetAttr<bool>("upper");
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  if (upper) {
    out = MatrixTranspose(ib, out);
    dout = MatrixTranspose(ib, dout);
  }
  auto dx = ib->Emit("CholeskyGrad", {out, dout});
  return {dx};
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
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
