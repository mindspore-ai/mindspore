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

#include "pipeline/pynative/grad/bprop_expander/bprop_irbuilder.h"
#include "include/common/utils/utils.h"
#include "pipeline/pynative/grad/bprop_expander/grad_ops/common_utils.h"
#include "utils/ms_context.h"

namespace mindspore::expander::bprop {
NodePtrList CheckBpropExpander(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
}

NodePtrList CompareBpropExpander(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(y)};
}

NodePtrList AddnGradFunc(const BpropIRBuilder *ib) {
  auto dout = ib->GetInput(kIndex2);
  auto x_abs = ib->GetInput(kIndex0)->abstract();
  auto x_len = x_abs->cast<abstract::AbstractSequencePtr>()->elements().size();
  NodePtrList result(x_len, dout);
  if (x_abs->isa<abstract::AbstractList>()) {
    return {ib->MakeList(result)};
  }
  return {ib->MakeTuple(result)};
}

NodePtrList IgammaBpropExpanderDyn(const BpropIRBuilder *ib) {
  auto a = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto sa = ib->Shape(a);
  auto sx = ib->Shape(x);
  auto rax = ib->DynamicBroadcastGradientArgs(sa, sx);
  auto ra = ib->TupleGetItem(rax, 0);
  auto rx = ib->TupleGetItem(rax, 1);
  auto partial_a = ib->Emit("IgammaGradA", {a, x});
  auto lgamma = LGamma(ib, a);
  auto partial_x = ib->Exp(
    ib->Sub((ib->Add((ib->Neg(x)), (ib->Mul((ib->Sub(a, (ib->Tensor(1, ib->GetDtype(a))))), (ib->Log(x)))))), lgamma));
  auto r1 = ib->Reshape(SumGradReduceAxisWithCast(ib, ib->Mul(partial_a, dout), ra), sa);
  auto r2 = ib->Reshape(SumGradReduceAxisWithCast(ib, ib->Mul(partial_x, dout), rx), sx);
  return {r1, r2};
}

NodePtrList IgammaBpropExpander(const BpropIRBuilder *ib) {
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
  auto lgamma = LGamma(ib, a);
  auto partial_x = ib->Exp(
    ib->Sub((ib->Add((ib->Neg(x)), (ib->Mul((ib->Sub(a, (ib->Tensor(1, ib->GetDtype(a))))), (ib->Log(x)))))), lgamma));
  auto dout = ib->GetInput(kIndex3);
  NodePtr r1, r2;
  if (ra.size()) {
    r1 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_a, dout), ra), sa);
  } else {
    r1 = ib->Reshape(ib->Mul(partial_a, dout), sa);
  }
  if (rx.size()) {
    r2 = ib->Reshape(ib->ReduceSum(ib->Mul(partial_x, dout), rx), sx);
  } else {
    r2 = ib->Reshape(ib->Mul(partial_x, dout), sx);
  }
  return {r1, r2};
}

inline bool IsScalar(const ShapeVector &shape) { return shape.empty() || (shape.size() == 1 && shape[0] == 1); }

bool IsExpandMinMaxGrad(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y) {
  auto is_ascend = (ib->GetTargetFromContext() == kAscendDevice);
  if (!is_ascend) {
    return false;
  }

  auto x_shp = ib->GetShape(x);
  auto y_shp = ib->GetShape(y);
  // Expand to ops for dynamic case.
  if (IsDynamic(x_shp) || IsDynamic(y_shp)) {
    return true;
  }

  // Only expand to ops in scalar broadcast static case.
  return (IsScalar(x_shp) ^ IsScalar(y_shp));
}

NodePtrList MinimumMaximumGrad(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &y, const NodePtr &dout,
                               bool is_minimum) {
  auto zeros = ib->ZerosLike(dout);
  NodePtr x_mask;
  if (is_minimum) {
    x_mask = ib->LessEqual(x, y);
  } else {
    x_mask = ib->GreaterEqual(x, y);
  }

  auto grad_x = ib->Select(x_mask, dout, zeros);
  auto grad_y = ib->Select(x_mask, zeros, dout);

  return BinopGradCommon(ib, x, y, grad_x, grad_y);
}

REG_BPROP_BUILDERS_BEGIN(GradMathOps)
REG_BPROP_BUILDER("MatMul").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto ta = ib->GetAttr<bool>("transpose_a");
  auto tb = ib->GetAttr<bool>("transpose_b");
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr dx;
  NodePtr dw;
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
  return BinopGradCommon(ib, x, y, bc_x, bc_y);
});

REG_BPROP_BUILDER("Less").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("LessEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("LogicalNot").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("LogicalAnd").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("LogicalOr").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("AssignAdd").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("AssignSub").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

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
  auto conj_x = ib->Emit("Conj", {x});
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
  auto dx = ib->Emit("Conj", {dout});
  return {dx};
});

REG_BPROP_BUILDER("ScalarCast").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto t = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Cast(dout, ib->GetDtype(x));
  return {dx, ib->ZerosLike(t)};
});

REG_BPROP_BUILDER("Sign").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("Round").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

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
    ib->Cast(ib->RealDiv(ib->Tensor(2, ib->GetDtype(x)), (ib->Emit("Sqrt", {ib->Tensor(pi, ib->GetDtype(x))}))),
             ib->GetDtype(x));
  auto x_square = ib->Emit("Square", {x});
  auto dx = ib->Mul((ib->Mul(dout, half_root_pi)), (ib->Exp(ib->Emit("Neg", {x_square}))));
  return {dx};
});

REG_BPROP_BUILDER("Erfc").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto half_root_pi =
    ib->Cast(ib->RealDiv(ib->Tensor(2, ib->GetDtype(x)), (ib->Emit("Sqrt", {ib->Tensor(pi, ib->GetDtype(x))}))),
             ib->GetDtype(x));
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

  if (IsExpandMinMaxGrad(ib, x, y)) {
    MS_LOG(DEBUG) << "Expand MinimumGrad to ops.";
    return MinimumMaximumGrad(ib, x, y, dout, true);
  }
  auto tmp = ib->Emit("MinimumGrad", {x, y, dout}, {{"grad_x", MakeValue(true)}, {"grad_y", MakeValue(true)}});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dy = ib->TupleGetItem(tmp, 1);
  return {dx, dy};
});

REG_BPROP_BUILDER("Maximum").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);

  if (IsExpandMinMaxGrad(ib, x, y)) {
    MS_LOG(DEBUG) << "Expand MaximumGrad to ops.";
    return MinimumMaximumGrad(ib, x, y, dout, false);
  }

  auto tmp = ib->Emit("MaximumGrad", {x, y, dout}, {{"grad_x", MakeValue(true)}, {"grad_y", MakeValue(true)}});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dy = ib->TupleGetItem(tmp, 1);
  return {dx, dy};
});

REG_BPROP_BUILDER("CumSum").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto reverse = GetValue<bool>(ib->GetAttr("reverse"));
  return {ib->Emit("CumSum", {dout, axis}, {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", MakeValue(!reverse)}}),
          ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("MulNoNan").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = ib->Mul(dout, y);
  auto bc_y = ib->Mul(x, dout);
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
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

REG_BPROP_BUILDER("LinSpace").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(BODYFUNC(ib) {
  auto start = ib->GetInput(kIndex0);
  auto stop = ib->GetInput(kIndex1);
  auto num = ib->GetInput(kIndex2);
  return {ib->ZerosLike(start), ib->ZerosLike(stop), ib->ZerosLike(num)};
});

REG_BPROP_BUILDER("IndexAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto axis = ib->EmitValue(ib->GetAttr(kAttrAxis));
  auto dy = ib->Emit(kGatherOpName, {dout, indices, axis});
  return {dout, ib->ZerosLike(indices), dy};
});

REG_BPROP_BUILDER("Logit").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("LogitGrad", {dout, x}, {{"eps", ib->GetAttr("eps")}});
  return {dx};
});

REG_BPROP_BUILDER("Cdist").SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto input_y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
    auto dout_shape = inputs.at(0);
    auto dout_dim = dout_shape.size();
    ShapeVector perm;
    for (uint64_t i = 0; i < dout_dim - 2; ++i) {
      perm.push_back(i);
    }
    perm.push_back(dout_dim - 1);
    perm.push_back(dout_dim - 2);
    return {perm};
  };
  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &) -> ShapeVector {
    auto dout_shape = inputs.at(0);
    return {IsDynamicRank(dout_shape) ? -1 : static_cast<int64_t>(dout_shape.size())};
  };
  auto res = ib->ShapeCalc({dout}, shape_func, infer_func, {})[0];
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
  return {lu_data_grad, ib->ZerosLike(lu_pivots)};
});

REG_BPROP_BUILDER("Sinc").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto product = ib->Mul(ib->Tensor(pi, ib->GetDtype(x)), x);
  auto reciprocal = ib->RealDiv(
    (ib->Sub((ib->Mul(product, (ib->Emit("Cos", {product})))), (ib->Emit("Sin", {product})))), (ib->Mul(product, x)));
  TypeId rec_type = ib->GetDtypeId(reciprocal);
  if (rec_type == kNumberTypeComplex64 || rec_type == kNumberTypeComplex128) {
    reciprocal = ib->Emit("Conj", {reciprocal});
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
  auto prod =
    ib->Emit("CumProd", {x, axis}, {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", MakeValue(reverse)}});
  out = ib->Emit("CumSum", {ib->Mul(prod, dout), axis},
                 {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", MakeValue(!reverse)}});
  return {ib->RealDiv(out, x), ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ReduceAll").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ReduceAny").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("IsFinite").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("IsNan").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("IsInf").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("ApproximateEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("Equal").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("NotEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("Greater").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("GreaterEqual").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("MatrixInverse").SetUnusedInputs({i0}).SetBody(BODYFUNC(ib) {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto out_shape = ib->GetShape(out);
  auto dx = out;
  if (out_shape.size() == 2) {
    dx = ib->MatMul(dout, dx, false, true);
    dx = ib->MatMul(out, dx, true, false);
  } else if (out_shape.size() > 2) {
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

REG_BPROP_BUILDER("FloorDiv").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

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

REG_BPROP_BUILDER("TruncateDiv").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

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

REG_BPROP_BUILDER("Floor").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("Ceil").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

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

REG_BPROP_BUILDER("Trunc").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("Ger").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto input_y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto m1 = ib->ExpandDims(input_y, 1);
  auto m2 = ib->ExpandDims(input_x, 1);
  ShapeVector axis = {1};
  auto dx = ib->Emit("Squeeze", {ib->MatMul(dout, m1, false, false)}, {{"axis", MakeValue(axis)}});
  ShapeVector perm = {1, 0};
  auto transpose = ib->Transpose(dout, perm);
  auto dy = ib->Emit("Squeeze", {ib->MatMul(transpose, m2, false, false)}, {{"axis", MakeValue(axis)}});
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
  auto sqrt = ib->Emit("Sqrt", {ib->Tensor(pi, out_type)});
  auto root_pi_over_two = ib->RealDiv(sqrt, ib->Tensor(2, ib->GetDtype(sqrt)));
  auto out_square = ib->Emit("Square", {out});
  auto dx = ib->Mul((ib->Mul(dout, root_pi_over_two)), (ib->Exp(out_square)));
  return {dx};
});

REG_BPROP_BUILDER("Bernoulli").SetUnusedInputs({i0, i1, i2, i3}).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("ReduceSum").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = SumGrad(ib, x, axis, dout);
  return {dx, ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ReduceProd").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto input_shape = ib->GetShape(x);
  NodePtr output_shape_kept_dims_p;
  NodePtr tile_scaling_p;
  NodePtr pack_shape_p;
  NodePtr perm_p;
  NodePtr invert_perm_p;
  if (IsDynamic(input_shape)) {
    auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
      auto input_shape = inputs.at(0);
      auto axis = inputs.at(1);
      auto output_shape_kept_dims = ReduceShape(input_shape, axis);
      auto tile_scaling = TupleDiv(input_shape, output_shape_kept_dims);
      auto [pack_shape, perm] = SplitShapeIndex(input_shape, axis);
      return {output_shape_kept_dims, tile_scaling, pack_shape, perm, InvertPermutation(perm)};
    };
    auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) -> ShapeVector {
      if (!invalid_indices.empty()) {
        return {-1, -1, 2, -1, -1};
      }
      auto size = SizeToLong(inputs.at(0).size());
      return {size, size, 2, size, size};
    };
    auto res = ib->ShapeCalc({x, axis}, shape_func, infer_func, {1});
    output_shape_kept_dims_p = res[0];
    tile_scaling_p = res[1];
    pack_shape_p = res[2];
    perm_p = res[3];
    invert_perm_p = res[4];
  } else {
    auto output_shape_kept_dims = ReduceShape(input_shape, GetIntList(axis));
    auto tile_scaling = TupleDiv(input_shape, output_shape_kept_dims);
    auto [pack_shape, perm] = SplitShapeIndex(input_shape, GetIntList(axis));
    auto invert_perm = InvertPermutation(perm);
    output_shape_kept_dims_p = ib->Value(output_shape_kept_dims);
    tile_scaling_p = ib->Value(tile_scaling);
    pack_shape_p = ib->Value(pack_shape);
    perm_p = ib->Value(perm);
    invert_perm_p = ib->Value(invert_perm);
  }
  dout = ib->Reshape(dout, output_shape_kept_dims_p);
  auto grad = ib->Tile(dout, tile_scaling_p);
  auto permuted = ib->Transpose(x, perm_p);
  auto permuted_shape = ib->Shape(permuted);
  auto reshaped = ib->Reshape(permuted, pack_shape_p);
  auto left = ib->Emit("CumProd", {reshaped, ib->Value<int64_t>(0)},
                       {{"exclusive", MakeValue(true)}, {"reverse", MakeValue(false)}});
  auto right = ib->Emit("CumProd", {reshaped, ib->Value<int64_t>(0)},
                        {{"exclusive", MakeValue(true)}, {"reverse", MakeValue(true)}});
  auto y = ib->Reshape(ib->Mul(left, right), permuted_shape);
  auto out = ib->Mul(ib->Transpose(y, invert_perm_p), grad);
  auto x_dtype_id = ib->GetDtypeId(x);
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'ReduceProd', gradient not support for complex type currently.";
  } else {
    auto dx = ib->Reshape(out, ib->Shape(x));
    return {dx, ib->ZerosLike(axis)};
  }
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
    return {dx, ib->ZerosLike(axis)};
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
    return {dx, ib->ZerosLike(axis)};
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
  auto grad = SumGrad(ib, x, axis, dout);
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
  return {dx, ib->ZerosLike(axis)};
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
  return {ib->DivNoNan(ib->Mul(ib->Emit("Complex", {dout, ib->ZerosLike(dout)}), x),
                       ib->Emit("Complex", {out, ib->ZerosLike(out)}))};
});

REG_BPROP_BUILDER("Real").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto zero = ib->ZerosLike(dout);
  return {ib->Emit("Complex", {dout, zero})};
});

REG_BPROP_BUILDER("Imag").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto zero = ib->ZerosLike(dout);
  return {ib->Emit("Complex", {zero, dout})};
});

REG_BPROP_BUILDER("Betainc").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto input_a = ib->GetInput(kIndex0);
  auto input_b = ib->GetInput(kIndex1);
  auto input_x = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto sx = ib->Shape(input_x);
  auto log_beta = (LGamma(ib, input_a) + LGamma(ib, input_b)) - LGamma(ib, ib->Add(input_a, input_b));
  auto partial_x = ib->Exp(ib->Sub(
    (ib->Add(
      (ib->Mul((ib->Sub(input_b, ib->Tensor(1, ib->GetDtype(input_b)))), (ib->Emit("Log1p", {ib->Neg(input_x)})))),
      (ib->Xlogy(ib->Sub(input_a, ib->Tensor(1, ib->GetDtype(input_b))), input_x)))),
    log_beta));
  return {ib->ZerosLike(input_a), ib->ZerosLike(input_b), ib->Reshape(ib->Mul(partial_x, dout), sx)};
});

REG_BPROP_BUILDER("LogMatrixDeterminant").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_adj_inv = ib->Emit("MatrixInverse", {x}, {{"adjoint", MakeValue(true)}});
  auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
    auto new_shape = inputs.at(0);
    new_shape.push_back(1);
    new_shape.push_back(1);
    return {new_shape};
  };
  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &) -> ShapeVector {
    auto new_shape = inputs.at(0);
    return {IsDynamicRank(new_shape) ? -1 : static_cast<int64_t>(new_shape.size()) + 2};
  };
  auto res = ib->ShapeCalc({ib->TupleGetItem(out, 1)}, shape_func, infer_func, {})[0];
  auto multipliers = ib->Reshape(ib->TupleGetItem(dout, 1), res);
  auto dx = ib->Mul(multipliers, x_adj_inv);
  return {dx};
});

REG_BPROP_BUILDER("MatrixDeterminant").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_adj_inv = ib->Emit("MatrixInverse", {x}, {{"adjoint", MakeValue(true)}});
  auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
    auto new_shape = inputs.at(0);
    new_shape.push_back(1);
    new_shape.push_back(1);
    return {new_shape};
  };
  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &) -> ShapeVector {
    auto new_shape = inputs.at(0);
    return {IsDynamicRank(new_shape) ? -1 : static_cast<int64_t>(new_shape.size()) + 2};
  };
  auto res = ib->ShapeCalc({out}, shape_func, infer_func, {})[0];
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
  auto EmitBmmPowerGrad = [&ib, &dx, &dout](int repeat, const NodePtr &a, const NodePtr &b) {
    for (int i = 0; i < repeat; ++i) {
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

REG_BPROP_BUILDER("MatrixExp").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
    auto shape_x = inputs.at(0);
    auto x_len = static_cast<int64_t>(shape_x.size());
    auto input_perm = Range(x_len);
    input_perm[x_len - 2] = x_len - 1;
    input_perm[x_len - 1] = x_len - 2;
    auto n = shape_x[x_len - 1];
    ShapeVector begins(x_len, 0);
    begins[x_len - 1] = n;
    auto sizes = shape_x;
    sizes[x_len - 1] = n;
    sizes[x_len - 2] = n;
    return {input_perm, begins, sizes};
  };
  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &) -> ShapeVector {
    auto shape_x = inputs.at(0);
    auto rank = IsDynamicRank(shape_x) ? -1 : static_cast<int64_t>(shape_x.size());
    return {rank, rank, rank};
  };
  auto res = ib->ShapeCalc({x}, shape_func, infer_func, {});
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
  return {ib->Emit("Slice", {meta_grad, begins, sizes})};
});

REG_BPROP_BUILDER("Complex").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("Real", {dout});
  auto dy = ib->Emit("Imag", {dout});
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
  auto DealWithUpper = [&upper, &dx2, &ib, &x2](const NodePtr &common_term) {
    if (upper) {
      dx2 = ib->Emit("Neg", {ib->MatMul(x2, common_term, false, false)});
    } else {
      dx2 = ib->Emit("Neg", {ib->MatMul(common_term, x2, false, false)});
    }
  };
  auto dx1 = ib->Emit("CholeskySolve", {dout, x2}, {{"upper", ib->GetAttr("upper")}});
  if (len_x1 == 2) {
    auto common_term = ib->MatMul(dx1, ib->Transpose(out, input_perm), false, false);
    common_term = ib->Add(common_term, (ib->Transpose(common_term, input_perm)));
    DealWithUpper(common_term);
  } else {
    auto x2_dim_size = static_cast<int64_t>(ib->GetShape(x2).size());
    auto target_order = Range(x2_dim_size - 2);
    target_order.push_back(x2_dim_size - 1);
    target_order.push_back(x2_dim_size - 2);
    auto common_term = ib->BatchMatMul(dx1, ib->Transpose(out, target_order), false, false);
    common_term = ib->Add(common_term, (ib->Transpose(common_term, target_order)));
    DealWithUpper(common_term);
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

REG_BPROP_BUILDER("MinimumGrad").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto tmp = ib->Emit("MinimumGradGrad", {x1, x2, ib->TupleGetItem(dout, 0), ib->TupleGetItem(dout, 1)});
  auto sopd_x1 = ib->ZerosLike(x1);
  auto sopd_x2 = ib->ZerosLike(x2);
  auto sopd_grads = ib->TupleGetItem(tmp, 2);
  return {sopd_x1, sopd_x2, sopd_grads};
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
    auto v = weight->get()->abstract()->BuildValue();
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
    dweight = ib->ZerosLike(weight);
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
  auto LeftShift = [](const BpropIRBuilder *ib, NodePtr x) {
    auto x_shape = ib->GetShape(x);
    std::vector<std::vector<int64_t>> paddings;
    auto rank = x_shape.size();
    for (size_t i = 0; i < rank - 2; i++) {
      paddings.emplace_back(ShapeVector{0LL, 0LL});
    }
    paddings.emplace_back(ShapeVector{0LL, 1LL});
    paddings.emplace_back(ShapeVector{0LL, 0LL});
    ShapeVector begin, end, strides;
    for (size_t i = 0; i < rank; i++) {
      begin.emplace_back(0LL);
      end.emplace_back(x_shape[i]);
      strides.emplace_back(1LL);
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
  auto RightShift = [](const BpropIRBuilder *ib, NodePtr x) {
    auto x_shape = ib->GetShape(x);
    std::vector<std::vector<int64_t>> paddings;
    auto rank = x_shape.size();
    for (size_t i = 0; i < rank - 2; i++) {
      paddings.emplace_back(ShapeVector{0LL, 0LL});
    }
    paddings.emplace_back(ShapeVector{1LL, 0LL});
    paddings.emplace_back(ShapeVector{0LL, 0LL});
    ShapeVector begin, end, strides;
    for (size_t i = 0; i < rank; i++) {
      begin.emplace_back(0LL);
      end.emplace_back(x_shape[i]);
      strides.emplace_back(1LL);
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
  auto MatrixTranspose = [](const BpropIRBuilder *ib, NodePtr x) {
    auto x_shape = ib->GetShape(x);
    auto rank = x_shape.size();
    ShapeVector perm;
    if (rank > IntToSize(2)) {
      for (size_t i = 0; i < rank - 2; i++) {
        perm.emplace_back(SizeToLong(i));
      }
    }
    perm.emplace_back(rank - 1);
    perm.emplace_back(rank - 2);
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
    superdiag_conj = ib->Emit("Conj", {superdiag_conj});
    maindiag_conj = ib->Emit("Conj", {maindiag_conj});
    subdiag_conj = ib->Emit("Conj", {subdiag_conj});
    rhs_conj = ib->Emit("Conj", {rhs});
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
  auto input_data = ib->GetInput(kIndex0);
  auto x1 = ib->GetInput(kIndex1);
  auto x2 = ib->GetInput(kIndex2);
  auto value = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto dinput_data = dout;
  std::unordered_set<TypeId> type_list{TypeId::kNumberTypeInt64, TypeId::kNumberTypeFloat16,
                                       TypeId::kNumberTypeFloat64};
  auto dout_typeptr = ib->GetDtype(dout);
  bool need_cast = type_list.count(dout_typeptr->type_id()) > 0;
  if (need_cast) {
    input_data = ib->Cast(input_data, kFloat32);
    x1 = ib->Cast(x1, kFloat32);
    x2 = ib->Cast(x2, kFloat32);
    value = ib->Cast(value, kFloat32);
    dinput_data = ib->Cast(dinput_data, kFloat32);
  }
  auto inner_out = ib->Add((ib->Mul(value, ib->Div(x1, x2))), input_data);
  auto dx2 = ib->Neg(ib->Mul(ib->Mul(ib->Mul(x1, value), ib->Pow(x2, ib->Tensor(-2, ib->GetDtype(x2)))), dinput_data));
  auto dx1 = ib->Mul(dinput_data, ib->Div(value, x2));
  auto dvalue = ib->Mul(dinput_data, ib->Div(x1, x2));
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
});

REG_BPROP_BUILDER("Addcmul").SetUnusedInputs({i4}).SetBody(BODYFUNC(ib) {
  auto input_data = ib->GetInput(kIndex0);
  auto x1 = ib->GetInput(kIndex1);
  auto x2 = ib->GetInput(kIndex2);
  auto value = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  std::unordered_set<TypeId> type_list{TypeId::kNumberTypeInt8,    TypeId::kNumberTypeInt16,
                                       TypeId::kNumberTypeInt64,   TypeId::kNumberTypeUInt8,
                                       TypeId::kNumberTypeFloat16, TypeId::kNumberTypeFloat64};
  auto dout_typeptr = ib->GetDtype(dout);
  bool need_cast = type_list.count(dout_typeptr->type_id()) > 0;
  if (need_cast) {
    input_data = ib->Cast(input_data, kFloat32);
    x1 = ib->Cast(x1, kFloat32);
    x2 = ib->Cast(x2, kFloat32);
    value = ib->Cast(value, kFloat32);
  }
  auto dinput_data = dout;
  auto dx1 = ib->Mul(dout, ib->Mul(value, x2));
  auto dx2 = ib->Mul(dout, ib->Mul(value, x1));
  auto inner_out = ib->Add((ib->Mul((ib->Mul(x1, x2)), value)), input_data);
  auto dvalue = ib->Mul(dout, ib->Mul(x1, x2));
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
    return {ib->ZerosLike(input_x)};
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
  auto p = GetValue<float>(ib->GetAttr("p"));
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
  norm = ib->Emit("BroadcastTo", {norm}, {{"shape", MakeValue(shape)}});
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
  auto x_shape = ib->GetShape(x);
  if (axis.empty() && !x_shape.empty()) {
    for (int64_t i = 0; i < SizeToLong(x_shape.size()); i++) {
      axis.push_back(i);
    }
  }
  (void)std::transform(axis.begin(), axis.end(), axis.begin(), [&x_shape](const int64_t &c) {
    if (c < 0) {
      return c + SizeToLong(x_shape.size());
    }
    return c;
  });
  for (size_t i = 1; i < axis.size(); ++i) {
    for (size_t j = 0; j < axis.size() - i; ++j) {
      if (axis[j] > (axis[j + 1])) {
        std::swap(axis[j], axis[j + 1]);
      }
    }
  }
  if (!keep_dims && !x_shape.empty()) {
    for (const auto &i : axis) {
      std_d = ib->ExpandDims(std_d, i);
      std = ib->ExpandDims(std, i);
      mean_d = ib->ExpandDims(mean_d, i);
      mean = ib->ExpandDims(mean, i);
    }
  }
  auto dx = ib->Sub(x, mean);
  dx = ib->Mul(dx, std_d);
  dx = ib->Div(dx, std);
  int64_t num = 1;
  for (const auto &i : axis) {
    num *= x_shape[LongToSize(i)];
  }
  if (unbiased) {
    dx = ib->Div(dx, ib->Tensor(num - 1, ib->GetDtype(dx)));
  } else {
    dx = ib->Div(dx, ib->Tensor(num, ib->GetDtype(dx)));
  }
  auto temp = ib->Div(mean_d, ib->Tensor(num, ib->GetDtype(mean_d)));
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
  return {ib->Sub(output_pos, output_neg), ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("NPUAllocFloatStatus").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) { return {}; });

REG_BPROP_BUILDER("NPUGetFloatStatus").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("NPUClearFloatStatus").SetUnusedInputs({i0, i1, i2}).SetBody(CheckBpropExpander);

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
    dx = ib->BatchMatMul(w, dout, (ta && tb), (ta || !tb));
  } else {
    dx = ib->BatchMatMul(dout, w, (ta && tb), (ta || !tb));
  }

  NodePtr dw;
  if (tb) {
    dw = ib->BatchMatMul(dout, x, ((!ta) || tb), (ta && tb));
  } else {
    dw = ib->BatchMatMul(x, dout, ((!ta) || tb), (ta && tb));
  }

  return BinopGradCommonWithShift(ib, x, w, dx, dw, 2);
});

REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
