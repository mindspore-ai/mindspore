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
#include "include/common/utils/utils.h"
#include "common/graph_kernel/bprop/expander/common_utils.h"

namespace mindspore::expander::bprop {
constexpr auto pi = acos(-1.0);

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
  auto n = LongToSize(ib->GetAttr<int64_t>("n"));
  NodePtrList result(n, dout);
  return {ib->MakeTuple(result)};
}

REG_BPROP_BUILDER(kMatMulOpName).SetBody([](const BpropIRBuilder *builder) -> NodePtrList {
  auto ta = builder->GetAttr<bool>("transpose_a");
  auto tb = builder->GetAttr<bool>("transpose_b");
  auto x = builder->GetInput(kIndex0);
  auto w = builder->GetInput(kIndex1);
  auto dout = builder->GetInput(kIndex3);
  NodePtr dx;
  NodePtr dw;
  if (ta) {
    dx = builder->MatMul(w, dout, (ta && tb), (ta || (!tb)));
  } else {
    dx = builder->MatMul(dout, w, (ta && tb), (ta || (!tb)));
  }
  if (tb) {
    dw = builder->MatMul(dout, x, ((!ta) || tb), (ta && tb));
  } else {
    dw = builder->MatMul(x, dout, ((!ta) || tb), (ta && tb));
  }
  return {dx, dw};
});

REG_BPROP_BUILDER(kAddOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return BinopGradCommon(ib, x, y, dout, dout);
});

REG_BPROP_BUILDER(kMulOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto bc_dx = ib->Mul(y, dout);
  auto bc_dy = ib->Mul(x, dout);
  return BinopGradCommon(ib, x, y, bc_dx, bc_dy);
});

REG_BPROP_BUILDER(kSubOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return BinopGradCommon(ib, x, y, dout, ib->Emit(kNegOpName, {dout}));
});

REG_BPROP_BUILDER(kDivOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = ib->Emit(kDivOpName, {dout, y});
  auto bc_y = -(bc_x * out);
  return BinopGradCommon(ib, x, y, bc_x, bc_y);
});

REG_BPROP_BUILDER(kLessOpName).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER(kLessEqualOpName).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("LogicalNot").SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("LogicalAnd").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("LogicalOr").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER(kAssignAddOpName).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER(kAssignSubOpName).SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("Sin").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(dout, (ib->Emit("Cos", {x})));
  return {dx};
});

REG_BPROP_BUILDER("Asin").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AsinGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("AsinGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto one = ib->Tensor(1, ib->GetDtype(x));
  auto minus_one_p5 = ib->Tensor(-1.5, ib->GetDtype(x));
  auto d2x =
    ib->Mul((ib->Mul((ib->Mul(dout, grad)), x)), (ib->Emit("Pow", {ib->Sub(one, (ib->Mul(x, x))), minus_one_p5})));
  auto ddy = ib->Emit("AsinGrad", {x, dout});
  return {d2x, ddy};
});

REG_BPROP_BUILDER("Asinh").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AsinhGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("AsinhGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto minus_one = ib->Tensor(-1.0, ib->GetDtype(out));
  auto dy = ib->Mul((ib->Mul((ib->Mul(dout, out)), minus_one)), (ib->Emit("Tanh", {y})));
  auto dgrad = ib->Emit("AsinhGrad", {y, dout});
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Sinh").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul((ib->Emit("Cosh", {x})), dout);
  return {dx};
});

REG_BPROP_BUILDER("Cos").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(dout, (ib->Emit("Neg", {ib->Emit("Sin", {x})})));
  return {dx};
});

REG_BPROP_BUILDER("ACos").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("ACosGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("ACosGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto one = ib->Tensor(1, ib->GetDtype(x));
  auto minus_one_p5 = ib->Tensor(-1.5, ib->GetDtype(x));
  auto d2x = ib->Mul((ib->Mul((ib->Mul((ib->Emit("Neg", {dout})), grad)), x)),
                     (ib->Emit("Pow", {ib->Sub(one, (ib->Mul(x, x))), minus_one_p5})));
  auto ddy = ib->Emit("ACosGrad", {x, dout});
  return {d2x, ddy};
});

REG_BPROP_BUILDER("Acosh").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AcoshGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("AcoshGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dy = ib->RealDiv((ib->Mul((ib->Mul(dout, out)), ib->Tensor(-1.0, ib->GetDtype(out)))), (ib->Emit("Tanh", {y})));
  auto dgrad = ib->Emit("AcoshGrad", {y, dout});
  return {dy, dgrad};
});

REG_BPROP_BUILDER("Cosh").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul((ib->Emit("Sinh", {x})), dout);
  return {dx};
});

REG_BPROP_BUILDER("Abs").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AbsGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("Conj").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Conj", {dout});
  return {dx};
});

REG_BPROP_BUILDER("ScalarCast").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto t = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Cast(dout, ib->GetDtype(x));
  return {dx, ib->ZerosLike(t)};
});

REG_BPROP_BUILDER("Sign").SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("Round").SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("Atan2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp = ib->RealDiv(dout, (ib->Add((ib->Emit("Square", {x})), (ib->Emit("Square", {y})))));
  auto bc_dx = ib->Mul(tmp, y);
  auto bc_dy = ib->Mul(tmp, (ib->Emit("Neg", {x})));
  return {BinopGradCommon(ib, x, y, bc_dx, bc_dy)};
});

REG_BPROP_BUILDER("BesselI0e").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(dout, (ib->Sub((ib->Emit("BesselI1e", {x})), (ib->Mul((ib->Emit("Sign", {x})), out)))));
  return {dx};
});

REG_BPROP_BUILDER("Atan").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AtanGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("AtanGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dgrad = ib->Emit("AtanGrad", {x, dout});
  auto dx = ib->Mul((ib->Mul((ib->Mul(out, dgrad)), ib->Tensor(-2.0, ib->GetDtype(x)))), x);
  return {dx, dgrad};
});

REG_BPROP_BUILDER("Log1p").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_1p = ib->Add(x, ib->Tensor(1, ib->GetDtype(x)));
  auto g = ib->Emit("Reciprocal", {x_1p});
  auto dx = ib->Mul(g, dout);
  return {dx};
});

REG_BPROP_BUILDER("Erf").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto half_root_pi =
    ib->Cast(ib->RealDiv(ib->Tensor(2, ib->GetDtype(x)), (ib->Emit("Sqrt", {ib->Tensor(pi, ib->GetDtype(x))}))),
             ib->GetDtype(x));
  auto x_square = ib->Emit("Square", {x});
  auto dx = ib->Mul((ib->Mul(dout, half_root_pi)), (ib->Emit("Exp", {ib->Emit("Neg", {x_square})})));
  return {dx};
});

REG_BPROP_BUILDER("Erfc").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto half_root_pi =
    ib->Cast(ib->RealDiv(ib->Tensor(2, ib->GetDtype(x)), (ib->Emit("Sqrt", {ib->Tensor(pi, ib->GetDtype(x))}))),
             ib->GetDtype(x));
  auto x_square = ib->Emit("Square", {x});
  auto dx =
    ib->Mul(dout, (ib->Mul((ib->Emit("Neg", {half_root_pi})), (ib->Emit("Exp", {ib->Emit("Neg", {x_square})})))));
  return {dx};
});

REG_BPROP_BUILDER("Pow").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto power = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto bc_dx =
    ib->Mul((ib->Mul(power, (ib->Emit("Pow", {x, ib->Sub(power, ib->Tensor(1.0, ib->GetDtype(x)))})))), dout);
  auto shape_x = ib->GetShape(x);
  x = ib->Emit("Select", {ib->Emit("Less", {x, ib->Tensor(0, ib->GetDtype(x))}),
                          ib->Emit("Fill", {ib->EmitValue(ib->GetDtype(x)), ib->Value<ShapeVector>(ib->GetShape(x)),
                                            ib->Tensor(1, ib->GetDtype(x))}),
                          x});
  auto bc_dpower = ib->Mul((ib->Mul(out, (ib->Emit("Log", {x})))), dout);
  return {BinopGradCommon(ib, x, power, bc_dx, bc_dpower)};
});

REG_BPROP_BUILDER("Exp").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto g = ib->Emit("Exp", {x});
  auto dx = ib->Mul(g, dout);
  return {dx};
});

REG_BPROP_BUILDER("Expm1").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto g = ib->Emit("Exp", {x});
  auto dx = ib->Mul(g, dout);
  return {dx};
});

REG_BPROP_BUILDER("Minimum").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp = ib->Emit("MinimumGrad", {x, y, dout}, {{"grad_x", MakeValue(true)}, {"grad_y", MakeValue(true)}});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dy = ib->TupleGetItem(tmp, 1);
  return {dx, dy};
});

REG_BPROP_BUILDER("Maximum").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp = ib->Emit("MaximumGrad", {x, y, dout}, {{"grad_x", MakeValue(true)}, {"grad_y", MakeValue(true)}});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dy = ib->TupleGetItem(tmp, 1);
  return {dx, dy};
});

REG_BPROP_BUILDER("CumSum").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {
    ib->Emit("CumSum", {dout, axis}, {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", ib->GetAttr("reverse")}}),
    ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("MulNoNan").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_shape = ib->GetShape(x);
  auto y_shape = ib->GetShape(y);
  auto dx = ib->Emit("MulNoNan", {dout, y});
  auto dy = ib->Emit("MulNoNan", {x, dout});
  std::vector<std::vector<int64_t>> bc_axis = BroadcastGradientArgs(x_shape, y_shape);
  auto broadcast_x = bc_axis[0];
  auto broadcast_y = bc_axis[1];
  if (!broadcast_x.empty()) {
    dx = ib->Reshape(ib->ReduceSum(dx, broadcast_x), x_shape);
  }
  if (!broadcast_y.empty()) {
    dy = ib->Reshape(ib->ReduceSum(dy, broadcast_y), y_shape);
  }
  return {dx, dy};
});

REG_BPROP_BUILDER("BesselI0").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_i1 = ib->Emit("BesselI1", {x});
  auto dx = ib->Mul(dout, bessel_i1);
  return {dx};
});

REG_BPROP_BUILDER("BesselI1").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_i0 = ib->Emit("BesselI0", {x});
  auto zero = ib->ZerosLike(x);
  auto one = ib->Emit(
    "Fill", {ib->EmitValue(ib->GetDtype(x)), ib->Value<ShapeVector>(ib->GetShape(x)), ib->Tensor(1, ib->GetDtype(x))});
  auto dout_dx =
    ib->Emit("Select", {ib->Emit("Equal", {x, zero}), one, ib->Sub(bessel_i0, (ib->Emit("Div", {out, x})))});
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselJ0").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_j1 = ib->Emit("BesselJ1", {x});
  auto dx = ib->Mul(ib->Emit("Neg", {dout}), bessel_j1);
  return {dx};
});

REG_BPROP_BUILDER("BesselJ1").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_j0 = ib->Emit("BesselJ0", {x});
  auto zero = ib->ZerosLike(x);
  auto zero_p5 = ib->Emit("Fill", {ib->EmitValue(ib->GetDtype(x)), ib->Value<ShapeVector>(ib->GetShape(x)),
                                   ib->Tensor(0.5, ib->GetDtype(x))});
  auto dout_dx =
    ib->Emit("Select", {ib->Emit("Equal", {x, zero}), zero_p5, ib->Sub(bessel_j0, (ib->Emit("Div", {out, x})))});
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselK0").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_k1 = ib->Emit("BesselK1", {x});
  auto dx = ib->Mul(ib->Emit("Neg", {dout}), bessel_k1);
  return {dx};
});

REG_BPROP_BUILDER("BesselK1").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_k0 = ib->Emit("BesselK0", {x});
  auto dout_dx = ib->Emit("Neg", {ib->Add(bessel_k0, (ib->Emit("Div", {out, x})))});
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselK0e").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_k1e = ib->Emit("BesselK1e", {x});
  auto dx = ib->Mul(dout, (ib->Sub(out, bessel_k1e)));
  return {dx};
});

REG_BPROP_BUILDER("BesselK1e").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_k0e = ib->Emit("BesselK0e", {x});
  auto one = ib->Tensor(1, ib->GetDtype(x));
  auto dout_dx = ib->Sub((ib->Mul(out, (ib->Sub(one, (ib->Emit("Reciprocal", {x})))))), bessel_k0e);
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER("BesselY0").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_y1 = ib->Emit("BesselY1", {x});
  auto dx = ib->Mul(ib->Emit("Neg", {dout}), bessel_y1);
  return {dx};
});

REG_BPROP_BUILDER("BesselY1").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto bessel_y0 = ib->Emit("BesselY0", {x});
  auto dout_dx = ib->Sub(bessel_y0, (ib->Emit("Div", {out, x})));
  auto dx = ib->Mul(dout, dout_dx);
  return {dx};
});

REG_BPROP_BUILDER(kAddNOpName).SetBody(AddnGradFunc);

REG_BPROP_BUILDER("AccumulateNV2").SetBody(AddnGradFunc);

REG_BPROP_BUILDER("Tan").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto cosx = ib->Emit("Cos", {x});
  auto secx2 = ib->Square(ib->Reciprocal(cosx));
  auto dx = secx2 * dout;
  return {dx};
});

REG_BPROP_BUILDER("BesselI1e").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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

REG_BPROP_BUILDER("Atanh").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_dtype = ib->GetDtype(x);
  auto one = ib->Tensor(1, x_dtype);
  auto tmp = one - ib->Pow(x, ib->Tensor(2, x_dtype));
  auto dx = ib->Div(one, tmp) * dout;
  return {dx};
});

REG_BPROP_BUILDER(prim::kInv).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(prim::kPrimInvGrad->name(), {out, dout});
  return {dx};
});

REG_BPROP_BUILDER(kLinSpaceOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto start = ib->GetInput(kIndex0);
  auto stop = ib->GetInput(kIndex1);
  auto num = ib->GetInput(kIndex2);
  return {ib->ZerosLike(start), ib->ZerosLike(stop), ib->ZerosLike(num)};
});

REG_BPROP_BUILDER("IndexAdd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto axis = ib->EmitValue(ib->GetAttr(kAttrAxis));
  auto dy = ib->Emit(kGatherOpName, {dout, indices, axis});
  return {dout, ib->ZerosLike(indices), dy};
});

REG_BPROP_BUILDER("ACos").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("ACosGrad", {input_x, dout});
  return {dx};
});

REG_BPROP_BUILDER("Logit").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("LogitGrad", {dout, x}, {{"eps", ib->GetAttr("eps")}});
  return {dx};
});

REG_BPROP_BUILDER("Cdist").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto input_y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dout_shape = ib->GetShape(dout);
  auto dout_dim = dout_shape.size();
  ShapeVector perm;
  for (uint64_t i = 0; i < dout_dim - 2; ++i) {
    perm.push_back(i);
  }
  perm.push_back(dout_dim - 1);
  perm.push_back(dout_dim - 2);
  auto dout_transpose = ib->Emit("Transpose", {dout, ib->Tensor(perm)});
  auto out_transpose = ib->Emit("Transpose", {out, ib->Tensor(perm)});
  auto dx = ib->Emit("CdistGrad", {dout, input_x, input_y, out}, {{"p", ib->GetAttr("p")}});
  auto dy = ib->Emit("CdistGrad", {dout_transpose, input_y, input_x, out_transpose}, {{"p", ib->GetAttr("p")}});
  return {dx, dy};
});

REG_BPROP_BUILDER("LuUnpack").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto lu_data = ib->GetInput(kIndex0);
  auto lu_pivots = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp = ib->Emit("LuUnpackGrad", {ib->TupleGetItem(dout, 1), ib->TupleGetItem(dout, 2), lu_data},
                      {{"L_grad_flag", MakeValue(true)}, {"U_grad_flag", MakeValue(true)}});
  auto dl = ib->TupleGetItem(tmp, 0);
  auto du = ib->TupleGetItem(tmp, 1);
  auto lu_data_grad = ib->Add(dl, du);
  return {lu_data_grad, ib->ZerosLike(lu_pivots)};
});

REG_BPROP_BUILDER("Sinc").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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

REG_BPROP_BUILDER("CumProd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto prod =
    ib->Emit("CumProd", {x, axis}, {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", ib->GetAttr("reverse")}});
  out = ib->Emit("CumSum", {ib->Mul(prod, dout), axis},
                 {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", ib->GetAttr("reverse")}});
  return {ib->RealDiv(out, x), ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ReduceAll").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ReduceAny").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("IsFinite").SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("IsNan").SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("IsInf").SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("ApproximateEqual").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("Equal").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("NotEqual").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("Greater").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("GreaterEqual").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("MatrixInverse").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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

REG_BPROP_BUILDER(kNegOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  return {-dout};
});

REG_BPROP_BUILDER(kRealDivOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = ib->RealDiv(dout, y);
  auto bc_y = -(bc_x * out);
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("DivNoNan").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = ib->Emit("DivNoNan", {dout, y});
  auto bc_y = -(bc_x * out);
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("Xdivy").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype = ib->GetDtype(x);
  auto not_zero_x = ib->Cast(ib->Emit("NotEqual", {x, ib->Tensor(0.0)}), x_dtype);
  auto bc_x = (ib->Emit("Xdivy", {not_zero_x, y})) * dout;
  auto bc_y = (ib->Emit("Xdivy", {-x, ib->Emit("Square", {y})})) * dout;
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("FloorDiv").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("FloorMod").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = dout;
  auto bc_y = (-dout) * (ib->Emit("FloorDiv", {x, y}));
  bc_x = ib->Cast(bc_x, ib->GetDtype(x));
  bc_y = ib->Cast(bc_y, ib->GetDtype(y));
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("TruncateDiv").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("TruncateMod").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = dout;
  auto bc_y = (-dout) * (ib->Emit("TruncateDiv", {x, y}));
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("Mod").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto bc_x = dout;
  auto bc_y = (-dout) * (ib->Emit("FloorDiv", {x, y}));
  bc_x = ib->Cast(bc_x, ib->GetDtype(x));
  bc_y = ib->Cast(bc_y, ib->GetDtype(y));
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER("SquaredDifference").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Tensor(2) * dout * (x - y);
  return {BinopGradCommon(ib, x, y, dx, -dx)};
});

REG_BPROP_BUILDER("Xlogy").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype = ib->GetDtype(x);
  auto not_zero_x = ib->Cast(ib->Emit("NotEqual", {x, ib->Tensor(0.0)}), x_dtype);
  auto bc_x = ib->Emit("Xlogy", {not_zero_x, y}) * dout;
  auto bc_y = ib->Emit("Xdivy", {x, y}) * dout;
  return {BinopGradCommon(ib, x, y, bc_x, bc_y)};
});

REG_BPROP_BUILDER(kSqrtOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SqrtGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("SqrtGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto gy = ib->RealDiv(dout, y);
  auto dy = (-gy) * out;
  auto gy_dtype = ib->GetDtype(gy);
  auto dgrad = ib->Tensor(0.5, gy_dtype) * gy;
  return {dy, dgrad};
});

REG_BPROP_BUILDER(kRsqrtOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("RsqrtGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER(kRsqrtGradOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto grad_dtype = ib->GetDtype(grad);
  auto dy = ib->Tensor(-1.5, grad_dtype) * grad * y * y * dout;
  auto dgrad = ib->Emit("RsqrtGrad", {y, dout});
  return {dy, dgrad};
});

REG_BPROP_BUILDER(kReciprocalOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("ReciprocalGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER(kLogOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto g = ib->Emit("Reciprocal", {x});
  auto dx = g * dout;
  return {dx};
});

REG_BPROP_BUILDER("Floor").SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("Ceil").SetBody(CheckBpropExpander);

REG_BPROP_BUILDER(kSquareOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = dout * x * ib->Tensor(2.0, ib->GetDtype(x));
  return {dx};
});

REG_BPROP_BUILDER("SquaredDifference").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = dout * (x - y) * ib->Tensor(2.0, ib->GetDtype(x));
  return {BinopGradCommon(ib, x, y, dx, -dx)};
});

REG_BPROP_BUILDER(kSquareSumAllOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dout_0 = ib->TupleGetItem(dout, kIndex0);
  auto dout_1 = ib->TupleGetItem(dout, kIndex1);
  auto dx = dout_0 * x * ib->Tensor(2.0, ib->GetDtype(x));
  auto dy = dout_1 * y * ib->Tensor(2.0, ib->GetDtype(y));
  return {dx, dy};
});

REG_BPROP_BUILDER("Hypot").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto x1_f32 = ib->Cast(x1, kFloat32);
  auto x2_f32 = ib->Cast(x2, kFloat32);
  auto out_f32 = ib->Cast(out, kFloat32);
  auto dout_f32 = ib->Cast(dout, kFloat32);
  auto dx1 = ib->Mul(ib->Emit("Div", {x1_f32, out_f32}), dout_f32);
  auto dx2 = ib->Mul(ib->Emit("Div", {x2_f32, out_f32}), dout_f32);
  auto tmp = BinopGradCommon(ib, x1_f32, x2_f32, dx1, dx2);
  auto result_dx1 = ib->Cast(tmp[0], ib->GetDtype(x1));
  auto result_dx2 = ib->Cast(tmp[1], ib->GetDtype(x2));
  return {result_dx1, result_dx2};
});

REG_BPROP_BUILDER("Trunc").SetBody(CheckBpropExpander);

REG_BPROP_BUILDER("Ger").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto input_y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto m1 = ib->Emit("ExpandDims", {input_y, ib->Value<int64_t>(1)});
  auto m2 = ib->Emit("ExpandDims", {input_x, ib->Value<int64_t>(1)});
  ShapeVector axis = {1};
  auto dx = ib->Emit("Squeeze", {ib->MatMul(dout, m1, false, false)}, {{"axis", MakeValue(axis)}});
  ShapeVector perm = {1, 0};
  auto transpose = ib->Emit("Transpose", {dout, ib->EmitValue(MakeValue(perm))});
  auto dy = ib->Emit("Squeeze", {ib->MatMul(transpose, m2, false, false)}, {{"axis", MakeValue(axis)}});
  return {dx, dy};
});

REG_BPROP_BUILDER("Cross").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input1 = ib->GetInput(kIndex0);
  auto input2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {ib->Emit("Cross", {input2, dout}, {{"dim", ib->GetAttr("dim")}}),
          ib->Emit("Cross", {dout, input1}, {{"dim", ib->GetAttr("dim")}})};
});

REG_BPROP_BUILDER("Median").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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

REG_BPROP_BUILDER("Trace").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shape = ib->GetShape(x);
  auto dx = ib->Emit("TraceGrad", {dout, ib->Tensor(shape, kInt64)});
  return {dx};
});

REG_BPROP_BUILDER("Erfinv").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto out_type = ib->GetDtype(dout);
  auto sqrt = ib->Emit("Sqrt", {ib->Tensor(pi, out_type)});
  auto root_pi_over_two = ib->RealDiv(sqrt, ib->Tensor(2, ib->GetDtype(sqrt)));
  auto out_square = ib->Emit("Square", {out});
  auto dx = ib->Mul((ib->Mul(dout, root_pi_over_two)), (ib->Emit("Exp", {out_square})));
  return {dx};
});

REG_BPROP_BUILDER("Bernoulli").SetBody(CompareBpropExpander);

REG_BPROP_BUILDER("ReduceSum").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = SumGrad(ib, x, GetAxisValue(axis), dout);
  return {dx, ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("CumSum").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {
    ib->Emit("CumSum", {dout, axis}, {{"exclusive", ib->GetAttr("exclusive")}, {"reverse", ib->GetAttr("reverse")}}),
    ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ReduceProd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto input_shape = ib->GetShape(x);
  auto output_shape_kept_dims = ReduceShape(input_shape, GetAxisValue(axis));
  dout = ib->Emit("Reshape", {dout, ib->Value<ShapeVector>(output_shape_kept_dims)});
  auto tile_scaling = TupleDiv(input_shape, output_shape_kept_dims);
  auto grad = ib->Emit("Tile", {dout, ib->Value<ShapeVector>(tile_scaling)});
  auto [pack_shape, perm] = SplitShapeIndex(input_shape, GetAxisValue(axis));
  auto permuted = ib->Emit("Transpose", {x, ib->Value<ShapeVector>(perm)});
  auto permuted_shape = ib->GetShape(permuted);
  auto reshaped = ib->Reshape(permuted, pack_shape);
  auto left = ib->Emit("CumProd", {reshaped, ib->Tensor(0, ib->GetDtype(axis))},
                       {{"exclusive", MakeValue(true)}, {"reverse", MakeValue(false)}});
  auto right = ib->Emit("CumProd", {reshaped, ib->Tensor(0, ib->GetDtype(axis))},
                        {{"exclusive", MakeValue(true)}, {"reverse", MakeValue(true)}});
  auto y = ib->Reshape(ib->Mul(left, right), permuted_shape);
  out = ib->Mul((ib->Emit("Transpose", {y, ib->Value<ShapeVector>(InvertPermutation(perm))})), grad);
  auto dx = ib->Reshape(out, input_shape);
  return {dx, ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ReduceMax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dx = MinOrMaxGrad(ib, x, GetAxisValue(axis), out, dout);
  return {dx, ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ReduceMin").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dx = MinOrMaxGrad(ib, x, GetAxisValue(axis), out, dout);
  return {dx, ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ReduceMean").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto grad = SumGrad(ib, x, GetAxisValue(axis), dout);
  auto shape_x = ib->GetShape(x);
  auto shape_out = ib->GetShape(out);
  auto getSize = [](const ShapeVector shape) {
    int64_t size = 1;
    for (auto &i : shape) {
      size *= i;
    }
    return size;
  };
  auto div_shape = getSize(shape_x) / getSize(shape_out);
  auto dx = ib->RealDiv(grad, ib->Tensor(div_shape, ib->GetDtype(grad)));
  return {dx, ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("ArgMaxWithValue").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto axis = GetValue<int64_t>(ib->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(ib->GetAttr("keep_dims"));
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, true);
  return {dx};
});

REG_BPROP_BUILDER("ArgMinWithValue").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto axis = GetValue<int64_t>(ib->GetAttr("axis"));
  auto keep_dims = GetValue<bool>(ib->GetAttr("keep_dims"));
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ArgminOrArgmaxGrad(ib, x, axis, keep_dims, out, dout, false);
  return {dx};
});

REG_BPROP_BUILDER("ComplexAbs").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("DivNoNan", {ib->Mul(ib->Emit("Complex", {dout, ib->ZerosLike(dout)}), x),
                                ib->Emit("Complex", {out, ib->ZerosLike(out)})})};
});

REG_BPROP_BUILDER("Real").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto zero = ib->ZerosLike(dout);
  return {ib->Emit("Complex", {dout, zero})};
});

REG_BPROP_BUILDER("Imag").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex2);
  auto zero = ib->ZerosLike(dout);
  return {ib->Emit("Complex", {zero, dout})};
});

REG_BPROP_BUILDER("Betainc").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_a = ib->GetInput(kIndex0);
  auto input_b = ib->GetInput(kIndex1);
  auto input_x = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto sx = ib->GetShape(input_x);
  auto log_beta =
    (ib->Emit("LGamma", {input_a}) + ib->Emit("LGamma", {input_b})) - (ib->Emit("LGamma", {ib->Add(input_a, input_b)}));
  auto partial_x = ib->Emit(
    "Exp", {ib->Sub((ib->Add((ib->Mul((ib->Sub(input_b, ib->Tensor(1, ib->GetDtype(input_b)))),
                                      (ib->Emit("Log1p", {ib->Neg(input_x)})))),
                             (ib->Emit("Xlogy", {ib->Sub(input_a, ib->Tensor(1, ib->GetDtype(input_b))), input_x})))),
                    log_beta)});
  return {ib->ZerosLike(input_a), ib->ZerosLike(input_b), ib->Reshape(ib->Mul(partial_x, dout), sx)};
});

REG_BPROP_BUILDER("LogMatrixDeterminant").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_adj_inv = ib->Emit("MatrixInverse", {x}, {{"adjoint", MakeValue(true)}});
  auto new_shape = ib->GetShape(ib->TupleGetItem(out, 1));
  new_shape.push_back(1);
  new_shape.push_back(1);
  auto multipliers = ib->Reshape(ib->TupleGetItem(dout, 1), new_shape);
  auto dx = ib->Mul(multipliers, x_adj_inv);
  return {dx};
});

REG_BPROP_BUILDER("MatrixDeterminant").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_adj_inv = ib->Emit("MatrixInverse", {x}, {{"adjoint", MakeValue(true)}});
  auto new_shape = ib->GetShape(out);
  new_shape.push_back(1);
  new_shape.push_back(1);
  auto multipliers = ib->Reshape(ib->Mul(dout, out), new_shape);
  auto dx = ib->Mul(multipliers, x_adj_inv);
  return {dx};
});

REG_BPROP_BUILDER("MatrixPower").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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

REG_BPROP_BUILDER("MatrixSolve").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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
    auto grad_a = ib->BatchMatMul(a, b, false, false);
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

REG_BPROP_BUILDER("MatrixInverse").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->MatMul(dout, out, false, true);
  dx = ib->MatMul(out, dx, true, false);
  dx = ib->Emit("Neg", {dx});
  return {dx};
});

REG_BPROP_BUILDER("MatrixExp").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shape_x = ib->GetShape(x);
  auto x_len = static_cast<int64_t>(shape_x.size());
  auto input_perm = Range(x_len);
  input_perm[x_len - 2] = x_len - 1;
  input_perm[x_len - 1] = x_len - 2;
  auto x_transpose = ib->Emit("Transpose", {x, ib->EmitValue(MakeValue(input_perm))});
  auto zero_matrix = ib->ZerosLike(x);
  zero_matrix = ib->Cast(zero_matrix, ib->GetDtype(dout));
  auto meta_grad_up = ib->Emit("Concat", {ib->MakeTuple({x_transpose, dout})}, {{"axis", MakeValue<int64_t>(-1)}});
  auto meta_grad_down =
    ib->Emit("Concat", {ib->MakeTuple({zero_matrix, x_transpose})}, {{"axis", MakeValue<int64_t>(-1)}});
  auto meta_grad =
    ib->Emit("Concat", {ib->MakeTuple({meta_grad_up, meta_grad_down})}, {{"axis", MakeValue<int64_t>(-2)}});
  meta_grad = ib->Emit("MatrixExp", {meta_grad});
  auto n = shape_x[x_len - 1];
  ShapeVector begins(x_len, 0);
  begins[x_len - 1] = n;
  auto sizes = shape_x;
  sizes[x_len - 1] = n;
  sizes[x_len - 2] = n;
  return {ib->Emit("Slice", {meta_grad, ib->EmitValue(MakeValue(begins)), ib->EmitValue(MakeValue(sizes))})};
});

REG_BPROP_BUILDER("Complex").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("Real", {dout});
  auto dy = ib->Emit("Imag", {dout});
  return {dx, dy};
});

REG_BPROP_BUILDER("CholeskyInverse").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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
    auto common_term = ib->Add(dout, ib->Emit("Transpose", {dout, ib->EmitValue(MakeValue(input_perm))}));
    common_term = ib->Cast(common_term, kFloat32);
    common_term = ib->MatMul(out, ib->MatMul(common_term, out, false, false), false, false);
    DealWithUpper(common_term);
    dx = ib->Cast(dx, kFloat64);
    return {dx};
  }
  auto common_term = ib->Add(dout, ib->Emit("Transpose", {dout, ib->EmitValue(MakeValue(input_perm))}));
  common_term = ib->MatMul(out, ib->MatMul(common_term, out, false, false), false, false);
  DealWithUpper(common_term);
  return {dx};
});

REG_BPROP_BUILDER("CholeskySolve").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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
    auto common_term =
      ib->MatMul(dx1, ib->Emit("Transpose", {out, ib->EmitValue(MakeValue(input_perm))}), false, false);
    common_term = ib->Add(common_term, (ib->Emit("Transpose", {common_term, ib->EmitValue(MakeValue(input_perm))})));
    DealWithUpper(common_term);
  } else {
    auto x2_dim_size = static_cast<int64_t>(ib->GetShape(x2).size());
    auto target_order = Range(x2_dim_size - 2);
    target_order.push_back(x2_dim_size - 1);
    target_order.push_back(x2_dim_size - 2);
    auto common_term =
      ib->BatchMatMul(dx1, ib->Emit("Transpose", {out, ib->EmitValue(MakeValue(target_order))}), false, false);
    common_term = ib->Add(common_term, (ib->Emit("Transpose", {common_term, ib->EmitValue(MakeValue(target_order))})));
    DealWithUpper(common_term);
  }
  if (flag == 1) {
    dx1 = ib->Cast(dx1, kFloat64);
    dx2 = ib->Cast(dx2, kFloat64);
  }
  return {dx1, dx2};
});

REG_BPROP_BUILDER("NextAfter").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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
  auto s_x1 = ib->GetShape(x1);
  auto partial_x1 = ib->Emit("Fill", {ib->EmitValue(x1_type), ib->Value<ShapeVector>(s_x1), ib->Tensor(1, x1_type)});
  auto s_x2 = ib->GetShape(x2);
  auto partial_x2 = ib->ZerosLike(x2);
  auto dx1 = ib->Reshape(ib->Mul(partial_x1, dout), s_x1);
  auto dx2 = ib->Reshape(ib->Mul(partial_x2, dout), s_x2);
  return {ib->Cast(dx1, dout_type), ib->Cast(dx2, dout_type)};
});

REG_BPROP_BUILDER("MinimumGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto tmp = ib->Emit("MinimumGradGrad", {x1, x2, ib->TupleGetItem(dout, 0), ib->TupleGetItem(dout, 1)});
  auto sopd_x1 = ib->ZerosLike(x1);
  auto sopd_x2 = ib->ZerosLike(x2);
  auto sopd_grads = ib->TupleGetItem(tmp, 2);
  return {sopd_x1, sopd_x2, sopd_grads};
});

REG_BPROP_BUILDER("Lerp").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto start = ib->GetInput(kIndex0);
  auto end = ib->GetInput(kIndex1);
  auto weight = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  dout = ib->Cast(dout, kFloat32);
  NodePtr sub_w, mul_w;
  if (weight->isa<ValueNode>()) {
    auto val_weight = weight->get<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(val_weight);
    auto v = val_weight->value();
    MS_EXCEPTION_IF_NULL(v);
    auto val = GetValue<float>(v);
    sub_w = ib->Tensor(1.0 - val, ib->GetDtype(dout));
    mul_w = ib->Tensor(val, ib->GetDtype(dout));
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

REG_BPROP_BUILDER("TridiagonalMatMul").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
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
    return ib->Emit("Transpose", {x, ib->Value<ShapeVector>(perm)});
  };
  auto superdiag = ib->GetInput(kIndex0);
  auto maindiag = ib->GetInput(kIndex1);
  auto subdiag = ib->GetInput(kIndex2);
  auto rhs = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto grad = ib->GetInput(kIndex5);

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
  auto superdiag_grad = ib->ReduceSum(LeftShift(ib, rhs_conj) * grad, ShapeVector{-1LL});
  auto maindiag_grad = ib->ReduceSum(rhs_conj * grad, ShapeVector{-1LL});
  auto subdiag_grad = ib->ReduceSum(RightShift(ib, rhs_conj) * grad, ShapeVector{-1LL});
  auto rhs_grad = RightShift(ib, superdiag_conj * grad) + maindiag_conj * grad + LeftShift(ib, subdiag_conj * grad);
  superdiag_grad = ib->Emit("ExpandDims", {superdiag_grad, ib->Value<int64_t>(-2)});
  maindiag_grad = ib->Emit("ExpandDims", {maindiag_grad, ib->Value<int64_t>(-2)});
  subdiag_grad = ib->Emit("ExpandDims", {subdiag_grad, ib->Value<int64_t>(-2)});
  return {superdiag_grad, maindiag_grad, subdiag_grad, rhs_grad};
});

REG_BPROP_BUILDER("AddV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  return {BinopGradCommon(ib, x, y, dout, dout)};
});
}  // namespace mindspore::expander::bprop
