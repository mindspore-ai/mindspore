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

REG_BPROP_BUILDER(kLessOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(y)};
});

REG_BPROP_BUILDER(kLessEqualOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("LogicalNot").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("LogicalAnd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("LogicalOr").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(y)};
});

REG_BPROP_BUILDER(kAssignAddOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(y)};
});

REG_BPROP_BUILDER(kAssignSubOpName).SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(y)};
});

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

REG_BPROP_BUILDER("Sign").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Round").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

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
    ib->ZerosLike(ib->Tensor(0, ib->GetDtype(axis)))};
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
    dx = ib->Reshape(ib->Emit("ReduceSum", {dx}, {{"axis", MakeValue(broadcast_x)}}), x_shape);
  }
  if (!broadcast_y.empty()) {
    dy = ib->Reshape(ib->Emit("ReduceSum", {dy}, {{"axis", MakeValue(broadcast_y)}}), y_shape);
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

NodePtrList AddnGradFunc(const BpropIRBuilder *ib) {
  auto dout = ib->GetInput(kIndex2);
  auto n = LongToSize(ib->GetAttr<int64_t>("n"));
  NodePtrList result(n, dout);
  return {ib->MakeTuple(result)};
}
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
  return {ib->RealDiv(out, x), ib->ZerosLike(ib->Tensor(0, ib->GetDtype(axis)))};
});

REG_BPROP_BUILDER("ReduceAll").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(ib->Tensor(0, ib->GetDtype(axis)))};
});

REG_BPROP_BUILDER("ReduceAny").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(ib->Tensor(0, ib->GetDtype(axis)))};
});

NodePtrList CheckBpropExpander(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
}

REG_BPROP_BUILDER("IsFinite").SetBody(CheckBpropExpander);
REG_BPROP_BUILDER("IsNan").SetBody(CheckBpropExpander);
REG_BPROP_BUILDER("IsInf").SetBody(CheckBpropExpander);

NodePtrList CompareBpropExpander(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(y)};
}

REG_BPROP_BUILDER("ApproximateEqual").SetBody(CompareBpropExpander);
REG_BPROP_BUILDER("Equal").SetBody(CompareBpropExpander);
REG_BPROP_BUILDER("NotEqual").SetBody(CompareBpropExpander);
REG_BPROP_BUILDER("Greater").SetBody(CompareBpropExpander);
REG_BPROP_BUILDER("GreaterEqual").SetBody(CompareBpropExpander);
}  // namespace mindspore::expander::bprop
