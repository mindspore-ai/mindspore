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
  auto bc_y = ib->Emit(kNegOpName, {ib->Mul(bc_x, out)});
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
}  // namespace mindspore::expander::bprop
