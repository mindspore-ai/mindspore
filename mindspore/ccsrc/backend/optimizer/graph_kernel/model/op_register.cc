/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/graph_kernel/model/op_register.h"
#include <memory>

namespace mindspore::graphkernel::inner {
namespace {
class OpRegister {
 public:
  OpRegister(const std::string &name, const CreatorFunc &func) : name_(name) {
    OpRegistry::Instance().Register(name_, func);
  }
  ~OpRegister() = default;

 private:
  // for pclint-plus
  std::string name_;
};

#define JOIN(x, y) x##y
#define UNIQUE_NAME(prefix, cnt) JOIN(prefix, cnt)
#define OP_REGISTER(name, cls)                                                     \
  static_assert(std::is_base_of<PrimOp, cls>::value, " should be base of PrimOp"); \
  static const OpRegister UNIQUE_NAME(g_graphkernel_op, __COUNTER__)(              \
    name, [](const std::string &op) -> PrimOpPtr { return std::make_shared<cls>(op); })
}  // namespace

// All nodes supported by GraphKernel are listed below.
OP_REGISTER("_opaque", OpaqueOp);
OP_REGISTER("Add", ElemwiseOp);
OP_REGISTER("Sub", ElemwiseOp);
OP_REGISTER("RealDiv", ElemwiseOp);
OP_REGISTER("Mul", ElemwiseOp);
OP_REGISTER("Log", ElemwiseOp);
OP_REGISTER("Exp", ElemwiseOp);
OP_REGISTER("Pow", ElemwiseOp);
OP_REGISTER("Sqrt", ElemwiseOp);
OP_REGISTER("Rsqrt", ElemwiseOp);
OP_REGISTER("Neg", ElemwiseOp);
OP_REGISTER("Reciprocal", ElemwiseOp);
OP_REGISTER("Abs", ElemwiseOp);
OP_REGISTER("BroadcastTo", BroadcastToOp);
OP_REGISTER("Reshape", ReshapeOp);
OP_REGISTER("ReduceSum", ReduceOp);
OP_REGISTER("ReduceMax", ReduceOp);
OP_REGISTER("ReduceMin", ReduceOp);
OP_REGISTER("Cast", CastOp);
OP_REGISTER("InplaceAssign", InplaceAssignOp);
OP_REGISTER("Select", SelectOp);
OP_REGISTER("Less", CompareOp);
OP_REGISTER("Equal", CompareOp);
OP_REGISTER("LessEqual", CompareOp);
OP_REGISTER("GreaterEqual", CompareOp);
OP_REGISTER("Greater", CompareOp);
OP_REGISTER("Transpose", TransposeOp);
OP_REGISTER("MatMul", MatMulOp);
OP_REGISTER("PadAkg", PadAkgOp);
OP_REGISTER("UnPadAkg", UnPadAkgOp);
OP_REGISTER("CReal", CRealOp);
OP_REGISTER("CImag", CImagOp);
OP_REGISTER("Complex", ComplexOp);
OP_REGISTER("StandardNormal", StandardNormalOp);
}  // namespace mindspore::graphkernel::inner
