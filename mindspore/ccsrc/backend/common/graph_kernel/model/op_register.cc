/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/model/op_register.h"
#include <memory>

namespace mindspore::graphkernel::inner {
namespace {
class OpRegister {
 public:
  OpRegister(const std::string &name, const CreatorFunc &func) : name_(name) {
    OpRegistry::Instance().Register(name, func);
  }
  ~OpRegister() = default;

 protected:
  // for pclint-plus
  std::string name_;
};

#define JOIN(x, y) x##y
#define UNIQUE_NAME(prefix, cnt) JOIN(prefix, cnt)
#define OP_REGISTER(name, cls)                                                     \
  static_assert(std::is_base_of<PrimOp, cls>::value, " should be base of PrimOp"); \
  static const OpRegister UNIQUE_NAME(g_graphkernel_op, __COUNTER__)(              \
    name, [](const std::string &op) noexcept -> PrimOpPtr { return std::make_shared<cls>(op); })
}  // namespace

/* All nodes supported by GraphKernel are listed below. */
// reshape ops
OP_REGISTER("Reshape", ReshapeOp);
// elemwise ops
OP_REGISTER("Abs", ElemwiseOp);
OP_REGISTER("Add", ElemwiseOp);
OP_REGISTER("Sub", ElemwiseOp);
OP_REGISTER("RealDiv", ElemwiseOp);
OP_REGISTER("Div", ElemwiseOp);
OP_REGISTER("Mul", ElemwiseOp);
OP_REGISTER("Log", ElemwiseOp);
OP_REGISTER("Exp", ElemwiseOp);
OP_REGISTER("Pow", ElemwiseOp);
OP_REGISTER("Sqrt", ElemwiseOp);
OP_REGISTER("Rsqrt", ElemwiseOp);
OP_REGISTER("Neg", ElemwiseOp);
OP_REGISTER("Reciprocal", ElemwiseOp);
OP_REGISTER("Cast", ElemwiseOp);
OP_REGISTER("Round", ElemwiseOp);
OP_REGISTER("Maximum", ElemwiseOp);
OP_REGISTER("Minimum", ElemwiseOp);
OP_REGISTER("Select", ElemwiseOp);
OP_REGISTER("Less", ElemwiseOp);
OP_REGISTER("Equal", ElemwiseOp);
OP_REGISTER("NotEqual", ElemwiseOp);
OP_REGISTER("LessEqual", ElemwiseOp);
OP_REGISTER("GreaterEqual", ElemwiseOp);
OP_REGISTER("Greater", ElemwiseOp);
OP_REGISTER("CReal", CImagRealOp);
OP_REGISTER("CImag", CImagRealOp);
OP_REGISTER("Complex", ComplexOp);
OP_REGISTER("StandardNormal", StandardNormalOp);
OP_REGISTER("IsNan", ElemwiseOp);
OP_REGISTER("IsInf", ElemwiseOp);
OP_REGISTER("IsFinite", ElemwiseOp);
OP_REGISTER("FloorDiv", ElemwiseOp);
OP_REGISTER("Mod", ElemwiseOp);
OP_REGISTER("Floor", ElemwiseOp);
OP_REGISTER("FloorMod", ElemwiseOp);
OP_REGISTER("Erf", ElemwiseOp);
OP_REGISTER("Erfc", ElemwiseOp);
OP_REGISTER("LogicalNot", ElemwiseOp);
OP_REGISTER("LogicalAnd", ElemwiseOp);
OP_REGISTER("LogicalOr", ElemwiseOp);
OP_REGISTER("Sign", ElemwiseOp);
OP_REGISTER("Sin", ElemwiseOp);
OP_REGISTER("Cos", ElemwiseOp);
OP_REGISTER("Asin", ElemwiseOp);
OP_REGISTER("ACos", ElemwiseOp);
OP_REGISTER("Tanh", ElemwiseOp);
OP_REGISTER("Asinh", ElemwiseOp);
OP_REGISTER("Acosh", ElemwiseOp);
OP_REGISTER("Atan", ElemwiseOp);
OP_REGISTER("Atan2", ElemwiseOp);
OP_REGISTER("Expm1", ElemwiseOp);
// broadcast ops
OP_REGISTER("BroadcastTo", BroadcastOp);
OP_REGISTER("Tile", BroadcastOp);
// reduce ops
OP_REGISTER("ReduceSum", ReduceOp);
OP_REGISTER("ReduceMax", ReduceOp);
OP_REGISTER("ReduceMin", ReduceOp);
OP_REGISTER("Argmax", ArgReduceOp);
OP_REGISTER("Argmin", ArgReduceOp);
// opaque ops
OP_REGISTER("_opaque", OpaqueOp);  // default opaque node
OP_REGISTER("Transpose", TransposeOp);
OP_REGISTER("LayoutTransform", LayoutTransformOp);
OP_REGISTER("MatMul", MatMulOp);
OP_REGISTER("PadAkg", PadAkgOp);
OP_REGISTER("UnPadAkg", UnPadAkgOp);
OP_REGISTER("BatchMatMul", OpaqueOp);
OP_REGISTER("CumSum", OpaqueOp);
OP_REGISTER("OneHot", OpaqueOp);
OP_REGISTER("StridedSlice", StridedSliceOp);
OP_REGISTER("StridedSliceOnnx", StridedSliceOnnxOp);
OP_REGISTER("Concat", ConcatOp);
OP_REGISTER("Gather", GatherOp);
OP_REGISTER("Shape", ShapeOp);
OP_REGISTER("ConstantOfShape", ConstantOfShapeOp);
OP_REGISTER("TensorScatterAdd", OpaqueOp);
OP_REGISTER("GatherNd", OpaqueOp);
OP_REGISTER("UnsortedSegmentSum", OpaqueOp);
OP_REGISTER("Conv2D", Conv2dOp);
OP_REGISTER("TransData", OpaqueOp);
OP_REGISTER("ElemAny", ElemAnyOp);
OP_REGISTER("Pool2D", Pool2DOp);
// virtual ops
OP_REGISTER("Assign", VirtualOp);
OP_REGISTER("TupleGetItem", TupleGetItemOp);
}  // namespace mindspore::graphkernel::inner
