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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_OP_REGISTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_OP_REGISTER_H_

#include <unordered_map>
#include <functional>
#include <string>
#include <memory>

#include "backend/optimizer/graph_kernel/model/node.h"

namespace mindspore {
namespace opt {
namespace graphkernel {
#define OP_CREATOR(cls) \
  [](const std::string &op, const std::string &name) -> PrimOpPtr { return std::make_shared<cls>(op, name); }

class OpRegistry {
 public:
  static OpRegistry &Instance() {
    static OpRegistry instance{};
    return instance;
  }
  void Register(const std::string &op_name,
                const std::function<PrimOpPtr(const std::string &, const std::string &)> &func) {
    creators.insert({op_name, func});
  }

  PrimOpPtr NewOp(const std::string &op, const std::string &name) {
    return creators.find(op) == creators.end() ? creators["Opaque"](op, name) : creators[op](op, name);
  }

 private:
  OpRegistry() {
    Register("Add", OP_CREATOR(ElemwiseOp));
    Register("Sub", OP_CREATOR(ElemwiseOp));
    Register("RealDiv", OP_CREATOR(ElemwiseOp));
    Register("Mul", OP_CREATOR(ElemwiseOp));
    Register("Log", OP_CREATOR(ElemwiseOp));
    Register("Exp", OP_CREATOR(ElemwiseOp));
    Register("Pow", OP_CREATOR(ElemwiseOp));
    Register("Sqrt", OP_CREATOR(ElemwiseOp));
    Register("Rsqrt", OP_CREATOR(ElemwiseOp));
    Register("Neg", OP_CREATOR(ElemwiseOp));
    Register("Reciprocal", OP_CREATOR(ElemwiseOp));
    Register("Abs", OP_CREATOR(ElemwiseOp));
    Register("BroadcastTo", OP_CREATOR(BroadcastToOp));
    Register("Reshape", OP_CREATOR(ReshapeOp));
    Register("ReduceSum", OP_CREATOR(ReduceOp));
    Register("ReduceMax", OP_CREATOR(ReduceOp));
    Register("ReduceMin", OP_CREATOR(ReduceOp));
    Register("Cast", OP_CREATOR(CastOp));
    Register("InplaceAssign", OP_CREATOR(InplaceAssignOp));
    Register("Select", OP_CREATOR(SelectOp));
    Register("Less", OP_CREATOR(LessOp));
    Register("Equal", OP_CREATOR(EqualOp));
    Register("LessEqual", OP_CREATOR(LessEqualOp));
    Register("GreaterEqual", OP_CREATOR(GreaterEqualOp));
    Register("Greater", OP_CREATOR(GreaterOp));
    Register("Transpose", OP_CREATOR(TransposeOp));
    Register("MatMul", OP_CREATOR(MatMulOp));
    Register("PadAkg", OP_CREATOR(PadAkgOp));
    Register("UnPadAkg", OP_CREATOR(UnPadAkgOp));
    Register("CReal", OP_CREATOR(CRealOp));
    Register("CImag", OP_CREATOR(CImagOp));
    Register("Complex", OP_CREATOR(ComplexOp));
    Register("Opaque", OP_CREATOR(OpaqueOp));
    Register("StandardNormal", OP_CREATOR(StandardNormalOp));
  }
  ~OpRegistry() = default;
  std::unordered_map<std::string, std::function<PrimOpPtr(const std::string &, const std::string &)>> creators;
};

}  // namespace graphkernel
}  // namespace opt
}  // namespace mindspore
#endif
