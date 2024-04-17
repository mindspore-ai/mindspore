/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/pi/graph_compiler/pi_ir/ir_visitor.h"
#include <algorithm>

namespace mindspore {
namespace pijit {
namespace ir {
IRVisitor::FVisit &IRVisitor::vtable() {  // NOLINT(*)
  static FVisit inst;
  return inst;
}

#define DEFINE_LEAF_NODE_VISIT_FUNC_(OP) \
  void IRVisitor::Visit_(const OP &node) {}

DEFINE_LEAF_NODE_VISIT_FUNC_(PlaceHolderPtr)
DEFINE_LEAF_NODE_VISIT_FUNC_(RefNodePtr)
DEFINE_LEAF_NODE_VISIT_FUNC_(ValuePtr)
DEFINE_LEAF_NODE_VISIT_FUNC_(ParameterPtr)

#define DEFINE_OP_NODE_VISIT_FUNC_(OP) \
  void IRVisitor::Visit_(const OP &node) { VISIT_NODE_LIST(node->GetArgs()) }

DEFINE_OP_NODE_VISIT_FUNC_(CastNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(DeleteNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(GetNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(InvertNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(NegativeNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(NotNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(ReturnNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(LoadValueNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(UnaryOperationPtr)

DEFINE_OP_NODE_VISIT_FUNC_(AddNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(SubNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(MulNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(DivNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(BitwiseNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(CompareNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(ContainsNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(IsNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(JumpNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(StoreNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(UpdateNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(BinaryOperationPtr)

DEFINE_OP_NODE_VISIT_FUNC_(LoadFieldNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(BuildNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(CallNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(NaryWithFlagNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(FormatNodePtr)
DEFINE_OP_NODE_VISIT_FUNC_(NaryOperationPtr)

void IRVisitor::Visit_(const FunctionNodePtr &node) {
  VISIT_NODE_LIST(node->GetParameters())
  VISIT_NODE_LIST(node->GetNodes())
}

void IRVisitor::Visit_(const IfNodePtr &node) {
  Visit(node->GetCondition());
  VISIT_NODE_LIST(node->GetThen())
  VISIT_NODE_LIST(node->GetElse())
}

void IRVisitor::Visit_(const WhileNodePtr &node) {
  Visit(node->GetCondition());
  VISIT_NODE_LIST(node->GetBody())
}

void IRVisitor::Visit_(const SubscrNodePtr &node) {
  Visit(node->GetObject());
  Visit(node->GetSubscr());
}

void IRVisitor::Visit_(const AttrNodePtr &node) {
  Visit(node->GetObject());
  Visit(node->GetAttr());
}

void IRVisitor::Visit_(const PairNodePtr &node) {
  Visit(node->GetFirst());
  Visit(node->GetSecond());
}

STATIC_IR_FUNCTOR(IRVisitor, vtable)
  .DISPATCH_TO_VISIT(PlaceHolder)
  .DISPATCH_TO_VISIT(RefNode)
  .DISPATCH_TO_VISIT(Value)
  .DISPATCH_TO_VISIT(Parameter)
  .DISPATCH_TO_VISIT(CastNode)
  .DISPATCH_TO_VISIT(DeleteNode)
  .DISPATCH_TO_VISIT(GetNode)
  .DISPATCH_TO_VISIT(InvertNode)
  .DISPATCH_TO_VISIT(NegativeNode)
  .DISPATCH_TO_VISIT(NotNode)
  .DISPATCH_TO_VISIT(ReturnNode)
  .DISPATCH_TO_VISIT(LoadValueNode)
  .DISPATCH_TO_VISIT(UnaryOperation)
  .DISPATCH_TO_VISIT(AddNode)
  .DISPATCH_TO_VISIT(SubNode)
  .DISPATCH_TO_VISIT(MulNode)
  .DISPATCH_TO_VISIT(DivNode)
  .DISPATCH_TO_VISIT(BitwiseNode)
  .DISPATCH_TO_VISIT(CompareNode)
  .DISPATCH_TO_VISIT(ContainsNode)
  .DISPATCH_TO_VISIT(IsNode)
  .DISPATCH_TO_VISIT(JumpNode)
  .DISPATCH_TO_VISIT(StoreNode)
  .DISPATCH_TO_VISIT(UpdateNode)
  .DISPATCH_TO_VISIT(BinaryOperation)
  .DISPATCH_TO_VISIT(LoadFieldNode)
  .DISPATCH_TO_VISIT(BuildNode)
  .DISPATCH_TO_VISIT(CallNode)
  .DISPATCH_TO_VISIT(NaryWithFlagNode)
  .DISPATCH_TO_VISIT(FormatNode)
  .DISPATCH_TO_VISIT(NaryOperation)
  .DISPATCH_TO_VISIT(FunctionNode)
  .DISPATCH_TO_VISIT(IfNode)
  .DISPATCH_TO_VISIT(WhileNode)
  .DISPATCH_TO_VISIT(AttrNode)
  .DISPATCH_TO_VISIT(PairNode)
  .DISPATCH_TO_VISIT(SubscrNode);
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore
