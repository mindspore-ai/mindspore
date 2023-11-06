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
namespace jit {
namespace graph {
namespace ir {
IRVisitor::FVisit &IRVisitor::vtable() {  // NOLINT(*)
  static FVisit inst;
  return inst;
}

#define DEFINE_LEAF_NODE_VISIT_(OP) \
  void IRVisitor::Visit_(const OP &node) {}

DEFINE_LEAF_NODE_VISIT_(PlaceHolderPtr)
DEFINE_LEAF_NODE_VISIT_(RefNodePtr)
DEFINE_LEAF_NODE_VISIT_(ValuePtr)
DEFINE_LEAF_NODE_VISIT_(ParameterPtr)

#define DEFINE_UN_NODE_VISIT_(OP) \
  void IRVisitor::Visit_(const OP &node) { Visit(node->GetArg()); }

DEFINE_UN_NODE_VISIT_(CastNodePtr)
DEFINE_UN_NODE_VISIT_(DeleteNodePtr)
DEFINE_UN_NODE_VISIT_(GetNodePtr)
DEFINE_UN_NODE_VISIT_(InvertNodePtr)
DEFINE_UN_NODE_VISIT_(NegativeNodePtr)
DEFINE_UN_NODE_VISIT_(NotNodePtr)
DEFINE_UN_NODE_VISIT_(ReturnNodePtr)
DEFINE_UN_NODE_VISIT_(UnaryOperationPtr)

#define DEFINE_BIN_NODE_VISIT_(OP)         \
  void IRVisitor::Visit_(const OP &node) { \
    Visit(node->GetLeftArg());             \
    Visit(node->GetRightArg());            \
  }

DEFINE_BIN_NODE_VISIT_(AddNodePtr)
DEFINE_BIN_NODE_VISIT_(SubNodePtr)
DEFINE_BIN_NODE_VISIT_(MulNodePtr)
DEFINE_BIN_NODE_VISIT_(DivNodePtr)
DEFINE_BIN_NODE_VISIT_(BitwiseNodePtr)
DEFINE_BIN_NODE_VISIT_(CompareNodePtr)
DEFINE_BIN_NODE_VISIT_(ContainsNodePtr)
DEFINE_BIN_NODE_VISIT_(IsNodePtr)
DEFINE_BIN_NODE_VISIT_(JumpNodePtr)
DEFINE_BIN_NODE_VISIT_(StoreNodePtr)
DEFINE_BIN_NODE_VISIT_(UpdateNodePtr)
DEFINE_BIN_NODE_VISIT_(BinaryOperationPtr)

#define DEFINE_N_NODE_VISIT_(OP) \
  void IRVisitor::Visit_(const OP &node) { VISIT_NODE_LIST(node->GetArgs()) }

DEFINE_N_NODE_VISIT_(LoadNodePtr)
DEFINE_N_NODE_VISIT_(BuildNodePtr)
DEFINE_N_NODE_VISIT_(CallNodePtr)
DEFINE_N_NODE_VISIT_(NaryWithFlagNodePtr)
DEFINE_N_NODE_VISIT_(FormatNodePtr)
DEFINE_N_NODE_VISIT_(NaryOperationPtr)

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
  .DISPATCH_TO_VISIT(LoadNode)
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
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
