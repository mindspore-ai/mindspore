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
#include "pipeline/jit/pi/graph_compiler/pi_ir/ir_mutator.h"

namespace mindspore {
namespace jit {
namespace graph {
namespace ir {
IRMutator::FMutate &IRMutator::vtable() {  // NOLINT(*)
  static FMutate inst;
  return inst;
}

#define DEFINE_LEAF_NODE_MUTATE_FUNC_(OP) \
  NodePtr IRMutator::Mutate_(const OP &node) { return node; }

DEFINE_LEAF_NODE_MUTATE_FUNC_(RefNodePtr)
DEFINE_LEAF_NODE_MUTATE_FUNC_(ValuePtr)
DEFINE_LEAF_NODE_MUTATE_FUNC_(ParameterPtr)

#define DEFINE_OP_NODE_MUTATE_(OP)             \
  NodePtr IRMutator::Mutate_(const OP &node) { \
    MUTATE_NODE_LIST(node->GetArgs())          \
    return node;                               \
  }

DEFINE_OP_NODE_MUTATE_(CastNodePtr)
DEFINE_OP_NODE_MUTATE_(InvertNodePtr)
DEFINE_OP_NODE_MUTATE_(NegativeNodePtr)
DEFINE_OP_NODE_MUTATE_(NotNodePtr)
DEFINE_OP_NODE_MUTATE_(ReturnNodePtr)
DEFINE_OP_NODE_MUTATE_(LoadValueNodePtr)
DEFINE_OP_NODE_MUTATE_(UnaryOperationPtr)

DEFINE_OP_NODE_MUTATE_(AddNodePtr)
DEFINE_OP_NODE_MUTATE_(SubNodePtr)
DEFINE_OP_NODE_MUTATE_(MulNodePtr)
DEFINE_OP_NODE_MUTATE_(DivNodePtr)
DEFINE_OP_NODE_MUTATE_(BitwiseNodePtr)
DEFINE_OP_NODE_MUTATE_(CompareNodePtr)
DEFINE_OP_NODE_MUTATE_(ContainsNodePtr)
DEFINE_OP_NODE_MUTATE_(IsNodePtr)
DEFINE_OP_NODE_MUTATE_(StoreNodePtr)
DEFINE_OP_NODE_MUTATE_(UpdateNodePtr)
DEFINE_OP_NODE_MUTATE_(BinaryOperationPtr)

DEFINE_OP_NODE_MUTATE_(LoadFieldNodePtr)
DEFINE_OP_NODE_MUTATE_(BuildNodePtr)
DEFINE_OP_NODE_MUTATE_(CallNodePtr)
DEFINE_OP_NODE_MUTATE_(FormatNodePtr)
DEFINE_OP_NODE_MUTATE_(NaryOperationPtr)
DEFINE_OP_NODE_MUTATE_(NaryWithFlagNodePtr)

NodePtr IRMutator::Mutate_(const FunctionNodePtr &node) {
  MUTATE_NODE_LIST(node->GetParameters())
  MUTATE_NODE_LIST(node->GetNodes())
  return node;
}

NodePtr IRMutator::Mutate_(const IfNodePtr &node) {
  node->SetCondition(Mutate(node->GetCondition()));
  MUTATE_NODE_LIST(node->GetThen())
  MUTATE_NODE_LIST(node->GetElse())
  return node;
}

NodePtr IRMutator::Mutate_(const WhileNodePtr &node) {
  node->SetCondition(Mutate(node->GetCondition()));
  MUTATE_NODE_LIST(node->GetBody())
  return node;
}

NodePtr IRMutator::Mutate_(const SubscrNodePtr &node) {
  node->SetObject(Mutate(node->GetObject()));
  node->SetSubscr(Mutate(node->GetSubscr()));
  return node;
}

NodePtr IRMutator::Mutate_(const AttrNodePtr &node) {
  node->SetObject(Mutate(node->GetObject()));
  node->SetAttr(Mutate(node->GetAttr()));
  return node;
}

NodePtr IRMutator::Mutate_(const PairNodePtr &node) {
  node->SetFirst(Mutate(node->GetFirst()));
  node->SetSecond(Mutate(node->GetSecond()));
  return node;
}

STATIC_IR_FUNCTOR(IRMutator, vtable)
  .DISPATCH_TO_MUTATE(RefNode)
  .DISPATCH_TO_MUTATE(Value)
  .DISPATCH_TO_MUTATE(Parameter)
  .DISPATCH_TO_MUTATE(CastNode)
  .DISPATCH_TO_MUTATE(InvertNode)
  .DISPATCH_TO_MUTATE(NegativeNode)
  .DISPATCH_TO_MUTATE(NotNode)
  .DISPATCH_TO_MUTATE(ReturnNode)
  .DISPATCH_TO_MUTATE(LoadValueNode)
  .DISPATCH_TO_MUTATE(UnaryOperation)
  .DISPATCH_TO_MUTATE(AddNode)
  .DISPATCH_TO_MUTATE(SubNode)
  .DISPATCH_TO_MUTATE(MulNode)
  .DISPATCH_TO_MUTATE(DivNode)
  .DISPATCH_TO_MUTATE(BitwiseNode)
  .DISPATCH_TO_MUTATE(CompareNode)
  .DISPATCH_TO_MUTATE(ContainsNode)
  .DISPATCH_TO_MUTATE(IsNode)
  .DISPATCH_TO_MUTATE(StoreNode)
  .DISPATCH_TO_MUTATE(UpdateNode)
  .DISPATCH_TO_MUTATE(BinaryOperation)
  .DISPATCH_TO_MUTATE(LoadFieldNode)
  .DISPATCH_TO_MUTATE(BuildNode)
  .DISPATCH_TO_MUTATE(CallNode)
  .DISPATCH_TO_MUTATE(FormatNode)
  .DISPATCH_TO_MUTATE(NaryOperation)
  .DISPATCH_TO_MUTATE(NaryWithFlagNode)
  .DISPATCH_TO_MUTATE(FunctionNode)
  .DISPATCH_TO_MUTATE(IfNode)
  .DISPATCH_TO_MUTATE(WhileNode)
  .DISPATCH_TO_MUTATE(SubscrNode)
  .DISPATCH_TO_MUTATE(AttrNode)
  .DISPATCH_TO_MUTATE(PairNode);
}  // namespace ir
}  // namespace graph
}  // namespace jit
}  // namespace mindspore
