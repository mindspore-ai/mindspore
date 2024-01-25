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
#ifndef MINDSPORE_PI_JIT_IR_VISITOR_H_
#define MINDSPORE_PI_JIT_IR_VISITOR_H_

#include "pipeline/jit/pi/graph_compiler/pi_ir/ctrl_flow.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/custom_nodes.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/functor.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/value.h"

namespace mindspore {
namespace pijit {
namespace ir {
class IRVisitor {
 public:
  /*
   * \brief recursively visit an IR node
   */
  virtual void Visit(const NodePtr &node) {
    static const FVisit &f = vtable();
    if (node != nullptr) {
      f(node, this);
    }
  }

  /// \brief destructor
  virtual ~IRVisitor() {}

  /*! \brief functor type of visitor */
  using FVisit = NodeFunctor<void(const NodePtr &, IRVisitor *)>;
  /*! \return internal vtable */
  static FVisit &vtable();

  // overloadable visit function.
  virtual void Visit_(const PlaceHolderPtr &node);
  virtual void Visit_(const RefNodePtr &node);
  virtual void Visit_(const ValuePtr &node);
  virtual void Visit_(const ParameterPtr &node);
  virtual void Visit_(const CastNodePtr &node);
  virtual void Visit_(const DeleteNodePtr &node);
  virtual void Visit_(const GetNodePtr &node);
  virtual void Visit_(const InvertNodePtr &node);
  virtual void Visit_(const NegativeNodePtr &node);
  virtual void Visit_(const NotNodePtr &node);
  virtual void Visit_(const ReturnNodePtr &node);
  virtual void Visit_(const LoadValueNodePtr &node);
  virtual void Visit_(const UnaryOperationPtr &node);
  virtual void Visit_(const AddNodePtr &node);
  virtual void Visit_(const SubNodePtr &node);
  virtual void Visit_(const MulNodePtr &node);
  virtual void Visit_(const DivNodePtr &node);
  virtual void Visit_(const BitwiseNodePtr &node);
  virtual void Visit_(const CompareNodePtr &node);
  virtual void Visit_(const ContainsNodePtr &node);
  virtual void Visit_(const IsNodePtr &node);
  virtual void Visit_(const JumpNodePtr &node);
  virtual void Visit_(const StoreNodePtr &node);
  virtual void Visit_(const UpdateNodePtr &node);
  virtual void Visit_(const BinaryOperationPtr &node);
  virtual void Visit_(const LoadFieldNodePtr &node);
  virtual void Visit_(const BuildNodePtr &node);
  virtual void Visit_(const CallNodePtr &node);
  virtual void Visit_(const NaryWithFlagNodePtr &node);
  virtual void Visit_(const FormatNodePtr &node);
  virtual void Visit_(const NaryOperationPtr &node);
  virtual void Visit_(const FunctionNodePtr &node);
  virtual void Visit_(const IfNodePtr &node);
  virtual void Visit_(const WhileNodePtr &node);
  virtual void Visit_(const AttrNodePtr &node);
  virtual void Visit_(const PairNodePtr &node);
  virtual void Visit_(const SubscrNodePtr &node);
};

#define VISIT_NODE_LIST(LIST)                                                                  \
  do {                                                                                         \
    std::for_each(LIST.begin(), LIST.end(), [this](const ir::NodePtr &node) { Visit(node); }); \
  } while (0);

#define DISPATCH_TO_VISIT(OP) \
  set_dispatch<OP>([](const NodePtr &node, IRVisitor *v) { v->Visit_(std::static_pointer_cast<OP>(node)); })
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_IR_VISITOR_H_
