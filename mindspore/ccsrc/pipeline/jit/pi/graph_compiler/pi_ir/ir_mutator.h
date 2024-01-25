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
#ifndef MINDSPORE_PI_JIT_IR_MUTATOR_H_
#define MINDSPORE_PI_JIT_IR_MUTATOR_H_

#include "pipeline/jit/pi/graph_compiler/pi_ir/ctrl_flow.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/custom_nodes.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/functor.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/value.h"

namespace mindspore {
namespace pijit {
namespace ir {
class IRMutator {
 public:
  /*
   * \brief recursively Mutate an IR node
   */
  virtual NodePtr Mutate(const NodePtr &node) {
    if (node == nullptr) {
      return nullptr;
    }
    static const FMutate &f = vtable();
    return f(node, this);
  }

  /// \brief destructor
  virtual ~IRMutator() {}

  /*! \brief functor type of visitor */
  using FMutate = NodeFunctor<NodePtr(const NodePtr &, IRMutator *)>;
  /*! \return internal vtable */
  static FMutate &vtable();

  // overloadable Mutate function.
  virtual NodePtr Mutate_(const RefNodePtr &node);
  virtual NodePtr Mutate_(const ParameterPtr &node);
  virtual NodePtr Mutate_(const FunctionNodePtr &node);
  virtual NodePtr Mutate_(const ValuePtr &node);
  virtual NodePtr Mutate_(const IfNodePtr &node);
  virtual NodePtr Mutate_(const WhileNodePtr &node);
  virtual NodePtr Mutate_(const UnaryOperationPtr &node);
  virtual NodePtr Mutate_(const BinaryOperationPtr &node);
  virtual NodePtr Mutate_(const NaryOperationPtr &node);
  virtual NodePtr Mutate_(const NegativeNodePtr &node);
  virtual NodePtr Mutate_(const NotNodePtr &node);
  virtual NodePtr Mutate_(const InvertNodePtr &node);
  virtual NodePtr Mutate_(const ReturnNodePtr &node);
  virtual NodePtr Mutate_(const LoadValueNodePtr &node);
  virtual NodePtr Mutate_(const CastNodePtr &node);
  virtual NodePtr Mutate_(const FormatNodePtr &node);
  virtual NodePtr Mutate_(const AddNodePtr &node);
  virtual NodePtr Mutate_(const SubNodePtr &node);
  virtual NodePtr Mutate_(const MulNodePtr &node);
  virtual NodePtr Mutate_(const DivNodePtr &node);
  virtual NodePtr Mutate_(const BitwiseNodePtr &node);
  virtual NodePtr Mutate_(const IsNodePtr &node);
  virtual NodePtr Mutate_(const ContainsNodePtr &node);
  virtual NodePtr Mutate_(const StoreNodePtr &node);
  virtual NodePtr Mutate_(const CompareNodePtr &node);
  virtual NodePtr Mutate_(const LoadFieldNodePtr &node);
  virtual NodePtr Mutate_(const BuildNodePtr &node);
  virtual NodePtr Mutate_(const CallNodePtr &node);
  virtual NodePtr Mutate_(const NaryWithFlagNodePtr &node);
  virtual NodePtr Mutate_(const UpdateNodePtr &node);
  virtual NodePtr Mutate_(const SubscrNodePtr &node);
  virtual NodePtr Mutate_(const AttrNodePtr &node);
  virtual NodePtr Mutate_(const PairNodePtr &node);
};

#define MUTATE_NODE_LIST(LIST)        \
  do {                                \
    for (ir::NodePtr & node : LIST) { \
      node = Mutate(node);            \
    }                                 \
  } while (0);

#define DISPATCH_TO_MUTATE(OP) \
  set_dispatch<OP>([](const NodePtr &node, IRMutator *m) { return m->Mutate_(std::static_pointer_cast<OP>(node)); })
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_IR_MUTATOR_H_
