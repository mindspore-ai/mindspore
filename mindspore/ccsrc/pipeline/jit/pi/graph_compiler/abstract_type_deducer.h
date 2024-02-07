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
#ifndef MINDSPORE_PI_JIT_ABSTRACT_TYPE_DEDUCER_H_
#define MINDSPORE_PI_JIT_ABSTRACT_TYPE_DEDUCER_H_

#include <map>
#include <memory>
#include <string>
#include "pipeline/jit/pi/graph_compiler/pi_ir/custom_nodes.h"
#include "pipeline/jit/pi/graph_compiler/pi_ir/ir_visitor.h"

namespace mindspore {
namespace pijit {
class AbstractTypeDeducer : public ir::IRVisitor {
 public:
  /// \brief The default constructor for Type.
  AbstractTypeDeducer() {}

  /// \brief Destructor.
  virtual ~AbstractTypeDeducer() = default;

  /// This method will deduce all nodes of function
  static void Deduce(const ir::FunctionNodePtr &func, const ir::py::tuple &args, const ir::py::dict &kwargs);

  // overloadable visit function.
  void Visit_(const ir::RefNodePtr &node) override;
  void Visit_(const ir::ValuePtr &node) override;
  void Visit_(const ir::CastNodePtr &node) override;
  void Visit_(const ir::DeleteNodePtr &node) override;
  void Visit_(const ir::GetNodePtr &node) override;
  void Visit_(const ir::InvertNodePtr &node) override;
  void Visit_(const ir::NegativeNodePtr &node) override;
  void Visit_(const ir::NotNodePtr &node) override;
  void Visit_(const ir::ReturnNodePtr &node) override;
  void Visit_(const ir::LoadValueNodePtr &node) override;
  void Visit_(const ir::UnaryOperationPtr &node) override;
  void Visit_(const ir::AddNodePtr &node) override;
  void Visit_(const ir::SubNodePtr &node) override;
  void Visit_(const ir::MulNodePtr &node) override;
  void Visit_(const ir::DivNodePtr &node) override;
  void Visit_(const ir::BitwiseNodePtr &node) override;
  void Visit_(const ir::CompareNodePtr &node) override;
  void Visit_(const ir::ContainsNodePtr &node) override;
  void Visit_(const ir::IsNodePtr &node) override;
  void Visit_(const ir::JumpNodePtr &node) override;
  void Visit_(const ir::StoreNodePtr &node) override;
  void Visit_(const ir::UpdateNodePtr &node) override;
  void Visit_(const ir::BinaryOperationPtr &node) override;
  void Visit_(const ir::LoadFieldNodePtr &node) override;
  void Visit_(const ir::BuildNodePtr &node) override;
  void Visit_(const ir::CallNodePtr &node) override;
  void Visit_(const ir::NaryWithFlagNodePtr &node) override;
  void Visit_(const ir::FormatNodePtr &node) override;
  void Visit_(const ir::NaryOperationPtr &node) override;
  void Visit_(const ir::FunctionNodePtr &node) override;
  void Visit_(const ir::IfNodePtr &node) override;
  void Visit_(const ir::WhileNodePtr &node) override;
  void Visit_(const ir::AttrNodePtr &node) override;
  void Visit_(const ir::PairNodePtr &node) override;
  void Visit_(const ir::SubscrNodePtr &node) override;

 private:
  std::map<std::string, ir::NodePtr> assigned_local_vars_;
  std::map<std::string, ir::NodePtr> assigned_global_vars_;
};

using AbstractTypeDeducerPtr = std::shared_ptr<AbstractTypeDeducer>;
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_ABSTRACT_TYPE_DEDUCER_H_
