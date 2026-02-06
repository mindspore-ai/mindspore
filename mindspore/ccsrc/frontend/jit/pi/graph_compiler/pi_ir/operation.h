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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_OPERATION_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_OPERATION_H_

#include <memory>
#include <string>
#include <vector>
#include "frontend/jit/pi/graph_compiler/pi_ir/node.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace pijit {
namespace ir {
using OpCode = int;

namespace py = pybind11;

/// \brief Operation is the parent class of all class which represent the operation of a instruction.
class Operation : public Node {
 public:
  /**
   * \brief The constructor of operation.
   *
   * \param[in] op the opcode of this operation.
   *
   * \return The instance of operation.
   */
  explicit Operation(OpCode op) : Operation(op, {}) {}

  /**
   * \brief The constructor of operation.
   *
   * \param[in] op the opcode of this operation.
   * \param[in] args the opands of this operation.
   *
   * \return The instance of operation.
   */
  explicit Operation(OpCode op, const NodePtrList &args) : opcode_(op), need_ext_instr_(false), args_(args) {}

  /// \brief Destructor.
  ~Operation() override = default;
  JIT_DECLARE_PARENT(Operation, Node);

  /**
   * \brief Judge whether this node is an operation(instruction).
   *
   * \return The result of the judgment.
   */
  bool IsOperation() const override { return true; }

  /**
   * \brief Set the id of this operation.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetNodeId(size_t *id) override;

  /**
   * \brief Set the offset of this operation.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetOffset(size_t *offset) override;

  /**
   * \brief Get opcode of this operation.
   *
   * \return the opcode of this operation.
   */
  OpCode GetOpCode() const { return opcode_; }

  /**
   * \brief Judge whether need to insert a EXTENDED_ARG instruction before this operation.
   *
   * \return The result of the judgment.
   */
  bool NeedExtInstr() const override { return need_ext_instr_; }

  /**
   * \brief Mark whether this operation need to insert a EXTENDED_ARG instruction.
   *
   * \param[in] need the result.
   */
  void SetNeedExtInstr(bool need) override { need_ext_instr_ = need; }

  /**
   * \brief Get the count of args.
   *
   * \return the count of args.
   */
  size_t GetArgsCnt() const { return args_.size(); }

  /**
   * \brief Get the specified positional operand of this operation.
   *
   * \return The specified positional operand
   */
  const NodePtr &GetArg(size_t index = 0) const { return args_[index]; }

  /**
   * \brief Set the operand of this operation.
   *
   * \param[in] index the position of the arg.
   * \param[in] arg the value of the arg.
   */
  void SetArg(size_t index, const NodePtr &arg) { args_[index] = arg; }

  /**
   * \brief Get the operands of this operation.
   *
   * \return the operands of this operation.
   */
  const NodePtrList &GetArgs() const { return args_; }

  /**
   * \brief Get the operands of this operation.
   *
   * \return the operands of this operation.
   */
  NodePtrList &GetArgs() { return args_; }

  /**
   * \brief Set the operands of this operation.
   *
   * \param[in] args the new value of the operation.
   */
  void SetArgs(const NodePtrList &args) { args_ = args; }

 private:
  /// \brief The opcode of this operation.
  OpCode opcode_;
  /// \brief The EXTENDED_ARG instruction is required.
  bool need_ext_instr_;
  /// \brief The operands of this operation.
  NodePtrList args_;
};

using OperationPtr = std::shared_ptr<Operation>;

/// \brief UnaryOperation is is the parent class of all class which represent the operation of instruction with one
/// operand.
class UnaryOperation : public Operation {
 public:
  /**
   * \brief The constructor of unary operation.
   *
   * \param[in] op the opcode of this unary operation.
   * \param[in] arg the operand of this unary operation.
   *
   * \return The instance of unary operation.
   */
  UnaryOperation(OpCode op, const NodePtr &arg) : Operation(op, {arg}) {}

  /// \brief Destructor.
  ~UnaryOperation() override = default;
  JIT_DECLARE_PARENT(UnaryOperation, Operation);

  /**
   * \brief Set the operand of this unary operation.
   *
   * \param[in] arg the value of the arg.
   */
  void SetArg(const NodePtr &arg) { Operation::SetArg(0, arg); }

  /**
   * \brief Get the description of this unary operation.
   * \return The description.
   */
  std::string ToString() const override;
};

using UnaryOperationPtr = std::shared_ptr<UnaryOperation>;

/// \brief BinaryOperation is is the parent class of all class which represent the operation of instruction with two
/// operand.
class BinaryOperation : public Operation {
 public:
  /**
   * \brief The constructor of binary operation node.
   *
   * \param[in] op the opcode of this binary operation node.
   * \param[in] left the first operand of this binary operation node.
   * \param[in] right the second operand of this binary operation node.
   *
   * \return The instance of binary operation node.
   */
  BinaryOperation(OpCode op, const NodePtr &left, const NodePtr &right) : Operation(op, {left, right}) {}

  /// \brief Destructor.
  ~BinaryOperation() override = default;
  JIT_DECLARE_PARENT(BinaryOperation, Operation);

  /**
   * \brief Get the operand of this binary operation.
   *
   * \return the operand of this binary operation.
   */
  const NodePtr &GetLeftArg() const { return GetArg(0); }

  /**
   * \brief Set the first operand of this binary operation.
   */
  void SetLeftArg(const NodePtr &arg) { SetArg(0, arg); }

  /**
   * \brief Get the operand of this binary operation.
   *
   * \return the operand of this binary operation.
   */
  const NodePtr &GetRightArg() const { return GetArg(1); }

  /**
   * \brief Set the second operand of this binary operation.
   */
  void SetRightArg(const NodePtr &arg) { SetArg(1, arg); }

  /**
   * \brief Get the description of this binary operation.
   * \return The description.
   */
  std::string ToString() const override;
};

using BinaryOperationPtr = std::shared_ptr<BinaryOperation>;

/// \brief NaryOperation is is the parent class of all class which represent the operation of instruction with
/// indeterminate number of operands.
class NaryOperation : public Operation {
 public:
  /**
   * \brief The constructor of nary operation node.
   *
   * \param[in] op the opcode of this nary operation node.
   *
   * \return The instance of nary operation node.
   */
  explicit NaryOperation(OpCode op) : NaryOperation(op, {}) {}

  /**
   * \brief The constructor of nary operation node.
   *
   * \param[in] op the opcode of this nary operation node.
   * \param[in] args the operands of this nary operation node.
   *
   * \return The instance of nary operation node.
   */
  NaryOperation(OpCode op, const NodePtrList &args) : Operation(op, args) {}

  /// \brief Destructor.
  ~NaryOperation() override = default;
  JIT_DECLARE_PARENT(NaryOperation, Operation);

  /**
   * \brief Get the description of this nary operation.
   * \return The description.
   */
  std::string ToString() const override;
};

using NaryOperationPtr = std::shared_ptr<NaryOperation>;

std::string GetOpName(OpCode op);
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_OPERATION_H_
