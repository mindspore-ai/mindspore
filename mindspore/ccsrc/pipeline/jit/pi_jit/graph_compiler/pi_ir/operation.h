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
#ifndef MINDSPORE_JIT_GRAPH_OPERATION_H_
#define MINDSPORE_JIT_GRAPH_OPERATION_H_

#include <memory>
#include <string>
#include <vector>
#include "pipeline/jit/pi_jit/graph_compiler/pi_ir/node.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace jit {
namespace graph {
namespace ir {
using OpCode = int;

namespace py = pybind11;

static std::string GetOpName(OpCode op) {
  static const std::vector<std::string> op_names =
    py::cast<std::vector<std::string>>(py::module::import("opcode").attr("opname"));
  return op_names[op];
}

/// \brief Operation is the parent class of all class which represent the operation of a instruction.
class Operation : public Node {
 public:
  /**
   * \brief The constructor of operation node.
   *
   * \param[in] kind the kind of this operation node.
   * \param[in] op the opcode of this operation node.
   *
   * \return The instance of operation node.
   */
  explicit Operation(OpCode op) : opcode_(op), need_ext_instr_(false) {}

  /// \brief Destructor.
  ~Operation() override = default;
  JIT_DECLARE_PARENT(Operation, Node);

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
  virtual size_t GetArgsCnt() const = 0;

  /**
   * \brief Judge whether this node is an operation(instruction).
   *
   * \return The result of the judgment.
   */
  bool IsOperation() const override { return true; }

  /**
   * \brief Get the specified positional operand of this operation.
   *
   * \return the the specified positional operand
   */
  virtual const NodePtr &GetArg(size_t index) const = 0;

 private:
  /// \brief The opcode of this operation.
  OpCode opcode_;
  /// \brief The EXTENDED_ARG instruction is required.
  bool need_ext_instr_;
};

/// \brief UnaryOperation is is the parent class of all class which represent the operation of instruction with one
/// operand.
class UnaryOperation : public Operation {
 public:
  /**
   * \brief The constructor of unary operation node.
   *
   * \param[in] op the opcode of this unary operation node.
   * \param[in] arg the operand of this unary operation node.
   *
   * \return The instance of unary operation node.
   */
  UnaryOperation(OpCode op, const NodePtr &arg) : Operation(op), arg_(arg) {}

  /// \brief Destructor.
  ~UnaryOperation() override = default;
  JIT_DECLARE_PARENT(UnaryOperation, Operation);

  /**
   * \brief Set the id of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetNodeId(size_t *id) override {
    arg_->SetNodeId(id);
    Node::SetNodeId(id);
  }

  /**
   * \brief Set the offset of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetOffset(size_t *offset) override {
    arg_->SetOffset(offset);
    Node::SetOffset(offset);
  }

  /**
   * \brief Get the count of args.
   *
   * \return the count of args.
   */
  size_t GetArgsCnt() const override { return 1; }

  /**
   * \brief Get the specified positional operand of this operation.
   *
   * \return the the specified positional operand
   */
  const NodePtr &GetArg(size_t index) const override { return arg_; }

  /**
   * \brief Get the operand of this unary operation.
   *
   * \return the operand of this unary operation.
   */
  const NodePtr &GetArg() const { return arg_; }

  /**
   * \brief Set the operand of this unary operation.
   */
  void SetArg(const NodePtr &arg) { arg_ = arg; }

  /**
   * \brief Get the description of this unary operation.
   * \return The description.
   */
  std::string ToString() const override {
    return arg_->ToString() + "\n%" + std::to_string(GetNodeId()) + " = " + GetNodeName() + "(" +
           GetOpName(GetOpCode()) + ", %" + std::to_string(arg_->GetNodeId()) + ")\n";
  }

 private:
  /// \brief The operand of this unary operation.
  NodePtr arg_;
};

using UnaryOperationPtr = std::shared_ptr<UnaryOperation>;

/// \brief UnaryOperation is is the parent class of all class which represent the operation of instruction with two
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
  BinaryOperation(OpCode op, const NodePtr &left, const NodePtr &right) : Operation(op), left_(left), right_(right) {}

  /// \brief Destructor.
  ~BinaryOperation() override = default;
  JIT_DECLARE_PARENT(BinaryOperation, Operation);

  /**
   * \brief Set the id of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetNodeId(size_t *id) override {
    left_->SetNodeId(id);
    right_->SetNodeId(id);
    Node::SetNodeId(id);
  }

  /**
   * \brief Set the offset of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetOffset(size_t *offset) override {
    left_->SetOffset(offset);
    right_->SetOffset(offset);
    Node::SetOffset(offset);
  }

  /**
   * \brief Get the count of args.
   *
   * \return the count of args.
   */
  size_t GetArgsCnt() const override { return 2; }

  /**
   * \brief Get the specified positional operand of this operation.
   *
   * \return the the specified positional operand
   */
  const NodePtr &GetArg(size_t index = 0) const override {
    if (index == 0) {
      return left_;
    }
    return right_;
  }

  /**
   * \brief Get the operand of this unary operation.
   *
   * \return the operand of this unary operation.
   */
  const NodePtr &GetLeftArg() const { return left_; }

  /**
   * \brief Set the first operand of this binary operation.
   */
  void SetLeftArg(const NodePtr &arg) { left_ = arg; }

  /**
   * \brief Get the operand of this unary operation.
   *
   * \return the operand of this unary operation.
   */
  const NodePtr &GetRightArg() const { return right_; }

  /**
   * \brief Set the second operand of this binary operation.
   */
  void SetRightArg(const NodePtr &arg) { right_ = arg; }

  /**
   * \brief Get the description of this binary operation.
   * \return The description.
   */
  std::string ToString() const override {
    return left_->ToString() + "\n" + right_->ToString() + "\n%" + std::to_string(GetNodeId()) + " = " + GetNodeName() +
           "(" + GetOpName(GetOpCode()) + ", %" + std::to_string(left_->GetNodeId()) + ", %" +
           std::to_string(right_->GetNodeId()) + ")\n";
  }

 private:
  /// \brief The operand of this binary operation.
  NodePtr left_;
  /// \brief The operand of this binary operation.
  NodePtr right_;
};

using BinaryOperationPtr = std::shared_ptr<BinaryOperation>;

/// \brief UnaryOperation is is the parent class of all class which represent the operation of instruction with
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
  NaryOperation(OpCode op, const NodePtrList &args) : Operation(op), args_(args) {}

  /// \brief Destructor.
  ~NaryOperation() override = default;
  JIT_DECLARE_PARENT(NaryOperation, Operation);

  /**
   * \brief Set the id of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetNodeId(size_t *id) override {
    for (const auto &arg : args_) {
      arg->SetNodeId(id);
    }
    Node::SetNodeId(id);
  }

  /**
   * \brief Set the offset of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetOffset(size_t *offset) override {
    for (const auto &arg : args_) {
      arg->SetOffset(offset);
    }
    Node::SetOffset(offset);
  }

  /**
   * \brief Get the count of args.
   *
   * \return the count of args.
   */
  size_t GetArgsCnt() const override { return args_.size(); }

  /**
   * \brief Get the specified positional operand of this nary operation.
   *
   * \return the specified positional operand
   */
  const NodePtr &GetArg(size_t index) const override { return args_[index]; }

  /**
   * \brief Get the operands of this nary operation.
   *
   * \return the operands of this nary operation.
   */
  const NodePtrList &GetArgs() const { return args_; }

  /**
   * \brief Get the operands of this nary operation.
   *
   * \return the operands of this nary operation.
   */
  NodePtrList &GetArgs() { return args_; }

  /**
   * \brief Set the specified positional operand of this nary operation.
   */
  void SetArg(size_t index, const NodePtr &arg) { args_[index] = arg; }

  /**
   * \brief Set the operands of this nary operation.
   */
  void SetArgs(const NodePtrList &args) { args_ = args; }

  /**
   * \brief Get the description of this nary operation.
   * \return The description.
   */
  std::string ToString() const override {
    std::string str;
    for (const auto &arg : args_) {
      str += arg->ToString() + "\n";
    }
    str += "%" + std::to_string(GetNodeId()) + " = " + GetNodeName() + "(" + GetOpName(GetOpCode());
    for (const auto &arg : args_) {
      str += ", %" + std::to_string(arg->GetNodeId());
    }
    str += ")\n";
    return str;
  }

 private:
  /// \brief The operands of this nary operation.
  NodePtrList args_;
};

using NaryOperationPtr = std::shared_ptr<NaryOperation>;
}  // namespace ir
}  // namespace graph
}  // namespace jit
}  // namespace mindspore

#endif  // MINDSPORE_JIT_GRAPH_OPERATION_H_
