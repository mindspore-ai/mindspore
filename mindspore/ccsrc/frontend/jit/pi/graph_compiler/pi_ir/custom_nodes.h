/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_CUSTOM_NODES_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_CUSTOM_NODES_H_

#include <memory>
#include <string>
#include "frontend/jit/pi/python_adapter/pydef.h"
#include "frontend/jit/pi/utils/opcode_declare.h"
#include "frontend/jit/pi/graph_compiler/pi_ir/operation.h"

namespace mindspore {
namespace pijit {
namespace ir {
/// \brief RefNode is the class which represent that this node is defined elsewhere and is only used here.
class RefNode : public Node {
 public:
  /**
   * \brief The constructor of reference node.
   *
   * \return The instance of reference node.
   */
  explicit RefNode(const NodePtr &node) : real_node_(node) {}

  // \brief Destructor.
  ~RefNode() override = default;
  JIT_DECLARE_PARENT(RefNode, Node);

  /**
   * \brief Get the node this reference node represents.
   *
   * \return The node this reference node represents.
   */
  const NodePtr &GetRealNode() const { return real_node_; }

  /**
   * \brief Set the real node of the ref node.
   *
   * \param[in] node the real object.
   */
  void SetRealNode(const NodePtr &node) { real_node_ = node; }

  /**
   * \brief Get the description of this node.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  /// \brief The node this reference node represents
  NodePtr real_node_;
};

using RefNodePtr = std::shared_ptr<RefNode>;

/// \brief PlaceHolder is the class which represent a symbol, and don't care about object specific information.
class PlaceHolder : public Node {
 public:
  /**
   * \brief The constructor of PlaceHolder node.
   *
   * \return The instance of PlaceHolder node.
   */
  explicit PlaceHolder(const std::string &tag) : tag_(tag) {}

  // \brief Destructor.
  ~PlaceHolder() override = default;
  JIT_DECLARE_PARENT(PlaceHolder, Node);

  /**
   * \brief Get the tag of PlaceHolder node.
   *
   * \return The tag of PlaceHolder node.
   */
  const std::string &GetTag() const { return tag_; }

  /**
   * \brief Set the id of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetNodeId(size_t *id) override {}

  /**
   * \brief Get the description of this node.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  /// \brief The mark of PlaceHolder used to explain the special meaning
  const std::string tag_;
};

using PlaceHolderPtr = std::shared_ptr<PlaceHolder>;

/// \brief SubscrNode is the class which represent a subscript access of object.
class SubscrNode : public Node {
 public:
  /**
   * \brief The constructor of subscript node.
   *
   * \param[in] base the object being accessed.
   * \param[in] subscr the subscript.
   *
   * \return The instance of subscript node.
   */
  SubscrNode(const NodePtr &base, const NodePtr &subscr) : base_(base), subscr_(subscr) {}

  // \brief Destructor.
  ~SubscrNode() override = default;
  JIT_DECLARE_PARENT(SubscrNode, Node);

  /**
   * \brief Get the object being accessed.
   *
   * \return The object being accessed.
   */
  const NodePtr &GetObject() const { return base_; }

  /**
   * \brief Set the the object being accessed.
   *
   * \param[in] obj the object.
   */
  void SetObject(const NodePtr &obj) { base_ = obj; }

  /**
   * \brief Get the subscr want to accessed.
   *
   * \return The subscr want to accessed.
   */
  const NodePtr &GetSubscr() const { return subscr_; }

  /**
   * \brief Set the subscr want to accessed.
   *
   * \param[in] subscr the element.
   */
  void SetSubscr(const NodePtr &subscr) { subscr_ = subscr; }

  /**
   * \brief Set the id of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetNodeId(size_t *id) override;

  /**
   * \brief Set the offset of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetOffset(size_t *offset) override;

  /**
   * \brief Get the description of this node.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  NodePtr base_;
  NodePtr subscr_;
};

using SubscrNodePtr = std::shared_ptr<SubscrNode>;

/// \brief SubscrNode is the class which represent a attr or method of the object.
class AttrNode : public Node {
 public:
  /**
   * \brief The constructor of attribute node.
   *
   * \param[in] base the object being accessed.
   * \param[in] attr the attribute name.
   *
   * \return The instance of attribute node.
   */
  AttrNode(const NodePtr &base, const NodePtr &attr) : base_(base), attr_(attr) {}

  // \brief Destructor.
  ~AttrNode() override = default;
  JIT_DECLARE_PARENT(AttrNode, Node);

  /**
   * \brief Get the object being accessed.
   *
   * \return The object being accessed.
   */
  const NodePtr &GetObject() const { return base_; }

  /**
   * \brief Set the object being accessed.
   *
   * \param[in] obj the object.
   */
  void SetObject(const NodePtr &obj) { base_ = obj; }

  /**
   * \brief Get the attribute name of the object.
   *
   * \return The attribute name of the object.
   */
  const NodePtr &GetAttr() const { return attr_; }

  /**
   * \brief Set the attribute name of the object.
   *
   * \param[in] attr the attribute name.
   */
  void SetAttr(const NodePtr &attr) { attr_ = attr; }

  /**
   * \brief Set the id of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetNodeId(size_t *id) override;

  /**
   * \brief Set the offset of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetOffset(size_t *offset) override;

  /**
   * \brief Get the description of this node.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  NodePtr base_;
  NodePtr attr_;
};

using AttrNodePtr = std::shared_ptr<AttrNode>;

/// \brief PairNode is the class which represent the object subscript access.
class PairNode : public Node {
 public:
  /**
   * \brief The constructor of pair node.
   *
   * \param[in] first the first element of the pair.
   * \param[in] second the second element of the pair.
   *
   * \return The instance of pair node.
   */
  PairNode(const NodePtr &first, const NodePtr &second) : first_(first), second_(second) {}

  // \brief Destructor.
  ~PairNode() override = default;
  JIT_DECLARE_PARENT(PairNode, Node);

  /**
   * \brief Get the first element of the pair.
   *
   * \return The first element of the pair.
   */
  const NodePtr &GetFirst() const { return first_; }

  /**
   * \brief Set the first element of the pair.
   *
   * \param[in] arg the element.
   */
  void SetFirst(const NodePtr &arg) { first_ = arg; }

  /**
   * \brief Get the second element of the pair.
   *
   * \return The second element of the pair.
   */
  const NodePtr &GetSecond() const { return second_; }

  /**
   * \brief Set the second element of the pair.
   *
   * \param[in] arg the element.
   */
  void SetSecond(const NodePtr &arg) { second_ = arg; }

  /**
   * \brief Get the description of this node.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  NodePtr first_;
  NodePtr second_;
};

using PairNodePtr = std::shared_ptr<PairNode>;

/// \brief InstrArg is the base class which represent the arg of instruction.
class InstrArg {
 public:
  /**
   * \brief The constructor of InstrArg.
   *
   * \param[in] arg the value of arg.
   *
   * \return The instance of InstrArg.
   */
  explicit InstrArg(int arg) : instr_arg_(arg) {}
  // \brief Destructor.
  virtual ~InstrArg() = default;

  /**
   * \brief Get the value of the instruction arg.
   *
   * \return The value of the instruction arg.
   */
  int GetInstrArg() const { return instr_arg_; }

  /**
   * \brief Set the value of the instruction arg.
   *
   * \param[in] arg the value of the instruction arg.
   */
  void SetInstrArg(int arg) { instr_arg_ = arg; }

 private:
  /// \brief The value of the instruction arg.
  int instr_arg_;
};

/// \brief NegativeNode is the class which represent operation that take negative value.
class NegativeNode : public UnaryOperation {
 public:
  /**
   * \brief The constructor of negative node.
   *
   * \param[in] opnd the value of negative node.
   *
   * \return The instance of negative node.
   */
  explicit NegativeNode(const NodePtr &opnd) : UnaryOperation(UNARY_NEGATIVE, opnd) {}

  // \brief Destructor.
  ~NegativeNode() override = default;
  JIT_DECLARE_PARENT(NegativeNode, UnaryOperation);
};

using NegativeNodePtr = std::shared_ptr<NegativeNode>;

/// \brief NotNode is the class which represent the operation that take logical negation.
class NotNode : public UnaryOperation {
 public:
  /**
   * \brief The constructor of logical not node.
   *
   * \param[in] opnd the value of logical not node.
   *
   * \return The instance of logical not node.
   */
  explicit NotNode(const NodePtr &opnd) : UnaryOperation(UNARY_NOT, opnd) {}

  // \brief Destructor.
  ~NotNode() override = default;
  JIT_DECLARE_PARENT(NotNode, UnaryOperation);
};

using NotNodePtr = std::shared_ptr<NotNode>;

/// \brief InvertNode is the class which represent the operation that take bitwise inversion.
class InvertNode : public UnaryOperation {
 public:
  /**
   * \brief The constructor of invert node.
   *
   * \param[in] opnd the value of invert node.
   *
   * \return The instance of invert node.
   */
  explicit InvertNode(const NodePtr &opnd) : UnaryOperation(UNARY_INVERT, opnd) {}

  // \brief Destructor.
  ~InvertNode() override = default;
  JIT_DECLARE_PARENT(InvertNode, UnaryOperation);
};

using InvertNodePtr = std::shared_ptr<InvertNode>;

/// \brief ReturnNode is the class which represent the return of function.
class ReturnNode : public UnaryOperation {
 public:
  /**
   * \brief The constructor of return node.
   *
   * \param[in] res the value of return node.
   *
   * \return The instance of return node.
   */
  explicit ReturnNode(const NodePtr &res) : UnaryOperation(RETURN_VALUE, res) {}

  // \brief Destructor.
  ~ReturnNode() override = default;
  JIT_DECLARE_PARENT(ReturnNode, UnaryOperation);

  /**
   * \brief Get the value of return node.
   *
   * \return the return value.
   */
  const NodePtr &GetReturn() const { return GetArg(); }
};

using ReturnNodePtr = std::shared_ptr<ReturnNode>;

/// \brief CastNode is the class which represent convert one type to another.
class CastNode : public UnaryOperation {
 public:
  /**
   * \brief The constructor of cast node.
   *
   * \param[in] opnd the value of cast node.
   *
   * \return The instance of cast node.
   */
  explicit CastNode(const NodePtr &opnd) : UnaryOperation(LIST_TO_TUPLE, opnd) {}

  // \brief Destructor.
  ~CastNode() override = default;
  JIT_DECLARE_PARENT(CastNode, UnaryOperation);
};

using CastNodePtr = std::shared_ptr<CastNode>;

/// \brief DeleteNode is the class which represent delete a object.
class DeleteNode : public UnaryOperation {
 public:
  /**
   * \brief The constructor of delete node.
   *
   * \param[in] opnd the object will be deleted.
   *
   * \return The instance of cast node.
   */
  explicit DeleteNode(OpCode op, const NodePtr &opnd) : UnaryOperation(op, opnd) {}

  // \brief Destructor.
  ~DeleteNode() override = default;
  JIT_DECLARE_PARENT(DeleteNode, UnaryOperation);
};

using DeleteNodePtr = std::shared_ptr<DeleteNode>;

/// \brief GetNode is the class which represent get a property of an object with `Get_*`.
class GetNode : public UnaryOperation {
 public:
  /**
   * \brief The constructor of get node.
   *
   * \param[in] opnd the object.
   *
   * \return The instance of get node.
   */
  explicit GetNode(OpCode op, const NodePtr &opnd) : UnaryOperation(op, opnd) {}

  // \brief Destructor.
  ~GetNode() override = default;
  JIT_DECLARE_PARENT(GetNode, UnaryOperation);
};

using GetNodePtr = std::shared_ptr<GetNode>;

/// \brief LoadValueNode is the class which represent load a value to stack.
class LoadValueNode : public UnaryOperation {
 public:
  /**
   * \brief The constructor of load node.
   *
   * \param[in] value the value will be load.
   *
   * \return The instance of load node.
   */
  LoadValueNode(OpCode op, const NodePtr &value) : UnaryOperation(op, value) {}

  // \brief Destructor.
  ~LoadValueNode() override = default;
  JIT_DECLARE_PARENT(LoadValueNode, NaryOperation);
};

using LoadValueNodePtr = std::shared_ptr<LoadValueNode>;

/// \brief LoadFieldNode is the class which represent load a filed of class to stack.
class LoadFieldNode : public BinaryOperation {
 public:
  /**
   * \brief The constructor of load node.
   *
   * \param[in] cls_ins the instance of class.
   * \param[in] field the field will be load.
   *
   * \return The instance of load node.
   */
  LoadFieldNode(OpCode op, const NodePtr &cls_ins, const NodePtr &field) : BinaryOperation(op, cls_ins, field) {}

  // \brief Destructor.
  ~LoadFieldNode() override = default;
  JIT_DECLARE_PARENT(LoadFieldNode, BinaryOperation);
};

using LoadFieldNodePtr = std::shared_ptr<LoadFieldNode>;

/// \brief AddNode is the class which represent the addition of two operands.
class AddNode : public BinaryOperation {
 public:
  /**
   * \brief The constructor of add node.
   *
   * \param[in] left the first operand of add.
   * \param[in] right the second operand of add.
   * \param[in] is_inplace whether the sum store to the first operand.
   *
   * \return The instance of add node.
   */
  AddNode(OpCode op, const NodePtr &left, const NodePtr &right) : BinaryOperation(op, left, right) {}

  // \brief Destructor.
  ~AddNode() override = default;
  JIT_DECLARE_PARENT(AddNode, BinaryOperation);

  /**
   * \brief Judge whether the opcode of this node is INPLACE_ADD.
   *
   * \return The result of the judgment.
   */
  bool IsInplace() const { return INPLACE_ADD == GetOpCode(); }
};

using AddNodePtr = std::shared_ptr<AddNode>;

/// \brief SubNode is the class which represent the subtraction of two operands.
class SubNode : public BinaryOperation {
 public:
  /**
   * \brief The constructor of sub node.
   *
   * \param[in] left the first operand of sub.
   * \param[in] right the second operand of sub.
   * \param[in] is_inplace whether the difference store to the first operand.
   *
   * \return The instance of sub node.
   */
  SubNode(OpCode op, const NodePtr &left, const NodePtr &right) : BinaryOperation(op, left, right) {}

  // \brief Destructor.
  ~SubNode() override = default;
  JIT_DECLARE_PARENT(SubNode, BinaryOperation);

  /**
   * \brief Judge whether the opcode of this node is INPLACE_ADD.
   *
   * \return The result of the judgment.
   */
  bool IsInplace() const { return INPLACE_SUBTRACT == GetOpCode(); }
};

using SubNodePtr = std::shared_ptr<SubNode>;

/// \brief MulNode is the class which represent the multiplication of two operands.
class MulNode : public BinaryOperation {
 public:
  /**
   * \brief The constructor of mul node.
   *
   * \param[in] left the first operand of mul.
   * \param[in] right the second operand of mul.
   * \param[in] is_inplace whether the product store to the first operand.
   *
   * \return The instance of mul node.
   */
  MulNode(OpCode op, const NodePtr &left, const NodePtr &right) : BinaryOperation(op, left, right) {}

  // \brief Destructor.
  ~MulNode() override = default;
  JIT_DECLARE_PARENT(MulNode, BinaryOperation);

  /**
   * \brief Judge whether the opcode of this node is INPLACE_MULTIPLY.
   *
   * \return The result of the judgment.
   */
  bool IsInplace() const { return (INPLACE_MULTIPLY == GetOpCode()) || (INPLACE_MATRIX_MULTIPLY == GetOpCode()); }
};

using MulNodePtr = std::shared_ptr<MulNode>;

/// \brief DivNode is the class which represent the division of two operands.
class DivNode : public BinaryOperation {
 public:
  /**
   * \brief The constructor of div node.
   *
   * \param[in] left the first operand of div.
   * \param[in] right the second operand of div.
   * \param[in] is_inplace whether the quotient of division store to the first operand.
   *
   * \return The instance of div node.
   */
  DivNode(OpCode op, const NodePtr &left, const NodePtr &right) : BinaryOperation(op, left, right) {}

  // \brief Destructor.
  ~DivNode() override = default;
  JIT_DECLARE_PARENT(DivNode, BinaryOperation);

  /**
   * \brief Judge whether the opcode of this node is INPLACE_TRUE_DIVIDE.
   *
   * \return The result of the judgment.
   */
  bool IsInplace() const { return INPLACE_TRUE_DIVIDE == GetOpCode(); }
};

using DivNodePtr = std::shared_ptr<DivNode>;

/// \brief BitwiseNode is the class which represent the addition of two operands.
class BitwiseNode : public BinaryOperation {
 public:
  /**
   * \brief The constructor of add node.
   *
   * \param[in] left the first operand of add.
   * \param[in] right the second operand of add.
   * \param[in] is_inplace whether the sum store to the first operand.
   *
   * \return The instance of add node.
   */
  BitwiseNode(OpCode op, const NodePtr &left, const NodePtr &right) : BinaryOperation(op, left, right) {}

  // \brief Destructor.
  ~BitwiseNode() override = default;
  JIT_DECLARE_PARENT(BitwiseNode, BinaryOperation);

  /**
   * \brief Judge whether the opcode of this node is INPLACE_ADD.
   *
   * \return The result of the judgment.
   */
  bool IsInplace() const {
    return INPLACE_LSHIFT == GetOpCode() || INPLACE_RSHIFT == GetOpCode() || INPLACE_AND == GetOpCode() ||
           INPLACE_XOR == GetOpCode() || INPLACE_OR == GetOpCode();
  }
};

using BitwiseNodePtr = std::shared_ptr<BitwiseNode>;

/// \brief IsNode is the class which represent whether two operands are same or not.
class IsNode : public BinaryOperation, public InstrArg {
 public:
  /**
   * \brief The constructor of is node.
   *
   * \param[in] left the first operand of is node.
   * \param[in] right the second operand of is node.
   * \param[in] is_invert the flag whether invert the result.
   *
   * \return The instance of is node.
   */
  IsNode(const NodePtr &left, const NodePtr &right, int arg) : BinaryOperation(IS_OP, left, right), InstrArg(arg) {}

  // \brief Destructor.
  ~IsNode() override = default;
  JIT_DECLARE_PARENT(IsNode, BinaryOperation);

  /**
   * \brief Judge whether invert the result.
   *
   * \return The result of the judgment.
   */
  bool IsInvert() const { return GetInstrArg() != 0; }
};

using IsNodePtr = std::shared_ptr<IsNode>;

/// \brief ContainsNode is the class which represent whether one contains another or not.
class ContainsNode : public BinaryOperation, public InstrArg {
 public:
  /**
   * \brief The constructor of is node.
   *
   * \param[in] left the first operand of is node.
   * \param[in] right the second operand of is node.
   * \param[in] is_invert the flag whether invert the result.
   *
   * \return The instance of contains node.
   */
  ContainsNode(const NodePtr &left, const NodePtr &right, int arg)
      : BinaryOperation(CONTAINS_OP, left, right), InstrArg(arg) {}

  // \brief Destructor.
  ~ContainsNode() override = default;
  JIT_DECLARE_PARENT(ContainsNode, BinaryOperation);

  /**
   * \brief Judge whether invert the result.
   *
   * \return The result of the judgment.
   */
  bool IsInvert() const { return GetInstrArg() != 0; }
};

using ContainsNodePtr = std::shared_ptr<ContainsNode>;

/// \brief StoreNode is the class which represent whether two operands are same.
class StoreNode : public BinaryOperation {
 public:
  /**
   * \brief The constructor of store node.
   *
   * \param[in] left the first operand of store node.
   * \param[in] right the second operand of store node.
   *
   * \return The instance of store node.
   */
  StoreNode(OpCode op, const NodePtr &source, const NodePtr &target) : BinaryOperation(op, source, target) {}

  // \brief Destructor.
  ~StoreNode() override = default;
  JIT_DECLARE_PARENT(StoreNode, BinaryOperation);
};

using StoreNodePtr = std::shared_ptr<StoreNode>;

/// \brief JumpNode is the class which represent jump stmt.
class JumpNode : public BinaryOperation {
 public:
  /**
   * \brief The constructor of jump node.
   *
   * \param[in] condition the condition for judging whether to jump.
   * \param[in] target the jump target.
   *
   * \return The instance of jump node.
   */
  JumpNode(OpCode op, const NodePtr &condition, const NodePtr &target) : BinaryOperation(op, condition, target) {}

  // \brief Destructor.
  ~JumpNode() override = default;
  JIT_DECLARE_PARENT(JumpNode, BinaryOperation);

  /**
   * \brief Get the condition for judging whether to jump.
   *
   * \return The condition for judging whether to jump.
   */
  NodePtr GetCondition() const { return GetLeftArg(); }

  /**
   * \brief Get the target for jump.
   *
   * \return The target for jump.
   */
  NodePtr GetTarget() const { return GetRightArg(); }

  /**
   * \brief Set the target of jump.
   *
   * \param[in] target the jump target.
   */
  void SetTarget(const NodePtr &target) { SetRightArg(target); }

  /**
   * \brief Set the id of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetNodeId(size_t *id) override;

  /**
   * \brief Set the offset of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetOffset(size_t *offset) override;

  /**
   * \brief Get the description of this jump node.
   * \return The description.
   */
  std::string ToString() const override;
};

using JumpNodePtr = std::shared_ptr<JumpNode>;

class CompareNode : public BinaryOperation, public InstrArg {
 public:
  /**
   * \brief The constructor of compare node.
   *
   * \param[in] category the category of compare.
   * \param[in] left the first operand of compare node.
   * \param[in] right the second operand of compare node.
   *
   * \return The instance of compare node.
   */
  CompareNode(int arg, const NodePtr &left, const NodePtr &right)
      : BinaryOperation(COMPARE_OP, left, right), InstrArg(arg) {}

  // \brief Destructor.
  ~CompareNode() override = default;
  JIT_DECLARE_PARENT(CompareNode, BinaryOperation);

  /**
   * \brief Get the description of this node.
   * \return The description.
   */
  std::string ToString() const override;
};

using CompareNodePtr = std::shared_ptr<CompareNode>;

/// \brief CallNode is the class which represent merge several dicts/lists into one.
class UpdateNode : public BinaryOperation, public InstrArg {
 public:
  /**
   * \brief The constructor of build node.
   *
   * \param[in] opnds the operand of build node.
   *
   * \return The instance of build node.
   */
  UpdateNode(OpCode op, const NodePtr &left, const NodePtr &right, int arg)
      : BinaryOperation(op, left, right), InstrArg(arg) {}

  // \brief Destructor.
  ~UpdateNode() override = default;
  JIT_DECLARE_PARENT(UpdateNode, BinaryOperation);
};

using UpdateNodePtr = std::shared_ptr<UpdateNode>;

/// \brief FormatNode is the class which represent format an object as required.
class FormatNode : public NaryOperation, public InstrArg {
 public:
  /**
   * \brief The constructor of format node.
   *
   * \param[in] opnd the value of format node.
   * \param[in] fmt the format type.
   *
   * \return The instance of format node.
   */
  FormatNode(const NodePtrList &opnds, int fmt) : NaryOperation(FORMAT_VALUE, opnds), InstrArg(fmt) {}

  // \brief Destructor.
  ~FormatNode() override = default;
  JIT_DECLARE_PARENT(FormatNode, NaryOperation);

  /**
   * \brief Get the format type of format node.
   *
   * \return the format type.
   */
  int GetFormatType() const { return GetInstrArg(); }
};

using FormatNodePtr = std::shared_ptr<FormatNode>;

/// \brief BuildNode is the class which represent build a value.
class BuildNode : public NaryOperation {
 public:
  /**
   * \brief The constructor of build node.
   *
   * \param[in] opnds the operand of build node.
   *
   * \return The instance of build node.
   */
  BuildNode(OpCode op, const NodePtrList &opnds) : NaryOperation(op, opnds) {}

  // \brief Destructor.
  ~BuildNode() override = default;
  JIT_DECLARE_PARENT(BuildNode, NaryOperation);
};

using BuildNodePtr = std::shared_ptr<BuildNode>;

/// \brief CallNode is the class which represent call a function.
class CallNode : public NaryOperation {
 public:
  /**
   * \brief The constructor of build node.
   *
   * \param[in] opnds the operand of build node.
   *
   * \return The instance of build node.
   */
  CallNode(OpCode op, const NodePtrList &opnds) : NaryOperation(op, opnds) {}

  // \brief Destructor.
  ~CallNode() override = default;
  JIT_DECLARE_PARENT(CallNode, NaryOperation);
};

using CallNodePtr = std::shared_ptr<CallNode>;

/// \brief NaryWithFlagNode is the class which represent make function.
class NaryWithFlagNode : public NaryOperation, public InstrArg {
 public:
  /**
   * \brief The constructor of nary with flag node.
   *
   * \param[in] opnds the operand of nary with flag node.
   *
   * \return The instance of nary with flag node.
   */
  NaryWithFlagNode(OpCode op, const NodePtrList &opnds, int flag) : NaryOperation(op, opnds), InstrArg(flag) {}

  // \brief Destructor.
  ~NaryWithFlagNode() override = default;
  JIT_DECLARE_PARENT(NaryWithFlagNode, NaryOperation);

  /**
   * \brief Get the flag of make function node.
   *
   * \return the flag.
   */
  int GetFlag() const { return GetInstrArg(); }
};

using NaryWithFlagNodePtr = std::shared_ptr<NaryWithFlagNode>;
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_CUSTOM_NODES_H_
