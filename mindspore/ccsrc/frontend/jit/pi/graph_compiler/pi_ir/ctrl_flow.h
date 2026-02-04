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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_CTRL_FLOW_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_CTRL_FLOW_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "frontend/jit/pi/graph_compiler/pi_ir/custom_nodes.h"

namespace mindspore {
namespace pijit {
namespace ir {

/// \brief Parameter is is the class which represent a parameter of function or method
class Parameter : public Node {
 public:
  /**
   * \brief The constructor of parameter node.
   *
   * \param[in] index the index of parameter.
   * \param[in] name the name of parameter.
   *
   * \return The instance of function node.
   */
  Parameter(size_t index, const std::string &name)
      : index_(index), name_(name), value_(nullptr), default_value_(nullptr), category_(0) {}

  // \brief Destructor.
  ~Parameter() override = default;
  JIT_DECLARE_PARENT(Parameter, Node);

  static constexpr int POSITIONAL = 0;
  static constexpr int VARIABLE = 1;
  static constexpr int KEYWORD_ONLY = 2;
  static constexpr int KEYWORD = 3;

  /**
   * \brief Set the id of this node.
   *
   * \note This method should not be actively called by the program writer, it should only be called by the method
   * Sort()
   */
  void SetNodeId(size_t *id) override;

  /**
   * \brief Get the index of parameter.
   *
   * \return the index of parameter.
   */
  size_t GetIndex() const { return index_; }

  /**
   * \brief Get the name of parameter.
   *
   * \return the name of parameter.
   */
  const std::string &GetName() const { return name_; }

  /**
   * \brief Set the name of parameter.
   *
   * \param[in] name the name of parameter.
   */
  void SetName(const std::string &name) { name_ = name; }

  /**
   * \brief Get the value of parameter.
   *
   * \return the value of parameter.
   */
  const NodePtr &GetValue() const { return value_; }

  /**
   * \brief Set the value of parameter.
   *
   * \param[in] value the value of parameter.
   */
  void SetValue(const NodePtr &value) { value_ = value; }

  /**
   * \brief Get the default value of parameter.
   *
   * \return the default value of parameter.
   */
  const NodePtr &GetDefaultValue() const { return default_value_; }

  /**
   * \brief Set the default value of parameter.
   *
   * \param[in] default_value the default value of parameter.
   */
  void SetDefaultValue(const NodePtr &default_value) { default_value_ = default_value; }

  /**
   * \brief Get the category of parameter.
   *
   * \return the category of parameter.
   */
  int GetCategory() const { return category_; }

  /**
   * \brief Set the category of parameter.
   *
   * \param[in] category the category of parameter.
   */
  void SetCategory(int category) { category_ = category; }

  /**
   * \brief Get the description of this parameter.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  /// \brief The index of parameter.
  size_t index_;
  /// \brief The name of parameter.
  std::string name_;
  /// \brief The value of parameter.
  NodePtr value_;
  /// \brief The default value of parameter.
  NodePtr default_value_;
  /// \brief The category of parameter, 0 : positional, 1 : varargs, 2 : kwonly, 3 : kw
  int category_;
};

using ParameterPtr = std::shared_ptr<Parameter>;

using ParameterPtrList = std::vector<ParameterPtr>;

/// \brief FunctionNode is is the class which represent a python function or method
class FunctionNode : public Node {
 public:
  /**
   * \brief The constructor of function node.
   *
   * \param[in] name the name of function.
   *
   * \return The instance of function node.
   */
  explicit FunctionNode(const std::string &name) : FunctionNode(name, {}) {}

  /**
   * \brief The constructor of function node.
   *
   * \param[in] name the name of function.
   * \param[in] nodes the body of function.
   * \param[in] use_global the global will be used in code generator.
   *
   * \return The instance of function node.
   */
  FunctionNode(const std::string &name, const NodePtrList &nodes, const NodePtr &use_global = nullptr)
      : name_(name), nodes_(nodes), use_global_(use_global) {}

  // \brief Destructor.
  ~FunctionNode() override = default;
  JIT_DECLARE_PARENT(FunctionNode, Node);

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
   * \brief Get the name of function.
   *
   * \return the name of function.
   */
  const std::string &GetName() const { return name_; }

  /**
   * \brief Get the count of positional args.
   *
   * \return the count of positional args.
   */
  int GetPosArgsCnt() const { return pos_args_cnt_; }

  /**
   * \brief Set the count of positional args.
   *
   * \param[in] cnt the count of positional args.
   */
  void SetPosArgsCnt(int cnt) { pos_args_cnt_ = cnt; }

  /**
   * \brief Get the count of keyword only args.
   *
   * \return the count of keyword only args.
   */
  int GetKwOnlyArgsCnt() const { return kw_only_args_cnt_; }

  /**
   * \brief Set the count of keyword only args.
   *
   * \param[in] cnt the count of keyword only args.
   */
  void SetKwOnlyArgsCnt(int cnt) { kw_only_args_cnt_ = cnt; }

  /**
   * \brief Get the flags of function.
   *
   * \return The flags of function.
   */
  int GetFlags() const { return flags_; }

  /**
   * \brief Set the flags of function.
   *
   * \param[in] flags the flags of function.
   */
  void SetFlags(uint32_t flags) { flags_ = flags; }

  /**
   * \brief Judgment whether has var args.
   *
   * \return The result of the judgment.
   */
  bool HasVarArg() const;

  /**
   * \brief Set whether has var args.
   *
   * \param[in] has_var_arg the result of whether has var args.
   */
  void SetHasVarArg(bool has_var_arg);

  /**
   * \brief Judgment whether has kw args.
   *
   * \return The result of the judgment.
   */
  bool HasKwArg() const;

  /**
   * \brief Set whether has kw args.
   *
   * \param[in] has_kw_arg the result of whether has kw args.
   */
  void SetHasKwArg(bool has_kw_arg);

  /**
   * \brief Judgment whether has the attr whose name is key.
   *
   * \param[in] key the name of the attr.
   *
   * \return The result of the judgment.
   */
  bool HasAttr(const std::string &key) const { return attrs_.find(key) != attrs_.end(); }

  /**
   * \brief Get the value of the attr whose name is key.
   *
   * \param[in] key the name of the attr.
   *
   * \return The value of the attr.
   */
  bool GetAttr(const std::string &key) const { return HasAttr(key) && attrs_.at(key); }

  /**
   * \brief Set the attr whose name is key.
   *
   * \param[in] key the name of the attr.
   * \param[in] value the value of the attr.
   */
  void SetAttr(const std::string &key, bool value) { attrs_[key] = value; }

  /**
   * \brief Judgment whether need generate parameters.
   *
   * \return Whether need generate parameters.
   */
  bool NeedGenParameters() const { return !without_params_gen_; }

  /**
   * \brief Mark no need generate parameters.
   */
  void MarkNoNeedGenParameters() { without_params_gen_ = true; }

  /**
   * \brief Get the parameters of function.
   *
   * \return the parameters of function.
   */
  const NodePtrList &GetParameters() const { return parameters_; }

  /**
   * \brief Get the parameters of function.
   *
   * \return the parameters of function.
   */
  NodePtrList &GetParameters() { return parameters_; }

  /**
   * \brief Get the specified positional parameter of function.
   *
   * \return the specified positional parameter of function.
   */
  const NodePtr &GetParameter(size_t index) const { return parameters_[index]; }

  /**
   * \brief Add the new input to function node.
   *
   * \param[in] parameter the new parameter of function.
   */
  void AddParameter(const ParameterPtr &parameter) { parameters_.push_back(parameter); }

  /**
   * \brief Set the specified positional parameters of function.
   *
   * \param[in] input the new parameter of function.
   */
  void SetParameter(size_t index, const ParameterPtr &parameter) { parameters_[index] = parameter; }

  /**
   * \brief Set the parameters of function.
   *
   * \param[in] parameters the new parameters of function.
   */
  void SetParameters(const NodePtrList &parameters) { parameters_ = parameters; }

  /**
   * \brief Get the nodes of function.
   *
   * \return the nodes of function.
   */
  const NodePtrList &GetNodes() const { return nodes_; }

  /**
   * \brief Get the nodes of function.
   *
   * \return the nodes of function.
   */
  NodePtrList &GetNodes() { return nodes_; }

  /**
   * \brief Add the new node to function node.
   *
   * \param[in] node the new node of function.
   *
   * \note The node after the return will be ignored.
   */
  void AddNode(const NodePtr &node);

  /**
   * \brief Get the global will be used in code generator.
   *
   * \return The global will be used in code generator.
   */
  const NodePtr &GetUseGlobal() const { return use_global_; }

  /**
   * \brief Get the global will be used in code generator.
   *
   * \param[in] use_global the global will be used in code generator.
   */
  void SetUseGlobal(const NodePtr &use_global) { use_global_ = use_global; }

  /**
   * \brief Get the file name of the function.
   *
   * \return The file name of the function.
   */
  const std::string &GetFileName() const { return file_names_[0]; }

  /**
   * \brief Get the file names of the function, maybe include inline functions.
   *
   * \return The the file names of the function.
   */
  const std::vector<std::string> &GetFileNames() const { return file_names_; }

  /**
   * \brief Add file name to file names of function.
   *
   * \param[in] name the file name of sub function.
   */
  void AddFileName(const std::string &name) { file_names_.push_back(name); }

  /**
   * \brief Get the number of the first line.
   *
   * \return the number of the first line.
   */
  int GetFirstLineNo() const { return first_line_no_; }

  /**
   * \brief Set the number of the first line.
   *
   * \param[in] line_no the number of the first line.
   */
  void SetFirstLineNo(int line_no) { first_line_no_ = line_no; }

  /**
   * \brief Get the stack size of function.
   *
   * \return The stack size of function.
   */
  int GetStackSize() const { return stack_size_; }

  /**
   * \brief Set the stack size of function.
   *
   * \param[in] size the stack size of function.
   */
  void SetStackSize(int size) { stack_size_ = size; }

  /**
   * \brief Get the description of this function.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  /// \brief The name of function.
  const std::string name_;
  /// \brief whether the node represents a method.
  bool is_method_{false};
  /// \brief The count of positional args.
  int pos_args_cnt_{0};
  /// \brief The count of keyword only args.
  int kw_only_args_cnt_{0};
  /// \brief An integer encoding a number of flags for the function.
  uint32_t flags_{0};
  /// \brief the attrs of function.
  std::map<std::string, bool> attrs_;
  /// \brief the flag whether generate the parameters of function.
  bool without_params_gen_{false};
  /// \brief The parameters of function
  NodePtrList parameters_;
  /// \brief The body of function
  NodePtrList nodes_;
  /// \brief The global will be used in code generator
  NodePtr use_global_;
  /// \brief The name of the file where the function resides.
  std::vector<std::string> file_names_;
  /// \brief The number of the first line.
  int first_line_no_{0};
  /// \brief The size of stack.
  int stack_size_{0};
};

using FunctionNodePtr = std::shared_ptr<FunctionNode>;

/// \brief IfNode is is the class which represent a if statement
class IfNode : public Node {
 public:
  /**
   * \brief The constructor of if node.
   *
   * \param[in] condition the condition of if node.
   *
   * \return The instance of if node.
   */
  explicit IfNode(const NodePtr &condition) : condition_jump_(condition) {}

  // \brief Destructor.
  ~IfNode() override = default;
  JIT_DECLARE_PARENT(IfNode, Node);

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
   * \brief Get the condition of if node.
   *
   * \return the condition of if node.
   */
  const NodePtr &GetCondition() const { return condition_jump_; }

  /**
   * \brief Set the condition of if node.
   *
   * \param[in] condition the condition of if node.
   */
  void SetCondition(const NodePtr &condition) { condition_jump_ = condition; }

  /**
   * \brief Get the then body of if node.
   *
   * \return the nodes of then body of if node.
   */
  const NodePtrList &GetThen() const { return then_; }

  /**
   * \brief Get the then body of if node.
   *
   * \return the nodes of then body of if node.
   */
  NodePtrList &GetThen() { return then_; }

  /**
   * \brief Add the new node to then body of if node.
   *
   * \param[in] node the new node.
   *
   * \note The node after the return will be ignored.
   */
  void AddThen(const NodePtr &node);

  /**
   * \brief Get the else body of if node.
   *
   * \return the nodes of else body.
   */
  const NodePtrList &GetElse() const { return else_; }

  /**
   * \brief Get the else body of if node.
   *
   * \return the nodes of else body.
   */
  NodePtrList &GetElse() { return else_; }

  /**
   * \brief Add the new node to else body of if node.
   *
   * \param[in] node the new node.
   *
   * \note The node after the return will be ignored.
   */
  void AddElse(const NodePtr &node);

  /**
   * \brief Get the description of this If node.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  /// \brief The condition of if, it must be a jump.
  NodePtr condition_jump_;
  /// \brief The body of if will be executed when no need to jump, maybe empty
  NodePtrList then_;
  /// \brief The body of if will be executed when need to jump, maybe empty
  NodePtrList else_;
};

using IfNodePtr = std::shared_ptr<IfNode>;

/// \brief IfNode is is the class which represent a if statement
class WhileNode : public Node {
 public:
  /**
   * \brief The constructor of while node.
   *
   * \param[in] condition the condition of function.
   *
   * \return The instance of while node.
   */
  explicit WhileNode(const NodePtr &condition) : condition_jump_(condition) {}

  // \brief Destructor.
  ~WhileNode() override = default;
  JIT_DECLARE_PARENT(WhileNode, Node);

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
   * \brief Get the condition of while node.
   *
   * \return the condition of while node.
   */
  const NodePtr &GetCondition() const { return condition_jump_; }

  /**
   * \brief Set the condition of while node.
   *
   * \param[in] condition the condition of while node.
   */
  void SetCondition(const NodePtr &condition) { condition_jump_ = condition; }

  /**
   * \brief Get the body of while node.
   *
   * \return the body nodes of while node.
   */
  const NodePtrList &GetBody() const { return body_; }

  /**
   * \brief Get the body of while node.
   *
   * \return the body nodes of while node.
   */
  NodePtrList &GetBody() { return body_; }

  /**
   * \brief Add the new node to body of while node.
   *
   * \param[in] node the new node.
   */
  void AddBody(const NodePtr &node) { body_.push_back(node); }

  /**
   * \brief Set the new nodes as body of while node.
   *
   * \param[in] nodes the new nodes of then body.
   */
  void SetBody(const NodePtrList &nodes) { body_ = nodes; }

  /**
   * \brief Get the description of this While node.
   * \return The description.
   */
  std::string ToString() const override;

 private:
  /// \brief The condition of while, it must be a jump.
  NodePtr condition_jump_;
  /// \brief The body executed in a loop.
  NodePtrList body_;
};

using WhileNodePtr = std::shared_ptr<WhileNode>;
}  // namespace ir
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_CTRL_FLOW_H_
