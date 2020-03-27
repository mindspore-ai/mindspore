/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_IR_ANF_H_
#define MINDSPORE_CCSRC_IR_ANF_H_

#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <utility>

#include "ir/base.h"
#include "debug/info.h"
#include "ir/scope.h"

// A MindSpore ANF IR defined here.
// with BNF followed:
// <ANode> ::= Scalar | Named | Tensor  | Var |
//             Prim   | MetaFuncGraph | FuncGraph | Type|
//             Shape  | Param
// <CNode> ::= (<ANode> ...)
// <AnfNode> ::= <CNode> | <ANode>
// ANode: Atomic  Node
// CNode: Complex Node
namespace mindspore {
namespace parallel {
class TensorLayout;
class OperatorInfo;
}  // namespace parallel
using OperatorInfoPtr = std::shared_ptr<parallel::OperatorInfo>;

namespace abstract {
class BaseShape;
class AbstractBase;
}  // namespace abstract
using BaseShapePtr = std::shared_ptr<abstract::BaseShape>;
using AbstractBasePtr = std::shared_ptr<abstract::AbstractBase>;

class ValueNode;
using ValueNodePtr = std::shared_ptr<ValueNode>;
class CNode;
using CNodePtr = std::shared_ptr<CNode>;

class FuncGraph;
using FuncGraphSet = OrderedSet<FuncGraphPtr>;
using FuncGraphPtrList = std::vector<FuncGraphPtr>;

class Primitive;
using PrimitivePtr = std::shared_ptr<Primitive>;

class BaseRef;

class Var;
using VarPtr = std::shared_ptr<Var>;

namespace device {
class KernelInfo;
}  // namespace device
using KernelInfoDevice = device::KernelInfo;
using KernelInfoDevicePtr = std::shared_ptr<KernelInfoDevice>;

class AnfVisitor;

// AnfNode is the basic class of the IR definition derived from Base.
// Only two types of nodes are derived: CNode and ANode.
// Methods:
// func_graph: return FuncGraph that this AnfNode belongs to.
// scope: return the scope namespace of this AnfNode. Set it using set_scope.
// abstract: return the cached inferred abstract value. It cantains type, shape
// value. Set New cache using set_abstract.
// intermediate_abstract: return the cached inferring abstract value.
// Type/Shape: return the related info of this AnfNode. When this AnfNode is an
// input of other CNodes, you can get the related info by this method.
// debug_info: return the information retrived from parser. Set it using set_debug_info.
// fullname_with_scope: return the detailed debug info.
class AnfNode : public Base {
 public:
  explicit AnfNode(const FuncGraphPtr &func_graph)
      : func_graph_(FuncGraphWeakPtr(func_graph)),
        abstract_(nullptr),
        intermediate_abstract_(nullptr),
        debug_info_(std::make_shared<NodeDebugInfo>()),
        fullname_with_scope_(""),
        hash_(std::hash<const AnfNode *>()),
        kernel_info_(nullptr) {
    scope_ = ScopeManager::GetInstance().GetCurrentScope();
  }

  ~AnfNode() override = default;
  MS_DECLARE_PARENT(AnfNode, Base);

  virtual void accept(AnfVisitor *) {}
  FuncGraphPtr func_graph() const { return func_graph_.lock(); }

  void set_func_graph(const FuncGraphPtr &func_graph) { func_graph_ = FuncGraphWeakPtr(func_graph); }

  ScopePtr scope() { return scope_; }
  void set_scope(const ScopePtr &scope) { scope_ = scope; }

  const KernelInfoDevice *kernel_info() const { return kernel_info_.get(); }
  KernelInfoDevice *kernel_info() { return kernel_info_.get(); }
  void set_kernel_info(const KernelInfoDevicePtr &kernel_info) { kernel_info_ = kernel_info; }

  AbstractBasePtr abstract() const { return abstract_; }
  void set_abstract(const AbstractBasePtr &abs) { abstract_ = abs; }

  AbstractBasePtr intermediate_abstract() { return intermediate_abstract_; }
  void set_intermediate_abstract(const AbstractBasePtr &abs) { intermediate_abstract_ = abs; }

  NodeDebugInfoPtr debug_info() {
    MS_EXCEPTION_IF_NULL(debug_info_);
    if (debug_info_->get_node() == nullptr) {
      debug_info_->set_node(shared_from_base<AnfNode>());
    }
    return debug_info_;
  }
  void set_debug_info(const NodeDebugInfoPtr &debug_info) {
    debug_info_ = debug_info;
    if (debug_info_->get_node() == nullptr) {
      debug_info_->set_node(shared_from_base<AnfNode>());
    }
  }

  TypePtr Type() const;
  BaseShapePtr Shape() const;

  std::size_t hash() const override { return this->hash_(this); }
  virtual std::string fullname_with_scope() { return ""; }

  virtual std::string DebugString(int recursive_level = 1) const { return ToString(); }
  virtual std::string DebugString(bool recursive) const { return DebugString(recursive ? 1 : 0); }
  std::string ToString() const override;
  void dump() const override { std::cout << DebugString() << std::endl; }
  std::string UniqueId() { return std::to_string(debug_info()->unique_id()); }
  std::string UniqueIdThroughCopy() { return std::to_string(debug_info()->unique_id_through_copy()); }
  virtual bool operator==(const AnfNode &other) const { return &other == this; }
  friend std::ostream &operator<<(std::ostream &os, const AnfNode &node) {
    os << node.ToString();
    return os;
  }

 protected:
  // Hold a weak ref to Graph as Graph also hold ref to AnfNode.
  // Otherwise, func_graph_ and AnfNode will make a reference cycle.
  FuncGraphWeakPtr func_graph_;
  AbstractBasePtr abstract_;
  AbstractBasePtr intermediate_abstract_;
  NodeDebugInfoPtr debug_info_;
  std::string fullname_with_scope_;

 private:
  std::hash<const AnfNode *> hash_;
  ScopePtr scope_;
  KernelInfoDevicePtr kernel_info_;
};

// CNode represents the complex node with a set of arguments.
// Fields:
// inputs_: represents all of the inputs for this CNode.
// Using input(i) to get the index i input.
// Using inputs() to get all the inputs as a vector.
// Using add_input(input) to append a new input for a CNode.
// Using set_input(i, input) to change some input of these inputs.
// Using set_inputs(inputs) to refresh all of the inputs of a CNode.
// func_graph_as_var_: used in opt pattern matching to match a real FuncGraph.
// stop_gradient_: a flag used to stop gradient.
// Using stop_gradient() to get this flag, mainly used in ad.
// Using set_stop_gradient() to set this flag.
class CNode : public AnfNode {
 public:
  CNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph);
  CNode(const std::vector<AnfNodePtr> &inputs, const VarPtr &func_graph_as_var)
      : AnfNode(nullptr), inputs_(inputs), func_graph_as_var_(func_graph_as_var), stop_gradient_(false) {}

  ~CNode() override = default;
  MS_DECLARE_PARENT(CNode, AnfNode);

  void accept(AnfVisitor *v) override;
  // check whether this cnode has some primitive value as the first input.
  bool IsApply(const PrimitivePtr &) const;

  const size_t size() const { return inputs_.size(); }
  const AnfNodePtr input(size_t i) const { return inputs_[i]; }
  const std::vector<AnfNodePtr> &inputs() const { return inputs_; }
  void add_input(const AnfNodePtr &input) { inputs_.push_back(input); }
  void set_input(size_t i, const AnfNodePtr &input);
  void set_inputs(const std::vector<AnfNodePtr> &inputs) { inputs_ = inputs; }

  bool stop_gradient() const { return stop_gradient_; }
  void set_stop_gradient(bool stop_gradient) { stop_gradient_ = stop_gradient; }

  std::string fullname_with_scope() override;
  std::string DebugString(int recursive_level = 1) const override;
  std::string DebugString(bool recursive) const override { return DebugString(recursive ? 1 : 0); }

  OperatorInfoPtr set_operator_info(const OperatorInfoPtr &operator_info);
  OperatorInfoPtr operator_info() { return operator_info_; }

  void set_in_forward_flag(bool flag) { in_forward_flag_ = flag; }
  bool in_forward_flag() const { return in_forward_flag_; }

  VarPtr func_graph_as_var() const { return func_graph_as_var_; }

 private:
  std::vector<AnfNodePtr> inputs_;
  VarPtr func_graph_as_var_;
  bool stop_gradient_;
  OperatorInfoPtr operator_info_ = nullptr;
  bool in_forward_flag_ = false;
};

// ANode represents the atomic node. It's derived Parameter and ValueNode.
class ANode : public AnfNode {
 public:
  ANode() : AnfNode(nullptr) {}
  explicit ANode(const FuncGraphPtr &func_graph) : AnfNode(func_graph) {}
  virtual ~ANode() = default;

  MS_DECLARE_PARENT(ANode, AnfNode);
};

// Parameter represents the parameter inputs of a function. They have no value.
// Attributes:
// default_param_: used to hold the inputting tensor of the model.
class Parameter : public ANode {
 public:
  explicit Parameter(const FuncGraphPtr &func_graph)
      : ANode(func_graph), name_(""), has_default_(false), default_param_(py::none()), tensor_layout_(nullptr) {}
  ~Parameter() override = default;
  MS_DECLARE_PARENT(Parameter, ANode);

  void accept(AnfVisitor *v) override;

  std::string name() const { return name_; }
  void set_name(const std::string &name) { name_ = name; }
  std::string fullname_with_scope() override { return name(); };

  bool has_default() const { return has_default_; }

  py::object default_param() { return default_param_; }
  void set_default_param(const py::object &obj) {
    default_param_ = obj;
    has_default_ = true;
  }

  std::shared_ptr<parallel::TensorLayout> tensor_layout() const { return tensor_layout_; }
  void set_tensor_layout(const std::shared_ptr<parallel::TensorLayout> &tensor_layout) {
    tensor_layout_ = tensor_layout;
  }

  bool operator==(const AnfNode &other) const override {
    if (!other.isa<Parameter>()) {
      return false;
    }
    auto p = static_cast<const Parameter &>(other);
    if (name_.length() > 0 && p.name_.length() > 0) {
      return p.name_ == name_;
    }
    return shared_from_this() == other.shared_from_this();
  }

 private:
  std::string name_;
  bool has_default_;
  py::object default_param_;
  std::shared_ptr<parallel::TensorLayout> tensor_layout_;
};
using ParameterPtr = std::shared_ptr<Parameter>;

// Value is used to represent the atomic expression metioned in BNF.
// It mainly be stored in ValueNode. Value and ValueNode is related definition.
class Value : public Base {
 public:
  Value() = default;
  explicit Value(const TypePtr t) : type_(t) {}
  Value(const Value &other) : Base(other) { this->type_ = other.type_; }
  ~Value() override = default;
  MS_DECLARE_PARENT(Value, Base)

  TypePtr type() const { return type_; }
  virtual abstract::AbstractBasePtr ToAbstract() { MS_LOG(EXCEPTION) << "ToAbstract error"; }

  virtual bool operator==(const Value &rhs) const = 0;
  virtual Value &operator=(const Value &other) {
    if (&other == this) {
      return *this;
    }
    this->type_ = other.type_;
    return *this;
  }

 protected:
  TypePtr type_{nullptr};
};
using ValuePtr = std::shared_ptr<Value>;
using ValuePtrList = std::vector<ValuePtr>;

// ValueNode is used to hold value. Unlike CNode and Parameter, ValueNode
// do not belong to any particular function graph.
class ValueNode : public ANode {
 public:
  explicit ValueNode(const ValuePtr &value) : value_(value) {}
  ~ValueNode() override = default;
  MS_DECLARE_PARENT(ValueNode, ANode);

  void accept(AnfVisitor *v) override;
  const ValuePtr &value() const { return value_; }
  std::string fullname_with_scope() override;

  std::string ToString() const override;
  std::string DebugString(int recursive_level = 1) const override;
  std::string DebugString(bool recursive) const override { return DebugString(recursive ? 1 : 0); }

  bool operator==(const AnfNode &other) const override {
    if (!other.isa<ValueNode>()) {
      return false;
    }
    auto v = static_cast<const ValueNode &>(other);
    return *v.value() == *value();
  }
  friend std::ostream &operator<<(std::ostream &os, const ValueNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    os << node->ToString();
    return os;
  }

 private:
  ValuePtr value_;
};

template <typename T>
struct ImmTraits {};

#define IMM_TRAITS(typeimm, prototype) \
  template <>                          \
  struct ImmTraits<prototype> {        \
    using type = typeimm;              \
  };

inline ValuePtr MakeValue(const ValuePtr &value) { return value; }

template <typename S, typename U = typename ImmTraits<S>::type::element_type>
inline ValuePtr MakeValue(S v) {
  return std::make_shared<U>(v);
}

template <typename S, typename U = typename ImmTraits<S>::type>
static S GetValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);

  U imm = value->cast<U>();
  if (imm == nullptr) {
    MS_LOG(EXCEPTION) << "Cast failed, original value: " << value->ToString() << ", type: " << value->type_name();
  }
  return imm->value();
}

template <typename S,
          typename std::enable_if<is_shared_ptr<S>::value && std::is_base_of<Value, typename S::element_type>::value,
                                  S>::type * = nullptr>
static S GetValue(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  S v = value->cast<S>();
  if (v == nullptr) {
    MS_LOG(EXCEPTION) << "Cast failed, original value: " << value->ToString() << ", type: " << value->type_name();
  }
  return v;
}

// used to check whether an AnfNode is a cnode with a kind of Primitive as first input
bool IsPrimitiveCNode(const AnfNodePtr &node, const PrimitivePtr &value);

// used to check whether an AnfNode is a cnode with a Primitive as first input
PrimitivePtr GetCNodePrimitive(const AnfNodePtr &node);

// used to check whether an AnfNode is a valuenode having some Primitive value
bool IsPrimitive(const AnfNodePtr &node, const PrimitivePtr &value);

// used to check whether a ValueNode has some kind of value
template <typename T>
static bool IsValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto anode = node->cast<ValueNodePtr>();
  if (anode != nullptr) {
    auto value = anode->value();
    if (value == nullptr) {
      MS_LOG(EXCEPTION) << "Const value is nullptr.";
    }
    return value->isa<T>();
  }
  return false;
}

inline ValuePtr GetValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<ValueNode>()) {
    return nullptr;
  }
  return node->cast<ValueNodePtr>()->value();
}

template <typename S,
          typename std::enable_if<is_shared_ptr<S>::value && std::is_base_of<Value, typename S::element_type>::value,
                                  S>::type * = nullptr>
inline S GetValueNode(const AnfNodePtr &node) {
  auto value = GetValueNode(node);
  if (value == nullptr) {
    return nullptr;
  }
  auto s = value->cast<S>();
  return s;
}
namespace id_generator {
std::string get_id(const AnfNodePtr &node);
void reset_id();
}  // namespace id_generator
using TaggedNodeMap = std::unordered_map<AnfNodePtr, size_t>;
using TaggedGraph = std::pair<FuncGraphPtr, TaggedNodeMap>;

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_IR_ANF_H_
