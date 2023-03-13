/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_ANF_H_
#define MINDSPORE_CORE_IR_ANF_H_

#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <set>
#include <bitset>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "base/base.h"
#include "base/user_data.h"
#include "base/effect_info.h"
#include "ir/kernel_info_dev.h"
#include "ir/scope.h"
#include "ir/primal_attr.h"
#include "ir/primal_debug_info.h"
#include "utils/info.h"
#include "utils/hashing.h"
#include "utils/ms_utils.h"
#include "utils/os.h"

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
namespace abstract {
class BaseShape;
class AbstractBase;
}  // namespace abstract
using BaseShapePtr = std::shared_ptr<abstract::BaseShape>;
using AbstractBasePtr = std::shared_ptr<abstract::AbstractBase>;
using AbstractBasePtrList = std::vector<AbstractBasePtr>;
using NodeDebugInfoSet = std::set<NodeDebugInfoPtr, DebugInfoCompare>;
using SeenNum = uint32_t;

class Value;
using ValuePtr = std::shared_ptr<Value>;
using ValuePtrList = std::vector<ValuePtr>;

class ValueNode;
using ValueNodePtr = std::shared_ptr<ValueNode>;

class CNode;
using CNodePtr = std::shared_ptr<CNode>;
using CNodePtrList = std::vector<CNodePtr>;
using CNodeWeakPtr = std::weak_ptr<CNode>;

class FuncGraph;
using FuncGraphSet = OrderedSet<FuncGraphPtr>;
using FuncGraphVector = std::vector<FuncGraphPtr>;

class Primitive;
using PrimitivePtr = std::shared_ptr<Primitive>;
struct PrimitiveHasher;
struct PrimitiveEqual;
using PrimitiveSet = mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual>;

class BaseRef;

class Var;
using VarPtr = std::shared_ptr<Var>;

class AnfIrVisitor;

class ParamInfo;
using ParamInfoPtr = std::shared_ptr<ParamInfo>;

// AnfNode is the basic class of the IR definition derived from Base.
// Only two types of nodes are derived: CNode and ANode.
// Methods:
// func_graph: return FuncGraph that this AnfNode belongs to.
// scope: return the scope namespace of this AnfNode. Set it using set_scope.
// abstract: return the cached inferred abstract value. It contains type, shape
// value. Set New cache using set_abstract.
// Type/Shape: return the related info of this AnfNode. When this AnfNode is an
// input of other CNodes, you can get the related info by this method.
// debug_info: return the information retrieved from parser. Set it using set_debug_info.
// fullname_with_scope: return the detailed debug info.

/// \brief AnfNode is the basic class of the IR definition derived from Base.
class MS_CORE_API AnfNode : public Base {
 public:
  /// \brief Constructor.
  ///
  /// \param[in] func_graph The FuncGraph to which this AnfNode belongs.
  /// \param[in] debug_info The debug info to be used for this AnfNode.
  AnfNode(const FuncGraphPtr &func_graph, NodeDebugInfoPtr &&debug_info)
      : func_graph_(FuncGraphWeakPtr(func_graph)),
        abstract_(nullptr),
        debug_info_(std::move(debug_info)),
        fullname_with_scope_(""),
        scope_(ScopeManager::GetInstance().GetCurrentScope()) {}

  /// \brief Constructor.
  ///
  /// \param[in] func_graph The FuncGraph to which this AnfNode belongs.
  explicit AnfNode(const FuncGraphPtr &func_graph) : AnfNode(func_graph, std::make_shared<NodeDebugInfo>()) {}

  /// \brief Destructor.
  ~AnfNode() override = default;
  MS_DECLARE_PARENT(AnfNode, Base);

  /// \brief Use the method of the AnfIrVisitor class to process the node.
  virtual void accept(AnfIrVisitor *) {}

  /// \brief Obtain the FuncGraph to which this AnfNode belongs.
  ///
  /// \return The FuncGraph to which this AnfNode belongs.
  FuncGraphPtr func_graph() const { return func_graph_.lock(); }

  /// \brief Set the FuncGraph to which this AnfNode belongs.
  ///
  /// \param[in] func_graph The input FuncGraph.
  virtual void set_func_graph(const FuncGraphPtr &func_graph) { func_graph_ = FuncGraphWeakPtr(func_graph); }

  /// \brief Obtain the scope namespace of this AnfNode.
  ///
  /// \return The scope namespace.
  ScopePtr scope() { return scope_; }

  /// \brief Set the scope namespace of this AnfNode.
  ///
  /// \param[in] scope New scope namespace.
  void set_scope(const ScopePtr &scope) { scope_ = scope; }

  /// \brief Obtain device kernel program information.
  ///
  /// \return Device kernel program information.
  const KernelInfoDevice *kernel_info() const { return kernel_info_ptr().get(); }

  /// \brief Obtain device kernel program information.
  ///
  /// \return Device kernel program information.
  KernelInfoDevice *kernel_info() { return kernel_info_ptr().get(); }

  /// \brief Obtain the pointer of KernelInfoDevice.
  ///
  /// \return The pointer of KernelInfoDevice.
  KernelInfoDevicePtr kernel_info_ptr() const { return user_data<KernelInfoDevice>(kKernelInfoKey); }

  /// \brief Set device kernel program information.
  ///
  /// \param[in] kernel_info New device kernel program information.
  void set_kernel_info(const KernelInfoDevicePtr &kernel_info) { set_user_data(kKernelInfoKey, kernel_info); }

  /// \brief Obtain the inferred abstract value of this AnfNode.
  ///
  /// \return The inferred abstract value.
  const AbstractBasePtr &abstract() const;

  /// \brief Set the abstract value of this AnfNode.
  ///
  /// \param[in] abs New abstract value.
  void set_abstract(const AbstractBasePtr &abs);

  /// \brief Obtain the debugging information of this AnfNode.
  ///
  /// \return The debugging information of this AnfNode.
  NodeDebugInfoPtr debug_info() {
    MS_EXCEPTION_IF_NULL(debug_info_);
    if (debug_info_->get_node() == nullptr) {
      debug_info_->set_node(shared_from_base<AnfNode>());
    }
    return debug_info_;
  }

  /// \brief Set the debugging information of this AnfNode.
  ///
  /// \return New debugging information.
  void set_debug_info(const NodeDebugInfoPtr &debug_info) {
    MS_EXCEPTION_IF_NULL(debug_info);
    debug_info_ = debug_info;
    if (debug_info_->get_node() == nullptr) {
      debug_info_->set_node(shared_from_base<AnfNode>());
    }
  }

  /// \brief Obtain the type of the element in this AnfNode.
  ///
  /// \return The type of the element.
  TypePtr Type() const;

  /// \brief Obtain the shape of the element in this AnfNode.
  ///
  /// \return The shape of the element.
  BaseShapePtr Shape() const;

  std::size_t hash() const final { return PointerHash<AnfNode>{}(this); }

  /// \brief Obtain detailed information about scope namespace.
  ///
  /// \return Detailed information about scope namespace.
  virtual std::string fullname_with_scope() { return ""; }

  /// \brief Obtain the unique name of this AnfNode.
  ///
  /// \return The unique name of this AnfNode.
  std::string UniqueName() { return fullname_with_scope() + "_" + UniqueId(); }

  /// \brief Obtain the display information of this AnfNode.
  ///
  /// \param[in] recursive_level Recursion level when displayed.
  /// \return Information to be displayed.
  virtual std::string DebugString(int recursive_level = 1) const { return ToString(); }

  /// \brief Obtain the display information of this AnfNode.
  ///
  /// \param[in] recursive Whether to display AnfNode recursively.
  /// \return Information to be displayed.
  virtual std::string DebugString(bool recursive) const { return DebugString(recursive ? 1 : 0); }

  std::string ToString() const override;

  void dump() const override { std::cout << DebugString() << std::endl; }

  /// \brief Obtain the unique id of the debug information of this AnfNode.
  ///
  /// \return Unique id.
  std::string UniqueId() { return std::to_string(debug_info()->unique_id()); }

  /// \brief Obtain the unique id through copied traced information.
  ///
  /// \return Unique id.
  std::string UniqueIdThroughCopy() { return std::to_string(debug_info()->unique_id_through_copy()); }

  /// \brief Determine whether two AnfNodes are the same.
  ///
  /// \param[in] other Another ANfNode.
  /// \return True if the same, otherwise False.
  virtual bool operator==(const AnfNode &other) const { return &other == this; }

  /// \brief Obtain the display information of this AnfNode.
  ///
  /// \param[in] os Output stream.
  /// \param[in] node AnfNode to be displayed.
  /// \return Output stream.
  friend std::ostream &operator<<(std::ostream &os, const AnfNode &node) {
    os << node.ToString();
    return os;
  }

  /// \brief Set user data.
  ///
  /// \param[in] key The key of user data.
  /// \param[in] value The value of user data.
  template <typename T>
  void set_user_data(const std::string &key, const std::shared_ptr<T> &value) {
    user_data_.set<T>(key, value);
  }

  /// \brief Set user data.
  ///
  /// \param[in] value The value of user data.
  template <typename T>
  void set_user_data(const std::shared_ptr<T> &value) {
    user_data_.set<T>(T::key, value);
  }

  /// \brief Get user data.
  ///
  /// \param[in] key The key of user data.
  /// \return Pointer to user data.
  template <typename T>
  std::shared_ptr<T> user_data(const std::string &key) const {
    return user_data_.get<T>(key);
  }

  /// \brief Set user data.
  ///
  /// \return Pointer to user data.
  template <typename T>
  std::shared_ptr<T> user_data() const {
    return user_data_.get<T>(T::key);
  }

  /// \brief Check whether there is corresponding user data by the given key.
  ///
  /// \param[in] key The key of user data.
  /// \return True if it exists, otherwise false.
  bool has_user_data(const std::string &key) const { return user_data_.has(key); }

  /// \brief Check if there is user data.
  ///
  /// \return True if it exists, otherwise false.
  template <typename T>
  bool has_user_data() const {
    return user_data_.has(T::key);
  }

  /// \brief Clone user data.
  ///
  /// \param[in] node Node used to copy user data.
  void CloneUserData(const AnfNodePtr &node) { user_data_ = node->user_data_; }

  /// \brief Check if there is an interpret node.
  ///
  /// \return True if there is an interpret node, otherwise false.
  bool interpret() const { return interpret_flags_[kInterpret]; }

  /// \brief Whether to use interpretation
  ///
  /// \param[in] interpret Boolean.
  void set_interpret(const bool &interpret) { interpret_flags_[kInterpret] = interpret; }

  /// \brief Check if there is an interpret node related to the unsupported internal type.
  ///
  /// \return True if there is an interpret node related to the unsupported internal type, otherwise false.
  bool interpret_internal_type() { return interpret_flags_[kInterpretInternalType]; }

  /// \brief Whether there is an interpret node with unsupported internal type.
  ///
  /// \param[in] interpret_internal_type Boolean.
  void set_interpret_internal_type(const bool &interpret_internal_type) {
    interpret_flags_[kInterpretInternalType] = interpret_internal_type;
  }

  SeenNum seen_{0};
  SeenNum extra_seen_{0};

 protected:
  // Hold a weak ref to Graph as Graph also hold ref to AnfNode.
  // Otherwise, func_graph_ and AnfNode will make a reference cycle.
  FuncGraphWeakPtr func_graph_;
  AbstractBasePtr abstract_;
  NodeDebugInfoPtr debug_info_;
  std::string fullname_with_scope_;

 private:
  static constexpr size_t kInterpret = 0;
  static constexpr size_t kInterpretInternalType = 1;
  static constexpr size_t kNumInterpretFlags = 2;
  static constexpr auto kKernelInfoKey = "kernel_info";

  ScopePtr scope_;
  UserData user_data_;
  std::bitset<kNumInterpretFlags> interpret_flags_;
};

// CNode represents the complex node with a set of arguments.
// Fields:
// inputs_: represents all of the inputs for this CNode.
// Using input(i) to get the index i input.
// Using inputs() to get all the inputs as a vector.
// Using add_input(input) to append a new input for a CNode.
// Using set_input(i, input) to change some input of these inputs.
// Using set_inputs(inputs) to refresh all of the inputs of a CNode.
// func_graph_as_var: used in opt pattern matching to match a real FuncGraph.
// stop_gradient: a flag used to stop gradient.
// Using stop_gradient() to get this flag, mainly used in ad.
// Using set_stop_gradient() to set this flag.
class MS_CORE_API CNode final : public AnfNode, public EffectInfoHolder {
 public:
  /// \brief Constructor.
  ///
  /// \param[in] inputs Input nodes of this Cnode.
  /// \param[in] func_graph The FuncGraph to which this CNode belongs.
  CNode(std::vector<AnfNodePtr> &&inputs, const FuncGraphPtr &func_graph);

  /// \brief Constructor.
  ///
  /// \param[in] inputs Input nodes of this Cnode.
  /// \param[in] func_graph The FuncGraph to which this CNode belongs.
  CNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph);

  /// \brief Constructor.
  ///
  /// \param[in] inputs Input nodes of this Cnode.
  /// \param[in] func_graph_as_var The FuncGraph of type VarPtr to which this CNode belongs,
  CNode(const std::vector<AnfNodePtr> &inputs, const VarPtr &func_graph_as_var) : AnfNode(nullptr), inputs_(inputs) {
    primal_attrs_ = PrimalAttrManager::GetInstance().GetCurrentPrimalAttr();
    primal_debug_infos_ = PrimalDebugInfoManager::GetInstance().GetCurrentPrimalDebugInfo();
    set_user_data(kFuncGraphVarKey, func_graph_as_var);
  }

  /// \brief Constructor.
  ///
  /// \param[in] inputs Input nodes of this Cnode.
  /// \param[in] func_graph The FuncGraph to which this CNode belongs.
  /// \param[in] debug_info The debug info to be used for this CNode.
  CNode(std::vector<AnfNodePtr> &&inputs, const FuncGraphPtr &func_graph, NodeDebugInfoPtr &&debug_info);

  /// \brief Destructor.
  ~CNode() override = default;
  MS_DECLARE_PARENT(CNode, AnfNode);

  void accept(AnfIrVisitor *v) override;

  /// \brief Check whether this cnode has the same primitive value as the first input.
  ///
  /// \return True if they have the same primitive value, otherwise false.
  bool IsApply(const PrimitivePtr &value) const;

  /// \brief Obtain the size of input nodes of this CNode.
  ///
  /// \return Size of input nodes.
  const size_t size() const { return inputs_.size(); }

  /// \brief Get the input node of the given index.
  ///
  /// \param[in] i The given index.
  /// \return The input node of the given index.
  const AnfNodePtr &input(size_t i) const;

  /// \brief Get the input nodes.
  ///
  /// \return The input nodes of this CNode.
  const std::vector<AnfNodePtr> &inputs() const { return inputs_; }

  /// \brief Add the input node to this CNode.
  ///
  /// \param[in] input Node.
  void add_input(const AnfNodePtr &input);

  /// \brief Set the input node of the given index.
  ///
  /// \param[in] i The given index.
  /// \param[in] input Node.
  void set_input(size_t i, const AnfNodePtr &new_input);

  /// \brief Set the input nodes for this CNode.
  ///
  /// \param[in] inputs Input nodes.
  void set_inputs(const std::vector<AnfNodePtr> &inputs);

  // output_value store cnode value and id in pynative mode.
  using OutputValue = std::pair<ValueNodePtr, std::string>;

  /// \brief Record the cnode value and id to output_value_.
  ///
  /// \param[in] forward The cnode value.
  /// \param[in] id The id.
  void set_forward(const ValueNodePtr &forward, const std::string &id) {
    set_user_data(kOutputValueKey, std::make_shared<OutputValue>(forward, id));
  }

  /// \brief Get the record of output value of this CNode.
  ///
  /// \return The output value of this CNode.
  const OutputValue &forward() const {
    static const OutputValue empty_value;
    auto ptr = user_data<OutputValue>(kOutputValueKey);
    if (ptr == nullptr) {
      return empty_value;
    }
    return *ptr;
  }

  /// \brief Check if stop_gradient is set.
  ///
  /// \return True if stop_gradient is set, otherwise false.
  bool stop_gradient() const { return flags_[kStopGradient]; }

  /// \brief Set stop_gradient.
  ///
  /// \param[in] stop_gradient Boolean.
  void set_stop_gradient(bool stop_gradient) { flags_[kStopGradient] = stop_gradient; }

  std::string fullname_with_scope() override;

  /// \brief Set fullname_with_scope for this CNode.
  ///
  /// \param[in] full_name The fullname_with_scope.
  void set_fullname_with_scope(const std::string full_name) { fullname_with_scope_ = full_name; }

  std::string DebugString(int recursive_level = 1) const override;
  std::string DebugString(bool recursive) const override { return DebugString(recursive ? 1 : 0); }

  /// \brief Set in_forward_flag for this CNode.
  ///
  /// \param[in] flag Boolean.
  void set_in_forward_flag(bool flag) { flags_[kInForwardFlag] = flag; }
  /// \brief Check if in_forward_flag is set.
  ///
  /// \return True if in_forward_flag is set, otherwise false.
  bool in_forward_flag() const { return flags_[kInForwardFlag]; }

  /// \brief Check if the primitive of this CNode is load.
  ///
  /// \param[in] is_load Boolean.
  void set_load_flag(bool is_load) { flags_[kIsLoad] = is_load; }
  /// \brief Check if is_load_ is set.
  ///
  /// \return True if is_load_ is set, otherwise false.
  bool get_load_flag() const { return flags_[kIsLoad]; }

  /// \brief Get func_graph_as_var of this CNode.
  ///
  /// \return func_graph_as_var.
  VarPtr func_graph_as_var() const { return user_data<Var>(kFuncGraphVarKey); }

  /// \brief Get all attributes of this CNode.
  ///
  /// \return Attributes of this CNode.
  const mindspore::HashMap<std::string, ValuePtr> &attrs() const { return attrs_; }
  void set_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
    attrs_.insert(attrs.cbegin(), attrs.cend());
  }

  /// \brief Add a new attribute to this CNode.
  ///
  /// \param[in] name The name of the new attribute.
  /// \param[in] attr The value of the new attribute.
  void AddAttr(const std::string &name, const ValuePtr &attr) { attrs_[name] = attr; }

  /// \brief Erase the attribute with the given name.
  ///
  /// \param[in] name The name of attribute.
  void EraseAttr(const std::string &name) { (void)attrs_.erase(name); }

  /// \brief Get the attribute with the given name.
  ///
  /// \param[in] name The name of attribute.
  /// \return Attribute.
  ValuePtr GetAttr(const std::string &name) const {
    auto iter = attrs_.find(name);
    return iter == attrs_.cend() ? nullptr : iter->second;
  }

  /// \brief Check whether this CNode has an attribute with the given name.
  ///
  /// \param[in] name The name of attribute.
  /// \return Boolean.
  bool HasAttr(const std::string &name) const { return attrs_.find(name) != attrs_.cend(); }

  /// \brief Get the number of input tensors.
  ///
  /// \return The number of input tensors.
  ssize_t input_tensor_num() const { return input_tensor_num_; }

  /// \brief Get the primal attributes of this CNode.
  ///
  /// \return The primal attributes.
  const mindspore::HashMap<std::string, ValuePtr> &primal_attrs() const { return primal_attrs_; }

  /// \brief Set the primal attributes of this CNode.
  ///
  /// \param[in] attrs The primal attributes.
  void set_primal_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
    primal_attrs_.insert(attrs.cbegin(), attrs.cend());
  }

  /// \brief Add the primal attribute to this CNode.
  ///
  /// \param[in] name The name of the attribute.
  /// \param[in] attr The attribute.
  void AddPrimalAttr(const std::string &name, const ValuePtr &attr) { primal_attrs_[name] = attr; }

  /// \brief Erase the primal attribute with the given name.
  ///
  /// \param[in] name The name of the attribute.
  void ErasePrimalAttr(const std::string &name) { (void)primal_attrs_.erase(name); }

  /// \brief Get the primal attribute with the given name.
  ///
  /// \param[in] name The name of the attribute.
  /// \return The primal attribute with the given name.
  ValuePtr GetPrimalAttr(const std::string &name) const {
    auto iter = primal_attrs_.find(name);
    return iter == primal_attrs_.cend() ? nullptr : iter->second;
  }

  /// \brief Check whether this CNode has an attribute with the given name.
  ///
  /// \param[in] name The name of the attribute.
  /// \return True if it exists, otherwise false.
  bool HasPrimalAttr(const std::string &name) const { return primal_attrs_.find(name) != primal_attrs_.end(); }

  /// \brief Get primal debug information.
  ///
  /// \return The primal debug information.
  NodeDebugInfoSet primal_debug_infos() const;

  /// \brief Set primal debug information.
  ///
  /// \param[in] debug_infos Debug information of this CNode.
  void set_primal_debug_infos(const NodeDebugInfoSet &debug_infos);

  /// \brief Add a primal debug information.
  ///
  /// \param[in] debug_info A debug information.
  void AddPrimalDebugInfo(const NodeDebugInfoPtr &debug_info);

  void CloneCNodeInfo(const CNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    set_abstract(node->abstract());
    set_forward(node->forward().first, node->forward().second);
    set_attrs(node->attrs());
    set_primal_attrs(node->primal_attrs());
    set_load_flag(node->get_load_flag());
    CloneUserData(node);
    set_kernel_info(node->kernel_info_ptr());
    set_primal_debug_infos(node->primal_debug_infos());
    set_fused_debug_infos(node->fused_debug_infos());
  }

  /// \brief Set the number of input tensors.
  ///
  /// \param[in] The number of input tensors.
  void set_input_tensor_num(ssize_t input_tensor_num) { input_tensor_num_ = input_tensor_num; }

  /// \brief Is effect have been handled.
  ///
  /// \return True if effect have been handled, otherwise false.
  bool IsEffectHandled() const { return flags_[kEffectHandled]; }

  /// \brief Set effect handled or not.
  ///
  /// \param[in] handled Boolean.
  void SetEffectHandled(bool handled) { flags_[kEffectHandled] = handled; }

  /// \brief Get the debug infos of fused nodes.
  ///
  /// \return A vector of debug infos.
  NodeDebugInfoSet fused_debug_infos() const { return fused_debug_infos_; }

  /// \brief Set the debug infos for CNode.
  ///
  /// \param fused_debug_infos The debug infos to be set.
  void set_fused_debug_infos(const NodeDebugInfoSet &fused_debug_infos) { fused_debug_infos_ = fused_debug_infos; }

  /// \brief Add a node's debug info or fused debug info.
  ///
  /// \param node An anf node.
  void AddFusedDebugInfo(const AnfNodePtr &node);

  /// \brief Add a vector of nodes' debug info or fused debug info.
  ///
  /// \param nodes A vector of anf nodes.
  void AddFusedDebugInfoList(const std::vector<AnfNodePtr> &nodes);

  /// \brief Add a node debug info.
  ///
  /// \param debug_info A node debug info of an anf node.
  void AddFusedDebugInfo(const NodeDebugInfoPtr &debug_info);

  /// \brief Add a list of node debug infos.
  ///
  /// \param debug_infos A node debug info of an anf node.
  void AddFusedDebugInfoList(const std::vector<NodeDebugInfoPtr> &debug_infos);

  /// \brief Check whether contains a input or indirect input, which is Depend CNode with isolated side-effect node.
  ///
  /// \return True if contains, otherwise false.
  bool has_side_effect_node() const { return has_side_effect_node_; }

  /// \brief Set whether contains a input or indirect input, which is Depend CNode with isolated side-effect node.
  ///
  /// \param[in] has_side_effect_node Boolean.
  void set_has_side_effect_node(bool has_side_effect_node) { has_side_effect_node_ = has_side_effect_node; }

 private:
  static constexpr size_t kStopGradient = 0;
  static constexpr size_t kInForwardFlag = 1;
  static constexpr size_t kEffectHandled = 2;
  static constexpr size_t kIsLoad = 3;
  static constexpr size_t kNumFlags = 4;
  static constexpr auto kFuncGraphVarKey = "fg_var";
  static constexpr auto kOutputValueKey = "out_value";

  std::vector<AnfNodePtr> inputs_;
  ssize_t input_tensor_num_ = -1;
  std::bitset<kNumFlags> flags_;

  mindspore::HashMap<std::string, ValuePtr> attrs_;
  mindspore::HashMap<std::string, ValuePtr> primal_attrs_;
  NodeDebugInfoSet primal_debug_infos_;
  NodeDebugInfoSet fused_debug_infos_;

  // If the inputs or their inputs contain Depend CNode with isolated side-effect node.
  bool has_side_effect_node_{false};
};

// ANode represents the atomic node. It's derived Parameter and ValueNode.
class MS_CORE_API ANode : public AnfNode {
 public:
  ANode() : AnfNode(nullptr) {}

  /// \brief Constructor.
  ///
  /// \param[in] func_graph The FuncGraph to which this ANode belongs.
  explicit ANode(const FuncGraphPtr &func_graph) : AnfNode(func_graph) {}

  /// \brief Constructor.
  ///
  /// \param[in] func_graph The FuncGraph to which this ANode belongs.
  /// \param[in] debug_info The debug info to be used for this ANode.
  ANode(const FuncGraphPtr &func_graph, NodeDebugInfoPtr &&debug_info) : AnfNode(func_graph, std::move(debug_info)) {}

  /// \brief Destructor.
  virtual ~ANode() = default;

  MS_DECLARE_PARENT(ANode, AnfNode);
};

// Parameter represents the parameter inputs of a function. They have no value.
// Attributes:
// default_param_value_: used to hold the inputting tensor of the model.
class MS_CORE_API Parameter final : public ANode {
 public:
  /// \brief Constructor.
  ///
  /// \param[in] func_graph The FuncGraph to which this Parameter belongs.
  explicit Parameter(const FuncGraphPtr &func_graph) : ANode(func_graph) {}

  /// \brief Constructor.
  ///
  /// \param[in] func_graph The FuncGraph to which this Parameter belongs.
  /// \param[in] debug_info The debug info to be used for this Parameter.
  Parameter(const FuncGraphPtr &func_graph, NodeDebugInfoPtr &&debug_info) : ANode(func_graph, std::move(debug_info)) {}

  /// \brief Destructor.
  ~Parameter() override = default;
  MS_DECLARE_PARENT(Parameter, ANode);

  void accept(AnfIrVisitor *v) override;
  std::string DebugString(int recursive_level = 1) const override;

  /// \brief Get the name of this Parameter.
  ///
  /// \return The name.
  std::string name() const { return name_; }

  /// \brief Set the name of this Parameter.
  ///
  /// \param[in] The name.
  void set_name(const std::string &name) { name_ = name; }

  std::string fullname_with_scope() override { return name(); }

  /// \brief Check if there is a default parameter.
  ///
  /// \return True if this Parameter has a default parameter, otherwise false.
  bool has_default() const { return has_default_; }

  /// \brief Set the default parameter.
  ///
  /// \param[in] param The default parameter.
  void set_default_param(const ValuePtr &param) {
    default_param_ = param;
    has_default_ = true;
  }

  /// \brief Get the default parameter.
  ///
  /// \return The default parameter.
  const ValuePtr &default_param() const { return default_param_; }

  /// \brief Get the parameter information.
  ///
  /// \return The parameter information.
  ParamInfoPtr param_info() const;

  /// \brief Increase used_graph_count.
  void IncreaseUsedGraphCount() { used_graph_count_++; }
  /// \brief Decrease used_graph_count.
  void DecreaseUsedGraphCount() { used_graph_count_--; }
  /// \brief Get used_graph_count.
  ///
  /// \return used_graph_count.
  int used_graph_count() const { return used_graph_count_; }

  bool is_top_graph_param() const { return is_top_graph_param_; }
  void set_is_top_graph_param(bool flag) { is_top_graph_param_ = flag; }

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

  /// \brief This parameter is not used in graph with id.
  ///
  /// \param[in] graph_id The graph id.
  void SetNotUsedByRealKernelInGraph(uint32_t graph_id) { (void)not_used_in_graphs_.insert(graph_id); }

  /// \brief Check if this Parameter is used in graph with id.
  ///
  /// \param[in] graph_id True if used, otherwise false.
  bool IsUsedByRealKernelInGraph(uint32_t graph_id) const {
    return not_used_in_graphs_.find(graph_id) == not_used_in_graphs_.end();
  }

  /// \brief Set whether this Parameter has a dynamic shape.
  ///
  /// \param[in] flag Boolean.
  void set_has_dynamic_shape(bool flag) { has_dynamic_shape_ = flag; }

  /// \brief Check whether this Parameter has a dynamic shape.
  ///
  /// \return True if this Parameter has a dynamic shape, otherwise false.
  bool has_dynamic_shape() const { return has_dynamic_shape_; }

  /// \brief Set whether this Parameter is dynamic len.
  ///
  /// \param[in] flag Boolean.
  void set_dynamic_len(bool flag) { is_dynamic_len_ = flag; }

  /// \brief Check whether this Parameter is dynamic len.
  ///
  /// \return True if this Parameter is dynamic len, otherwise false.
  bool dynamic_len() const { return is_dynamic_len_; }

  /// \brief Set groups attr in FRACTAL_Z format.
  ///
  /// \param[in] fracz_group Groups attr in FRACTAL_Z format.
  void set_fracz_group(int64_t fracz_group) { format_attrs_.fracz_group = fracz_group; }

  /// \brief Get groups attr in FRACTAL_Z format.
  ///
  /// \return Groups attr in FRACTAL_Z format.
  int64_t fracz_group() const { return format_attrs_.fracz_group; }

  /// \brief Set input_size attr in FracNZ_RNN or ND_RNN_Bias format.
  ///
  /// \param[in] input_size input_size attr in FracNZ_RNN or ND_RNN_Bias format.
  void set_input_size(int64_t input_size) { format_attrs_.input_size = input_size; }

  /// \brief Get input_size attr in FracNZ_RNN or ND_RNN_Bias format.
  ///
  /// \return input_size attr in FracNZ_RNN or ND_RNN_Bias format.
  int64_t input_size() const { return format_attrs_.input_size; }

  /// \brief Set hidden_size attr in FracNZ_RNN or ND_RNN_Bias format.
  ///
  /// \param[in] hidden_size hidden_size attr in FracNZ_RNN or ND_RNN_Bias format.
  void set_hidden_size(int64_t hidden_size) { format_attrs_.hidden_size = hidden_size; }

  /// \brief Get hidden_size attr in FracNZ_RNN or ND_RNN_Bias format.
  ///
  /// \return hidden_size attr in FracNZ_RNN or ND_RNN_Bias format.
  int64_t hidden_size() const { return format_attrs_.hidden_size; }

 private:
  struct FormatAttr {
    int64_t fracz_group = 1;
    int64_t input_size = 0;
    int64_t hidden_size = 0;
  };
  std::string name_;
  ValuePtr default_param_;
  // Some attrs used in special format.
  FormatAttr format_attrs_;
  std::set<uint32_t> not_used_in_graphs_;
  int used_graph_count_ = 0;
  bool has_default_ = false;
  bool has_dynamic_shape_ = false;
  // Dynamic len is a flag indicating whether the parameter is dynamic sequence.
  bool is_dynamic_len_ = false;
  bool is_top_graph_param_ = false;
};
using ParameterPtr = std::shared_ptr<Parameter>;
using ParameterWeakPtr = std::weak_ptr<Parameter>;

// Value is used to represent the atomic expression mentioned in BNF.
// It mainly be stored in ValueNode. Value and ValueNode is related definition.
class MS_CORE_API Value : public Base {
 public:
  /// \brief Default constructor.
  Value() = default;

  /// \brief Constructor of Value.
  ///
  /// \param[in] t The type of this Value.
  explicit Value(const TypePtr t) : type_(t) {}

  /// \brief Constructor of Value.
  ///
  /// \param[in] other Another Value.
  Value(const Value &other) : Base(other) { this->type_ = other.type_; }

  /// \brief Destructor.
  ~Value() override = default;
  MS_DECLARE_PARENT(Value, Base)

  /// \brief Get the type of this Value.
  ///
  /// \return The type.
  TypePtr type() const { return type_; }

  /// \brief Get the abstract value of Value.
  ///
  /// \return Abstract value of Value.
  virtual abstract::AbstractBasePtr ToAbstract() {
    MS_LOG(EXCEPTION) << "ToAbstract error : The class " << type_name() << "has no implement ToAbstract yet.";
  }

  /// \brief Check whether the input is the current Value object.
  ///
  /// \param[in] rhs The Value object to be compared.
  /// \return Whether the input is the current Value object.
  virtual bool operator==(const Value &rhs) const = 0;

  /// \brief Check whether the input is the current Value object.
  ///
  /// \param[in] other The Value object to be compared.
  /// \return Whether the input is the current Value object.
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

// ValueNode is used to hold value. Unlike CNode and Parameter, ValueNode
// does not belong to any particular function graph.
class MS_CORE_API ValueNode final : public ANode {
 public:
  /// \brief Constructor of ValueNode.
  ///
  /// \param[in] value The value of this ValueNode.
  explicit ValueNode(const ValuePtr &value) : value_(value) {}

  /// \brief Constructor of ValueNode.
  ///
  /// \param[in] value The value of this ValueNode.
  /// \param[in] debug_info The debug info to be used for this ValueNode.
  ValueNode(const ValuePtr &value, NodeDebugInfoPtr &&debug_info)
      : ANode(nullptr, std::move(debug_info)), value_(value) {}

  /// \brief Destructor.
  ~ValueNode() override = default;
  MS_DECLARE_PARENT(ValueNode, ANode);

  void set_func_graph(const FuncGraphPtr &) override {
    MS_EXCEPTION(ValueError) << "ValueNode should not set its func_graph.";
  }

  void accept(AnfIrVisitor *v) override;

  /// \brief Set the value of this ValueNode.
  ///
  /// \param[in] value The value.
  void set_value(const ValuePtr &value) { value_ = value; }

  /// \brief Get the value of this ValueNode.
  ///
  /// \return The value.
  const ValuePtr &value() const { return value_; }

  std::string fullname_with_scope() override;

  /// \brief Set whether this ValueNode has a new value.
  ///
  /// \param[in] flag Whether this ValueNode has a new value.
  void set_has_new_value(bool flag) { has_new_value_ = flag; }

  /// \brief Check whether this ValueNode has a new value.
  ///
  /// \return Whether this ValueNode has a new value.
  bool has_new_value() const { return has_new_value_; }

  /// \brief Get the count of graphs using this ValueNode.
  ///
  /// \return The count of graphs using this ValueNode.
  size_t used_graph_count() const { return used_graph_count_; }

  /// \brief Set the count of groups using this ValueNode.
  ///
  /// \param[in] group The count of groups using this ValueNode.
  void set_fracz_group(int64_t group) { format_attr_.fracz_group = group; }

  /// \brief Get groups attr in FRACTAL_Z format.
  ///
  /// \return Groups attr in FRACTAL_Z format.
  int64_t fracz_group() const { return format_attr_.fracz_group; }

  /// \brief Set the count of graphs using this ValueNode.
  ///
  /// \param[in] used_graph_count The count of graphs using this ValueNode.
  void set_used_graph_count(size_t used_graph_count) { used_graph_count_ = used_graph_count; }

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
  struct FormatAttr {
    int64_t fracz_group = 1;
    int64_t input_size = 0;
    int64_t hidden_size = 0;
  };
  FormatAttr format_attr_;
  ValuePtr value_;
  size_t used_graph_count_{0};
  bool has_new_value_ = false;
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
  auto imm = value->cast_ptr<typename U::element_type>();
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

MS_CORE_API std::string GetCNodeFuncName(const CNodePtr &cnode);

// Used to get FuncGraphPtr from a cnode first input
MS_CORE_API FuncGraphPtr GetCNodeFuncGraph(const AnfNodePtr &node);

// Used to check whether an AnfNode is a cnode with a kind of Primitive as first input.
MS_CORE_API bool IsPrimitiveCNode(const AnfNodePtr &node, const PrimitivePtr &value = nullptr);

// Used to get PrimitivePtr from a cnode first input
MS_CORE_API PrimitivePtr GetCNodePrimitive(const AnfNodePtr &node);

// Return the function Primitive if DoSignaturePrimitive,
// otherwise return the Primitive directly.
MS_CORE_API PrimitivePtr GetPrimitiveWithoutDoSignature(const AnfNodePtr &node);
// Check the first input of CNode.
// Return the function Primitive if DoSignaturePrimitive,
// otherwise return the Primitive directly.
MS_CORE_API PrimitivePtr GetCNodePrimitiveWithoutDoSignature(const AnfNodePtr &node);

// Return the function value if DoSignaturePrimitive,
// otherwise return the value directly.
MS_CORE_API ValuePtr GetValueWithoutDoSignature(const ValuePtr &value);
// Return the function value if DoSignaturePrimitive,
// otherwise return the value directly.
MS_CORE_API ValuePtr GetValueWithoutDoSignature(const AnfNodePtr &node);
// Check the first input of CNode.
// Return the function value if DoSignaturePrimitive,
// otherwise return the value directly.
MS_CORE_API ValuePtr GetCNodeValueWithoutDoSignature(const AnfNodePtr &node);

/// \brief Used to check whether the given node is a ValueNode with some Primitive value.
///
/// \param[in] node The input node.
/// \param[in] value Primitive value.
/// \return Whether the given node is a ValueNode with some Primitive value.
MS_CORE_API bool IsPrimitive(const AnfNodePtr &node, const PrimitivePtr &value);

// Check whether the given node is a ValueNode belonging to a primitive set.
bool IsOneOfPrimitive(const AnfNodePtr &node, const PrimitiveSet &prim_set);

/// \brief Used to check whether the given node is a CNode belonging to a primitive set.
///
/// \param[in] node The input node.
/// \param[in] prim_set Primitive set.
/// \return Whether the given node is a CNode belonging to a primitive set.
MS_CORE_API bool IsOneOfPrimitiveCNode(const AnfNodePtr &node, const PrimitiveSet &prim_set);

// Check whether two primitives are same.
MS_CORE_API bool IsPrimitiveEquals(const PrimitivePtr &prim1, const PrimitivePtr &prim2);

// Get number of AbstractMonad
MS_CORE_API size_t GetAbstractMonadNum(const AbstractBasePtrList &args);

// Check whether the given node has monad abstract.
MS_CORE_API bool HasAbstractMonad(const AnfNodePtr &node);

// Check whether the given node has U monad abstract.
MS_CORE_API bool HasAbstractUMonad(const AnfNodePtr &node);

// Check whether the given node has IO monad abstract.
MS_CORE_API bool HasAbstractIOMonad(const AnfNodePtr &node);

// Gets primitive attribute value as a bool flag.
MS_CORE_API bool GetPrimitiveFlag(const PrimitivePtr &prim, const std::string &attr);

// Gets effect info from a primitive by its attributes.
MS_CORE_API EffectInfo GetPrimEffectInfo(const PrimitivePtr &prim);

// Check if monad state is equivalent for the connected two nodes, not strict but more faster.
MS_CORE_API bool IsStateEquivalent(const AnfNodePtr &outer, const AnfNodePtr &inner);

// Check if the node is DeadNode.
MS_CORE_API bool IsDeadNode(const AnfNodePtr &node);

// Check if the node is PolyNode.
MS_CORE_API bool IsPolyNode(const AnfNodePtr &node);

// Used to check whether a ValueNode has some kind of value.
template <typename T>
inline bool IsValueNode(const AnfNodePtr &node) {
  auto value_node = dyn_cast_ptr<ValueNode>(node);
  if (value_node == nullptr) {
    return false;
  }
  const auto &value = value_node->value();
  return (value != nullptr) && (value->isa<T>());
}

inline ValuePtr GetValueNode(const AnfNodePtr &node) {
  auto value_node = dyn_cast_ptr<ValueNode>(node);
  return (value_node == nullptr) ? nullptr : value_node->value();
}

inline Value *GetValuePtr(const AnfNodePtr &node) {
  auto value_node = dyn_cast_ptr<ValueNode>(node);
  return (value_node == nullptr) ? nullptr : value_node->value().get();
}

template <typename S,
          typename std::enable_if<is_shared_ptr<S>::value && std::is_base_of<Value, typename S::element_type>::value,
                                  S>::type * = nullptr>
inline S GetValueNode(const AnfNodePtr &node) {
  auto value = GetValuePtr(node);
  return (value == nullptr) ? nullptr : value->cast<S>();
}

template <typename S, typename std::enable_if<std::is_base_of<Value, S>::value, S>::type * = nullptr>
inline S *GetValuePtr(const AnfNodePtr &node) {
  auto value_node = dyn_cast_ptr<ValueNode>(node);
  if (value_node == nullptr) {
    return nullptr;
  }
  const auto &value = value_node->value();
  return (value == nullptr) ? nullptr : value->cast_ptr<S>();
}

MS_CORE_API SeenNum NewSeenGeneration();

namespace id_generator {
MS_CORE_API std::string get_id(const AnfNodePtr &node);
MS_CORE_API void reset_id();
MS_CORE_API void reset_id_with_offset();
}  // namespace id_generator
using TaggedNodeMap = mindspore::HashMap<AnfNodePtr, size_t>;
using TaggedGraph = std::pair<FuncGraphPtr, TaggedNodeMap>;
MS_CORE_API std::string GetCNodeTarget(const AnfNodePtr &node);
std::string GetOriginNodeTarget(const AnfNodePtr &node);
MS_CORE_API bool ContainMultiTarget(const std::vector<AnfNodePtr> &nodes);
struct GraphSegment {
  GraphSegment(const std::vector<AnfNodePtr> &nodes, bool is_cut) : nodes_(nodes), is_cut_(is_cut) {}
  void AddPreSegment(const std::shared_ptr<GraphSegment> &segment) { (void)pre_segments_.insert(segment); }
  std::vector<AnfNodePtr> nodes_;
  std::set<std::shared_ptr<GraphSegment>> pre_segments_;
  bool is_cut_{false};
  uint32_t graph_id_{0};
};
using GraphSegmentPtr = std::shared_ptr<GraphSegment>;

constexpr auto kElementsUseFlagsKey = "elements_use_flags";
inline std::shared_ptr<std::vector<bool>> GetSequenceNodeElementsUseFlags(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->template user_data<std::vector<bool>>(kElementsUseFlagsKey);
}

inline void SetSequenceNodeElementsUseFlags(const AnfNodePtr &node, const std::shared_ptr<std::vector<bool>> &flags) {
  MS_EXCEPTION_IF_NULL(node);
  node->set_user_data(kElementsUseFlagsKey, flags);
}

// Set the sequence nodes' elements use flags to 'new_flag' at specific 'index' position.
MS_CORE_API void SetSequenceElementsUseFlags(const AbstractBasePtr &abs, std::size_t index, bool new_flag);
// Set the sequence nodes' elements use flags all to 'new_flag'.
MS_CORE_API void SetSequenceElementsUseFlags(const AbstractBasePtr &abs, bool new_flag);
// Set the sequence nodes' elements use flags all to 'new_flag' recursively.
MS_CORE_API void SetSequenceElementsUseFlagsRecursively(const AbstractBasePtr &abs, bool new_flag);
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_ANF_H_
