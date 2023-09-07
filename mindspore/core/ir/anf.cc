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

#include "ir/anf.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <queue>

#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "utils/hash_map.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "utils/ms_context.h"
#include "utils/anf_utils.h"

namespace mindspore {
AnfNode::AnfNode(const FuncGraphPtr &func_graph, NodeDebugInfoPtr &&debug_info)
    : func_graph_(FuncGraphWeakPtr(func_graph)),
      abstract_(nullptr),
      debug_info_(std::move(debug_info)),
      fullname_with_scope_(""),
      scope_(ScopeManager::GetInstance().GetCurrentScope()) {}

AnfNode::AnfNode(const FuncGraphPtr &func_graph) : AnfNode(func_graph, std::make_shared<NodeDebugInfo>()) {}

void AnfNode::accept(AnfIrVisitor *) {}

FuncGraphPtr AnfNode::func_graph() const { return func_graph_.lock(); }

void AnfNode::set_func_graph(const FuncGraphPtr &func_graph) { func_graph_ = FuncGraphWeakPtr(func_graph); }

ScopePtr AnfNode::scope() { return scope_; }

void AnfNode::set_scope(const ScopePtr &scope) { scope_ = scope; }

const KernelInfoDevice *AnfNode::kernel_info() const { return kernel_info_ptr().get(); }

KernelInfoDevice *AnfNode::kernel_info() { return kernel_info_ptr().get(); }

KernelInfoDevicePtr AnfNode::kernel_info_ptr() const { return user_data<KernelInfoDevice>(kKernelInfoKey); }

void AnfNode::set_kernel_info(const KernelInfoDevicePtr &kernel_info) { set_user_data(kKernelInfoKey, kernel_info); }

NodeDebugInfoPtr AnfNode::debug_info() {
  MS_EXCEPTION_IF_NULL(debug_info_);
  if (debug_info_->get_node() == nullptr) {
    debug_info_->set_node(shared_from_base<AnfNode>());
  }
  return debug_info_;
}

void AnfNode::set_debug_info(const NodeDebugInfoPtr &debug_info) {
  MS_EXCEPTION_IF_NULL(debug_info);
  debug_info_ = debug_info;
  if (debug_info_->get_node() == nullptr) {
    debug_info_->set_node(shared_from_base<AnfNode>());
  }
}

std::size_t AnfNode::hash() const { return PointerHash<AnfNode>{}(this); }

std::string AnfNode::fullname_with_scope() { return ""; }

std::string AnfNode::UniqueName() { return fullname_with_scope() + "_" + UniqueId(); }

std::string AnfNode::DebugString(int recursive_level) const { return ToString(); }

std::string AnfNode::DebugString(bool recursive) const { return DebugString(recursive ? 1 : 0); }

void AnfNode::dump() const { std::cout << DebugString() << std::endl; }

std::string AnfNode::UniqueId() { return std::to_string(debug_info()->unique_id()); }

std::string AnfNode::UniqueIdThroughCopy() { return std::to_string(debug_info()->unique_id_through_copy()); }

bool AnfNode::operator==(const AnfNode &other) const { return &other == this; }

std::ostream &operator<<(std::ostream &os, const AnfNode &node) {
  os << node.ToString();
  return os;
}

bool AnfNode::interpret() const { return interpret_flags_[kInterpret]; }

void AnfNode::set_interpret(const bool &interpret) { interpret_flags_[kInterpret] = interpret; }

bool AnfNode::interpret_internal_type() { return interpret_flags_[kInterpretInternalType]; }

void AnfNode::set_interpret_internal_type(const bool &interpret_internal_type) {
  interpret_flags_[kInterpretInternalType] = interpret_internal_type;
}

const AbstractBasePtr &AnfNode::abstract() const {
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(this);
  return abstract_;
}

void AnfNode::set_abstract(const AbstractBasePtr &abs) {
  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(this);
  abstract_ = abs;
}

// namespace to support intermediate representation definition
CNode::CNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &func_graph)
    : AnfNode(func_graph),
      inputs_(inputs),
      primal_attrs_(PrimalAttrManager::GetInstance().GetCurrentPrimalAttr()),
      primal_debug_infos_(PrimalDebugInfoManager::GetInstance().GetCurrentPrimalDebugInfo()) {}

CNode::CNode(std::vector<AnfNodePtr> &&inputs, const FuncGraphPtr &func_graph)
    : AnfNode(func_graph),
      inputs_(std::move(inputs)),
      primal_attrs_(PrimalAttrManager::GetInstance().GetCurrentPrimalAttr()),
      primal_debug_infos_(PrimalDebugInfoManager::GetInstance().GetCurrentPrimalDebugInfo()) {}

CNode::CNode(std::vector<AnfNodePtr> &&inputs, const FuncGraphPtr &func_graph, NodeDebugInfoPtr &&debug_info)
    : AnfNode(func_graph, std::move(debug_info)),
      inputs_(std::move(inputs)),
      primal_attrs_(PrimalAttrManager::GetInstance().GetCurrentPrimalAttr()),
      primal_debug_infos_(PrimalDebugInfoManager::GetInstance().GetCurrentPrimalDebugInfo()) {}

CNode::CNode(const std::vector<AnfNodePtr> &inputs, const VarPtr &func_graph_as_var)
    : AnfNode(nullptr), inputs_(inputs) {
  primal_attrs_ = PrimalAttrManager::GetInstance().GetCurrentPrimalAttr();
  primal_debug_infos_ = PrimalDebugInfoManager::GetInstance().GetCurrentPrimalDebugInfo();
  set_user_data(kFuncGraphVarKey, func_graph_as_var);
}

const size_t CNode::size() const { return inputs_.size(); }

const std::vector<AnfNodePtr> &CNode::inputs() const { return inputs_; }

// Check if CNode is an apply with the specific Primitive.
bool CNode::IsApply(const PrimitivePtr &value) const {
  if (value == nullptr || inputs_.empty()) {
    return false;
  }
  auto prim = GetValuePtr<Primitive>(inputs_[0]);
  return (prim != nullptr) && (prim->Hash() == value->Hash()) && (prim->name() == value->name());
}

void CNode::add_input(const AnfNodePtr &input) {
  (void)inputs_.emplace_back(input);
  input_tensor_num_ = -1;
}

void CNode::set_input(size_t i, const AnfNodePtr &new_input) {
  if (i >= inputs_.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "i: " << i << " out of range: " << inputs_.size() << ", cnode: " << DebugString();
  }
  inputs_[i] = new_input;
  input_tensor_num_ = -1;
}

void CNode::set_inputs(const std::vector<AnfNodePtr> &inputs) {
  inputs_ = inputs;
  input_tensor_num_ = -1;
}

const AnfNodePtr &CNode::input(size_t i) const {
  if (i >= inputs_.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "i: " << i << " out of range: " << inputs_.size() << ", cnode: " << DebugString();
  }
  return inputs_.at(i);
}

std::string CNode::DebugString(int recursive_level) const {
  std::ostringstream buffer;
  if (recursive_level > 0) {
    if (func_graph() != nullptr) {
      buffer << "@" << func_graph()->ToString() << ":";
    }
    buffer << ToString() << "{";
    bool is_first_node = true;
    int idx = 0;
    for (auto &node : inputs_) {
      MS_EXCEPTION_IF_NULL(node);
      if (is_first_node) {
        is_first_node = false;
      } else {
        buffer << ", ";
      }
      buffer << "[" << idx << "]: " << node->DebugString(recursive_level - 1);
      idx++;
    }
    buffer << "}";
  } else {
    buffer << ToString();
  }
  return buffer.str();
}

void CNode::AddFusedDebugInfo(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<CNode>()) {
    return;
  }
  if (shared_from_this() == node) {
    this->AddFusedDebugInfo(node->debug_info());
    return;
  }
  auto cnode = node->cast_ptr<CNode>();
  auto node_fused_debug_infos = cnode->fused_debug_infos();
  if (!node_fused_debug_infos.empty()) {
    (void)std::for_each(node_fused_debug_infos.begin(), node_fused_debug_infos.end(),
                        [this](const NodeDebugInfoPtr &debug_info) { this->AddFusedDebugInfo(debug_info); });
  } else {
    this->AddFusedDebugInfo(cnode->debug_info());
  }

  auto primal_debug_infos = cnode->primal_debug_infos();
  if (!primal_debug_infos.empty()) {
    (void)std::for_each(primal_debug_infos.begin(), primal_debug_infos.end(),
                        [this](const NodeDebugInfoPtr &debug_info) { this->AddPrimalDebugInfo(debug_info); });
  }
}

void CNode::AddFusedDebugInfoList(const std::vector<AnfNodePtr> &nodes) {
  (void)std::for_each(nodes.begin(), nodes.end(), [this](const AnfNodePtr &node) { this->AddFusedDebugInfo(node); });
}

void CNode::AddFusedDebugInfo(const NodeDebugInfoPtr &debug_info) {
  if (debug_info == nullptr) {
    return;
  }
  (void)fused_debug_infos_.emplace(debug_info);
}

void CNode::AddFusedDebugInfoList(const std::vector<NodeDebugInfoPtr> &debug_infos) {
  (void)std::for_each(debug_infos.begin(), debug_infos.end(),
                      [this](const NodeDebugInfoPtr &debug_info) { this->AddFusedDebugInfo(debug_info); });
}

NodeDebugInfoSet CNode::primal_debug_infos() const { return primal_debug_infos_; }

void CNode::set_primal_debug_infos(const NodeDebugInfoSet &debug_infos) {
  (void)std::for_each(debug_infos.begin(), debug_infos.end(),
                      [this](const NodeDebugInfoPtr &debug_info) { this->AddPrimalDebugInfo(debug_info); });
}

void CNode::AddPrimalDebugInfo(const NodeDebugInfoPtr &debug_info) { (void)primal_debug_infos_.emplace(debug_info); }

void CNode::set_forward(const ValueNodePtr &forward, const std::string &id) {
  set_user_data(kOutputValueKey, std::make_shared<OutputValue>(forward, id));
}

const CNode::OutputValue &CNode::forward() const {
  static const CNode::OutputValue empty_value;
  auto ptr = user_data<CNode::OutputValue>(kOutputValueKey);
  if (ptr == nullptr) {
    return empty_value;
  }
  return *ptr;
}

bool CNode::stop_gradient() const { return flags_[kStopGradient]; }

void CNode::set_stop_gradient(bool stop_gradient) { flags_[kStopGradient] = stop_gradient; }

void CNode::set_fullname_with_scope(const std::string full_name) { fullname_with_scope_ = full_name; }

std::string CNode::DebugString(bool recursive) const { return DebugString(recursive ? 1 : 0); }

void CNode::set_in_forward_flag(bool flag) { flags_[kInForwardFlag] = flag; }

bool CNode::in_forward_flag() const { return flags_[kInForwardFlag]; }

void CNode::set_load_flag(bool is_load) { flags_[kIsLoad] = is_load; }

bool CNode::get_load_flag() const { return flags_[kIsLoad]; }

VarPtr CNode::func_graph_as_var() const { return user_data<Var>(kFuncGraphVarKey); }

const mindspore::HashMap<std::string, ValuePtr> &CNode::attrs() const { return attrs_; }

void CNode::set_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  attrs_.insert(attrs.cbegin(), attrs.cend());
}

void CNode::AddAttr(const std::string &name, const ValuePtr &attr) { attrs_[name] = attr; }

void CNode::EraseAttr(const std::string &name) { (void)attrs_.erase(name); }

ValuePtr CNode::GetAttr(const std::string &name) const {
  auto iter = attrs_.find(name);
  return iter == attrs_.cend() ? nullptr : iter->second;
}

bool CNode::HasAttr(const std::string &name) const { return attrs_.find(name) != attrs_.cend(); }

ssize_t CNode::input_tensor_num() const { return input_tensor_num_; }

const mindspore::HashMap<std::string, ValuePtr> &CNode::primal_attrs() const { return primal_attrs_; }

void CNode::set_primal_attrs(const mindspore::HashMap<std::string, ValuePtr> &attrs) {
  primal_attrs_.insert(attrs.cbegin(), attrs.cend());
}

void CNode::AddPrimalAttr(const std::string &name, const ValuePtr &attr) { primal_attrs_[name] = attr; }

void CNode::ErasePrimalAttr(const std::string &name) { (void)primal_attrs_.erase(name); }

ValuePtr CNode::GetPrimalAttr(const std::string &name) const {
  auto iter = primal_attrs_.find(name);
  return iter == primal_attrs_.cend() ? nullptr : iter->second;
}

bool CNode::HasPrimalAttr(const std::string &name) const { return primal_attrs_.find(name) != primal_attrs_.end(); }

void CNode::CloneCNodeInfo(const CNodePtr &node) {
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

void CNode::set_input_tensor_num(ssize_t input_tensor_num) { input_tensor_num_ = input_tensor_num; }

bool CNode::IsEffectHandled() const { return flags_[kEffectHandled]; }

void CNode::SetEffectHandled(bool handled) { flags_[kEffectHandled] = handled; }

NodeDebugInfoSet CNode::fused_debug_infos() const { return fused_debug_infos_; }

void CNode::set_fused_debug_infos(const NodeDebugInfoSet &fused_debug_infos) { fused_debug_infos_ = fused_debug_infos; }

bool CNode::has_side_effect_node() const { return has_side_effect_node_; }

void CNode::set_has_side_effect_node(bool has_side_effect_node) { has_side_effect_node_ = has_side_effect_node; }

Parameter::Parameter(const FuncGraphPtr &func_graph) : ANode(func_graph) {}

Parameter::Parameter(const FuncGraphPtr &func_graph, NodeDebugInfoPtr &&debug_info)
    : ANode(func_graph, std::move(debug_info)) {}

std::string Parameter::name() const { return name_; }

void Parameter::set_name(const std::string &name) { name_ = name; }

std::string Parameter::fullname_with_scope() { return name(); }

bool Parameter::has_default() const { return has_default_; }

void Parameter::set_default_param(const ValuePtr &param) {
  default_param_ = param;
  has_default_ = true;
}

const ValuePtr &Parameter::default_param() const { return default_param_; }

void Parameter::IncreaseUsedGraphCount() { used_graph_count_++; }

void Parameter::DecreaseUsedGraphCount() { used_graph_count_--; }

int Parameter::used_graph_count() const { return used_graph_count_; }

bool Parameter::is_top_graph_param() const { return is_top_graph_param_; }

void Parameter::set_is_top_graph_param(bool flag) { is_top_graph_param_ = flag; }

bool Parameter::operator==(const AnfNode &other) const {
  if (!other.isa<Parameter>()) {
    return false;
  }
  auto &p = static_cast<const Parameter &>(other);
  if (name_.length() > 0 && p.name_.length() > 0) {
    return p.name_ == name_;
  }
  return shared_from_this() == other.shared_from_this();
}

void Parameter::SetNotUsedByRealKernelInGraph(uint32_t graph_id) { (void)not_used_in_graphs_.insert(graph_id); }

bool Parameter::IsUsedByRealKernelInGraph(uint32_t graph_id) const {
  return not_used_in_graphs_.find(graph_id) == not_used_in_graphs_.end();
}

void Parameter::set_has_dynamic_shape(bool flag) { has_dynamic_shape_ = flag; }

bool Parameter::has_dynamic_shape() const { return has_dynamic_shape_; }

void Parameter::set_dynamic_len(bool flag) { is_dynamic_len_ = flag; }

bool Parameter::dynamic_len() const { return is_dynamic_len_; }

void Parameter::set_fracz_group(int64_t fracz_group) { format_attrs_.fracz_group = fracz_group; }

int64_t Parameter::fracz_group() const { return format_attrs_.fracz_group; }

void Parameter::set_input_size(int64_t input_size) { format_attrs_.input_size = input_size; }

int64_t Parameter::input_size() const { return format_attrs_.input_size; }

void Parameter::set_hidden_size(int64_t hidden_size) { format_attrs_.hidden_size = hidden_size; }

int64_t Parameter::hidden_size() const { return format_attrs_.hidden_size; }

std::string Parameter::DebugString(int recursive_level) const {
  std::ostringstream buffer;
  if (recursive_level > 0) {
    if (func_graph() != nullptr) {
      buffer << "@" << func_graph()->ToString() << ":";
    }
  }
  buffer << "param_" << ToString();
  return buffer.str();
}

ParamInfoPtr Parameter::param_info() const {
  if (!has_default()) {
    return nullptr;
  }
  auto tensor = default_param()->cast_ptr<tensor::MetaTensor>();
  if (tensor == nullptr || !tensor->is_parameter()) {
    return nullptr;
  }
  return tensor->param_info();
}

Value::Value(const TypePtr t) : type_(t) {}

Value::Value(const Value &other) : Base(other) { this->type_ = other.type_; }

TypePtr Value::type() const { return type_; }

abstract::AbstractBasePtr Value::ToAbstract() {
  MS_LOG(INTERNAL_EXCEPTION) << "ToAbstract error : The class " << type_name() << "has no implement ToAbstract yet.";
}

Value &Value::operator=(const Value &other) {
  if (&other == this) {
    return *this;
  }
  this->type_ = other.type_;
  return *this;
}

ValueNode::ValueNode(const ValuePtr &value) : value_(value) {}

ValueNode::ValueNode(const ValuePtr &value, NodeDebugInfoPtr &&debug_info)
    : ANode(nullptr, std::move(debug_info)), value_(value) {}

void ValueNode::set_func_graph(const FuncGraphPtr &) {
  MS_INTERNAL_EXCEPTION(ValueError) << "ValueNode should not set its func_graph.";
}

void ValueNode::set_value(const ValuePtr &value) { value_ = value; }

const ValuePtr &ValueNode::value() const { return value_; }

void ValueNode::set_has_new_value(bool flag) { has_new_value_ = flag; }

bool ValueNode::has_new_value() const { return has_new_value_; }

size_t ValueNode::used_graph_count() const { return used_graph_count_; }

void ValueNode::set_fracz_group(int64_t group) { format_attr_.fracz_group = group; }

int64_t ValueNode::fracz_group() const { return format_attr_.fracz_group; }

void ValueNode::set_used_graph_count(size_t used_graph_count) { used_graph_count_ = used_graph_count; }

std::string ValueNode::DebugString(bool recursive) const { return DebugString(recursive ? 1 : 0); }

bool ValueNode::operator==(const AnfNode &other) const {
  if (!other.isa<ValueNode>()) {
    return false;
  }
  auto &v = static_cast<const ValueNode &>(other);
  return *v.value() == *value();
}

std::string ValueNode::ToString() const {
  MS_EXCEPTION_IF_NULL(value_);
  if (value_->isa<FuncGraph>()) {
    return value_->ToString();
  }
  std::ostringstream buffer;
  buffer << AnfNode::ToString();
  buffer << "(" << value_->ToString() << ")";
  return buffer.str();
}

std::string ValueNode::DebugString(int) const {
  MS_EXCEPTION_IF_NULL(value_);
  std::ostringstream buffer;
  buffer << "ValueNode<" << value_->type_name() << "> " << value_->ToString();
  return buffer.str();
}

std::string ValueNode::fullname_with_scope() {
  if (!fullname_with_scope_.empty()) {
    return fullname_with_scope_;
  }

  MS_EXCEPTION_IF_NULL(scope());
  fullname_with_scope_ = scope()->name() + "/" + "data-" + id_generator::get_id(shared_from_base<ValueNode>());
  return fullname_with_scope_;
}

bool IsPrimitiveCNode(const AnfNodePtr &node, const PrimitivePtr &value) {
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr || cnode->size() == 0) {
    return false;
  }
  auto prim = GetValuePtr<Primitive>(cnode->input(0));
  if (prim == nullptr) {
    return false;
  }
  return (value == nullptr) || ((prim->Hash() == value->Hash()) && (prim->name() == value->name()));
}

bool IsPrimitiveCNodeWithoutDoSignature(const AnfNodePtr &node, const PrimitivePtr &value) {
  auto prim = GetCNodePrimitiveWithoutDoSignature(node);
  if (prim == nullptr) {
    return false;
  }
  return (value == nullptr) || ((prim->Hash() == value->Hash()) && (prim->name() == value->name()));
}

PrimitivePtr GetCNodePrimitive(const AnfNodePtr &node) {
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr || cnode->size() == 0) {
    return nullptr;
  }
  return GetValueNode<PrimitivePtr>(cnode->input(0));
}

// Return the function Primitive if DoSignaturePrimitive,
// otherwise return the Primitive directly.
PrimitivePtr GetPrimitiveWithoutDoSignature(const AnfNodePtr &node) {
  auto do_signature_prim = GetValuePtr<prim::DoSignaturePrimitive>(node);
  if (do_signature_prim != nullptr) {
    return dyn_cast<Primitive>(do_signature_prim->function());
  }
  return GetValueNode<PrimitivePtr>(node);
}

// Check the first input of CNode.
// Return the function Primitive if DoSignaturePrimitive,
// otherwise return the Primitive directly.
PrimitivePtr GetCNodePrimitiveWithoutDoSignature(const AnfNodePtr &node) {
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr || cnode->size() == 0) {
    return nullptr;
  }
  return GetPrimitiveWithoutDoSignature(cnode->input(0));
}

// Return the function value if DoSignaturePrimitive,
// otherwise return the value directly.
ValuePtr GetValueWithoutDoSignature(const ValuePtr &value) {
  auto do_signature_prim = dyn_cast_ptr<prim::DoSignaturePrimitive>(value);
  if (do_signature_prim != nullptr) {
    return do_signature_prim->function();
  }
  return value;
}

// Return the function value if DoSignaturePrimitive,
// otherwise return the value directly.
ValuePtr GetValueWithoutDoSignature(const AnfNodePtr &node) {
  auto value = GetValueNode(node);
  if (value == nullptr) {
    return nullptr;
  }
  return GetValueWithoutDoSignature(value);
}

// Check the first input of CNode.
// Return the function value if DoSignaturePrimitive,
// otherwise return the value directly.
ValuePtr GetCNodeValueWithoutDoSignature(const AnfNodePtr &node) {
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr || cnode->size() == 0) {
    return nullptr;
  }
  return GetValueWithoutDoSignature(cnode->input(0));
}

std::string GetCNodeFuncName(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    return "";
  }

  AnfNodePtr valuenode = cnode->input(0);
  auto value = GetValuePtr(valuenode);
  if (value == nullptr) {
    return "";
  }
  auto prim = value->cast_ptr<Primitive>();
  if (prim != nullptr) {
    return prim->name();
  }
  return value->ToString();
}

FuncGraphPtr GetCNodeFuncGraph(const AnfNodePtr &node) {
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode != nullptr && cnode->size() > 0) {
    return GetValueNode<FuncGraphPtr>(cnode->input(0));
  }
  return nullptr;
}

bool IsPrimitive(const AnfNodePtr &node, const PrimitivePtr &value) {
  if (IsValueNode<Primitive>(node)) {
    auto prim = GetValuePtr<Primitive>(node);
    MS_EXCEPTION_IF_NULL(value);
    if (prim->Hash() == value->Hash() && prim->name() == value->name()) {
      return true;
    }
  }
  return false;
}

bool IsPrimitiveEquals(const PrimitivePtr &prim1, const PrimitivePtr &prim2) {
  if (prim1 == nullptr || prim2 == nullptr) {
    return false;
  }
  return (prim1 == prim2) || (prim1->Hash() == prim2->Hash() && prim1->name() == prim2->name());
}

size_t GetAbstractMonadNum(const AbstractBasePtrList &args) {
  size_t num = 0;
  for (auto &arg : args) {
    if (arg->isa<abstract::AbstractMonad>()) {
      ++num;
    }
  }
  return num;
}

template <typename T>
bool HasAbstract(const AnfNodePtr &node) {
  if (node == nullptr) {
    return false;
  }
  const auto &abs = node->abstract();
  return (abs != nullptr && abs->isa<T>());
}

bool HasAbstractMonad(const AnfNodePtr &node) { return HasAbstract<abstract::AbstractMonad>(node); }

bool HasAbstractUMonad(const AnfNodePtr &node) { return HasAbstract<abstract::AbstractUMonad>(node); }

bool HasAbstractIOMonad(const AnfNodePtr &node) { return HasAbstract<abstract::AbstractIOMonad>(node); }

bool GetPrimitiveFlag(const PrimitivePtr &prim, const std::string &attr) {
  if (prim != nullptr) {
    auto flag = prim->GetAttr(attr);
    if (flag && flag->isa<BoolImm>()) {
      return GetValue<bool>(flag);
    }
  }
  return false;
}

EffectInfo GetPrimEffectInfo(const PrimitivePtr &prim) {
  bool mem = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM);
  bool io = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_IO);
  bool back_mem = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP_MEM);
  return {EffectInfo::kDetected, mem, io, false, back_mem};
}

std::set<CNodePtr> GetLoadInputs(const AnfNodePtr &node) {
  std::set<CNodePtr> loads;
  auto cnode = dyn_cast_ptr<CNode>(node);
  if (cnode == nullptr) {
    return loads;
  }
  auto &inputs = cnode->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto &input = inputs.at(i);
    if (IsPrimitiveCNode(input, prim::kPrimLoad)) {
      loads.insert(input->cast<CNodePtr>());
    } else if (IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
      loads.merge(GetLoadInputs(input));
    }
  }
  return loads;
}

bool IsStateEquivalent(const AnfNodePtr &outer, const AnfNodePtr &inner) {
  constexpr size_t kMonadInput = 2;
  auto outer_loads = GetLoadInputs(outer);
  if (outer_loads.empty()) {
    return true;
  }
  auto inner_loads = GetLoadInputs(inner);
  if (inner_loads.empty()) {
    return true;
  }
  outer_loads.merge(inner_loads);
  auto &monad = (*outer_loads.begin())->inputs().at(kMonadInput);
  return std::all_of(++outer_loads.begin(), outer_loads.end(),
                     [&monad, kMonadInput](const CNodePtr &load) { return load->inputs().at(kMonadInput) == monad; });
}

// Check if the node is DeadNode.
bool IsDeadNode(const AnfNodePtr &node) {
  auto value = GetValuePtr<ValueProblem>(node);
  return (value != nullptr) && (value->IsDead());
}

// Check if the node is PolyNode.
bool IsPolyNode(const AnfNodePtr &node) {
  auto value = GetValuePtr<ValueProblem>(node);
  return (value != nullptr) && (value->IsPoly());
}

SeenNum NewSeenGeneration() {
  static SeenNum seen_generation = 0;
  return ++seen_generation;
}

namespace id_generator {
static mindspore::HashMap<std::string, int> node_ids;
static int offset = 0;
std::string get_id(const AnfNodePtr &node) {
  auto type_name = node->type_name();
  if (node_ids.find(type_name) == node_ids.end()) {
    node_ids[type_name] = 0;
  } else {
    node_ids[type_name]++;
  }
  std::string base_id = std::to_string(node_ids[type_name]);
  // The id with offset means the user called reset_id_with_offset() and expect the operated id generated from 0 with an
  // identified offset.
  if (offset != 0) {
    return base_id + '_' + std::to_string(offset);
  }
  return base_id;
}

void reset_id() { node_ids.clear(); }

void reset_id_with_offset() {
  node_ids.clear();
  offset++;
}
}  // namespace id_generator
auto constexpr kPrimitiveTarget = "primitive_target";
namespace {
PrimitivePtr GetPrimitiveFromValueNode(const AnfNodePtr &node) {
  auto value_node = dyn_cast_ptr<ValueNode>(node);
  if (value_node == nullptr) {
    return nullptr;
  }
  return dyn_cast<Primitive>(value_node->value());
}

static std::string GetNodeTargetForVarInputNode(const CNodePtr &cnode) {
  auto &inputs = cnode->inputs();
  std::vector<AnfNodePtr> real_inputs;
  const size_t update_state_valid_input_index = 2;
  const size_t make_tuple_valid_input_index = 1;
  if (cnode->IsApply(prim::kPrimUpdateState) && inputs.size() > update_state_valid_input_index) {
    (void)std::copy(inputs.begin() + SizeToLong(update_state_valid_input_index), inputs.end(),
                    std::back_inserter(real_inputs));
  } else if (cnode->IsApply(prim::kPrimMakeTuple) && inputs.size() > make_tuple_valid_input_index) {
    (void)std::copy(inputs.begin() + SizeToLong(make_tuple_valid_input_index), inputs.end(),
                    std::back_inserter(real_inputs));
  }
  std::string first_input_target = kDeviceUnDefined;
  bool has_diff_target =
    std::any_of(std::rbegin(real_inputs), std::rend(real_inputs), [&first_input_target](const AnfNodePtr &n) {
      auto target = GetOriginNodeTarget(n);
      if (target == kDeviceUnDefined) {
        return false;
      }
      if (first_input_target == kDeviceUnDefined) {
        first_input_target = target;
      }
      return target != first_input_target;
    });
  if (!has_diff_target) {
    return first_input_target;
  }
  return kDeviceUnDefined;
}

static inline bool IsSummaryPrimitiveCNode(const AnfNodePtr &node) {
  return IsPrimitiveCNode(node, prim::kPrimImageSummary) || IsPrimitiveCNode(node, prim::kPrimScalarSummary) ||
         IsPrimitiveCNode(node, prim::kPrimTensorSummary) || IsPrimitiveCNode(node, prim::kPrimHistogramSummary);
}

std::string GetVirtualNodeTargetFromInputs(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto &inputs = cnode->inputs();
#ifndef ENABLE_SECURITY
  if (IsSummaryPrimitiveCNode(node)) {
    if (inputs.size() > 1) {
      return GetOriginNodeTarget(inputs[1]);
    }
    return kDeviceUnDefined;
  }
#endif
  if (IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad)) {
    const size_t node_inputs_num = 3;
    if (inputs.size() >= node_inputs_num) {
      size_t use_index = 1;
      if (!inputs[use_index]->isa<CNode>()) {
        use_index = 2;
      }
      return GetOriginNodeTarget(inputs[use_index]);
    }
  } else if (IsPrimitiveCNode(node, prim::kPrimMakeTuple) || IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
    return GetNodeTargetForVarInputNode(node->cast<CNodePtr>());
  } else if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return GetOriginNodeTarget(cnode->input(1));
  }
  return kDeviceUnDefined;
}

std::string GetVirtualNodeTargetFromUsers(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    return kDeviceUnDefined;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    return kDeviceUnDefined;
  }
  auto users = manager->node_users()[cnode];
  std::string first_user_target = kDeviceUnDefined;
  bool has_diff_target =
    std::any_of(std::begin(users), std::end(users), [&first_user_target](const std::pair<AnfNodePtr, int> &u) {
      auto target = GetOriginNodeTarget(u.first);
      if (target == kDeviceUnDefined) {
        return false;
      }
      if (first_user_target == kDeviceUnDefined) {
        first_user_target = target;
      }
      return target != first_user_target;
    });
  if (!has_diff_target) {
    return first_user_target;
  }
  return kDeviceUnDefined;
}

std::string GetVirtualNodeTarget(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  node->set_user_data(kPrimitiveTarget, std::make_shared<std::string>(kDeviceUnDefined));
  auto target = GetVirtualNodeTargetFromInputs(node);
  node->set_user_data(kPrimitiveTarget, std::make_shared<std::string>(target));
  if (target != kDeviceUnDefined) {
    return target;
  }
  target = GetVirtualNodeTargetFromUsers(node);
  node->set_user_data(kPrimitiveTarget, std::make_shared<std::string>(target));
  return target;
}

std::string GetTargetFromAttr(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr_input = cnode->input(0);
  auto primitive = GetPrimitiveFromValueNode(attr_input);
  if (primitive == nullptr) {
    return kDeviceUnDefined;
  }
  auto att_target = primitive->GetAttr(kPrimitiveTarget);
  if (att_target != nullptr) {
    if (!att_target->isa<StringImm>()) {
      MS_LOG(EXCEPTION) << "Only support string CPU|GPU|Ascend for primitive_target";
    }
    auto target = GetValue<std::string>(att_target);
    if (kTargetSet.find(target) == kTargetSet.end()) {
      MS_LOG(EXCEPTION) << "Only support string CPU|GPU|Ascend for primitive_target, but get " << target;
    }
    return target;
  }
  return kDeviceUnDefined;
}
}  // namespace

std::string GetOriginNodeTarget(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return kDeviceUnDefined;
  }
  auto cnode = node->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto ud_target = cnode->user_data<std::string>(kPrimitiveTarget);
  if (ud_target != nullptr) {
    return *ud_target.get();
  }
  auto target = GetTargetFromAttr(node);
  if (target != kDeviceUnDefined) {
    return target;
  }
#ifndef ENABLE_SECURITY
  if (IsPrimitiveCNode(node, prim::kPrimImageSummary) || IsPrimitiveCNode(node, prim::kPrimScalarSummary) ||
      IsPrimitiveCNode(node, prim::kPrimTensorSummary) || IsPrimitiveCNode(node, prim::kPrimHistogramSummary) ||
      IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad) ||
      IsPrimitiveCNode(node, prim::kPrimUpdateState) || IsPrimitiveCNode(node, prim::kPrimMakeTuple) ||
      IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return GetVirtualNodeTarget(node);
  }
#else
  if (IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad) ||
      IsPrimitiveCNode(node, prim::kPrimUpdateState) || IsPrimitiveCNode(node, prim::kPrimMakeTuple) ||
      IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    return GetVirtualNodeTarget(node);
  }
#endif
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  return context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
}

std::string GetCNodeTarget(const AnfNodePtr &node) {
  auto kernel_info = node->kernel_info();
  if (kernel_info != nullptr) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (runtime_cache.runtime_cache().is_valid()) {
      auto tmp_target = runtime_cache.runtime_cache().device_target();
      if (!tmp_target.empty()) {
        return tmp_target;
      }
    }
  }

  std::string target;
  auto ori_target = GetOriginNodeTarget(node);
  if (ori_target != kDeviceUnDefined) {
    target = ori_target;
  } else {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  }

  if (kernel_info != nullptr) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (runtime_cache.runtime_cache().is_valid()) {
      runtime_cache.runtime_cache().set_device_target(target);
    }
  }
  return target;
}

bool ContainMultiTarget(const std::vector<AnfNodePtr> &nodes) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string last_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  for (auto &node : nodes) {
    if (node->isa<CNode>()) {
      std::string cur_target = GetCNodeTarget(node);
      if (last_target != cur_target) {
        return true;
      }
      last_target = cur_target;
    }
  }
  return false;
}

bool IsOneOfPrimitive(const AnfNodePtr &node, const PrimitiveSet &prim_set) {
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(node);
  return (prim && prim_set.find(prim) != prim_set.end());
}

bool IsOneOfPrimitiveCNode(const AnfNodePtr &node, const PrimitiveSet &prim_set) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast_ptr<CNode>();
  if (cnode == nullptr || cnode->size() == 0) {
    return false;
  }
  return IsOneOfPrimitive(cnode->input(0), prim_set);
}

// Set the sequence nodes' elements use flags to 'new_flag' at specific 'index' position.
void SetSequenceElementsUseFlags(const AbstractBasePtr &abs, std::size_t index, bool new_flag) {
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  if (!enable_eliminate_unused_element) {
    return;
  }

  auto sequence_abs = dyn_cast_ptr<abstract::AbstractSequence>(abs);
  if (sequence_abs == nullptr) {
    return;
  }
  if (sequence_abs->sequence_nodes() == nullptr || sequence_abs->sequence_nodes()->empty()) {
    return;
  }
  for (auto &node : *sequence_abs->sequence_nodes()) {
    auto sequence_node = node.lock();
    if (sequence_node == nullptr) {
      MS_LOG(DEBUG) << "The node in sequence_nodes is free.";
      continue;
    }
    auto flags = GetSequenceNodeElementsUseFlags(sequence_node);
    if (flags == nullptr) {
      continue;
    }
    if (index >= flags->size()) {
      MS_LOG(ERROR) << "The index " << index << " is out of range, size is " << flags->size() << ", for "
                    << sequence_node->DebugString();
      return;
    }
    (*flags)[index] = new_flag;
    MS_LOG(DEBUG) << "Set item[" << index << "] use flag as " << new_flag << ", for " << sequence_node->DebugString();
  }
}

// Set the sequence nodes' elements use flags all to 'new_flag'.
void SetSequenceElementsUseFlags(const AbstractBasePtr &abs, bool new_flag) {
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  if (!enable_eliminate_unused_element) {
    return;
  }

  auto sequence_abs = dyn_cast_ptr<abstract::AbstractSequence>(abs);
  if (sequence_abs == nullptr) {
    return;
  }
  if (sequence_abs->sequence_nodes() == nullptr || sequence_abs->sequence_nodes()->empty()) {
    return;
  }
  for (auto &weak_node : *sequence_abs->sequence_nodes()) {
    auto sequence_node = weak_node.lock();
    if (sequence_node == nullptr) {
      MS_LOG(DEBUG) << "The node in sequence_nodes is free.";
      continue;
    }
    auto flags = GetSequenceNodeElementsUseFlags(sequence_node);
    if (flags != nullptr) {
      auto &all_flags = (*flags);
      (void)std::transform(all_flags.begin(), all_flags.end(), all_flags.begin(),
                           [&new_flag](bool) -> bool { return new_flag; });
    }
  }
}

// Set the sequence nodes' elements use flags all to 'new_flag' recursively.
void SetSequenceElementsUseFlagsRecursively(const AbstractBasePtr &abs, bool new_flag) {
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  if (!enable_eliminate_unused_element) {
    return;
  }

  SetSequenceElementsUseFlags(abs, new_flag);

  // Check its elements if it's a sequence node.
  auto sequence_abs = dyn_cast_ptr<abstract::AbstractSequence>(abs);
  if (sequence_abs != nullptr) {
    for (auto &element : sequence_abs->elements()) {
      SetSequenceElementsUseFlagsRecursively(element, new_flag);
    }
    return;
  }

  // Check its elements if it's a dictionary node.
  auto dictionary_abs = dyn_cast_ptr<abstract::AbstractDictionary>(abs);
  if (dictionary_abs != nullptr) {
    for (auto &element : dictionary_abs->elements()) {
      SetSequenceElementsUseFlagsRecursively(element.second, new_flag);
    }
  }
}
}  // namespace mindspore
