/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "mindapi/ir/anf.h"
#include "mindapi/src/helper.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "ir/primitive.h"
#include "abstract/abstract_value.h"

namespace mindspore::api {
using ValueImpl = mindspore::Value;
using AnfNodeImpl = mindspore::AnfNode;
using PrimitiveImpl = mindspore::Primitive;
using AbstractBaseImpl = mindspore::abstract::AbstractBase;

MIND_API_BASE_IMPL(AnfNode, AnfNodeImpl, Base);

std::string AnfNode::fullname_with_scope() const { return ToRef<AnfNodeImpl>(impl_).fullname_with_scope(); }

AbstractBasePtr AnfNode::abstract() const {
  const auto &abs = ToRef<AnfNodeImpl>(impl_).abstract();
  return ToWrapper<AbstractBase>(abs);
}

void AnfNode::set_abstract(const AbstractBasePtr &abs) {
  ToRef<AnfNodeImpl>(impl_).set_abstract(ToImpl<AbstractBaseImpl>(abs));
}

using CNodeImpl = mindspore::CNode;

MIND_API_BASE_IMPL(CNode, CNodeImpl, AnfNode);

size_t CNode::size() const { return ToRef<CNodeImpl>(impl_).size(); }

AnfNodePtr CNode::input(size_t i) const {
  auto &input = ToRef<CNodeImpl>(impl_).input(i);
  return ToWrapper<AnfNode>(input);
}

std::vector<AnfNodePtr> CNode::inputs() const {
  auto &impl_inputs = ToRef<CNodeImpl>(impl_).inputs();
  return ToWrapperVector<AnfNode>(impl_inputs);
}

void CNode::set_inputs(const std::vector<AnfNodePtr> &inputs) {
  auto impl_inputs = ToImplVector<AnfNodeImpl>(inputs);
  ToRef<CNodeImpl>(impl_).set_inputs(impl_inputs);
}

void CNode::add_input(const AnfNodePtr &input) {
  auto impl_input = ToImpl<AnfNodeImpl>(input);
  MS_EXCEPTION_IF_NULL(impl_input);
  ToRef<CNodeImpl>(impl_).add_input(impl_input);
}

void CNode::set_fullname_with_scope(const std::string &full_name) {
  ToRef<CNodeImpl>(impl_).set_fullname_with_scope(full_name);
}

void CNode::AddAttr(const std::string &name, const ValuePtr &attr) {
  auto impl_attr = ToImpl<ValueImpl>(attr);
  MS_EXCEPTION_IF_NULL(impl_attr);
  ToRef<CNodeImpl>(impl_).AddAttr(name, impl_attr);
}

void CNode::EraseAttr(const std::string &name) { ToRef<CNodeImpl>(impl_).EraseAttr(name); }

ValuePtr CNode::GetAttr(const std::string &name) const {
  auto v = ToRef<CNodeImpl>(impl_).GetAttr(name);
  return ToWrapper<Value>(v);
}

using ParameterImpl = mindspore::Parameter;

MIND_API_BASE_IMPL(Parameter, ParameterImpl, AnfNode);

std::string Parameter::name() const { return ToRef<ParameterImpl>(impl_).name(); }

void Parameter::set_name(const std::string &name) { ToRef<ParameterImpl>(impl_).set_name(name); }

bool Parameter::has_default() const { return ToRef<ParameterImpl>(impl_).has_default(); }

void Parameter::set_default_param(const ValuePtr &param) {
  auto v = ToImpl<ValueImpl>(param);
  ToRef<ParameterImpl>(impl_).set_default_param(v);
}

ValuePtr Parameter::default_param() const {
  auto v = ToRef<ParameterImpl>(impl_).default_param();
  return ToWrapper<Value>(v);
}

using ValueNodeImpl = mindspore::ValueNode;

MIND_API_BASE_IMPL(ValueNode, ValueNodeImpl, AnfNode);

ValueNode::ValueNode(const ValuePtr &value) : AnfNode(std::make_shared<ValueNodeImpl>(ToImpl<ValueImpl>(value))) {}

ValuePtr ValueNode::value() const {
  auto v = ToRef<ValueNodeImpl>(impl_).value();
  return ToWrapper<Value>(v);
}

bool IsPrimitiveCNode(const AnfNodePtr &node, const PrimitivePtr &prim) {
  auto node_impl = ToImpl<AnfNodeImpl>(node);
  auto prim_impl = ToImpl<PrimitiveImpl>(prim);
  return mindspore::IsPrimitiveCNode(node_impl, prim_impl);
}

bool IsPrimitive(const AnfNodePtr &node, const PrimitivePtr &prim) {
  auto node_impl = ToImpl<AnfNodeImpl>(node);
  auto prim_impl = ToImpl<PrimitiveImpl>(prim);
  return mindspore::IsPrimitive(node_impl, prim_impl);
}

bool IsDataNode(const AnfNodePtr &node) {
  auto node_impl = ToImpl<AnfNodeImpl>(node);
  // We assume that node with monad abstract is not a data node.
  return !HasAbstractMonad(node_impl);
}
}  // namespace mindspore::api
