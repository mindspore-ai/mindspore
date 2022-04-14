/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include "mindapi/ir/func_graph.h"
#include "mindapi/src/helper.h"
#include "ir/anf.h"
#include "ir/value.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "ir/primitive.h"
#include "ir/graph_utils.h"

namespace mindspore::api {
using ValueImpl = mindspore::Value;
using AnfNodeImpl = mindspore::AnfNode;
using CNodeImpl = mindspore::CNode;
using PrimitiveImpl = mindspore::Primitive;
using ParameterImpl = mindspore::Parameter;
using FuncGraphImpl = mindspore::FuncGraph;
using FuncGraphManagerImpl = mindspore::FuncGraphManager;

MIND_API_BASE_IMPL(FuncGraph, FuncGraphImpl, Value);

std::vector<AnfNodePtr> FuncGraph::get_inputs() const {
  auto &inputs = ToRef<FuncGraphImpl>(impl_).get_inputs();
  return ToWrapperVector<AnfNode>(inputs);
}

std::vector<AnfNodePtr> FuncGraph::parameters() const {
  auto &params = ToRef<FuncGraphImpl>(impl_).parameters();
  return ToWrapperVector<AnfNode>(params);
}

void FuncGraph::add_parameter(const ParameterPtr &p) {
  auto param_impl = ToImpl<ParameterImpl>(p);
  ToRef<FuncGraphImpl>(impl_).add_parameter(param_impl);
}

ParameterPtr FuncGraph::add_parameter() {
  auto param_impl = ToRef<FuncGraphImpl>(impl_).add_parameter();
  return ToWrapper<Parameter>(param_impl);
}

AnfNodePtr FuncGraph::output() const {
  auto output = ToRef<FuncGraphImpl>(impl_).output();
  return ToWrapper<AnfNode>(output);
}

CNodePtr FuncGraph::get_return() const {
  auto ret = ToRef<FuncGraphImpl>(impl_).get_return();
  return ToWrapper<CNode>(ret);
}

void FuncGraph::set_output(const AnfNodePtr &value, bool force_new_ret) {
  auto output = ToImpl<AnfNodeImpl>(value);
  ToRef<FuncGraphImpl>(impl_).set_output(output, force_new_ret);
}

void FuncGraph::set_return(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto cnode_impl = ToImpl<CNodeImpl>(cnode);
  ToRef<FuncGraphImpl>(impl_).set_return(cnode_impl);
}

CNodePtr FuncGraph::NewCNode(const std::vector<AnfNodePtr> &inputs) {
  auto inputs_impl = ToImplVector<AnfNodeImpl>(inputs);
  auto cnode_impl = ToRef<FuncGraphImpl>(impl_).NewCNode(std::move(inputs_impl));
  return ToWrapper<CNode>(cnode_impl);
}

CNodePtr FuncGraph::NewCNode(const PrimitivePtr &primitive, const std::vector<AnfNodePtr> &prim_inputs) {
  auto prim_impl = ToImpl<PrimitiveImpl>(primitive);
  auto prim_inputs_impl = ToImplVector<AnfNodeImpl>(prim_inputs);
  auto cnode_impl = ToRef<FuncGraphImpl>(impl_).NewCNode(prim_impl, prim_inputs_impl);
  return ToWrapper<CNode>(cnode_impl);
}

std::vector<AnfNodePtr> FuncGraph::nodes() const {
  auto &nodes = ToRef<FuncGraphImpl>(impl_).nodes();
  return ToWrapperVector<AnfNode>(nodes);
}

bool FuncGraph::has_attr(const std::string &key) const { return ToRef<FuncGraphImpl>(impl_).has_attr(key); }

ValuePtr FuncGraph::get_attr(const std::string &key) const {
  auto v = ToRef<FuncGraphImpl>(impl_).get_attr(key);
  return ToWrapper<Value>(v);
}

void FuncGraph::set_attr(const std::string &key, const ValuePtr &value) {
  auto value_impl = ToImpl<ValueImpl>(value);
  ToRef<FuncGraphImpl>(impl_).set_attr(key, value_impl);
}

FuncGraphManagerPtr FuncGraph::manager() const {
  auto manager = ToRef<FuncGraphImpl>(impl_).manager();
  if (manager == nullptr) {
    return nullptr;
  }
  return MakeShared<FuncGraphManager>(manager);
}

FuncGraphPtr FuncGraph::Create() {
  auto fg = std::make_shared<FuncGraphImpl>();
  return ToWrapper<FuncGraph>(fg);
}

std::vector<AnfNodePtr> FuncGraph::TopoSort(const AnfNodePtr &node) {
  auto node_impl = ToImpl<AnfNodeImpl>(node);
  if (node_impl == nullptr) {
    return {};
  }
  auto sorted = mindspore::TopoSort(node_impl);
  return ToWrapperVector<AnfNode>(sorted);
}

// FuncGraphManager is not derived from Base, we implement it directly.
FuncGraphManager::FuncGraphManager(const std::shared_ptr<mindspore::FuncGraphManager> &impl) : impl_(impl) {
  MS_EXCEPTION_IF_NULL(impl_);
}

bool FuncGraphManager::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node) {
  return impl_->Replace(ToImpl<AnfNodeImpl>(old_node), ToImpl<AnfNodeImpl>(new_node));
}

void FuncGraphManager::SetEdge(const AnfNodePtr &node, int index, const AnfNodePtr &value) {
  return impl_->SetEdge(ToImpl<AnfNodeImpl>(node), index, ToImpl<AnfNodeImpl>(value));
}

void FuncGraphManager::AddEdge(const AnfNodePtr &node, const AnfNodePtr &value) {
  return impl_->AddEdge(ToImpl<AnfNodeImpl>(node), ToImpl<AnfNodeImpl>(value));
}

std::vector<std::pair<AnfNodePtr, int>> FuncGraphManager::GetUsers(const AnfNodePtr &node) const {
  auto &node_users = impl_->node_users();
  auto iter = node_users.find(ToImpl<AnfNodeImpl>(node));
  if (iter == node_users.end()) {
    return {};
  }
  auto &users_impl = iter->second;
  std::vector<std::pair<AnfNodePtr, int>> users;
  users.reserve(users_impl.size());
  (void)std::transform(users_impl.begin(), users_impl.end(), std::back_inserter(users),
                       [](const auto &user) { return std::make_pair(ToWrapper<AnfNode>(user.first), user.second); });
  return users;
}

FuncGraphManagerPtr FuncGraphManager::Manage(const FuncGraphPtr &func_graph, bool manage) {
  auto fg_impl = ToImpl<FuncGraphImpl>(func_graph);
  auto mgr_impl = mindspore::Manage(fg_impl, manage);
  if (mgr_impl == nullptr) {
    return nullptr;
  }
  return MakeShared<FuncGraphManager>(mgr_impl);
}
}  // namespace mindspore::api
