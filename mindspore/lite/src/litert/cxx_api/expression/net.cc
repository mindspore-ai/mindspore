/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "include/api/net.h"
#include "include/api/status.h"
#include "src/litert/cxx_api/expression/node_impl.h"
#include "src/litert/cxx_api/expression/net_impl.h"
#include "src/expression/ops.h"
#include "src/expression/cfg.h"

namespace mindspore {
uint32_t Node::type() { return kNodeType; }

std::vector<Expr *> Node::operator()(const std::vector<Expr *> &inputs) {
  auto in = Expr::convert(inputs);
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "empty implementation";
    return {};
  }
  if (impl_->node() == nullptr) {
    MS_LOG(ERROR) << "expression node is not attached";
    return {};
  }
  auto out = impl_->node()->construct(in);
  return Expr::convert(out);
}

Expr *Node::Create(std::string name) {
  auto expr = impl_->node()->create(name);
  return reinterpret_cast<Expr *>(expr);
}

Node::Node() {
  auto impl = std::make_shared<NodeImpl>();
  impl_ = impl;
  impl_->set_pnode(this);
}

Node::~Node() {
  impl_->set_pnode(nullptr);
  auto node = impl_->node();
  if (node != nullptr) {
    impl_->set_node(nullptr);
    delete node;
  }
}

Net::Net(std::string name) {
  auto impl = std::make_shared<NetImpl>();
  if (impl == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate network implementation";
    return;
  }
  impl_ = impl;
  impl_->set_pnet(std::shared_ptr<Net>(this));
  auto netl = new (std::nothrow) lite::Net(name);
  if (netl == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate network lite";
    return;
  }
  netl->set_impl(impl);
  impl_->set_net(netl);
}

Net::Net() : Net("") {}

Net::Net(const Graph &g) {
  auto net = NetImpl::GetNet(g);
  impl_ = net->impl_;
}

void Net::Add(NetBase *element) { MS_LOG(WARNING) << "Only sequential can add element"; }

uint32_t Net::type() { return kNetType; }

std::vector<Expr *> Net::construct(const std::vector<Expr *> &inputs) {
  auto in = Expr::convert(inputs);
  auto out = impl_->net()->construct(in);
  return Expr::convert(out);
}

std::vector<Expr *> Net::operator()(const std::vector<Expr *> &inputs) {
  auto in = Expr::convert(inputs);
  auto x = construct(inputs);
  impl_->net()->input_ = in;
  auto out = Expr::convert(x);
  impl_->net()->output_ = out;
  impl_->net()->real_output_ = out;
  return x;
}
void Net::Register(Net *net, std::string &&name) {
  if (net != nullptr) {
    auto net_lite = net->impl_->net();
    impl_->net()->Register(net_lite, std::move(name));
  }
}

void Net::Register(Node *node, std::string &&name) {
  if (node != nullptr) {
    auto impl = NodeImpl::GetImpl(node);
    if (impl == nullptr) {
      MS_LOG(ERROR) << "missing implementation";
      return;
    }
    auto node_lite = impl->node();
    impl_->net()->Register(node_lite, std::move(name));
  }
}

std::shared_ptr<NodeSet> Net::trainable_params() {
  auto node_set = std::make_shared<NodeSet>();
  if (node_set == nullptr) {
    MS_LOG(ERROR) << "new NodeSet failed.";
    return nullptr;
  }
  node_set->set_ = impl_->net()->trainable_params();
  return node_set;
}

const std::vector<int> Net::InputShape(int idx) { return impl_->InputShape(idx); }
const std::vector<int> Net::OutputShape(int idx) { return impl_->OutputShape(idx); }

Net::~Net() {
  if (impl_ != nullptr) {
    if ((impl_->pnet() == nullptr) || (impl_->pnet() == this)) {
      impl_->set_pnet(nullptr);
      impl_->set_net(nullptr);
      impl_.reset();
    }
  }
}
}  // namespace mindspore
