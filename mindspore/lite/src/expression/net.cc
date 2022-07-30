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

#include "src/expression/net.h"
#include <vector>
#include "src/litert/cxx_api/expression/net_impl.h"
#include "src/expression/ops.h"
#include "src/expression/export.h"
#include "src/expression/ops/addn.h"
#include "src/expression/ops/arithmetic.h"
#include "src/common/storage.h"
#include "tools/common/meta_graph_serializer.h"
namespace mindspore {
namespace lite {
void Net::update_name(std::string name) {
  if (!this->name().empty())
    Node::update_name(name);
  else
    set_name(name);
  for (auto &itr : ops_) {
    itr->update_name(name);
  }
}

std::vector<EXPR *> Net::operator()(const std::initializer_list<EXPR *> &&inputs) {
  std::vector<EXPR *> vec = inputs;
  std::vector<EXPR *> x;
  if (impl_ == nullptr) {
    x = construct(inputs);
  } else {
    x = impl_->construct(vec);
  }
  return x;
}

std::vector<EXPR *> Net::operator()(const std::vector<EXPR *> &inputs) {
  std::vector<EXPR *> x;
  if (impl_ == nullptr) {
    x = construct(inputs);
  } else {
    x = impl_->construct(inputs);
  }
  input_ = inputs;
  output_ = x;
  real_output_ = x;
  return x;
}

std::vector<EXPR *> Net::construct(const std::vector<EXPR *> &inputs) {
  if (!output_.empty()) {
    if (input_.size() != inputs.size()) {
      MS_LOG(ERROR) << "input size mismatch, should be " << input_.size() << " got " << inputs.size();
      return {};
    }
    auto in_ptr = inputs;
    EXPR::Replace(output_, &input_, &in_ptr);
  } else {
    MS_LOG(ERROR) << "no network construction function";
  }
  return output_;
}

void Net::TopoSortUtil(Node *node, std::stack<Node *> *stack) {
  visited_.insert(node);
  for (size_t i = 0; i < node->OutputsNum(); i++) {
    auto expr = node->expr(i);
    auto itr = outmap_.find(expr);
    if (itr != outmap_.end()) {
      for (auto &e : itr->second)
        if (visited_.find(e->node()) == visited_.end()) {
          TopoSortUtil(e->node(), stack);
        }
    }
  }
  stack->push(node);
}

std::vector<Node *> Net::Sort() {
  std::stack<Node *> stack;
  outmap_.clear();
  EXPR::CreateOutputMap(output_, &outmap_);
  for (auto &itr : outmap_) {
    EXPR *e = itr.first;
    if (visited_.find(e->node()) == visited_.end()) {
      TopoSortUtil(e->node(), &stack);
    }
  }
  std::vector<Node *> res;
  while (stack.empty() == false) {
    res.push_back(stack.top());
    stack.pop();
  }
  visited_.clear();
  return res;
}

std::unique_ptr<schema::MetaGraphT> Net::MakeMs() {
  auto nodes = Sort();
  auto s = new (std::nothrow) ExportSession(outmap_);
  if (s == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate export session";
    return nullptr;
  }
  session_.reset(s);
  session_->Init(name(), Version());
  for (auto node : nodes) {
    auto res = node->MakeEntry(session_.get());
    if (res != RET_OK) {
      MS_LOG(ERROR) << "failed in MakeEntry: " << node->name();
      return nullptr;
    }
  }
  session_->SetInputOutput(input_, real_output_);
  auto res = session_->meta_graph();
  return std::unique_ptr<schema::MetaGraphT>(res);
}

std::unique_ptr<schema::MetaGraphT> Net::MakeMs(const std::string file_name) {
  auto graph = MakeMs();
  Save(*graph, file_name);
  return graph;
}

std::set<Node *> Net::trainable_params() {
  std::set<Node *> res;
  for (auto &node : ops_) {
    res.merge(node->trainable_params());
  }
  return res;
}

int Net::BuildGrad(Node *optimizer) {
  std::set<Node *> learn = optimizer->trainable_params();
  auto NetOrder = Sort();
  optimizer_.reset(optimizer);
  optimizer->AddNetOutput(&output_);
  std::map<std::pair<EXPR *, EXPR *>, EXPR *> backprop;
  for (auto itr = NetOrder.rbegin(); itr != NetOrder.rend(); itr++) {
    Node *node = *itr;
    EXPR *yt = nullptr;
    if (node->primitive() == schema::PrimitiveType_NONE) continue;
    if (outmap_.find(node->expr()) == outmap_.end() || outmap_[node->expr()].size() == 0) {
      yt = node->expr();
    } else {
      std::vector<EXPR *> add_params;
      for (auto &output : outmap_[node->expr()]) {
        auto link = std::make_pair(node->expr(), output);
        auto grad = backprop[link];
        add_params.push_back(grad);
      }
      if (add_params.size() == 1) {
        yt = add_params.front();
      } else {
        auto addn = new (std::nothrow) AddN(0);
        if (addn == nullptr) {
          MS_LOG(ERROR) << "Cannot allocate add operator";
          return RET_ERROR;
        }
        PushOp(addn);
        addn->update_name(name());
        yt = (*addn)(add_params).front();
      }
    }
    auto inGrads = node->Grad(yt);
    for (size_t i = 0; i < node->inputs().size(); i++) {
      EXPR *inGrad{nullptr};
      if (i < inGrads.size()) {
        inGrad = inGrads[i];
      } else {
        inGrad = nullptr;
      }
      auto input = node->input(i);
      if (learn.find(input->node()) != learn.end()) {
        auto opt = optimizer->Clone(inGrad, input);
        if (opt.size() == 0) {
          MS_LOG(ERROR) << "failed to create optimizer";
          return RET_ERROR;
        }
        if (inGrad == nullptr) {
          MS_LOG(ERROR) << "illegal null value for grad";
          return RET_ERROR;
        }
        if (opt.size() == 0) {
          MS_LOG(ERROR) << "optimizer for " << input->node()->name() << " failure";
          return RET_ERROR;
        }
        auto opt_op = opt.at(0)->node();
        PushOp(opt_op);
        opt_op->update_name(node->name());
        output_.push_back(opt.at(0));
      }
      auto link = std::make_pair(input, node->expr());
      backprop[link] = inGrad;
    }
  }
  return RET_OK;
}

std::vector<EXPR *> Net::add(const std::vector<EXPR *> &input) {
  auto _add = NN::Add();
  _add->set_name(name() + "/" + _add->name());
  ops_.push_back(_add);
  return (*_add)(input);
}

Net *Net::TrainNet(Node *optimizer, Node *loss_fn, const std::vector<EXPR *> &inputs) {
  auto net = new (std::nothrow) NetWithLoss(this, loss_fn);
  if (net == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate loss network";
    return nullptr;
  }
  return net->TrainNet(optimizer, inputs);
}

Net *Net::TrainNet(Node *optimizer, const std::vector<EXPR *> &inputs) {
  auto x = (*this)(inputs);
  auto res = BuildGrad(optimizer);
  if (res != RET_OK) {
    MS_LOG(ERROR) << "Build gradient network failed";
    return nullptr;
  }
  real_output_ = x;
  return this;
}

int Net::Save(const schema::MetaGraphT &graph, std::string file_name) { return Storage::Save(graph, file_name); }

const std::vector<int> Net::OutputShape(int idx) {
  if (static_cast<size_t>(idx) >= real_output_.size()) {
    MS_LOG(ERROR) << "index (" << idx << ") exceed output size (" << real_output_.size() << ")";
    return {};
  }
  return real_output_.at(idx)->dims();
}

const std::vector<int> Net::InputShape(int idx) {
  if (static_cast<size_t>(idx) >= input_.size()) {
    MS_LOG(ERROR) << "index (" << idx << ") exceed input size (" << input_.size() << ")";
    return {};
  }
  return input_.at(idx)->dims();
}

Net::~Net() {
  if (impl_ != nullptr) {
    impl_->erase_net();
    auto pnet = impl_->pnet();
    if (pnet != nullptr) {
      impl_->set_pnet(nullptr);
    }
  }
  impl_ = nullptr;
}
}  // namespace lite
}  // namespace mindspore
