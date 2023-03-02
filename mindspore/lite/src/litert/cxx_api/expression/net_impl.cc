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

#include "src/litert/cxx_api/expression/net_impl.h"
#include <vector>
#include <utility>
#include "include/api/serialization.h"
#include "src/expression/import.h"
#include "src/expression/ops.h"
#include "src/litert/cxx_api/model/model_impl.h"

namespace {
constexpr size_t kFlatbuffersBuilderInitSize = 1024;
};

namespace mindspore {
Sequential::Sequential() {}

lite::Node *Sequential::GetNode(NetBase *element) {
  lite::Node *lite_node = nullptr;
  switch (element->type()) {
    case kNodeType: {
      Node *node = reinterpret_cast<Node *>(element);
      auto impl = NodeImpl::GetImpl(node);
      if (impl == nullptr) {
        MS_LOG(ERROR) << "cannot find node implement";
        return nullptr;
      }
      lite_node = impl->node();
      break;
    }
    case kNetType: {
      auto net = reinterpret_cast<Net *>(element);
      auto impl = NetImpl::GetImpl(net);
      if (impl == nullptr) {
        MS_LOG(ERROR) << "cannot find node implement";
        return nullptr;
      }
      lite_node = impl->net();
      break;
    }
  }
  return lite_node;
}

void Sequential::Add(NetBase *element) {
  lite::Node *node = GetNode(element);
  auto impl = NetImpl::GetImpl(this);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "No implementation";
    return;
  }
  impl->net()->Add(node);
}

NetWithLoss::NetWithLoss(Net *net, Node *loss) : net_(net), loss_fn_(loss) {
  REG(net_);
  Register(loss_fn_, "_loss_fn");
}

std::vector<Expr *> NetWithLoss::construct(const std::vector<Expr *> &inputs) {
  if (inputs.size() != C2NUM) {
    MS_LOG(ERROR) << "need 2 inputs for loss";
    return {};
  }
  auto input = inputs[FIRST_INPUT];
  auto label = inputs[SECOND_INPUT];
  auto x = (*net_)({input});
  x = (*loss_fn_)({x[FIRST_INPUT], label});
  return x;
}

NetImpl::NetImpl(std::shared_ptr<Net> p) { pnet_ = p; }

NetImpl::NetImpl(Graph *g) { pnet_ = g->net_data_->net(); }

std::vector<lite::EXPR *> NetImpl::construct(const std::vector<lite::EXPR *> &inputs) {
  auto in = Expr::convert(inputs);
  auto out = pnet_->construct(in);
  return Expr::convert(out);
}

Net *NetImpl::Connect(std::shared_ptr<Net> net, lite::Net *lnet) {
  auto impl = GetImpl(net.get());
  if (impl == nullptr) {
    MS_LOG(ERROR) << "missing implementation";
    return nullptr;
  }
  impl->set_pnet(net);
  lnet->set_impl(impl);
  impl->set_net(lnet);
  return net.get();
}

Status NetImpl::Import(const char *model_buf, Graph *graph) {
  auto mg = schema::GetMetaGraph(model_buf);
  auto net = new (std::nothrow) Net();
  if (net == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate network";
    return kLiteMemoryFailed;
  }
  lite::Import import;
  auto lite_net = import.ImportMs(mg);
  if (lite_net == nullptr) {
    MS_LOG(ERROR) << "failed to import net";
    return kLiteMemoryFailed;
  }
  lite_net->SetRealOutput();
  Connect(net->shared_from_this(), lite_net);
  *graph = Graph(net);
  return kSuccess;
}

Status NetImpl::TrainNet(Node *optimizer, const std::vector<Expr *> &inputs) {
  auto impl = NodeImpl::GetImpl(optimizer);
  if (impl == nullptr) {
    MS_LOG(ERROR) << "missing implementation ";
    return kLiteNullptr;
  }
  auto opt = impl->node();
  auto in = Expr::convert(inputs);
  auto ret_net = net()->TrainNet(opt, in);
  if (ret_net == nullptr) {
    MS_LOG(ERROR) << "failed to train network";
    return kLiteNullptr;
  }
  return kSuccess;
}

std::unique_ptr<Graph> NetImpl::MakeMs() {
  auto mgraph = std::make_unique<Graph>(Graph::Type::kExecutableGraph);
  if (mgraph == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate graph";
    return nullptr;
  }
  auto trained_graph = net()->MakeMs();
  if (trained_graph == nullptr) {
    MS_LOG(ERROR) << "cannot create flat buffer";
    return nullptr;
  }
  flatbuffers::FlatBufferBuilder builder(kFlatbuffersBuilderInitSize);
  auto offset = schema::MetaGraph::Pack(builder, trained_graph.get());
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  auto buffer = builder.GetBufferPointer();
  size_t size = builder.GetSize();
  auto status = Serialization::Load(buffer, size, mindspore::kMindIR, mgraph.get());
  if (status != kSuccess) {
    MS_LOG(ERROR) << "failed to load flatbuffer to graph";
    return nullptr;
  }
  return mgraph;
}

const std::vector<int> NetImpl::InputShape(int idx) { return net_->InputShape(idx); }

const std::vector<int> NetImpl::OutputShape(int idx) { return net_->OutputShape(idx); }

void NetImpl::ReplaceNet(Graph *g, std::shared_ptr<Net> n) { g->net_data_->net().swap(n); }

ExpressionLoader expression_registrator = CreateExpressionLoader(NetImpl::Import);

namespace NN {
Net *Sequential() {
  auto net = new (std::nothrow) mindspore::Sequential();
  if (net == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate ";
    return nullptr;
  }
  auto netl = lite::NN::Sequential();
  return NetImpl::Connect(net->shared_from_this(), netl);
}

Net *NetWithLoss(Net *net, Node *loss) {
  auto loss_net = new (std::nothrow) mindspore::NetWithLoss(net, loss);
  if (net == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate loss net";
    return nullptr;
  }
  return loss_net;
}

Graph *GraphWithLoss(Graph *graph, Node *loss) {
  auto net = NetImpl::GetNet(*graph);
  if (net == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate network";
    return nullptr;
  }
  auto loss_net = NetWithLoss(net.get(), loss);
  if (loss_net == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate network";
    return nullptr;
  }
  NetImpl::ReplaceNet(graph, loss_net->shared_from_this());
  return graph;
}

Net *NetWithLoss(Graph *g, Node *loss) {
  auto net = new (std::nothrow) Net(*g);
  if (net == nullptr) {
    MS_LOG(ERROR) << "Cannot allocate net";
    return nullptr;
  }
  return NetWithLoss(net, loss);
}
}  // namespace NN
}  // namespace mindspore
