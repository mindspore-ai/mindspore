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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_CXX_API_EXPRESSION_NET_IMPL_H_
#define MINDSPORE_LITE_SRC_RUNTIME_CXX_API_EXPRESSION_NET_IMPL_H_

#include <algorithm>
#include <set>
#include <memory>
#include <vector>
#include <utility>
#include "include/api/cfg.h"
#include "include/api/data_type.h"
#include "include/api/graph.h"
#include "include/api/status.h"
#include "include/api/net.h"
#include "src/litert/cxx_api/expression/node_impl.h"
#include "src/litert/cxx_api/graph/net_data.h"
#include "src/expression/net.h"
#include "src/expression/ops.h"

namespace mindspore {
constexpr uint32_t kNodeType = 1;
constexpr uint32_t kNetType = 2;
class Sequential : public Net {
 public:
  Sequential();
  void Add(NetBase *n) override;

 private:
  std::vector<NetBase *> ops_;
  lite::Node *GetNode(NetBase *element);
};

class NetWithLoss : public Net {
 public:
  NetWithLoss(Net *net, Node *loss);
  std::vector<Expr *> construct(const std::vector<Expr *> &inputs) override;

 private:
  Net *net_{nullptr};
  Node *loss_fn_{nullptr};
};

class MS_API NetImpl {
 public:
  virtual ~NetImpl() {}
  explicit NetImpl(std::shared_ptr<Net> p);
  explicit NetImpl(Graph *g);
  NetImpl() = default;
  void set_net(lite::Net *net) {
    if (net_ != nullptr) {
      net_->set_impl(nullptr);
      delete net_;
    }
    net_ = net;
  }
  void erase_net() { net_ = nullptr; }
  void set_pnet(std::shared_ptr<Net> net) { pnet_ = net; }
  Net *pnet() { return pnet_.get(); }
  lite::Net *net() { return net_; }

  std::vector<lite::EXPR *> construct(const std::vector<lite::EXPR *> &inputs);
  static std::shared_ptr<mindspore::NetImpl> &GetImpl(Net *net) { return net->impl_; }
  static Net *Connect(std::shared_ptr<Net> net, lite::Net *lnet);
  static std::shared_ptr<Net> &GetNet(const Graph &g) { return g.net_data_->net(); }
  static void SetNet(Graph *g, std::shared_ptr<Net> n) { g->net_data_->set_net(n); }
  static void ReplaceNet(Graph *g, std::shared_ptr<Net> n);
  static Status Import(const char *model_buf, Graph *graph);
  Status TrainNet(Node *optimizer, const std::vector<Expr *> &inputs);
  const std::vector<int> InputShape(int idx);
  const std::vector<int> OutputShape(int idx);
  std::unique_ptr<Graph> MakeMs();
  void Release() { pnet_.reset(); }

 private:
  std::shared_ptr<Net> pnet_;
  lite::Net *net_ = nullptr;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_CXX_API_EXPRESSION_NET_IMPL_H_
