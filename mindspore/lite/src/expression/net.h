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

#ifndef MINDSPORE_LITE_SRC_EXPRESSION_NET_H_
#define MINDSPORE_LITE_SRC_EXPRESSION_NET_H_
#include <stack>
#include <memory>
#include <set>
#include <map>
#include <utility>
#include <string>
#include <unordered_set>
#include <list>
#include <vector>
#include "src/expression/node.h"
#include "inner/model_generated.h"

namespace mindspore {
class Net;
class NetImpl;
namespace lite {
#define REG(_name) Register(_name, #_name)

class ExportSession;

class Net : public Node {
 public:
  Net() = default;
  virtual ~Net();
  explicit Net(std::string name) : Node(name) {}
  std::vector<EXPR *> construct(const std::vector<EXPR *> &inputs) override;
  std::vector<EXPR *> operator()(const std::vector<EXPR *> &inputs) override;
  std::vector<EXPR *> operator()(const std::initializer_list<EXPR *> &&inputs) override;
  void update_name(std::string name) override;
  Net *TrainNet(Node *optimizer, Node *loss_fn, const std::vector<EXPR *> &inputs);
  Net *TrainNet(Node *optimizer, const std::vector<EXPR *> &inputs);
  void PrintDot() { EXPR::PrintDot(output_); }

  void PushOutput(EXPR *e) { output_.push_back(e); }
  void PushInput(EXPR *e) { input_.push_back(e); }
  void SetRealOutput() { real_output_ = output_; }
  std::set<Node *> trainable_params() override;
  std::vector<Node *> Sort();
  int BuildGrad(Node *optimizer);
  int BuildGrad(Node *optimizer, std::set<Node *> learnable);
  std::unique_ptr<schema::MetaGraphT> MakeMs();
  std::unique_ptr<schema::MetaGraphT> MakeMs(std::string file_name);
  schema::MetaGraph *meta_graph() { return meta_graph_; }
  int Save(const schema::MetaGraphT &graph, const std::string filename);
  void set_impl(std::shared_ptr<mindspore::NetImpl> impl) { impl_ = impl; }
  const std::vector<int> InputShape(int idx);
  const std::vector<int> OutputShape(int idx);

 protected:
  std::vector<EXPR *> add(const std::vector<EXPR *> &input);
  void Register(Node *node, std::string &&name) {
    if (node != nullptr) {
      PushOp(node);
      node->update_name(name);
    }
  }

 private:
  friend mindspore::Net;
  std::unordered_set<Node *> visited_;
  std::map<EXPR *, std::list<EXPR *>> outmap_;  // outputs per expression
  std::map<EXPR *, std::list<EXPR *>> inmap_;   // inputs per expression
  std::vector<EXPR *> output_;                  // network output expression
  std::vector<EXPR *> real_output_;             // network output for export
  std::vector<EXPR *> input_;                   // network input expression
  schema::MetaGraph *meta_graph_;               // imported meta_graph
  std::unique_ptr<ExportSession> session_;      // export session
  std::unique_ptr<Node> optimizer_;
  void TopoSortUtil(Node *v, std::stack<Node *> *stack);
  void CreateOutputMap(std::vector<EXPR *> vec, std::map<Node *, std::list<Node *>> *outmap);
  std::shared_ptr<mindspore::NetImpl> impl_;
};

class NetWithLoss : public Net {
 public:
  NetWithLoss(Net *net, Node *loss) : net_(net), loss_fn_(loss) {
    REG(net_);
    REG(loss_fn_);
    loss_fn_->set_name("_loss_fn");
  }
  std::vector<EXPR *> construct(const std::vector<EXPR *> &inputs) {
    auto input = inputs[0];
    auto label = inputs[1];
    auto x = (*net_)({input});
    x = (*loss_fn_)({x[0], label});
    return {x.front()};
  }

 private:
  Net *net_{nullptr};
  Node *loss_fn_{nullptr};
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXPRESSION_NET_H_
