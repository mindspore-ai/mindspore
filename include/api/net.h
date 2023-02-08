/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_INCLUDE_API_NET_H
#define MINDSPORE_INCLUDE_API_NET_H

#include <memory>
#include <vector>
#include <unordered_set>
#include <string>
#include "include/api/types.h"
#include "include/api/data_type.h"
#include "include/api/cfg.h"

namespace mindspore {
/// \brief Register node or sub network
#define REG(_name) Register(_name, #_name)

class Expr;
class NodeImpl;
class NetImpl;
class NodeSet;
class Graph;
class NetData;

class MS_API NetBase {
 public:
  NetBase() = default;
  virtual std::vector<Expr *> operator()(const std::vector<Expr *> &inputs) = 0;
  virtual uint32_t type() = 0;
};

class MS_API Node : public NetBase {
 public:
  Node();
  virtual ~Node();
  /// \brief Create output expression from node

  /// \param[in] name Name of input (like "labels" etc.)
  ///
  /// \return Expression
  Expr *Create(std::string name);
  /// \brief Run node on inputs. This operator is used in Net::construct()
  ///
  /// \param[in] inputs Inputs expression for the node.
  /// \return Output node expression vector
  std::vector<Expr *> operator()(const std::vector<Expr *> &inputs) override;
  uint32_t type() final;

 private:
  friend NodeImpl;
  std::shared_ptr<NodeImpl> impl_ = nullptr;
};

class MS_API Net : public NetBase, public std::enable_shared_from_this<Net> {
 public:
  Net();
  virtual ~Net();
  explicit Net(std::string name);
  explicit Net(const Graph &g);
  /// \brief Define the relation between network inputs and outputs
  ///
  /// \param[in] inputs expression vector
  ///
  /// \return expression vector

  virtual std::vector<Expr *> construct(const std::vector<Expr *> &inputs);
  /// \brief Addition operation
  ///
  /// \param[in] inputs Two elements to add
  ///
  /// \return expression vector (single element)

  /// \brief Execution operator. Connect inputs to outputs via user defined construct
  ///
  /// \return expression vector

  std::vector<Expr *> operator()(const std::vector<Expr *> &inputs);
  void Register(Net *net, std::string &&name);
  void Register(Node *node, std::string &&name);
  /// \brief Find the trainable params for the trained network
  ///
  /// \return NodeSet for all trainable nodes
  std::shared_ptr<NodeSet> trainable_params();
  virtual void Add(NetBase *element);
  /// \brief Input shape
  ///
  /// \param[in] idx input index
  ///
  /// \return Specific input shape vector
  const std::vector<int> InputShape(int idx);
  /// \brief Output shape
  ///
  /// \param[in] idx Output index
  ///
  /// \return Specific output shape vector
  const std::vector<int> OutputShape(int idx);
  uint32_t type() final;

 private:
  friend NetImpl;
  friend NetData;
  std::shared_ptr<NetImpl> impl_;
};

class MS_API SoftMaxCrossEntropyCfg {
 public:
  std::string reduction = "mean"; /**<  Specifies reduction mode. The optional values are "none", "mean", "sum" */
};

class MS_API AdamConfig {
 public:
  float learning_rate_ = 1e-3;
  float beta1_ = 0.9;
  float beta2_ = 0.999;
  float eps_ = 1e-08;
  bool use_nesterov_ = false;
};

namespace NN {
MS_API Net *NetWithLoss(Net *net, Node *loss);
MS_API Graph *GraphWithLoss(Graph *g, Node *loss);
MS_API Node *Adam(std::shared_ptr<NodeSet> learn, const AdamConfig &cfg);
MS_API Node *SoftmaxCrossEntropy(const SoftMaxCrossEntropyCfg &cfg);
MS_API std::unique_ptr<Node> Input(std::vector<int> dims, DataType data_type = DataType::kNumberTypeFloat32,
                                   int fmt = NHWC);
};  // namespace NN
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_NET_H
