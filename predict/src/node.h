/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PREDICT_SRC_NODE_H_
#define PREDICT_SRC_NODE_H_

#include <unordered_set>
#include <string>
#include <vector>
#include "include/session.h"
#include "src/op.h"

namespace mindspore {
namespace predict {
using NODE_ID = std::string;

class Node {
 public:
  Node() = default;
  explicit Node(const NodeDef *nodeDef);
  virtual ~Node();
  NODE_ID ID();
  std::string Type();
  void SetTensors(const NodeDef &nodeDef, const std::vector<Tensor *> &allTensors);
  void SetDepends(const std::unordered_set<NODE_ID> &deps);
  std::unordered_set<NODE_ID> GetDepends();

  void AddInEdge(Node *node);
  void AddOutEdge(Node *node);
  std::unordered_set<Node *> &GetAllOutEdges();
  std::unordered_set<Node *> &GetAllInEdges();

  std::vector<Tensor *> &GetOutputTensors();
  std::vector<Tensor *> &GetInputTensors();

  int InitOp(const OpDef &opDef, const Context &ctx);
  int Run(const Context &ctx);
  int MallocOutput(const Context &ctx);
  void FreeInput();

 protected:
  friend class GraphExecution;
  NODE_ID id;
  std::string type;
  OpBase *op{};
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs;
  std::unordered_set<NODE_ID> depends;
  std::unordered_set<Node *> inEdges;
  std::unordered_set<Node *> outEdges;
};
}  // namespace predict
}  // namespace mindspore

#endif  // PREDICT_SRC_NODE_H_
