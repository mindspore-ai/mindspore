/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_OPTIMIZER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_OPTIMIZER_H_
#include <vector>
#include "schema/inner/model_generated.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
using namespace schema;
template <typename T>
class Pass {
 public:
  Pass() = default;
  virtual ~Pass() = default;
  virtual STATUS Run(T *t) = 0;
};

class GraphPass : public Pass<schema::MetaGraphT> {
 public:
  GraphPass() = default;

  ~GraphPass() override = default;

  STATUS Run(schema::MetaGraphT *graph) override = 0;
};

struct GraphNode {
  GraphNode(schema::MetaGraphT *subGraph, schema::CNodeT *opDefT) : sub_graph_(subGraph), op_def_(opDefT) {}
  ~GraphNode() = default;
  schema::MetaGraphT *sub_graph_ = nullptr;
  schema::CNodeT *op_def_ = nullptr;
};

class NodePass : public Pass<GraphNode> {
 public:
  NodePass() = default;

  ~NodePass() override = default;

  STATUS Run(GraphNode *graphNode) override = 0;
};

class Optimizer {
 public:
  Optimizer() = default;

  virtual ~Optimizer();

  void AddPass(GraphPass *graph_pass);

  void AddPass(NodePass *node_pass);

  STATUS Run(schema::MetaGraphT *graph_defT);

 private:
  std::vector<GraphPass *> graph_passes_;
  std::vector<NodePass *> node_passes_;
};
}  // namespace lite
}  // namespace mindspore

#endif
