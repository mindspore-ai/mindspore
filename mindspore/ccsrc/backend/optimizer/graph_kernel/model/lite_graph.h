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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_LITE_GRAPH_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_LITE_GRAPH_H_

#include <memory>
#include <vector>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <string>
#include "backend/optimizer/graph_kernel/model/node.h"
#include "backend/optimizer/graph_kernel/model/op_node.h"

namespace mindspore {
namespace opt {
namespace graphkernel {
class LiteGraph {
 public:
  class GraphBuilder;
  explicit LiteGraph(const std::string &name = "") : name_(name), output_(new OutputNode()) {}
  ~LiteGraph() = default;
  NodePtr &Add(PrimOpPtr op) {
    ops_.emplace_back(op);
    return ops_.back();
  }

  const NodePtrList &GetOrderedNodes();

  std::string Dump() const;
  const std::string &name() const { return name_; }
  const NodePtrList &ops() const { return ops_; }
  const NodePtrList &inputs() const { return inputs_; }
  const NodePtr &output() const { return output_; }
  const NodePtrList &GetOutputs() const { return output_->inputs(); }

 protected:
  std::string name_;
  NodePtrList ops_;  // save all operators in topo order
  NodePtrList inputs_;
  NodePtr output_;

 private:
  int name_id_{0};
};
using LiteGraphPtr = std::shared_ptr<LiteGraph>;

class LiteGraph::GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name = "") { graph_ = std::make_shared<LiteGraph>(name); }
  ~GraphBuilder() = default;
  NodePtr Parameter(const NodeBase &baseinfo, std::string name = "") {
    if (name.empty()) name = NewName();
    auto para = std::make_shared<ParamNode>(name, baseinfo);
    graph_->inputs_.push_back(para);
    return para;
  }
  NodePtr Value(const tensor::TensorPtr &data, const std::string &name = "") {
    return std::make_shared<ConstTensorNode>(data, name);
  }

  void SetOutputs(const NodePtrList &nodes) { graph_->output_->SetInputs(nodes); }

  NodePtr Emit(const std::string &op, const NodePtrList &inputs, const DAttrs &attrs = {}, std::string node_name = "");
  NodePtr Op(const std::string &op, const NodeBase &baseinfo, const NodePtrList &inputs, const DAttrs &attrs = {},
             std::string node_name = "");
  LiteGraphPtr Get() { return graph_; }

 private:
  PrimOpPtr CreateOp(const std::string &id, const std::string &name);
  std::string NewName(std::string prefix = "output_") { return prefix + std::to_string(graph_->name_id_++); }

  LiteGraphPtr graph_;
};
}  // namespace graphkernel
}  // namespace opt
}  // namespace mindspore
#endif
