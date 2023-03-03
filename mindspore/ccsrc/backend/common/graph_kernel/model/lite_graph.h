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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_LITE_GRAPH_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_MODEL_LITE_GRAPH_H_

#include <memory>
#include <string>
#include "backend/common/graph_kernel/model/node.h"

namespace mindspore::graphkernel::inner {
class LiteGraph {
 public:
  class GraphBuilderBase;
  explicit LiteGraph(const std::string &name = "") : name_(name), output_(new OutputNode()) {}

  const NodePtrList &GetOrderedNodes();
  std::string ToString(bool reset_node_name = false) const;
  const std::string &name() const { return name_; }
  const NodePtrList &ops() const { return ops_; }
  const NodePtrList &inputs() const { return inputs_; }
  const NodePtr &output(size_t i) const { return output_->input(i); }
  const NodePtrList &GetOutputs() const { return output_->inputs(); }

  void SetOutput(size_t i, const NodePtr &node) { output_->SetInput(i, node); }
  void SetOutputs(const NodePtrList &nodes) { output_->SetInputs(nodes); }

 protected:
  std::string name_;
  NodePtrList ops_;  // save all operators in topo order
  NodePtrList inputs_;
  NodePtr output_;

 private:
  std::string ParamName() const { return "input_" + std::to_string(param_id_++); }
  std::string NodeName() const { return "output_" + std::to_string(node_id_++); }
  mutable int param_id_{0};
  mutable int node_id_{0};
};
using LiteGraphPtr = std::shared_ptr<LiteGraph>;
class LiteGraph::GraphBuilderBase {
 public:
  explicit GraphBuilderBase(const std::string &name = "") { graph_ = std::make_shared<LiteGraph>(name); }
  ~GraphBuilderBase() = default;

  // Create a parameter of graph
  NodePtr Parameter(const NodeBase &baseinfo) const {
    auto para = std::make_shared<ParamNode>(baseinfo);
    para->SetDebugName(graph_->ParamName());
    graph_->inputs_.push_back(para);
    return para;
  }

  // Create a const value node
  NodePtr Value(const tensor::TensorPtr &data) const { return std::make_shared<ConstTensorNode>(data); }

  void SetOutputs(const NodePtrList &nodes) const { graph_->output_->SetInputs(nodes); }

  // Emit op, auto inferring the baseinfo of Node.
  NodePtr Emit(const std::string &op, const NodePtrList &inputs, const DAttrs &attrs = {}) const;

  // Create op node with given baseinfo.
  NodePtr Op(const std::string &op, const NodeBase &baseinfo, const NodePtrList &inputs,
             const DAttrs &attrs = {}) const;
  LiteGraphPtr Get() const { return graph_; }

 private:
  LiteGraphPtr graph_;
};
}  // namespace mindspore::graphkernel::inner
#endif
