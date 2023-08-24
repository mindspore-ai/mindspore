/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifdef ENABLE_DRAW
#ifndef MINDSPORE_LITE_SRC_COMMON_DRAW_ADAPTER_GRAPHS_COMPILE_RESULT_ADAPTER_GRAPH_H_
#define MINDSPORE_LITE_SRC_COMMON_DRAW_ADAPTER_GRAPHS_COMPILE_RESULT_ADAPTER_GRAPH_H_

#include <utility>
#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <unordered_map>
#include "src/common/log_adapter.h"
#include "src/common/draw/adapter_graph.h"
#include "src/common/draw/graphviz_graph_builder.h"
#include "include/errorcode.h"
#include "src/extendrt/graph_compiler/compile_result.h"

namespace mindspore::lite {
class CompileNodeAdapterNode : public AdapterNode {
 public:
  explicit CompileNodeAdapterNode(CompileNodePtr node) : node_(std::move(node)) {}

  std::string GetName() const override { return node_->GetName(); }
  std::vector<Tensor *> GetInputs() const override { return node_->GetInputs(); }
  Tensor *GetInput(const size_t &index) const override {
    if (index >= InputSize()) {
      return nullptr;
    }
    return node_->GetInput(index);
  }
  size_t InputSize() const override { return node_->InputSize(); }
  std::vector<Tensor *> GetOutputs() const override { return node_->GetOutputs(); }
  Tensor *GetOutput(const size_t &index) const override {
    if (index >= OutputSize()) {
      return nullptr;
    }
    return node_->GetOutput(index);
  }
  size_t OutputSize() const override { return node_->OutputSize(); }

 private:
  const CompileNodePtr node_;
};

class CompileResultAdapterGraph : public AdapterGraph {
 public:
  static std::shared_ptr<CompileResultAdapterGraph> Create(const CompileResult *graph) {
    auto adapter_graph = std::make_shared<CompileResultAdapterGraph>(graph);
    for (const auto &node : graph->GetNodes()) {
      adapter_graph->nodes_.emplace_back(new CompileNodeAdapterNode(node));
    }
    return adapter_graph;
  }

  explicit CompileResultAdapterGraph(const CompileResult *graph) : graph_(graph) {}
  ~CompileResultAdapterGraph() override {
    for (auto node : nodes_) {
      delete node;
    }
    nodes_.clear();
  }
  std::string GetName() const override { return "CompileResult"; }
  std::vector<AdapterNode *> GetNodes() const override { return nodes_; }
  std::vector<Tensor *> GetInputs() const override { return graph_->GetInputs(); }
  size_t InputSize() const override { return graph_->InputSize(); }
  std::vector<Tensor *> GetOutputs() const override { return graph_->GetOutputs(); }
  size_t OutputSize() const override { return graph_->OutputSize(); }

 private:
  const CompileResult *graph_;
  std::vector<AdapterNode *> nodes_;
};

std::shared_ptr<GVGraph> CreateGVGraph(const CompileResult *graph) {
  auto adapter_graph = CompileResultAdapterGraph::Create(graph);
  if (adapter_graph == nullptr) {
    MS_LOG(ERROR) << "Create CompileResultAdapterGraph failed.";
    return nullptr;
  }
  GVGraphBuilder builder;
  auto gv_graph = builder.Build(adapter_graph);
  if (gv_graph == nullptr) {
    MS_LOG(ERROR) << "Build gv_graph failed.";
    return nullptr;
  }
  return gv_graph;
}
}  // namespace mindspore::lite

#endif  // MINDSPORE_LITE_SRC_COMMON_DRAW_ADAPTER_GRAPHS_COMPILE_RESULT_ADAPTER_GRAPH_H_
#endif
