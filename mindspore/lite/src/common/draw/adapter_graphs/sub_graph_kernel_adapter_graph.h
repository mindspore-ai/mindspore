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
#ifndef MINDSPORE_LITE_SRC_COMMON_DRAW_ADAPTER_GRAPHS_SUB_GRAPH_KERNEL_ADAPTER_GRAPH_H_
#define MINDSPORE_LITE_SRC_COMMON_DRAW_ADAPTER_GRAPHS_SUB_GRAPH_KERNEL_ADAPTER_GRAPH_H_

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
#include "src/litert/kernel_exec_util.h"
#include "src/executor/kernel_exec.h"
#include "src/executor/sub_graph_kernel.h"
#include "src/common/draw/adapter_graphs/drawer_mark_filter.h"

namespace mindspore::lite {
class KernelExecAdapterNode : public AdapterNode {
 public:
  explicit KernelExecAdapterNode(const kernel::KernelExec *kernel, MarkFilter mark_filter = nullptr)
      : kernel_(kernel), filter_(std::move(mark_filter)) {}

  std::string GetName() const override { return kernel_->name(); }
  std::vector<Tensor *> GetInputs() const override { return kernel_->in_tensors(); }
  Tensor *GetInput(const size_t &index) const override {
    if (index >= InputSize()) {
      return nullptr;
    }
    return kernel_->in_tensors()[index];
  }
  size_t InputSize() const override { return kernel_->in_tensors().size(); }
  std::vector<Tensor *> GetOutputs() const override { return kernel_->out_tensors(); }
  Tensor *GetOutput(const size_t &index) const override {
    if (index >= OutputSize()) {
      return nullptr;
    }
    return kernel_->out_tensors()[index];
  }
  size_t OutputSize() const override { return kernel_->out_tensors().size(); }

  bool IsHighlight() const override {
    if (filter_ == nullptr) {
      return false;
    }
    return filter_(*kernel_);
  }

 private:
  const kernel::KernelExec *kernel_;
  const MarkFilter filter_;
};

class SubGraphKernelAdapterGraph : public AdapterGraph {
 public:
  static std::shared_ptr<SubGraphKernelAdapterGraph> Create(const kernel::SubGraphKernel *graph,
                                                            const MarkFilter &mark_filter = nullptr) {
    auto adapter_graph = std::make_shared<SubGraphKernelAdapterGraph>(graph);
    auto nodes = graph->immutable_nodes();
    auto ret = kernel::KernelExecUtil::TopologicalSortNodes(&nodes, graph->in_nodes());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "TopologicalSortNodes failed!";
      return nullptr;
    }
    for (auto node : nodes) {
      auto *adapter_node = new (std::nothrow) KernelExecAdapterNode(node, mark_filter);
      if (adapter_node == nullptr) {
        MS_LOG(ERROR) << "new KernelExecAdapterNode failed! Please check whether memory is enough!";
        return nullptr;
      }
      adapter_graph->nodes_.emplace_back(adapter_node);
    }
    return adapter_graph;
  }

  explicit SubGraphKernelAdapterGraph(const kernel::SubGraphKernel *graph) : graph_(graph) {}
  ~SubGraphKernelAdapterGraph() override {
    for (auto node : nodes_) {
      delete node;
    }
    nodes_.clear();
  }
  std::string GetName() const override { return graph_->name(); }
  std::vector<AdapterNode *> GetNodes() const override { return nodes_; }
  std::vector<Tensor *> GetInputs() const override { return graph_->in_tensors(); }
  size_t InputSize() const override { return graph_->in_tensors().size(); }
  std::vector<Tensor *> GetOutputs() const override { return graph_->out_tensors(); }
  size_t OutputSize() const override { return graph_->out_tensors().size(); }

 private:
  const kernel::SubGraphKernel *graph_;
  std::vector<AdapterNode *> nodes_;
};

std::shared_ptr<GVGraph> CreateGVGraph(const kernel::SubGraphKernel *graph, const MarkFilter &mark_filter = nullptr) {
  auto adapter_graph = SubGraphKernelAdapterGraph::Create(graph, mark_filter);
  if (adapter_graph == nullptr) {
    MS_LOG(ERROR) << "Create SubGraphKernelAdapterGraph failed.";
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

#endif  // MINDSPORE_LITE_SRC_COMMON_DRAW_ADAPTER_GRAPHS_SUB_GRAPH_KERNEL_ADAPTER_GRAPH_H_
#endif
