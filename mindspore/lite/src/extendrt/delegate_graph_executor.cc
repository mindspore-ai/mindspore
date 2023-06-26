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
#include "src/extendrt/delegate_graph_executor.h"
#include <set>
#include <memory>
#include "src/extendrt/subgraph_kernel.h"
#include "ops/fusion/partial_fusion.h"
namespace mindspore {
// Graph sink delegate, the whole FuncGraph as a node to execute.
void GraphSinkDelegate::ReplaceNodes(const std::shared_ptr<FuncGraph> &graph) {
  sink_graph_ = graph;
  return;
}

bool GraphSinkDelegate::IsDelegateNode(const std::shared_ptr<CNode> &node) {
  auto partial_prim = std::make_shared<mindspore::ops::PartialFusion>();
  if (!IsPrimitiveCNode(node, partial_prim->GetPrim())) {
    return false;
  }
  auto graph = GetCNodeFuncGraph(node);
  if (graph.get() == sink_graph_.get()) {
    return true;
  }
  return false;
}

std::shared_ptr<kernel::KernelMod> GraphExecutorDelegate::CreateKernel(const std::shared_ptr<CNode> &node) {
  if (!IsDelegateNode(node)) {
    return nullptr;
  }
  auto kernel = std::make_shared<kernel::SubgraphKernel>(sink_graph_, executor_);
  return kernel;
}
}  // namespace mindspore
