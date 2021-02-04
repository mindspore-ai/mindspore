/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_EXPANDER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_EXPANDER_H_
#include <memory>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include "backend/optimizer/common/pass.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace opt {
class GraphKernelExpander : public Pass {
 public:
  GraphKernelExpander() : Pass("graph_kernel_expander") {}
  ~GraphKernelExpander() override = default;
  bool Run(const FuncGraphPtr &func_graph);

 private:
  FuncGraphPtr CreateExpandFuncGraph(const CNodePtr &node);
  bool DoExpand(const FuncGraphPtr &func_graph);
  void EliminateRedundantParameters(const FuncGraphPtr &func_graph, AnfNodePtrList *inputs);
  AnfNodePtr CreateExpandGraphKernel(const FuncGraphPtr &func_graph, const FuncGraphPtr &new_func_graph,
                                     const CNodePtr &node);
  bool CanExpand(const CNodePtr &node) {
    return std::any_of(expand_ops_.begin(), expand_ops_.end(),
                       [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  }
  bool ExpandJsonInfo(const AnfNodePtr &node, nlohmann::json *kernel_json);

 private:
  std::unordered_set<PrimitivePtr> expand_ops_;
};
using GraphKernelExpanderPtr = std::shared_ptr<GraphKernelExpander>;
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_GRAPH_KERNEL_EXPANDER_H_
