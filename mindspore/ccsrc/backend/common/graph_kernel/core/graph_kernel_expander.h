/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_EXPANDER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_EXPANDER_H_
#include <memory>
#include <vector>
#include <string>
#include "include/backend/optimizer/pass.h"
#include "ir/func_graph.h"
#include "backend/common/graph_kernel/core/expander.h"
#include "backend/common/graph_kernel/expanders/op_desc_registry.h"

namespace mindspore::graphkernel {
class GraphKernelExpander : public opt::Pass {
 public:
  GraphKernelExpander() : Pass("graph_kernel_expander") {}
  explicit GraphKernelExpander(const std::string &name) : Pass(name) {}
  ~GraphKernelExpander() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  AnfNodePtr CreateExpandedNode(const CNodePtr &node, const std::string &name) const;
  virtual ExpanderPtr InitExpander(const AnfNodePtr &node) = 0;
  virtual std::vector<PrimitivePtr> InitOpList() = 0;
  bool DoExpand(const FuncGraphPtr &func_graph);
  virtual bool CanExpand(const CNodePtr &node) const {
    return std::any_of(expand_ops_.begin(), expand_ops_.end(),
                       [&node](const PrimitivePtr &prim) { return IsPrimitiveCNode(node, prim); });
  }
  virtual void PreProcessAllNode(const CNodePtr &node) {}

 private:
  std::vector<PrimitivePtr> expand_ops_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_EXPANDER_H_
