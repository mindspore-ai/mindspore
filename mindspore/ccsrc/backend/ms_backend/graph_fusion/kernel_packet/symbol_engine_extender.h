/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_KERNEL_PACKET_SYMBOL_ENGINE_EXTENDER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_KERNEL_PACKET_SYMBOL_ENGINE_EXTENDER_H_

#include <string>
#include <vector>
#include "include/backend/common/pass_manager/pass.h"
#include "abstract/symbolic_shape/operation_builder.h"

namespace mindspore::graphkernel::packet {
// Extend kernel to a bigger subgraph using a symbol engine,
// to include all the nodes that do shape calc for the kernel.
class SymbolEngineExtender : public opt::Pass {
 public:
  SymbolEngineExtender() : Pass("symbol_engine_extender") {}
  ~SymbolEngineExtender() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  bool CheckBaseNode(const AnfNodePtr &node) const;
  AnfNodePtrList FindCandidates(const CNodePtr &base_node);
  void SearchInputs(const CNodePtr &node, const std::vector<symshape::DependOn> &depends, size_t depth);
  void FindShapeDependNode(const CNodePtr &node, size_t depth);
  void FindValueDependNode(const CNodePtr &node, size_t depth);
  bool ExtendNode(const AnfNodePtr &node, const FuncGraphPtr &main_fg);
  void RemoveWildGetitem();
  ValuePtr FindOnlyDependShapeInputs(const FuncGraphPtr &fg) const;
  bool IsValidNode(const CNodePtr &node) const;
  CNodePtr CreatePacketNode(const FuncGraphPtr &main_fg, const FuncGraphPtr &sub_fg,
                            const AnfNodePtrList &inputs) const;
  void ProcessNopNode(const FuncGraphPtr &fg, AnfNodePtrList *inputs) const;
  void ConstValueToParam(const FuncGraphPtr &fg, AnfNodePtrList *inputs) const;

  HashSet<AnfNodePtr> visited_;
  HashSet<AnfNodePtr> valid_nodes_;
};
}  // namespace mindspore::graphkernel::packet
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_KERNEL_PACKET_SYMBOL_ENGINE_EXTENDER_H_
