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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_SPLITTER_REBUILD_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_SPLITTER_REBUILD_H_

#include <memory>
#include <vector>
#include <unordered_map>

#include "ir/func_graph.h"
#include "utils/hash_map.h"
#include "backend/ms_backend/graph_fusion/core/split_schemer.h"

namespace mindspore::graphkernel {
class Rebuilder {
 public:
  Rebuilder(const CNodePtr &main_cnode, const SplitSchemerPtr &split_schemer,
            const mindspore::HashMap<ParameterPtr, AnfNodePtr> &param_to_main_graph_node_map)
      : main_graph_(main_cnode->func_graph()),
        mng_(main_cnode->func_graph() != nullptr ? main_cnode->func_graph()->manager() : nullptr),
        main_cnode_(main_cnode),
        split_schemer_(split_schemer),
        param_to_main_graph_node_map_(param_to_main_graph_node_map) {}
  ~Rebuilder() = default;

  void Rebuild();

 private:
  CNodePtr InlineSubFuncGraph(const CNodePtr &main_node);
  void Inline();
  void ConnectToMainGraph();
  void ConnectSubGraphs();
  void SetSplitNodeName(const AnfNodePtr &callnode, size_t i) const;
  void CreateSubGraphs();

 private:
  FuncGraphPtr main_graph_;
  FuncGraphManagerPtr mng_;
  CNodePtr main_cnode_;
  SplitSchemerPtr split_schemer_;
  std::vector<CNodePtr> call_nodes_;
  std::unordered_map<AnfNodePtr, AnfNodePtr> old2new_;
  mindspore::HashMap<ParameterPtr, AnfNodePtr> param_to_main_graph_node_map_;
  std::vector<CNodePtr> need_inline_cnodes_;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_SPLITTER_REBUILD_H_
