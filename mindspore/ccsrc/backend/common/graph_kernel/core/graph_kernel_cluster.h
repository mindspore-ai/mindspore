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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_CLUSTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_CLUSTER_H_

#include <vector>
#include <memory>
#include <sstream>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "include/backend/optimizer/pass.h"

namespace mindspore {
namespace graphkernel {
class Graph;
using GraphPtr = std::shared_ptr<Graph>;
class GraphKernelCluster : public opt::Pass {
 public:
  GraphKernelCluster() : Pass("graph_kernel_cluster") {}
  ~GraphKernelCluster() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
  static std::vector<PrimitivePtr> GetClusterOps();

 protected:
  virtual std::vector<PrimitivePtr> GetClusterableOpList();
  virtual bool IsClusterableOp(const AnfNodePtr &node);

 private:
  void Init(const FuncGraphPtr &func_graph);
  bool Process(const FuncGraphPtr &func_graph);
  std::vector<size_t> FindCandidates(size_t basenode_id);
  void RemoveWildGetitem(std::vector<size_t> *candidates);
  void CreateFuncGraph(const FuncGraphPtr &func_graph, const std::vector<size_t> &nodes_id);
  void ConvertInputToAttrForClusterGraph(const AnfNodePtr &graph_kernel_node);
  void DumpClusterInfo(const AnfNodePtrList &old_nodes, const AnfNodePtr &new_node);
  void DumpToFile();
  void Clean() {
    nodes_.clear();
    node_idx_map_.clear();
    graph_ = nullptr;
  }

  GraphPtr graph_{nullptr};
  std::vector<AnfNodePtr> nodes_;
  mindspore::HashMap<AnfNodePtr, size_t> node_idx_map_;
  std::stringstream dump_buf_;
  std::vector<PrimitivePtr> op_list_;
};
}  // namespace graphkernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_CLUSTER_H_
