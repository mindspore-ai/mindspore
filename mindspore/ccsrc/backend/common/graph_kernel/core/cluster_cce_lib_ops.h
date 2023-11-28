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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_CLUSTER_CCE_LIB_OPS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_CLUSTER_CCE_LIB_OPS_H_
#include <vector>
#include "ir/func_graph.h"
#include "include/backend/optimizer/pass.h"
#include "backend/common/graph_kernel/core/graph_kernel_cluster.h"

namespace mindspore::graphkernel {
class ClusterCceLibOps : public GraphKernelCluster {
  /**
   * @brief This pass will cluster op in to subgraph and mark the cnode with "use_akg_cce" attr
   *  in order to run akg cce lib.
   *  This pass will only cluster one node into one subgraph.
   *  This subgraph will not cluster new ops in  `graph_kernel_cluster_lite` pass.
   */
 public:
  ClusterCceLibOps() : GraphKernelCluster("cluster_cce_lib_ops") {}
  ~ClusterCceLibOps() override = default;
  std::vector<PrimitivePtr> GetClusterableOpList() override;
  bool IsClusterableOp(const AnfNodePtr &node) override;
  void CreateFuncGraph(const FuncGraphPtr &func_graph, const std::vector<size_t> &nodes_id) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_CLUSTER_CCE_LIB_OPS_H_
