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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_CLUSTER_CLOUD_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_CLUSTER_CLOUD_H_
#include <vector>
#include "backend/ms_backend/graph_fusion/core/graph_kernel_cluster.h"

namespace mindspore::graphkernel {
class StaticShapeCluster : public GraphKernelCluster {
 public:
  StaticShapeCluster() = default;
  ~StaticShapeCluster() override = default;
  static std::vector<PrimitivePtr> GetClusterOps();
  static bool CanClusterableOp(const AnfNodePtr &node, const std::vector<PrimitivePtr> &op_list);

 protected:
  std::vector<PrimitivePtr> GetClusterableOpList() override;
  bool IsClusterableOp(const AnfNodePtr &node) override;
};

class DynamicShapeCluster : public GraphKernelCluster {
 public:
  DynamicShapeCluster() : GraphKernelCluster("graph_kernel_cluster_for_dynshape") {}
  ~DynamicShapeCluster() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 protected:
  std::vector<PrimitivePtr> GetClusterableOpList();
  bool IsClusterableOp(const AnfNodePtr &node) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_CLUSTER_CLOUD_H_
