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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_EXECUTION_MODE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_EXECUTION_MODE_H_

#include <string>
#include <vector>
#include <memory>
#include "ir/func_graph.h"
#include "include/backend/distributed/constants.h"
#include "include/backend/distributed/mode/distributed_label.h"
#include "include/backend/distributed/mode/inter_process_edge.h"

namespace mindspore {
namespace distributed {
// Base class for different execution modes. DistributedExecutionMode helps build distributed DAG and prepares for
// graphs partitioning. It encapsulates communication operator creation and fusion, pre and post graph building, etc.
class DistributedExecutionMode {
 public:
  DistributedExecutionMode() = default;
  virtual ~DistributedExecutionMode() = default;

 protected:
  // Dya nodes and graphs with distributed labels.
  virtual GraphPartitionLabels DyeGraph();

  // Prebuild graph before distributed graph partitioning.
  // Before common graph partitioning process, developer may also want to add/delete some nodes from the graph, or apply
  // some other optimization. This step is customized by developer so that they could do some modification to the graph
  // to meet their graph partitioning requirement.
  virtual void PreBuildDistributedGraph();

  // Postbuild graph after distributed graph partitioning.
  // Similar to PreBuildDistributedGraph, this method allows developer to customize some operations after common graph
  // partitioning process.
  virtual void PostBuildDistributedGraph();

  // Generate communication edges between processes according to distributed labels.
  virtual InterProcessEdgePtrList GenerateInterProcessEdges();

  // Lower communication traffic through operator fusion.
  virtual void DoCommOpFusion();

  // Root graph and all other sub-graphs.
  FuncGraphPtr root_;
  FuncGraphSet fgs_;

  // All communication edges in distributed DAG.
  InterProcessEdgePtrList comm_edges_;

  // Name of execution mode this process is running.
  std::string name_;
};
using DistributedExecutionModePtr = std::shared_ptr<DistributedExecutionMode>;
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_EXECUTION_MODE_H_
