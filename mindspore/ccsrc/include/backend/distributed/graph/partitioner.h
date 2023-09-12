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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_GRAPH_PARTITIONER_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_GRAPH_PARTITIONER_H_

#include <string>
#include <memory>
#include "ir/func_graph.h"
#include "include/backend/distributed/constants.h"
#include "include/backend/distributed/mode/distributed_label.h"
#include "include/backend/distributed/mode/execution_mode.h"

namespace mindspore {
namespace distributed {
class DistributedGraphPartitioner {
 public:
  DistributedGraphPartitioner() {}
  virtual ~DistributedGraphPartitioner();

 private:
  // Create DistributedExecutionMode's subclass object for current distributed traning job.
  // This method takes multiple elements such as context values, graph attributes, etc into consideration to judge what
  // execution mode this process is launching.
  DistributedExecutionModePtr GenerateExecMode();

  // Common graph partitioning procedure. After calling this method, each process only has the graph with corresponding
  // label.
  void DoGraphPartition();

  // Root graph and all other sub-graphs.
  FuncGraphPtr root_;
  FuncGraphSet fgs_;

  // Distributed execution mode pointer, it is core to graph partitioning. Customized operations like communication
  // operator creation and fusion, pre and post graph building, etc, are encapsulated in this object.
  DistributedExecutionModePtr exec_mode_;

  // Distributed labels of nodes and graphs used to generate distributed DAG.
  GraphPartitionLabels labels_;
};
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_GRAPH_PARTITIONER_H_
