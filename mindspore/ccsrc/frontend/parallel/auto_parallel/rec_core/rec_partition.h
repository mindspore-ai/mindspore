/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef PARALLEL_AUTO_PARALLEL_REC_PARTITION_H_
#define PARALLEL_AUTO_PARALLEL_REC_PARTITION_H_

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/auto_parallel/rec_core/rec_cost.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_strategy.h"
#include "frontend/parallel/status.h"
#include "ir/func_graph.h"

namespace mindspore {
namespace parallel {
constexpr bool ENABLE_PIPE_ALGO = false;

enum PartitionOrder { TopologyOrder, WeightOrder };

constexpr PartitionOrder PARTITION_ORDER = PartitionOrder::TopologyOrder;

std::vector<size_t> SortByWeight(const std::shared_ptr<Graph> &graph);

double GetWeights(const Graph::NodeType &node);

StrategyRec PartitionNode(const Graph::NodeType &node,
                          const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                          const std::shared_ptr<Graph> &graph, const bool isTraining);

Status PartitionForAllDevices(const size_t num_device, const double device_memory, const std::shared_ptr<Graph> &graph,
                              const bool isTraining);

Graph::NodeType ApplyStrToTensor(Graph::NodeType Node);

Status DevicesMemoryControl(const size_t num_device, const double device_memory, const std::shared_ptr<Graph> &graph);

StrategyRec GetOneLoopStrategy(size_t op_inputs_num, const StrategyRec &old_str, StrategyRec new_str);

Graph::NodeType ChangeStrategy(Graph::NodeType Node, size_t n_cut);

size_t GetDataTypeSize(const TensorType &type);
}  // namespace parallel
}  // namespace mindspore

#endif  // PARALLEL_AUTO_PARALLEL_REC_PARTITION_H_
