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

#include "parallel/auto_parallel/rec_core/rec_cost.h"
#include "parallel/auto_parallel/rec_core/rec_graph.h"
#include "parallel/auto_parallel/rec_core/rec_strategy.h"
#include "parallel/status.h"

namespace mindspore {
namespace parallel {
std::vector<size_t> SortByWeight(const std::shared_ptr<Graph> graph);

double GetWeights(const Graph::NodeType &node);

StrategyRec PartitionNode(const Graph::NodeType &node,
                          const std::vector<std::pair<std::string, StrategyRec>> &node_name_to_strategy,
                          std::shared_ptr<Graph> graph);

Status PartitionForAllDevices(const size_t num_device, const double device_memory, std::shared_ptr<Graph> graph);

Graph::NodeType ApplyStrToTensor(Graph::NodeType Node);

Status DevicesMemoryControl(const double device_memory, std::shared_ptr<Graph> graph);

size_t GetDataTypeSize(const TensorType &type);
}  // namespace parallel
}  // namespace mindspore

#endif  // PARALLEL_AUTO_PARALLEL_REC_PARTITION_H_
