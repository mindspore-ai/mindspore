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
#ifndef MINDSPORE_LITE_EXTENDRT_GRAPH_PARTITIONER_TYPE_H_
#define MINDSPORE_LITE_EXTENDRT_GRAPH_PARTITIONER_TYPE_H_

#include <vector>
#include <memory>

#include "ir/func_graph.h"

namespace mindspore {
enum GraphPartitionerType { kConditionPartitioner = 0, kCustomPartitioner, kNoneRuntime };

class GraphPartitioner : public std::enable_shared_from_this<GraphPartitioner> {
 public:
  virtual ~GraphPartitioner() = default;

  /// \brief Partition FuncGraph into several GraphSegment
  ///
  /// \param[in] graph FuncGraph need to partition.
  /// \param[out] multi_target if the graph need run on multi target
  ///
  /// \return list of GraphSegment for SubGraphs
  virtual std::vector<GraphSegmentPtr> Partition(const FuncGraphPtr &graph, bool *multi_target = nullptr) = 0;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_EXTENDRT_GRAPH_PARTITIONER_TYPE_H_
