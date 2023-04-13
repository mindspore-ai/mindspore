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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_PARTITIONER_CONDITION_PARTITIONER_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_PARTITIONER_CONDITION_PARTITIONER_H_

#include <memory>
#include <vector>
#include <map>

#include "extendrt/graph_partitioner/type.h"
#include "extendrt/graph_partitioner/graph_separator/type.h"

namespace mindspore {
class ConditionPartitioner : public GraphPartitioner {
 public:
  ConditionPartitioner() = default;
  ~ConditionPartitioner() = default;

  std::vector<GraphSegmentPtr> Partition(const FuncGraphPtr &graph, bool *multi_target = nullptr) override;

  void SetSeparators(std::vector<std::shared_ptr<GraphSeparator>> separator) { separators_ = separator; }

  std::vector<std::shared_ptr<GraphSeparetor>> GetSeparators() { return separators_; }

 private:
  SepareteType EvalSeparatorCondition(const std::vector<AnfNodePtr> &prev_segment, const AnfNodePtr &node);
  void NodesToSegments(const std::vector<AnfNodePtr> &segment_nodes, std::vector<GraphSegmentPtr> *segments,
                       std::map<AnfNodePtr, GraphSegmentPtr> *node_to_segment);
  void AddSegment(const std::vector<AnfNodePtr> &nodes, std::vector<GraphSegmentPtr> *segments,
                  std::map<AnfNodePtr, GraphSegmentPtr> *node_to_segment);
  void AddSegmentDependency(const FuncGraphPtr &graph, const std::map<AnfNodePtr, GraphSegmentPtr> &node_to_segment);
  void RemoveUselessDependency(const std::vector<GraphSegmentPtr> *segments);
  void CalcNodeRefCount(const FuncGraphPtr &graph, std::map<AnfNodePtr, size_t> *nodes_ref);

 private:
  std::vector<std::shared_ptr<GraphSeparator>> separators_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_GRAPH_PARTITIONER_CONDITION_PARTITIONER_H_
