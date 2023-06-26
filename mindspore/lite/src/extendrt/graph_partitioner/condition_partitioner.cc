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

#include <string>
#include <memory>
#include <stack>
#include <vector>
#include <map>
#include <queue>
#include <set>

#include "extendrt/graph_partitioner/condition_partitioner.h"

#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "utils/ms_context.h"

namespace mindspore {
std::vector<GraphSegmentPtr> ConditionPartitioner::Partition(const FuncGraphPtr &graph, bool *multi_target) {
  MS_EXCEPTION_IF_NULL(graph);
  auto nodes = TopoSort(graph->get_return());

  MS_LOG(DEBUG) << "Split all nodes size:" << nodes.size();
  bool contain_multi_target = ContainMultiTarget(nodes);
  if (multi_target != nullptr) {
    *multi_target = contain_multi_target;
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string default_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);

  std::vector<GraphSegmentPtr> segments;
  std::vector<AnfNodePtr> segment_nodes;
  std::map<AnfNodePtr, GraphSegmentPtr> node_to_segment;
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    auto separete_type = EvalSeparatorCondition(segment_nodes, node);
    switch (separete_type) {
      case kIsolatedNodeSeparate: {
        NodesToSegments(segment_nodes, &segments, &node_to_segment);
        segment_nodes.clear();
        segment_nodes.emplace_back(node);
        auto segment = std::make_shared<GraphSegment>(segment_nodes, true);
        segments.push_back(segment);
        segment_nodes.clear();
        break;
      }
      case kDirectSeparate: {
        NodesToSegments(segment_nodes, &segments, &node_to_segment);
        segment_nodes.clear();
        break;
      }
      case kNoSeparate: {
        segment_nodes.emplace_back(node);
        break;
      }
      default: {
        MS_LOG(ERROR) << "Separate graph failed";
        return std::vector<GraphSegmentPtr>{};
      }
    }
  }
  if (contain_multi_target) {
    AddSegmentDependency(graph, node_to_segment);
    RemoveUselessDependency(&segments);
  }
  return segments;
}

SeparateType ConditionPartitioner::EvalSeparatorCondition(const std::vector<AnfNodePtr> &prev_segment,
                                                          const AnfNodePtr &node) {
  for (auto separator : separators_) {
    MS_EXCEPTION_IF_NULL(separator);
    auto separate = separator->GraphSeparateCheck(prev_segment, node);
    if (separate == kNoSeparate) {
      continue;
    }
    return separate;
  }
  return kNoSeparate;
}

void ConditionPartitioner::NodesToSegments(const std::vector<AnfNodePtr> &segment_nodes,
                                           std::vector<GraphSegmentPtr> *segments,
                                           std::map<AnfNodePtr, GraphSegmentPtr> *node_to_segment) {
  if (segment_nodes.empty()) {
    return;
  }

  AddSegment(segment_nodes, segments, node_to_segment);
  return;
}

void ConditionPartitioner::AddSegment(const std::vector<AnfNodePtr> &nodes, std::vector<GraphSegmentPtr> *segments,
                                      std::map<AnfNodePtr, GraphSegmentPtr> *node_to_segment) {
  MS_EXCEPTION_IF_NULL(segments);
  MS_EXCEPTION_IF_NULL(node_to_segment);
  auto segment = std::make_shared<GraphSegment>(nodes, false);
  segments->emplace_back(segment);
  for (auto &node : nodes) {
    (*node_to_segment)[node] = segment;
  }
}

void ConditionPartitioner::AddSegmentDependency(const FuncGraphPtr &graph,
                                                const std::map<AnfNodePtr, GraphSegmentPtr> &node_to_segment) {
  MS_EXCEPTION_IF_NULL(graph);
  std::stack<AnfNodePtr> to_visit;
  std::map<AnfNodePtr, size_t> nodes_ref;
  CalcNodeRefCount(graph, &nodes_ref);
  to_visit.push(graph->get_return());
  while (!to_visit.empty()) {
    auto &node = to_visit.top();
    MS_EXCEPTION_IF_NULL(node);
    to_visit.pop();
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto node_inputs = cnode->inputs();
    GraphSegmentPtr node_segment{nullptr};
    auto node_iter = node_to_segment.find(node);
    if (node_iter != node_to_segment.end()) {
      node_segment = node_iter->second;
    }
    for (auto &input : node_inputs) {
      if (node_segment != nullptr && !node_segment->is_cut_ && input != nullptr && input->isa<CNode>()) {
        GraphSegmentPtr input_segment{nullptr};
        auto input_iter = node_to_segment.find(input);
        if (input_iter != node_to_segment.end()) {
          input_segment = input_iter->second;
        }
        if (input_segment != nullptr && input_segment != node_segment && !input_segment->is_cut_) {
          node_segment->AddPreSegment(input_segment);
        }
      }
      auto ref_iter = nodes_ref.find(input);
      if (ref_iter != nodes_ref.end()) {
        ref_iter->second--;
        if (ref_iter->second != 0) {
          continue;
        }
      }
      to_visit.push(input);
    }
  }
}

void ConditionPartitioner::RemoveUselessDependency(const std::vector<GraphSegmentPtr> *segments) {
  MS_EXCEPTION_IF_NULL(segments);
  for (auto &segment : *segments) {
    MS_EXCEPTION_IF_NULL(segment);
    if (segment->is_cut_) {
      continue;
    }
    bool total_virtual_node = true;
    for (auto &node : segment->nodes_) {
      if (IsPrimitiveCNode(node, prim::kPrimImageSummary) || IsPrimitiveCNode(node, prim::kPrimScalarSummary) ||
          IsPrimitiveCNode(node, prim::kPrimTensorSummary) || IsPrimitiveCNode(node, prim::kPrimHistogramSummary) ||
          IsPrimitiveCNode(node, prim::kPrimDepend) || IsPrimitiveCNode(node, prim::kPrimLoad) ||
          IsPrimitiveCNode(node, prim::kPrimUpdateState) || IsPrimitiveCNode(node, prim::kPrimMakeTuple) ||
          IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
        continue;
      }
      total_virtual_node = false;
      break;
    }
    if (total_virtual_node) {
      segment->pre_segments_.clear();
    }
  }
}

void ConditionPartitioner::CalcNodeRefCount(const FuncGraphPtr &graph, std::map<AnfNodePtr, size_t> *nodes_ref) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(nodes_ref);
  std::queue<AnfNodePtr> queue;
  queue.push(graph->get_return());
  std::set<AnfNodePtr> visited;
  while (!queue.empty()) {
    auto node = queue.front();
    queue.pop();
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (auto &input : cnode->inputs()) {
      auto iter = nodes_ref->find(input);
      if (iter != nodes_ref->end()) {
        iter->second++;
      } else {
        (void)nodes_ref->emplace(input, 1UL);
      }
      if (visited.find(input) != visited.end()) {
        continue;
      }
      visited.insert(input);
      queue.push(input);
    }
  }
}
}  // namespace mindspore
