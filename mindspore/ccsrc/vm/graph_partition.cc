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

#include "vm/graph_partition.h"
#include <string>
#include <functional>
#include <utility>
#include <map>
#include <queue>
#include <stack>
#include <set>
#include <algorithm>
#include "base/core_ops.h"
#include "utils/utils.h"
#include "utils/ms_context.h"
#include "ps/ps_context.h"
#ifdef ENABLE_GE
#include "transform/graph_ir/convert.h"
#endif
namespace mindspore {
const char kMsConvert[] = "ms";
const char kMsVm[] = "vm";
const char kGeVm[] = "ge";
namespace compile {
namespace {
std::string GetOtherTarget(const std::vector<AnfNodePtr> &nodes) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string default_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  for (auto &node : nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    std::string cur_target = GetCNodeTarget(node);
    if (cur_target != default_target) {
      return cur_target;
    }
  }
  return "";
}
bool ExtractNodes(const FuncGraphPtr &graph, const AnfNodePtr &prior_node, const AnfNodePtr &behind_node,
                  std::vector<AnfNodePtr> *prior_nodes, std::vector<AnfNodePtr> *depend_nodes) {
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(behind_node);
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  if (prior_node->isa<Parameter>()) {
    for (auto &user : node_users[prior_node]) {
      auto cnode = user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (!IsPrimitiveCNode(cnode, prim::kPrimControlDepend)) {
        prior_nodes->emplace_back(cnode);
      }
    }
  } else if (!IsPrimitiveCNode(prior_node, prim::kPrimControlDepend)) {
    prior_nodes->emplace_back(prior_node);
  } else {
    return false;
  }
  if (behind_node->isa<Parameter>()) {
    for (auto &user : node_users[behind_node]) {
      auto cnode = user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (!IsPrimitiveCNode(cnode, prim::kPrimControlDepend)) {
        depend_nodes->emplace_back(cnode);
      }
    }
  } else if (!IsPrimitiveCNode(behind_node, prim::kPrimControlDepend)) {
    depend_nodes->emplace_back(behind_node);
  } else {
    return false;
  }
  return true;
}

void AddControlEdge(const FuncGraphPtr &graph, const AnfNodePtr &node,
                    std::map<AnfNodePtr, std::vector<AnfNodePtr>> *control_edges,
                    std::map<AnfNodePtr, size_t> *nodes_ref) {
  MS_EXCEPTION_IF_NULL(node);
  auto input_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(input_cnode);
  auto prior_node = input_cnode->input(kControlDependPriorIndex);
  auto depend_node = input_cnode->input(kControlDependBehindIndex);
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(depend_node);
  auto prim_ptr = GetValueNode<PrimitivePtr>(input_cnode->input(0));
  MS_EXCEPTION_IF_NULL(prim_ptr);
  ValuePtr mode_ptr = prim_ptr->GetAttr("depend_mode");
  int64_t depend_mode = 0;
  if (mode_ptr != nullptr) {
    depend_mode = GetValue<int64_t>(mode_ptr);
  }
  if ((prior_node->isa<Parameter>() || depend_node->isa<Parameter>()) && depend_mode == 0) {
    return;
  }
  std::vector<AnfNodePtr> prior_nodes;
  std::vector<AnfNodePtr> behind_nodes;
  if (!ExtractNodes(graph, prior_node, depend_node, &prior_nodes, &behind_nodes)) {
    return;
  }
  for (auto &first_node : prior_nodes) {
    for (auto &second_node : behind_nodes) {
      MS_EXCEPTION_IF_NULL(first_node);
      MS_EXCEPTION_IF_NULL(second_node);
      auto iter = control_edges->find(second_node);
      if (iter == control_edges->end()) {
        (void)control_edges->insert(
          std::pair<AnfNodePtr, std::vector<AnfNodePtr>>(second_node, std::vector<AnfNodePtr>{first_node}));
      } else {
        iter->second.emplace_back(first_node);
      }
      auto ref_iter = nodes_ref->find(first_node);
      if (ref_iter != nodes_ref->end()) {
        ref_iter->second++;
      } else {
        (void)nodes_ref->insert(std::pair<AnfNodePtr, size_t>(first_node, 1));
      }
    }
  }
}

void CalcNodeRefCount(const FuncGraphPtr &graph, std::map<AnfNodePtr, size_t> *nodes_ref,
                      std::map<AnfNodePtr, std::vector<AnfNodePtr>> *control_edges) {
  std::queue<AnfNodePtr> queue;
  queue.push(graph->get_return());
  std::set<AnfNodePtr> visited;
  while (!queue.empty()) {
    auto &node = queue.front();
    queue.pop();
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (auto &input : cnode->inputs()) {
      if (IsPrimitiveCNode(input, prim::kPrimControlDepend)) {
        AddControlEdge(graph, input, control_edges, nodes_ref);
      }
      auto iter = nodes_ref->find(input);
      if (iter != nodes_ref->end()) {
        iter->second++;
      } else {
        (void)nodes_ref->insert(std::pair<AnfNodePtr, size_t>(input, 1));
      }
      if (visited.find(input) != visited.end()) {
        continue;
      }
      visited.insert(input);
      queue.push(input);
    }
  }
}

std::vector<AnfNodePtr> OptimizeGetItemOrder(const std::vector<AnfNodePtr> &nodes) {
  std::vector<AnfNodePtr> result;
  std::map<size_t, std::vector<AnfNodePtr>> insert_positions;
  std::map<AnfNodePtr, size_t> node_positions;
  for (auto &node : nodes) {
    if (node->isa<CNode>() && IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      auto cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto &inputs = cnode->inputs();
      if (inputs.size() < 2) {
        MS_LOG(EXCEPTION) << "Invalid get item node";
      }
      auto &parent = inputs[1];
      auto iter = node_positions.find(parent);
      if (iter != node_positions.end()) {
        size_t position = iter->second;
        auto iter_nodes = insert_positions.find(position);
        if (iter_nodes != insert_positions.end()) {
          iter_nodes->second.push_back(node);
        } else {
          (void)insert_positions.insert(
            std::pair<size_t, std::vector<AnfNodePtr>>(position, std::vector<AnfNodePtr>{node}));
        }
        continue;
      }
    }
    result.emplace_back(node);
    node_positions[node] = result.size();
  }

  size_t insert_num = 0;
  for (auto &item : insert_positions) {
    size_t position = item.first + insert_num;
    (void)result.insert(result.begin() + position, item.second.begin(), item.second.end());
    insert_num += item.second.size();
  }
  return result;
}

std::vector<AnfNodePtr> SplitSort(const FuncGraphPtr &graph, const std::string &default_target) {
  std::vector<AnfNodePtr> result;
  std::stack<AnfNodePtr> to_visit;
  std::stack<AnfNodePtr> next_to_visit;
  std::map<AnfNodePtr, size_t> nodes_ref;
  std::map<AnfNodePtr, std::vector<AnfNodePtr>> control_edges;
  CalcNodeRefCount(graph, &nodes_ref, &control_edges);
  std::string handle_target = default_target;
  std::string next_target;
  to_visit.push(graph->get_return());
  while (!to_visit.empty() || !next_to_visit.empty()) {
    if (to_visit.empty()) {
      to_visit.swap(next_to_visit);
      handle_target = next_target;
    }
    auto node = to_visit.top();
    MS_EXCEPTION_IF_NULL(node);
    to_visit.pop();
    result.emplace_back(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto node_inputs = cnode->inputs();
    if (!IsPrimitiveCNode(cnode, prim::kPrimSwitch)) {
      std::reverse(node_inputs.begin(), node_inputs.end());
    }
    auto ctrl_inputs = control_edges.find(node);
    if (ctrl_inputs != control_edges.end()) {
      node_inputs.insert(node_inputs.end(), ctrl_inputs->second.begin(), ctrl_inputs->second.end());
    }
    for (auto &input : node_inputs) {
      auto iter = nodes_ref.find(input);
      if (iter != nodes_ref.end()) {
        iter->second--;
        if (iter->second != 0) {
          continue;
        }
      }
      if (!input->isa<CNode>()) {
        to_visit.push(input);
        continue;
      }
      std::string input_target = GetCNodeTarget(input);
      if (input_target == handle_target) {
        to_visit.push(input);
      } else if (next_to_visit.empty() || input_target == next_target) {
        next_to_visit.push(input);
        next_target = input_target;
      } else {
        MS_LOG(EXCEPTION) << "Only support two different target";
      }
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
}

struct GraphNodesDependencyInfo {
  std::stack<AnfNodePtr> independent_nodes_;
  std::map<AnfNodePtr, size_t> input_num_;
  std::map<AnfNodePtr, std::vector<AnfNodePtr>> output_edges_;
};

GraphNodesDependencyInfo GetNodesDependencyInfo(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  GraphNodesDependencyInfo info;
  std::stack<AnfNodePtr> to_visit;
  std::map<AnfNodePtr, size_t> nodes_ref;
  std::map<AnfNodePtr, std::vector<AnfNodePtr>> control_edges;
  CalcNodeRefCount(graph, &nodes_ref, &control_edges);
  to_visit.push(graph->get_return());
  while (!to_visit.empty()) {
    auto node = to_visit.top();
    to_visit.pop();
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto node_inputs = cnode->inputs();
    auto ctrl_inputs = control_edges.find(node);
    if (ctrl_inputs != control_edges.end()) {
      node_inputs.insert(node_inputs.end(), ctrl_inputs->second.begin(), ctrl_inputs->second.end());
    }
    bool independent = true;
    for (auto &input : node_inputs) {
      if (input->isa<CNode>()) {
        independent = false;
        auto output_edge_iter = info.output_edges_.find(input);
        if (output_edge_iter != info.output_edges_.end()) {
          auto &edges = output_edge_iter->second;
          edges.emplace_back(node);
        } else {
          info.output_edges_[input] = {node};
        }
        auto input_num_iter = info.input_num_.find(node);
        if (input_num_iter != info.input_num_.end()) {
          input_num_iter->second++;
        } else {
          info.input_num_[node] = 1;
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
    if (independent) {
      info.independent_nodes_.push(node);
    }
  }
  return info;
}

struct VisitNodesInfo {
  std::queue<AnfNodePtr> default_target_nodes_;
  std::queue<AnfNodePtr> other_target_nodes_;
  std::map<AnfNodePtr, AnfNodePtr> seed_cast_next_node_;
};

VisitNodesInfo GetVisitNodesInfo(const GraphNodesDependencyInfo &dependency_info, const std::string &default_target,
                                 const std::string &other_target) {
  VisitNodesInfo result;
  auto independent_nodes = dependency_info.independent_nodes_;
  while (!independent_nodes.empty()) {
    auto seed_node = independent_nodes.top();
    independent_nodes.pop();
    MS_EXCEPTION_IF_NULL(seed_node);
    auto node_target = GetCNodeTarget(seed_node);
    if (IsPrimitiveCNode(seed_node, prim::kPrimCast)) {
      auto output_edges_iter = dependency_info.output_edges_.find(seed_node);
      if (output_edges_iter != dependency_info.output_edges_.end() && output_edges_iter->second.size() == 1) {
        auto &cast_next_node = output_edges_iter->second[0];
        auto input_num_iter = dependency_info.input_num_.find(cast_next_node);
        if (input_num_iter == dependency_info.input_num_.end()) {
          MS_LOG(EXCEPTION) << "Node input num not found!";
        }
        if (input_num_iter->second > 1 && node_target == GetCNodeTarget(cast_next_node)) {
          result.seed_cast_next_node_[cast_next_node] = seed_node;
          continue;
        }
      }
    }
    if (node_target == default_target) {
      result.default_target_nodes_.push(seed_node);
    } else if (node_target == other_target) {
      result.other_target_nodes_.push(seed_node);
    } else {
      MS_LOG(EXCEPTION) << "Only support two difference targets";
    }
  }
  return result;
}

std::string ParallelSortDecideNextHandleTarget(const std::vector<AnfNodePtr> &output_edges,
                                               const std::string &node_target,
                                               std::map<AnfNodePtr, std::string> *node_input_target_map) {
  MS_EXCEPTION_IF_NULL(node_input_target_map);
  std::string next_target = node_target;
  for (auto &dst_node : output_edges) {
    auto input_target_iter = node_input_target_map->find(dst_node);
    if (input_target_iter != node_input_target_map->end() && input_target_iter->second != node_target) {
      next_target = input_target_iter->second;
      break;
    }
    auto dst_node_target = GetCNodeTarget(dst_node);
    if (dst_node_target != node_target) {
      next_target = dst_node_target;
      break;
    }
  }
  for (auto &dst_node : output_edges) {
    (*node_input_target_map)[dst_node] = node_target;
  }
  return next_target;
}

void ParallelSortVisitNodeEdges(const std::vector<AnfNodePtr> &output_edges, GraphNodesDependencyInfo *dependency_info,
                                VisitNodesInfo *visit_nodes_info, const std::string &default_target) {
  MS_EXCEPTION_IF_NULL(dependency_info);
  MS_EXCEPTION_IF_NULL(visit_nodes_info);
  for (auto &dst_node : output_edges) {
    auto dst_node_target = GetCNodeTarget(dst_node);
    auto input_num_iter = dependency_info->input_num_.find(dst_node);
    if (input_num_iter == dependency_info->input_num_.end()) {
      MS_LOG(EXCEPTION) << "Node input num not found!";
    }
    input_num_iter->second--;
    if (input_num_iter->second == 1 &&
        visit_nodes_info->seed_cast_next_node_.find(dst_node) != visit_nodes_info->seed_cast_next_node_.end()) {
      input_num_iter->second--;
    }
    if (input_num_iter->second > 0) {
      continue;
    }
    if (dst_node_target == default_target) {
      visit_nodes_info->default_target_nodes_.push(dst_node);
    } else {
      visit_nodes_info->other_target_nodes_.push(dst_node);
    }
  }
}

std::vector<AnfNodePtr> ParallelSort(const FuncGraphPtr &graph, const std::string &default_target,
                                     const std::string &other_target) {
  MS_EXCEPTION_IF_NULL(graph);
  auto dependency_info = GetNodesDependencyInfo(graph);
  auto visit_nodes_info = GetVisitNodesInfo(dependency_info, default_target, other_target);
  std::vector<AnfNodePtr> result;
  std::string handle_target;
  if (!visit_nodes_info.default_target_nodes_.empty()) {
    handle_target = default_target;
  } else {
    handle_target = other_target;
  }
  std::map<AnfNodePtr, std::string> node_input_target_map;
  while (!visit_nodes_info.default_target_nodes_.empty() || !visit_nodes_info.other_target_nodes_.empty()) {
    AnfNodePtr ready_node;
    if ((!visit_nodes_info.default_target_nodes_.empty() && handle_target == default_target) ||
        visit_nodes_info.other_target_nodes_.empty()) {
      ready_node = visit_nodes_info.default_target_nodes_.front();
      visit_nodes_info.default_target_nodes_.pop();
      handle_target = default_target;
    } else {
      ready_node = visit_nodes_info.other_target_nodes_.front();
      visit_nodes_info.other_target_nodes_.pop();
      handle_target = other_target;
    }
    MS_EXCEPTION_IF_NULL(ready_node);
    auto cast_map_iter = visit_nodes_info.seed_cast_next_node_.find(ready_node);
    if (cast_map_iter != visit_nodes_info.seed_cast_next_node_.end()) {
      result.emplace_back(cast_map_iter->second);
    }
    result.emplace_back(ready_node);
    auto output_edges_iter = dependency_info.output_edges_.find(ready_node);
    if (output_edges_iter == dependency_info.output_edges_.end()) {
      continue;
    }
    auto &output_edges = output_edges_iter->second;
    handle_target = ParallelSortDecideNextHandleTarget(output_edges, handle_target, &node_input_target_map);
    ParallelSortVisitNodeEdges(output_edges, &dependency_info, &visit_nodes_info, default_target);
  }
  return result;
}

void AddSegmentDependency(const FuncGraphPtr &graph, const std::string &default_target,
                          const std::map<AnfNodePtr, GraphSegmentPtr> &node_to_segment) {
  std::stack<AnfNodePtr> to_visit;
  std::map<AnfNodePtr, size_t> nodes_ref;
  std::map<AnfNodePtr, std::vector<AnfNodePtr>> control_edges;
  CalcNodeRefCount(graph, &nodes_ref, &control_edges);
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
    auto ctrl_inputs = control_edges.find(node);
    if (ctrl_inputs != control_edges.end()) {
      node_inputs.insert(node_inputs.end(), ctrl_inputs->second.begin(), ctrl_inputs->second.end());
    }
    GraphSegmentPtr node_segment{nullptr};
    if (!IsPrimitiveCNode(cnode, prim::kPrimControlDepend)) {
      auto node_iter = node_to_segment.find(node);
      if (node_iter != node_to_segment.end()) {
        node_segment = node_iter->second;
      }
    }
    for (auto &input : node_inputs) {
      if (node_segment != nullptr && !node_segment->is_cut_ && input->isa<CNode>()) {
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

bool IsSubGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto &inputs = cnode->inputs();
    if (inputs.empty()) {
      MS_LOG(EXCEPTION) << "Inputs of apply node is empty";
    }

    AnfNodePtr fn = inputs[0];
    if (!IsValueNode<Primitive>(fn)) {
      return false;
    }
    auto node_prim = GetValueNode<PrimitivePtr>(fn);
    if (node_prim->name() == prim::kPrimPartial->name()) {
      return true;
    }
  } else if (IsValueNode<FuncGraph>(node)) {
    return true;
  }
  return false;
}

bool IsShapeDynamic(const abstract::ShapePtr &shape) {
  MS_EXCEPTION_IF_NULL(shape);
  return std::any_of(shape->shape().begin(), shape->shape().end(), [](int64_t s) { return s < 0; });
}

bool IsNodeOutputDynamicShape(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  if (base_shape == nullptr) {
    MS_LOG(INFO) << "Invalid base shape, node: " << node->fullname_with_scope();
    return false;
  }
  if (base_shape->isa<abstract::Shape>()) {
    if (IsShapeDynamic(base_shape->cast<abstract::ShapePtr>())) {
      return true;
    }
  } else if (base_shape->isa<abstract::TupleShape>()) {
    auto tuple_shape = base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    for (size_t i = 0; i < tuple_shape->size(); i++) {
      auto b_shape = (*tuple_shape)[i];
      if (!b_shape->isa<abstract::Shape>()) {
        continue;
      }
      if (IsShapeDynamic(b_shape->cast<abstract::ShapePtr>())) {
        return true;
      }
    }
  }
  return false;
}

void AddSegment(const std::vector<AnfNodePtr> &nodes, std::vector<GraphSegmentPtr> *segments,
                std::map<AnfNodePtr, GraphSegmentPtr> *node_to_segment) {
  MS_EXCEPTION_IF_NULL(segments);
  MS_EXCEPTION_IF_NULL(node_to_segment);
  auto segment = std::make_shared<GraphSegment>(nodes, false);
  segments->emplace_back(segment);
  for (auto &node : nodes) {
    (*node_to_segment)[node] = segment;
  }
}

struct SplitDynamicNodesHelper {
  void AddNode(const AnfNodePtr &node, bool is_dynamic_shape) {
    if (is_dynamic_shape) {
      pre_dynamic_nodes.emplace_back(node);
      pre_dynamic_nodes_set.insert(node);
    } else {
      pre_common_nodes.emplace_back(node);
      pre_common_nodes_set.insert(node);
    }
    pre_nodes.emplace_back(node);
  }

  void AddSegments(std::vector<GraphSegmentPtr> *segments, std::map<AnfNodePtr, GraphSegmentPtr> *node_to_segment) {
    if (pre_nodes.size() < merge_node_threshold) {
      AddSegment(pre_nodes, segments, node_to_segment);
    } else {
      if (!pre_common_nodes.empty()) {
        AddSegment(pre_common_nodes, segments, node_to_segment);
      }
      if (!pre_dynamic_nodes.empty()) {
        AddSegment(pre_dynamic_nodes, segments, node_to_segment);
      }
    }
    pre_common_nodes.clear();
    pre_common_nodes_set.clear();
    pre_dynamic_nodes.clear();
    pre_dynamic_nodes_set.clear();
    pre_nodes.clear();
  }
  std::vector<AnfNodePtr> pre_nodes;
  std::vector<AnfNodePtr> pre_dynamic_nodes;
  std::vector<AnfNodePtr> pre_common_nodes;
  std::set<AnfNodePtr> pre_common_nodes_set;
  std::set<AnfNodePtr> pre_dynamic_nodes_set;
  size_t merge_node_threshold = 6;
};

void SplitDynamicNodeSegment(const std::vector<AnfNodePtr> &segment_nodes, std::vector<GraphSegmentPtr> *segments,
                             std::map<AnfNodePtr, GraphSegmentPtr> *node_to_segment,
                             const std::set<AnfNodePtr> &dynamic_nodes_set) {
  SplitDynamicNodesHelper helper;
  bool is_last_node_dynamic = false;
  for (auto &node : segment_nodes) {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsPrimitiveCNode(cnode, prim::kPrimControlDepend)) {
      helper.AddNode(node, is_last_node_dynamic);
      continue;
    }
    auto &inputs = cnode->inputs();
    bool has_dynamic_shape = dynamic_nodes_set.find(node) != dynamic_nodes_set.end();
    bool depend_common_node = false;
    bool depend_dynamic_node = false;
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (dynamic_nodes_set.find(inputs[i]) != dynamic_nodes_set.end()) {
        has_dynamic_shape = true;
      }
      if (helper.pre_common_nodes_set.find(inputs[i]) != helper.pre_common_nodes_set.end()) {
        depend_common_node = true;
      }
      if (helper.pre_dynamic_nodes_set.find(inputs[i]) != helper.pre_dynamic_nodes_set.end()) {
        depend_dynamic_node = true;
      }
    }
    if (has_dynamic_shape) {
      if (depend_common_node) {
        helper.AddSegments(segments, node_to_segment);
      }
      is_last_node_dynamic = true;
    } else {
      if (depend_dynamic_node) {
        helper.AddSegments(segments, node_to_segment);
      }
      is_last_node_dynamic = false;
    }
    helper.AddNode(node, is_last_node_dynamic);
  }
  helper.AddSegments(segments, node_to_segment);
}

void NodesToSegments(const std::vector<AnfNodePtr> &segment_nodes, std::vector<GraphSegmentPtr> *segments,
                     std::map<AnfNodePtr, GraphSegmentPtr> *node_to_segment) {
  if (segment_nodes.empty()) {
    return;
  }
  auto segment_target = GetCNodeTarget(segment_nodes[0]);
  if (segment_target != kAscendDevice) {
    AddSegment(segment_nodes, segments, node_to_segment);
    return;
  }
  MS_EXCEPTION_IF_NULL(segments);
  MS_EXCEPTION_IF_NULL(node_to_segment);
  std::set<AnfNodePtr> dynamic_nodes_set;
  for (auto &node : segment_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (IsNodeOutputDynamicShape(cnode)) {
      (void)dynamic_nodes_set.insert(node);
    }
  }
  if (dynamic_nodes_set.empty()) {
    AddSegment(segment_nodes, segments, node_to_segment);
    return;
  }
  SplitDynamicNodeSegment(segment_nodes, segments, node_to_segment, dynamic_nodes_set);
}
}  // namespace

GraphPartition::GraphPartition(const std::vector<PrimitivePtr> &cut_list, const std::string &backend_name)
    : cut_list_(cut_list), backend_name_(backend_name) {}

bool GraphPartition::IsCut(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto &inputs = cnode->inputs();
    if (inputs.empty()) {
      MS_LOG(EXCEPTION) << "Inputs of apply node is empty";
    }
    AnfNodePtr fn = inputs[0];
    if (IsValueNode<FuncGraph>(fn)) {
      auto fg = GetValueNode<FuncGraphPtr>(fn);
      if (fg->has_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL)) {
        return false;
      }
    }
    if (!IsValueNode<Primitive>(fn)) {
      return true;
    }
    auto node_prim = GetValueNode<PrimitivePtr>(fn);
    for (auto &prim : cut_list_) {
      MS_EXCEPTION_IF_NULL(prim);
      if (prim->name() == node_prim->name()) {
        if (prim->name() == prim::kPrimBpropCut->name()) {
          auto ms_context = MsContext::GetInstance();
          MS_EXCEPTION_IF_NULL(ms_context);
          ms_context->set_param<bool>(MS_CTX_ENABLE_PYNATIVE_HOOK, true);
        }
        if (backend_name_ == kMsConvert && prim->name() == prim::kPrimMakeTuple->name()) {
          if (inputs.size() < 2) {
            return false;
          }
          auto ret = IsSubGraph(inputs[1]);
          return ret;
        }
        return true;
      }
    }
#ifdef ENABLE_GE
    if (backend_name_ == kGeVm) {
      auto name = GetCNodeFuncName(cnode);
      auto adpt = transform::DfGraphConvertor::FindAdapter(name);
      if (adpt == nullptr) {
        return true;
      }
    }
#endif
  }
  return false;
}

std::vector<GraphSegmentPtr> GraphPartition::Partition(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto nodes = TopoSort(graph->get_return());
  MS_LOG(DEBUG) << "Split all nodes size:" << nodes.size();
  bool contain_multi_target = ContainMultiTarget(nodes);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string default_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (contain_multi_target) {
    if (context_ptr->get_param<bool>(MS_CTX_ENABLE_PARALLEL_SPLIT)) {
      auto other_target = GetOtherTarget(nodes);
      nodes = ParallelSort(graph, default_target, other_target);
    } else {
      nodes = SplitSort(graph, default_target);
    }
    nodes = OptimizeGetItemOrder(nodes);
  }
  std::vector<GraphSegmentPtr> segments;
  std::vector<AnfNodePtr> segment_nodes;
  std::map<AnfNodePtr, GraphSegmentPtr> node_to_segment;
  std::string last_target;
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsCut(node)) {
      NodesToSegments(segment_nodes, &segments, &node_to_segment);
      segment_nodes.clear();
      segment_nodes.emplace_back(node);
      auto segment = std::make_shared<GraphSegment>(segment_nodes, true);
      segments.push_back(segment);
      segment_nodes.clear();
    } else if (node->isa<CNode>()) {
      if (contain_multi_target) {
        std::string cur_target = GetCNodeTarget(node);
        if (cur_target != last_target && !last_target.empty()) {
          NodesToSegments(segment_nodes, &segments, &node_to_segment);
          segment_nodes.clear();
        }
        last_target = cur_target;
      }
      segment_nodes.emplace_back(node);
    }
  }
  MS_LOG(DEBUG) << "Segment size:" << segments.size();
  if (contain_multi_target) {
    AddSegmentDependency(graph, default_target, node_to_segment);
  }
  return segments;
}
}  // namespace compile
}  // namespace mindspore
