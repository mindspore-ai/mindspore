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
#include "base/core_ops.h"
#include "utils/utils.h"
#include "utils/ms_context.h"
#ifdef ENABLE_GE
#include "transform/graph_ir/convert.h"
#endif
namespace mindspore {
const char kMsConvert[] = "ms";
const char kMsVm[] = "vm";
const char kGeVm[] = "ge";
namespace compile {
namespace {
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
  PrimitivePtr prim_ptr = GetValueNode<PrimitivePtr>(input_cnode->input(0));
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
  std::string next_target = "";
  to_visit.push(graph->get_return());
  while (!to_visit.empty() || !next_to_visit.empty()) {
    if (to_visit.empty()) {
      to_visit.swap(next_to_visit);
      handle_target = next_target;
    }
    auto &node = to_visit.top();
    MS_EXCEPTION_IF_NULL(node);
    to_visit.pop();
    result.emplace_back(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto node_inputs = cnode->inputs();
    std::reverse(node_inputs.begin(), node_inputs.end());
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
    auto node_iter = node_to_segment.find(node);
    if (node_iter != node_to_segment.end()) {
      node_segment = node_iter->second;
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

std::vector<AnfNodePtr> ParallelSplitSort(const FuncGraphPtr &graph, const std::string &default_target) {
  std::vector<AnfNodePtr> result;
  std::stack<AnfNodePtr> handle_nodes;
  std::stack<AnfNodePtr> next_handle_nodes;
  std::stack<AnfNodePtr> wait_handle_nodes;
  std::map<AnfNodePtr, size_t> nodes_ref;
  std::map<AnfNodePtr, std::vector<AnfNodePtr>> control_edges;
  CalcNodeRefCount(graph, &nodes_ref, &control_edges);
  std::string handle_target = default_target;
  std::string next_target = "";
  handle_nodes.push(graph->get_return());
  while (!handle_nodes.empty() || !next_handle_nodes.empty() || !wait_handle_nodes.empty()) {
    if (handle_nodes.empty()) {
      handle_nodes.swap(next_handle_nodes);
      handle_target.swap(next_target);
      next_handle_nodes.swap(wait_handle_nodes);
    }
    auto &node = handle_nodes.top();
    MS_EXCEPTION_IF_NULL(node);
    handle_nodes.pop();
    result.emplace_back(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto node_inputs = cnode->inputs();
    std::reverse(node_inputs.begin(), node_inputs.end());
    auto ctrl_inputs = control_edges.find(node);
    if (ctrl_inputs != control_edges.end()) {
      node_inputs.insert(node_inputs.end(), ctrl_inputs->second.begin(), ctrl_inputs->second.end());
    }
    std::vector<AnfNodePtr> same_target_ready_inputs;
    std::vector<AnfNodePtr> diff_target_ready_inputs;
    for (auto &input : node_inputs) {
      auto iter = nodes_ref.find(input);
      if (iter != nodes_ref.end()) {
        iter->second--;
        if (iter->second != 0) {
          continue;
        }
      }
      if (!input->isa<CNode>()) {
        handle_nodes.push(input);
        continue;
      }
      std::string input_target = GetCNodeTarget(input);
      if (input_target == handle_target) {
        same_target_ready_inputs.emplace_back(input);
      } else {
        if (next_target.empty()) {
          next_target = input_target;
        }
        if (input_target != next_target) {
          MS_LOG(EXCEPTION) << "Only support two different target";
        }
        diff_target_ready_inputs.emplace_back(input);
      }
    }
    if (diff_target_ready_inputs.size() == 0) {
      for (auto &input : same_target_ready_inputs) {
        handle_nodes.push(input);
      }
    } else {
      for (auto &input : same_target_ready_inputs) {
        wait_handle_nodes.push(input);
      }
      for (auto &input : diff_target_ready_inputs) {
        next_handle_nodes.push(input);
      }
    }
  }
  std::reverse(result.begin(), result.end());
  return result;
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
    PrimitivePtr node_prim = GetValueNode<PrimitivePtr>(fn);
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
    if (graph != nullptr) {
      nodes = SplitSort(graph, default_target);
    } else {
      nodes = ParallelSplitSort(graph, default_target);
    }
    nodes = OptimizeGetItemOrder(nodes);
  }
  std::vector<GraphSegmentPtr> segments;
  std::vector<AnfNodePtr> segment_nodes;
  std::map<AnfNodePtr, GraphSegmentPtr> node_to_segment;
  auto new_segment = [&segments, &segment_nodes, &node_to_segment]() {
    if (segment_nodes.size() != 0) {
      auto segment = std::make_shared<GraphSegment>(segment_nodes, false);
      segments.emplace_back(segment);
      for (auto node : segment_nodes) {
        node_to_segment[node] = segment;
      }
      segment_nodes.clear();
    }
  };
  std::string last_target;
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsCut(node)) {
      new_segment();
      segment_nodes.emplace_back(node);
      auto segment = std::make_shared<GraphSegment>(segment_nodes, true);
      segments.push_back(segment);
      segment_nodes.clear();
    } else if (node->isa<CNode>()) {
      if (contain_multi_target) {
        std::string cur_target = GetCNodeTarget(node);
        if (cur_target != last_target && !last_target.empty()) {
          new_segment();
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
