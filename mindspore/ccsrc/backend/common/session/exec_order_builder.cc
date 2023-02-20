/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "backend/common/session/exec_order_builder.h"
#include "include/common/utils/anfalgo.h"
#include "utils/ms_context.h"

namespace mindspore::session {
namespace {
std::string GetNodeGroup(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (common::AnfAlgo::HasNodeAttr(kAttrGroup, cnode)) {
    return common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrGroup);
  }
  return "";
}

bool NeedOptimize(const AnfNodePtr &node, const std::string &optimized_comm_group) {
  bool is_fused_comm = common::AnfAlgo::IsFusedCommunicationOp(node);
  if (!is_fused_comm) {
    return false;
  }
  auto node_group = GetNodeGroup(node);
  if (node_group.find(kSyncBnGroup) == string::npos) {
    if (optimized_comm_group.empty() || node_group == optimized_comm_group) {
      return true;
    }
  }
  return false;
}
}  // namespace

ExecOrderBuilder::~ExecOrderBuilder() { ClearLinkInfo(); }

std::vector<CNodePtr> ExecOrderBuilder::Build(FuncGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  graph_ = graph;
  ClearLinkInfo();
  BuildLinkInfo();
  FindIndependentNodes();
  return Build();
}

void ExecOrderBuilder::ClearLinkInfo() {
  node_input_num_.clear();
  node_output_num_.clear();
  node_input_edges_.clear();
  node_output_edges_.clear();
  trivial_nodes_.clear();
}

bool ExecOrderBuilder::IsTrivialNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return true;
  }

  if (AnfUtils::IsRealKernel(node)) {
    return false;
  }

  auto iter = trivial_nodes_.find(node);
  if (iter != trivial_nodes_.end()) {
    return true;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (auto &input : cnode->inputs()) {
    MS_EXCEPTION_IF_NULL(input);
    if (!IsTrivialNode(input)) {
      return false;
    }
  }
  (void)trivial_nodes_.insert(node);
  return true;
}

void ExecOrderBuilder::BuildLinkInfo() {
  std::queue<AnfNodePtr> to_visit;
  to_visit.push(graph_->get_return());
  std::set<AnfNodePtr> visited;
  while (!to_visit.empty()) {
    auto node = to_visit.front();
    to_visit.pop();
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>() || AnfUtils::IsCustomActorNode(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    for (auto &input : cnode->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (IsTrivialNode(input)) {
        continue;
      }
      (void)node_output_edges_[input].emplace_back(node);
      (void)node_input_edges_[node].emplace_back(input);
      node_input_num_[node] += 1;
      node_output_num_[input] += 1;
      if (visited.find(input) != visited.end()) {
        continue;
      }
      to_visit.push(input);
      (void)visited.insert(input);
    }
  }
}

void ExecOrderBuilder::FindIndependentNodes() {
  std::queue<AnfNodePtr> to_visit;
  std::queue<AnfNodePtr> vnode_to_visit;
  mindspore::HashSet<AnfNodePtr> visited;
  vnode_to_visit.push(graph_->get_return());
  bool visit_with_refcount = true;
  auto ms_context = MsContext::GetInstance();
  auto target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (target == kGPUDevice) {
    visit_with_refcount = false;
  }
  while (!to_visit.empty() || !vnode_to_visit.empty()) {
    AnfNodePtr node;
    if (vnode_to_visit.empty()) {
      node = to_visit.front();
      to_visit.pop();
    } else {
      node = vnode_to_visit.front();
      vnode_to_visit.pop();
    }

    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    if (AnfUtils::IsCustomActorNode(node)) {
      independent_nodes_.push(node);
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    bool independent = true;
    auto &inputs = cnode->inputs();
    for (auto iter = inputs.rbegin(); iter != inputs.rend(); ++iter) {
      auto &input = *iter;
      MS_EXCEPTION_IF_NULL(input);
      if (IsTrivialNode(input)) {
        continue;
      }
      independent = false;

      if (visit_with_refcount) {
        auto output_iter = node_output_num_.find(input);
        if (output_iter != node_output_num_.end()) {
          output_iter->second--;
          if (output_iter->second != 0) {
            continue;
          }
        }
      } else {
        if (visited.find(input) != visited.end()) {
          continue;
        }
        visited.insert(input);
      }

      if (AnfUtils::IsRealKernel(node)) {
        to_visit.push(input);
        if (!independent_nodes_.empty() && visit_with_refcount) {
          auto inode = independent_nodes_.top();
          node_output_edges_[input].emplace_back(inode);
          node_input_edges_[inode].emplace_back(input);
          node_input_num_[inode] += 1;
          independent_nodes_.pop();
        }
      } else {
        vnode_to_visit.push(input);
      }
    }

    if (independent) {
      independent_nodes_.push(node);
    }
  }
}

void ExecOrderBuilder::EnqueueReadyNodes(const AnfNodePtr &node, std::deque<AnfNodePtr> *visit_queue, bool comm_first) {
  MS_EXCEPTION_IF_NULL(visit_queue);
  auto it = node_output_edges_.find(node);
  if (it == node_output_edges_.end()) {
    return;
  }

  std::vector<AnfNodePtr> active_nodes;
  for (const auto &output_node : it->second) {
    MS_EXCEPTION_IF_NULL(output_node);
    auto input_num_iter = node_input_num_.find(output_node);
    if (input_num_iter == node_input_num_.end() || input_num_iter->second == 0) {
      continue;
    }
    input_num_iter->second--;
    if (input_num_iter->second > 0) {
      continue;
    }

    bool is_comm_node = common::AnfAlgo::IsCommunicationOp(output_node);
    if (!AnfUtils::IsRealKernel(output_node) || it->second.size() == 1) {
      visit_queue->push_front(output_node);
    } else if ((is_comm_node && comm_first) || (!is_comm_node && !comm_first)) {
      visit_queue->push_back(output_node);
    } else {
      active_nodes.emplace_back(output_node);
    }
  }

  (void)std::copy(active_nodes.begin(), active_nodes.end(), std::back_inserter(*visit_queue));
}

std::vector<CNodePtr> ExecOrderBuilder::Build() {
  std::vector<CNodePtr> execution_order;
  std::deque<AnfNodePtr> to_visit;
  std::deque<AnfNodePtr> delay_visit;
  std::deque<AnfNodePtr> high_priority_to_visit;
  std::deque<AnfNodePtr> *handle_queue_ptr;
  std::string optimized_comm_group;
  AnfNodePtr pending_node = nullptr;
  while (!independent_nodes_.empty() || pending_node != nullptr || !delay_visit.empty()) {
    if (!delay_visit.empty()) {
      EnqueueReadyNodes(delay_visit.front(), &high_priority_to_visit, false);
      delay_visit.pop_front();
    } else if (pending_node != nullptr) {
      EnqueueReadyNodes(pending_node, &high_priority_to_visit, false);
      pending_node = nullptr;
    } else {
      to_visit.push_back(independent_nodes_.top());
      independent_nodes_.pop();
    }
    // comm descendant first, then common queue
    while (!to_visit.empty() || !high_priority_to_visit.empty()) {
      AnfNodePtr node;
      if (!high_priority_to_visit.empty()) {
        handle_queue_ptr = &high_priority_to_visit;
        node = high_priority_to_visit.front();
        high_priority_to_visit.pop_front();
      } else {
        handle_queue_ptr = &to_visit;
        node = to_visit.front();
        to_visit.pop_front();
      }
      // add execute node
      MS_EXCEPTION_IF_NULL(node);
      if (node->isa<CNode>() && AnfUtils::IsRealKernel(node)) {
        execution_order.emplace_back(node->cast<CNodePtr>());
      }
      // delay execute comm ops that need optimize
      bool is_comm = common::AnfAlgo::IsCommunicationOp(node);
      bool optimize_comm = NeedOptimize(node, optimized_comm_group);
      if (optimize_comm) {
        optimized_comm_group = GetNodeGroup(node);
        if (pending_node != nullptr) {
          EnqueueReadyNodes(pending_node, &high_priority_to_visit, false);
        }
        pending_node = node;
      } else if (is_comm) {
        delay_visit.push_back(node);
      } else {
        EnqueueReadyNodes(node, handle_queue_ptr);
      }
    }
  }
  CheckLoop();
  return execution_order;
}

bool ExecOrderBuilder::PrintLoopNodesIfExist(const AnfNodePtr &node, std::set<AnfNodePtr> *visited_nodes,
                                             mindspore::HashMap<AnfNodePtr, AnfNodePtr> *next_nodes) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(visited_nodes);
  MS_EXCEPTION_IF_NULL(next_nodes);

  (void)visited_nodes->insert(node);
  for (auto &input_node : node_input_edges_[node]) {
    size_t input_num = node_input_num_[input_node];
    if (input_num == 0) {
      continue;
    }
    if (visited_nodes->find(input_node) == visited_nodes->end()) {
      MS_EXCEPTION_IF_NULL(input_node);
      (*next_nodes)[input_node] = node;
      if (PrintLoopNodesIfExist(input_node, visited_nodes, next_nodes)) {
        return true;
      }
    } else {
      auto cur_node = node;
      std::queue<AnfNodePtr> loop_nodes;
      while (cur_node != input_node && cur_node != nullptr) {
        loop_nodes.push(cur_node);
        cur_node = (*next_nodes)[cur_node];
      }

      if (cur_node == input_node) {
        loop_nodes.push(cur_node);
        MS_LOG(INFO) << "Print loop nodes start:";
        while (!loop_nodes.empty()) {
          cur_node = loop_nodes.front();
          node_input_num_[cur_node]--;
          MS_LOG(INFO) << "Get loop node:" << cur_node->DebugString();
          loop_nodes.pop();
        }
        MS_LOG(INFO) << "Print loop nodes end.";
        return true;
      }
    }
  }
  return false;
}

void ExecOrderBuilder::CheckLoop() {
  std::vector<AnfNodePtr> unvisited_nodes;
  for (auto &node_ref : node_input_num_) {
    MS_EXCEPTION_IF_NULL(node_ref.first);
    if (node_ref.second == 0) {
      continue;
    }
    std::string info;
    for (const auto &input_node : node_input_edges_[node_ref.first]) {
      MS_EXCEPTION_IF_NULL(input_node);
      info = info.append(input_node->DebugString()).append("|");
    }
    MS_LOG(WARNING) << "Node:" << node_ref.first->DebugString() << ",inputs:" << info
                    << ",input num:" << node_ref.second;
    (void)unvisited_nodes.emplace_back(node_ref.first);
  }

  if (unvisited_nodes.empty()) {
    return;
  }

  for (auto &node : unvisited_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    std::set<AnfNodePtr> visited_nodes;
    mindspore::HashMap<AnfNodePtr, AnfNodePtr> next_nodes;
    if (PrintLoopNodesIfExist(node, &visited_nodes, &next_nodes)) {
      break;
    }
  }
  MS_LOG(EXCEPTION) << "Graph has unvisited nodes and the number is :" << unvisited_nodes.size();
}
}  // namespace mindspore::session
