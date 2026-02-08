/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/interleave_split_concat_branches.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <queue>
#include <stack>
#include <unordered_map>
#include <utility>
#include "primitive/array_op_name.h"
#include "primitive/math_op_name.h"
#include "primitive/other_op_name.h"
#include "primitive/other_ops.h"
#include "include/frontend/optimizer/optimizer.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/pass/interleave_branches_utils.h"
#include "include/utils/utils.h"
#include "utils/ms_context.h"
#include "primitive/auto_generate/gen_ops_primitive_a.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_d.h"
#include "primitive/auto_generate/gen_ops_primitive_p.h"
#include "primitive/auto_generate/gen_ops_primitive_g.h"

namespace mindspore {
namespace parallel {
namespace {
auto const kAttrOverlapped = "overlapped";
auto const kGradFlag = "Gradients";
auto const kFilterCost = 5;
auto const kDefaultCostThreshold = 15;
auto const kDataCostTreshold = 1024 * 1024 * 1024;

struct BranchInterleaveNode {
  size_t begin{0};
  size_t end{0};
  float cost{0.0f};
  bool is_comm{false};
  bool overlapped{false};
  size_t branch_id{0};
};

using BranchInterleaveNodePtr = std::shared_ptr<BranchInterleaveNode>;

enum class InterleaveNodeType { kCommunication = 0, kComputation = 1, kVirtual = 2, kOverlapped = 3 };

enum class InterleaveNodeTrace { kUp = 0, kLeft = 1, kDiag = 2 };

inline bool IsForwardNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    return false;
  }
  return node->fullname_with_scope().find(kGradFlag) != 0;
}

inline bool IsNotMatMul(const std::string &node_name) {
  if (node_name != kMatMulOpName && node_name != kBatchMatMulOpName && node_name != ops::kNameGroupedMatmul) {
    return true;
  }
  return false;
}

// Branch id propagation to mask split independent branches
void PropagateBranchId(const InterLeaveScopePtr &interleave_scope, const CNodePtr &seed_node, size_t branch_id) {
  MS_EXCEPTION_IF_NULL(interleave_scope);
  MS_EXCEPTION_IF_NULL(seed_node);
  auto scope_id_value = MakeValue<size_t>(interleave_scope->scope_id);
  auto branch_id_value = MakeValue<size_t>(branch_id);
  static auto kSharedBranchIdValue = MakeValue<size_t>(kInterleaveSharedBranchId);
  seed_node->AddAttr(kInterleaveScopeId, scope_id_value);
  seed_node->AddAttr(kInterleaveBranchId, branch_id_value);
  std::queue<CNodePtr> to_visit;
  to_visit.emplace(seed_node);
  bool is_backward_scope = !interleave_scope->forward;
  while (!to_visit.empty()) {
    auto node = to_visit.front();
    to_visit.pop();
    MS_EXCEPTION_IF_NULL(node);
    if (node == interleave_scope->fork_node) {
      continue;
    }
    bool is_branch_node = true;
    bool is_depend = IsPrimitiveCNode(node, prim::kPrimDepend);
    if (node->HasAttr(kInterleaveBranchId) &&
        GetValue<size_t>(node->GetAttr(kInterleaveBranchId)) == kInterleaveSharedBranchId) {
      is_branch_node = false;
    }
    for (auto &input : node->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>()) {
        continue;
      }

      auto input_cnode = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      if (is_depend && node->input(1) != input) {
        continue;
      }

      if (is_backward_scope && common::AnfAlgo::IsRecompute(input_cnode)) {
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveScopeId)) {
        input_cnode->AddAttr(kInterleaveScopeId, scope_id_value);
      } else if (GetValue<size_t>(input_cnode->GetAttr(kInterleaveScopeId)) != interleave_scope->scope_id) {
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveBranchId)) {
        if (is_branch_node) {
          input_cnode->AddAttr(kInterleaveBranchId, branch_id_value);
        } else {
          input_cnode->AddAttr(kInterleaveBranchId, kSharedBranchIdValue);
        }
        to_visit.emplace(input_cnode);
        continue;
      }

      auto input_branch_id = GetValue<size_t>(input_cnode->GetAttr(kInterleaveBranchId));
      if (input_branch_id != branch_id && input_branch_id != kInterleaveSharedBranchId) {
        input_cnode->AddAttr(kInterleaveBranchId, kSharedBranchIdValue);
        to_visit.emplace(input_cnode);
      }
    }
  }
}

mindspore::HashMap<CNodePtr, size_t> GetBranchNodesRefCount(const InterLeaveScopePtr &interleave_scope,
                                                            const CNodePtr &seed_node, size_t branch_id) {
  MS_EXCEPTION_IF_NULL(interleave_scope);
  MS_EXCEPTION_IF_NULL(seed_node);
  auto seen = NewSeenGeneration();
  mindspore::HashMap<CNodePtr, size_t> ref_count;
  std::queue<CNodePtr> to_visit;
  to_visit.emplace(seed_node);
  while (!to_visit.empty()) {
    auto node = to_visit.front();
    to_visit.pop();
    MS_EXCEPTION_IF_NULL(node);
    if (node == interleave_scope->fork_node) {
      continue;
    }

    for (auto &input : node->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>()) {
        continue;
      }

      auto input_cnode = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      if (input->seen_ == seen) {
        ref_count[input_cnode] += 1;
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveScopeId) ||
          GetValue<size_t>(input_cnode->GetAttr(kInterleaveScopeId)) != interleave_scope->scope_id) {
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveBranchId)) {
        continue;
      }

      auto input_branch_id = GetValue<size_t>(input_cnode->GetAttr(kInterleaveBranchId));
      if (input_branch_id == branch_id) {
        input->seen_ = seen;
        ref_count[input_cnode] = 1;
        to_visit.emplace(input_cnode);
      }
    }
  }
  return ref_count;
}

void AppendMatMulGradDwNode(std::vector<CNodePtr> *ordered_nodes_ptr,
                            HashMap<CNodePtr, CNodePtr> *matmul_grad_dual_map_ptr) {
  MS_EXCEPTION_IF_NULL(ordered_nodes_ptr);
  if (matmul_grad_dual_map_ptr == nullptr) {
    return;
  }
  auto &matmul_grad_dual_map = *matmul_grad_dual_map_ptr;
  auto &nodes = *ordered_nodes_ptr;
  auto node_size = nodes.size();
  for (size_t i = 0; i < node_size; ++i) {
    auto node_name = common::AnfAlgo::GetCNodeName(nodes[i]);
    if (IsNotMatMul(node_name)) {
      continue;
    }

    auto iter = matmul_grad_dual_map.find(nodes[i]);
    if (iter != matmul_grad_dual_map.end() && iter->second != nullptr) {
      if (!iter->second->HasAttr(kInterleaveBranchId)) {
        (void)nodes.emplace_back(iter->second);
      }
    }
  }
}

// Get ordered branch nodes
std::vector<CNodePtr> GetBranchOrderedNodes(const InterLeaveScopePtr &interleave_scope, const CNodePtr &seed_node,
                                            size_t branch_id) {
  MS_EXCEPTION_IF_NULL(interleave_scope);
  MS_EXCEPTION_IF_NULL(seed_node);
  auto seen = NewSeenGeneration();
  mindspore::HashMap<CNodePtr, size_t> ref_count = GetBranchNodesRefCount(interleave_scope, seed_node, branch_id);
  std::vector<CNodePtr> ordered_nodes;
  std::queue<CNodePtr> compute_queue;
  std::queue<CNodePtr> communication_queue;
  std::queue<CNodePtr> *current_queue = &compute_queue;
  compute_queue.emplace(seed_node);
  while (!compute_queue.empty() || !communication_queue.empty()) {
    if (current_queue->empty()) {
      current_queue = &compute_queue;
      if (compute_queue.empty()) {
        current_queue = &communication_queue;
      }
    }
    auto node = current_queue->front();
    current_queue->pop();
    MS_EXCEPTION_IF_NULL(node);
    if (AnfUtils::IsRealKernel(node)) {
      ordered_nodes.emplace_back(node);
    }

    for (auto &input : node->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>() || input->seen_ == seen) {
        continue;
      }

      auto input_cnode = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      if (!input_cnode->HasAttr(kInterleaveScopeId) ||
          GetValue<size_t>(input_cnode->GetAttr(kInterleaveScopeId)) != interleave_scope->scope_id) {
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveBranchId)) {
        continue;
      }

      auto input_branch_id = GetValue<size_t>(input_cnode->GetAttr(kInterleaveBranchId));
      if (input_branch_id != branch_id) {
        continue;
      }

      auto iter_ref = ref_count.find(input_cnode);
      if (iter_ref == ref_count.end()) {
        continue;
      }

      if (iter_ref->second > 1) {
        --iter_ref->second;
        continue;
      }

      input->seen_ = seen;
      if (common::AnfAlgo::IsCommunicationOp(input_cnode)) {
        communication_queue.emplace(input_cnode);
      } else if (AnfUtils::IsRealKernel(input_cnode)) {
        compute_queue.emplace(input_cnode);
      } else {
        current_queue->emplace(input_cnode);
      }
    }
  }

  std::reverse(ordered_nodes.begin(), ordered_nodes.end());
  AppendMatMulGradDwNode(&ordered_nodes, interleave_scope->matmul_grad_dual_map);
  return ordered_nodes;
}

float GetNodeCost(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  static const float kDefaultCost = 0.1f;
  static const std::unordered_map<std::string, float> kBaseCostMap = {
    {kAllReduceOpName, 8.0f},    {kReduceScatterOpName, 8.0f}, {kAllGatherOpName, 8.0f},        {kAllToAllOpName, 8.0f},
    {kAlltoAllOpName, 8.0f},     {kAllToAllvOpName, 8.0f},     {kAlltoAllVOpName, 8.0f},        {kReshapeOpName, 0.01f},
    {kBatchMatMulOpName, 10.0f}, {kMatMulOpName, 10.0f},       {ops::kNameGroupedMatmul, 10.0f}};
  auto node_name = common::AnfAlgo::GetCNodeName(node);
  auto iter = kBaseCostMap.find(node_name);
  if (iter != kBaseCostMap.end()) {
    if (iter->second > kFilterCost) {
      auto shape = common::AnfAlgo::GetOutputInferShape(node, 0);
      auto data_size = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
      if (data_size > 0) {
        return iter->second + 1.0f * data_size / kDataCostTreshold;
      }
    }
    return iter->second;
  }

  return kDefaultCost;
}

// Interleave node is a range of the same compute type nodes
std::vector<BranchInterleaveNodePtr> GetInterleaveNodes(std::vector<CNodePtr> *node_vec_ptr, size_t branch_id,
                                                        bool filter_cost) {
  MS_EXCEPTION_IF_NULL(node_vec_ptr);
  auto &node_vec = *node_vec_ptr;
  BranchInterleaveNodePtr current_node = nullptr;
  std::vector<BranchInterleaveNodePtr> result;
  auto current_node_type = InterleaveNodeType::kVirtual;
  auto add_result = [filter_cost, &result](const BranchInterleaveNodePtr &current_node) {
    if (!filter_cost || current_node->cost >= kFilterCost) {
      result.emplace_back(current_node);
    }
  };

  for (size_t i = 0; i < node_vec.size(); ++i) {
    auto &node = node_vec[i];
    MS_EXCEPTION_IF_NULL(node);
    auto node_type = InterleaveNodeType::kVirtual;
    if (node->HasAttr(kAttrOverlapped) ||
        (IsPrimitiveCNode(node, prim::kPrimAllGather) && common::AnfAlgo::IsFromParallelOptimizer(node))) {
      node_type = InterleaveNodeType::kOverlapped;
    } else if (common::AnfAlgo::IsCommunicationOp(node)) {
      node_type = InterleaveNodeType::kCommunication;
    } else if (AnfUtils::IsRealKernel(node)) {
      node_type = InterleaveNodeType::kComputation;
    } else {
      continue;
    }

    bool exceed_threshold = false;
    auto node_cost = GetNodeCost(node);
    if (current_node != nullptr && current_node->cost + node_cost > kDefaultCostThreshold) {
      exceed_threshold = true;
    }

    if (current_node != nullptr && (current_node_type != node_type || exceed_threshold)) {
      add_result(current_node);
      current_node = nullptr;
    }

    if (node_type == InterleaveNodeType::kOverlapped) {
      continue;
    }

    if (current_node == nullptr) {
      current_node_type = node_type;
      current_node = std::make_shared<BranchInterleaveNode>();
      current_node->branch_id = branch_id;
      current_node->begin = i;
      if (node_type == InterleaveNodeType::kCommunication) {
        current_node->is_comm = true;
      }
    }

    current_node->cost += node_cost;
    current_node->end = i;
  }

  if (current_node != nullptr) {
    add_result(current_node);
  }
  return result;
}

void AddDependNode(const FuncGraphPtr &graph, const CNodePtr &post_node, const AnfNodePtr &pre_node,
                   bool eager_depend = false) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(post_node);
  MS_EXCEPTION_IF_NULL(pre_node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto post_node_input = post_node->input(1);
  MS_EXCEPTION_IF_NULL(post_node_input);
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), post_node_input, pre_node};
  auto depend_node = graph->NewCNode(depend_inputs);
  depend_node->AddAttr(kSplitConcatDepend, MakeValue(1));
  if (eager_depend) {
    // eager visit depend in execute order builder
    depend_node->AddAttr(kAttrEagerDepend, MakeValue(1));
  }
  depend_node->set_abstract(post_node_input->abstract());
  manager->SetEdge(post_node, 1, depend_node);
}

bool CheckInterleaveNodeValid(std::vector<std::vector<CNodePtr>> *branch_cnodes_ptr,
                              const BranchInterleaveNodePtr &pre_node, const BranchInterleaveNodePtr &cur_node) {
  MS_EXCEPTION_IF_NULL(branch_cnodes_ptr);
  MS_EXCEPTION_IF_NULL(pre_node);
  MS_EXCEPTION_IF_NULL(cur_node);

  auto &branch_cnodes = *branch_cnodes_ptr;
  auto pre_node_branch_size = branch_cnodes[pre_node->branch_id].size();
  auto cur_node_branch_size = branch_cnodes[cur_node->branch_id].size();
  if (pre_node->begin >= pre_node_branch_size || pre_node->end >= pre_node_branch_size ||
      cur_node->begin >= cur_node_branch_size || cur_node->end >= cur_node_branch_size) {
    return false;
  }
  return true;
}

void MarkOverlappedFlag(std::vector<std::vector<CNodePtr>> *branch_cnodes_ptr, const BranchInterleaveNodePtr &pre_node,
                        const BranchInterleaveNodePtr &cur_node) {
  auto overlapped_flag = MakeValue(1);
  auto add_overlapped_flag = [overlapped_flag](const std::vector<CNodePtr> &cnodes_list,
                                               const BranchInterleaveNodePtr &interleave_range) {
    for (size_t i = interleave_range->begin; i <= interleave_range->end; ++i) {
      auto &cnode = cnodes_list[i];
      if (cnode != nullptr) {
        cnode->AddAttr(kAttrOverlapped, overlapped_flag);
      }
    }
  };
  add_overlapped_flag((*branch_cnodes_ptr)[pre_node->branch_id], pre_node);
  add_overlapped_flag((*branch_cnodes_ptr)[cur_node->branch_id], cur_node);
}

void AddDependForOverlap(std::vector<std::vector<CNodePtr>> *branch_cnodes_ptr, const BranchInterleaveNodePtr &pre_node,
                         const BranchInterleaveNodePtr &cur_node, const FuncGraphPtr &graph) {
  if (!CheckInterleaveNodeValid(branch_cnodes_ptr, pre_node, cur_node)) {
    return;
  }
  pre_node->overlapped = true;
  cur_node->overlapped = true;
  MarkOverlappedFlag(branch_cnodes_ptr, pre_node, cur_node);
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto add_user_depend = [graph, manager](const CNodePtr &post_cnode, const CNodePtr &pre_cnode) {
    auto node_users = manager->node_users()[post_cnode];
    for (auto &node_user : node_users) {
      if (node_user.first == nullptr) {
        continue;
      }
      if (!node_user.first->isa<CNode>() || IsPrimitiveCNode(node_user.first, prim::kPrimPartial)) {
        continue;
      }
      auto cnode_user = node_user.first->cast<CNodePtr>();
      if (cnode_user == nullptr) {
        continue;
      }
      if (IsPrimitiveCNode(node_user.first, prim::kPrimDepend) && cnode_user->input(1) != post_cnode) {
        continue;
      }
      AddDependNode(graph, cnode_user, pre_cnode);
    }
  };

  auto add_input_depend = [graph, manager](const CNodePtr &post_cnode, const CNodePtr &pre_cnode) {
    auto node_inputs = pre_cnode->inputs();
    for (auto &node_input : node_inputs) {
      if (node_input == nullptr) {
        continue;
      }
      if (!node_input->isa<CNode>()) {
        continue;
      }
      AddDependNode(graph, post_cnode, node_input);
    }
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto lhs_node = pre_node;
  auto rhs_node = cur_node;
  if (!pre_node->is_comm) {
    lhs_node = cur_node;
    rhs_node = pre_node;
  }
  auto const &branch_cnodes = *branch_cnodes_ptr;
  auto const &lhs_cnodes = branch_cnodes[lhs_node->branch_id];
  auto const &rhs_cnodes = branch_cnodes[rhs_node->branch_id];
  auto lhs_begin_cnode = lhs_cnodes[lhs_node->begin];
  auto lhs_end_cnode = lhs_cnodes[lhs_node->end];
  auto rhs_begin_cnode = rhs_cnodes[rhs_node->begin];
  auto rhs_end_cnode = rhs_cnodes[rhs_node->end];

  // 1.Avoid communication users node block comm/comp parallel
  // 2.Insert communication nodes before computation nodes for kbk
  // lhs-begin              <----    rhs-allnode-input
  // lhs-end-users          <----    rhs-end
  // lhs-end                ----->   rhs-end-users
  // lhs-end                ----->   rhs-begin       for kbk
  // lhs-begin-input        ----->   rhs-begin       for ge

  if (ms_context->IsKByKExecutorMode()) {
    auto mm_begin = rhs_node->begin;
    while (mm_begin <= rhs_node->end && GetNodeCost(rhs_cnodes[mm_begin]) < kFilterCost) {
      ++mm_begin;
    }
    if (mm_begin > rhs_node->end) {
      mm_begin = rhs_node->begin;
    }
    rhs_begin_cnode = rhs_cnodes[mm_begin];
    add_input_depend(lhs_begin_cnode, rhs_begin_cnode);
    add_user_depend(lhs_end_cnode, rhs_end_cnode);
    add_user_depend(rhs_end_cnode, lhs_end_cnode);
    AddDependNode(graph, rhs_begin_cnode, lhs_end_cnode, true);
  } else {
    add_input_depend(lhs_begin_cnode, rhs_begin_cnode);
    add_user_depend(lhs_end_cnode, rhs_end_cnode);
    add_user_depend(rhs_end_cnode, lhs_end_cnode);
    AddDependNode(graph, rhs_begin_cnode, lhs_begin_cnode->input(1));
  }
}

// Interleave two branches nodes to overlap comm/comp
// Return unoverlapped nodes
std::vector<BranchInterleaveNodePtr> InterleaveTwoBranch(std::vector<std::vector<CNodePtr>> *branch_cnodes_ptr,
                                                         std::vector<BranchInterleaveNodePtr> *pre_nodes_ptr,
                                                         std::vector<BranchInterleaveNodePtr> *cur_nodes_ptr,
                                                         const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(pre_nodes_ptr);
  MS_EXCEPTION_IF_NULL(cur_nodes_ptr);
  auto &pre_nodes = *pre_nodes_ptr;
  auto &cur_nodes = *cur_nodes_ptr;
  std::vector<BranchInterleaveNodePtr> unoverlapped_nodes;
  size_t j = 0;
  size_t k = 0;
  for (size_t i = 0; i < cur_nodes.size(); ++i) {
    auto &cur_node = cur_nodes[i];
    MS_EXCEPTION_IF_NULL(cur_node);
    if (cur_node->cost < kFilterCost) {
      cur_node->overlapped = true;
      continue;
    }

    j = k;
    if (j == pre_nodes.size()) {
      if (!cur_node->overlapped) {
        (void)unoverlapped_nodes.emplace_back(cur_node);
      }
      continue;
    }

    bool get_next = false;
    while (j < pre_nodes.size()) {
      auto &pre_node = pre_nodes[j];
      MS_EXCEPTION_IF_NULL(pre_node);
      if (get_next) {
        if (pre_node->cost >= kFilterCost) {
          break;
        }
        pre_node->overlapped = true;
        ++j;
        k = j;
        continue;
      }

      if (pre_node->cost < kFilterCost) {
        pre_node->overlapped = true;
        ++j;
      } else if (cur_node->is_comm == pre_node->is_comm) {
        ++j;
      } else {
        AddDependForOverlap(branch_cnodes_ptr, pre_node, cur_node, graph);
        for (auto check_index = k; check_index < j; ++check_index) {
          auto &check_node = pre_nodes[check_index];
          if (!check_node->overlapped) {
            (void)unoverlapped_nodes.emplace_back(check_node);
          }
        }
        ++j;
        k = j;
        get_next = true;
      }
    }
    if (!cur_node->overlapped) {
      (void)unoverlapped_nodes.emplace_back(cur_node);
    }
  }

  return unoverlapped_nodes;
}

void FillBranchDPTraceMap(std::vector<BranchInterleaveNodePtr> *pre_nodes_ptr,
                          std::vector<BranchInterleaveNodePtr> *cur_nodes_ptr,
                          std::vector<std::vector<InterleaveNodeTrace>> *trace_map_ptr) {
  MS_EXCEPTION_IF_NULL(pre_nodes_ptr);
  MS_EXCEPTION_IF_NULL(cur_nodes_ptr);
  MS_EXCEPTION_IF_NULL(trace_map_ptr);
  auto &pre_nodes = *pre_nodes_ptr;
  auto &cur_nodes = *cur_nodes_ptr;
  auto &trace_map = *trace_map_ptr;
  std::vector<std::vector<float>> cost_map(pre_nodes.size(), std::vector<float>(cur_nodes.size(), 0.0f));

  for (size_t i = 0; i < pre_nodes.size(); ++i) {
    auto &pre_node = pre_nodes[i];
    for (size_t j = 0; j < cur_nodes.size(); ++j) {
      auto &cur_node = cur_nodes[j];
      float overlap = 0.0f;
      float left_cost = 0.0f;
      float up_cost = 0.0f;
      if (pre_node->is_comm != cur_node->is_comm) {
        overlap = std::min(pre_node->cost, cur_node->cost);
      }

      if (i > 0) {
        up_cost = cost_map[i - 1][j];
      }
      if (j > 0) {
        left_cost = cost_map[i][j - 1];
      }
      float diag_cost = overlap;
      if (i > 0 && j > 0) {
        diag_cost = cost_map[i - 1][j - 1] + overlap;
      }

      if (up_cost >= left_cost && up_cost >= diag_cost) {
        cost_map[i][j] = up_cost;
        trace_map[i][j] = InterleaveNodeTrace::kUp;
      } else if (left_cost > up_cost && left_cost >= diag_cost) {
        cost_map[i][j] = left_cost;
        trace_map[i][j] = InterleaveNodeTrace::kLeft;
      } else {
        cost_map[i][j] = diag_cost;
        trace_map[i][j] = InterleaveNodeTrace::kDiag;
      }
    }
  }
}

// Interleave two branches nodes to overlap comm/comp using dynamic programming
// Return unoverlapped nodes
std::vector<BranchInterleaveNodePtr> InterleaveTwoBranchDP(std::vector<std::vector<CNodePtr>> *branch_cnodes_ptr,
                                                           std::vector<BranchInterleaveNodePtr> *pre_nodes_ptr,
                                                           std::vector<BranchInterleaveNodePtr> *cur_nodes_ptr,
                                                           const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(pre_nodes_ptr);
  MS_EXCEPTION_IF_NULL(cur_nodes_ptr);
  auto &pre_nodes = *pre_nodes_ptr;
  auto &cur_nodes = *cur_nodes_ptr;
  std::vector<std::vector<InterleaveNodeTrace>> trace_map(
    pre_nodes.size(), std::vector<InterleaveNodeTrace>(cur_nodes.size(), InterleaveNodeTrace::kUp));
  FillBranchDPTraceMap(pre_nodes_ptr, cur_nodes_ptr, &trace_map);

  std::vector<BranchInterleaveNodePtr> unoverlapped_nodes;
  auto pre_index = SizeToLong(pre_nodes.size() - 1);
  auto cur_index = SizeToLong(cur_nodes.size() - 1);
  auto pre_last_index = pre_index;
  auto cur_last_index = cur_index;
  auto add_unoverlapped_node = [&unoverlapped_nodes](std::vector<BranchInterleaveNodePtr> *nodes_ptr,
                                                     int64_t begin_index, int64_t end_index) {
    MS_EXCEPTION_IF_NULL(nodes_ptr);
    for (auto i = end_index; i > begin_index; --i) {
      auto &check_node = nodes_ptr->at(i);
      if (!check_node->overlapped) {
        (void)unoverlapped_nodes.emplace_back(check_node);
      }
    }
  };

  std::vector<std::pair<int64_t, int64_t>> overlapped_node_indices;
  while (pre_index >= 0 && cur_index >= 0) {
    if (trace_map[pre_index][cur_index] == InterleaveNodeTrace::kUp) {
      --pre_index;
    } else if (trace_map[pre_index][cur_index] == InterleaveNodeTrace::kLeft) {
      --cur_index;
    } else {
      (void)overlapped_node_indices.emplace_back(std::make_pair(pre_index, cur_index));
      add_unoverlapped_node(cur_nodes_ptr, cur_index, cur_last_index);
      add_unoverlapped_node(pre_nodes_ptr, pre_index, pre_last_index);
      --pre_index;
      --cur_index;
      pre_last_index = pre_index;
      cur_last_index = cur_index;
    }
  }

  std::reverse(overlapped_node_indices.begin(), overlapped_node_indices.end());
  for (auto &overlapped_index : overlapped_node_indices) {
    AddDependForOverlap(branch_cnodes_ptr, pre_nodes[overlapped_index.first], cur_nodes[overlapped_index.second],
                        graph);
  }

  std::reverse(unoverlapped_nodes.begin(), unoverlapped_nodes.end());
  return unoverlapped_nodes;
}

void PrintBranchNodes(const std::vector<CNodePtr> &branch_nodes) {
  MS_LOG(INFO) << "Start print split concat interleave branch node name:";
  for (auto &node : branch_nodes) {
    if (node == nullptr) {
      continue;
    }
    MS_LOG(INFO) << node->fullname_with_scope();
  }
  MS_LOG(INFO) << "End print split concat interleave branch node name.";
}
}  // namespace

void InterleaveParallelBranches(const InterLeaveScopePtr &interleave_scope, bool use_dp) {
  MS_EXCEPTION_IF_NULL(interleave_scope);
  MS_EXCEPTION_IF_NULL(interleave_scope->merge_node);
  auto entry_node = interleave_scope->merge_node;
  if (IsPrimitiveCNode(entry_node, prim::kPrimConcat) || IsPrimitiveCNode(entry_node, prim::kPrimAddN)) {
    auto make_tuple = entry_node->input(1);
    MS_EXCEPTION_IF_NULL(make_tuple);
    entry_node = make_tuple->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(entry_node);
  }

  auto branch_outputs = entry_node->inputs();

  // Find independent branch nodes
  std::vector<std::vector<CNodePtr>> total_branch_nodes;
  size_t interleave_branch_num = branch_outputs.size() - 1;
  if (interleave_branch_num <= 1) {
    return;
  }

  for (size_t i = 0; i < interleave_branch_num; ++i) {
    auto branch_id = i + 1;
    auto &branch_output = branch_outputs[branch_id];
    auto branch_output_cnode = branch_output->cast<CNodePtr>();
    if (branch_output_cnode == nullptr) {
      return;
    }
    PropagateBranchId(interleave_scope, branch_output_cnode, branch_id);
  }

  // Get ordered branch nodes
  for (size_t i = 0; i < interleave_branch_num; ++i) {
    auto branch_id = i + 1;
    auto &branch_output = branch_outputs[branch_id];
    auto branch_output_cnode = branch_output->cast<CNodePtr>();
    if (branch_output_cnode == nullptr) {
      return;
    }
    auto branch_nodes = GetBranchOrderedNodes(interleave_scope, branch_output_cnode, branch_id);
    total_branch_nodes.emplace_back(branch_nodes);
    PrintBranchNodes(branch_nodes);
  }

  size_t cur_index = 1;
  auto filter_cost = !use_dp;
  auto pre_nodes = GetInterleaveNodes(&total_branch_nodes[0], 0, filter_cost);
  while (cur_index < total_branch_nodes.size()) {
    auto cur_nodes = GetInterleaveNodes(&total_branch_nodes[cur_index], cur_index, filter_cost);
    // Interleave two branches nodes to overlap comm/comp
    if (pre_nodes.empty()) {
      pre_nodes = cur_nodes;
    } else if (!cur_nodes.empty()) {
      if (use_dp) {
        pre_nodes = InterleaveTwoBranchDP(&total_branch_nodes, &pre_nodes, &cur_nodes, interleave_scope->graph);
      } else {
        pre_nodes = InterleaveTwoBranch(&total_branch_nodes, &pre_nodes, &cur_nodes, interleave_scope->graph);
      }
    }
    ++cur_index;
  }
}

void EraseInterLeaveBranchAttr(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->HasAttr(kInterleaveScopeId)) {
    node->EraseAttr(kInterleaveScopeId);
    if (node->HasAttr(kInterleaveBranchId)) {
      node->EraseAttr(kInterleaveBranchId);
    }
  }
}

void UpdateMatMulGradDualMap(const CNodePtr &node, HashMap<std::string, CNodePtr> *matmul_unique_id_map_ptr,
                             HashMap<CNodePtr, CNodePtr> *matmul_grad_dual_map_ptr) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(matmul_unique_id_map_ptr);
  MS_EXCEPTION_IF_NULL(matmul_grad_dual_map_ptr);
  auto node_name = common::AnfAlgo::GetCNodeName(node);
  if (IsNotMatMul(node_name)) {
    return;
  }

  auto &matmul_unique_id_map = *matmul_unique_id_map_ptr;
  if (IsForwardNode(node)) {
    if (node->HasPrimalAttr(kPrimalAttrUniqueId)) {
      matmul_unique_id_map[GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrUniqueId))] = nullptr;
    }
    return;
  }

  if (!node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
    return;
  }

  auto iter_first_node =
    matmul_unique_id_map.find(GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrForwardUniqueId)));
  if (iter_first_node == matmul_unique_id_map.end()) {
    return;
  }

  if (iter_first_node->second == nullptr) {
    iter_first_node->second = node;
  } else {
    auto &matmul_grad_dual_map = *matmul_grad_dual_map_ptr;
    matmul_grad_dual_map[node] = iter_first_node->second;
    matmul_grad_dual_map[iter_first_node->second] = node;
  }
}
}  // namespace parallel
}  // namespace mindspore
