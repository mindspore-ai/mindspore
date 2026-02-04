/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/overlap_1b1f.h"

#include <memory>
#include <queue>
#include <vector>
#include <utility>
#include <algorithm>

#include "primitive/other_ops.h"
#include "include/utils/utils.h"
#include "mindspore/core/include/utils/ms_context.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace opt {
using CNodeMapMap = std::unordered_map<size_t, std::unordered_map<size_t, CNodePtr>>;
using CNodeVectorMapMap = std::unordered_map<size_t, std::unordered_map<size_t, std::vector<CNodePtr>>>;
namespace {
std::vector<std::pair<AnfNodePtr, int>> GetRealOutputNodes(const AnfNodePtr &node) {
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<std::pair<AnfNodePtr, int>> res;
  std::queue<AnfNodePtr> node_queue;
  node_queue.push(node);
  while (!node_queue.empty()) {
    auto queue_end = node_queue.front();
    node_queue.pop();
    auto &user_set = manager->node_users()[queue_end];
    for (auto &pair : user_set) {
      if ((IsPrimitiveCNode(pair.first, prim::kPrimDepend) && pair.second == kIndex2) ||
          IsPrimitiveCNode(pair.first, prim::kPrimMakeTuple)) {
        continue;
      }
      if (IsPrimitiveCNode(pair.first) && AnfUtils::IsRealKernel(pair.first) &&
          !common::AnfAlgo::IsNopNode(pair.first)) {
        res.push_back(pair);
      } else {
        node_queue.push(pair.first);
      }
    }
  }
  return res;
}

void OverlapReceiveBpBegin(const KernelGraphPtr &kernel_graph, const FuncGraphManagerPtr &manager,
                           const std::unordered_map<size_t, CNodePtr> &backward_begins_map,
                           const std::unordered_map<size_t, std::vector<CNodePtr>> &pre_recv_users_map) {
  for (const auto &bp_begin_pair : backward_begins_map) {
    auto index = bp_begin_pair.first;
    if (pre_recv_users_map.count(index) == 0) {
      continue;
    }
    for (const auto &recv_user : pre_recv_users_map.at(index)) {
      common::AnfAlgo::InsertDepend(bp_begin_pair.second, recv_user, manager, kernel_graph, "1f1b_recv_depend");
    }
  }
}

void OverlapReceiveLastCNodes(const KernelGraphPtr &kernel_graph, const FuncGraphManagerPtr &manager,
                              const std::unordered_map<size_t, CNodePtr> &fp_last_cnodes_map,
                              const std::unordered_map<size_t, std::vector<CNodePtr>> &inter_recv_users_map) {
  for (const auto &cnode_pair : fp_last_cnodes_map) {
    auto index = cnode_pair.first;
    if (index == kIndex0) {
      continue;
    }
    if (inter_recv_users_map.count(index) == 0) {
      continue;
    }
    for (const auto &inter_recv_usr : inter_recv_users_map.at(index)) {
      common::AnfAlgo::InsertDepend(cnode_pair.second, inter_recv_usr, manager, kernel_graph, "1f1b_last_recv_depend");
    }
  }
}

void OverlapReceiveMiddleCNodes(const KernelGraphPtr &kernel_graph, const FuncGraphManagerPtr &manager,
                                const std::unordered_map<size_t, CNodePtr> &fp_middle_cnodes_map,
                                const std::unordered_map<size_t, std::vector<CNodePtr>> &inter_recv_map) {
  for (const auto &cnode_pair : fp_middle_cnodes_map) {
    auto index = cnode_pair.first;
    if (index == kIndex0) {
      continue;
    }
    if (inter_recv_map.count(index) == 0) {
      continue;
    }
    for (const auto &inter_recv : inter_recv_map.at(index)) {
      common::AnfAlgo::InsertDepend(inter_recv, cnode_pair.second, manager, kernel_graph, "1f1b_middle_recv_depend");
    }
  }
}

void OverlapAll2All(const KernelGraphPtr &kernel_graph, const CNodeMapMap &forward_input_points_map,
                    const CNodeMapMap &backward_input_points_map, const CNodeVectorMapMap &forward_output_points_map,
                    const CNodeVectorMapMap &backward_output_points_map) {
  auto manager = kernel_graph->manager();
  auto max_1b1f_size = std::max({forward_input_points_map.size(), forward_output_points_map.size(),
                                 backward_input_points_map.size(), backward_output_points_map.size()});
  for (size_t index_1f1b = 0; index_1f1b < max_1b1f_size; ++index_1f1b) {
    if (forward_input_points_map.count(index_1f1b) == 0 || forward_output_points_map.count(index_1f1b) == 0 ||
        backward_input_points_map.count(index_1f1b) == 0 || backward_output_points_map.count(index_1f1b) == 0) {
      MS_LOG(DEBUG) << "The 1b1f index " << index_1f1b << " is not found in pipeline scheduler.";
      continue;
    }
    auto forward_a2a_inputs = forward_input_points_map.at(index_1f1b);
    auto forward_a2a_outputs = forward_output_points_map.at(index_1f1b);
    auto backward_a2a_inputs = backward_input_points_map.at(index_1f1b);
    auto backward_a2a_outputs = backward_output_points_map.at(index_1f1b);
    auto min_a2a_size = std::min(
      {forward_a2a_inputs.size(), forward_a2a_outputs.size(), backward_a2a_inputs.size(), backward_a2a_outputs.size()});
    for (size_t j = 0; j < min_a2a_size; ++j) {
      if (forward_a2a_inputs.count(j) == 0 || forward_a2a_outputs.count(j) == 0 || backward_a2a_inputs.count(j) == 0 ||
          backward_a2a_outputs.count(j) == 0) {
        MS_LOG(DEBUG) << "The all2all index " << j << " is not found.";
        continue;
      }
      // foward_input -> backward_output
      // backwardinput -> next_forward_output
      for (const auto &backward_a2a_output : backward_a2a_outputs.at(j)) {
        common::AnfAlgo::InsertDepend(forward_a2a_inputs.at(j), backward_a2a_output, manager, kernel_graph,
                                      "1b1f_depend1");
      }
      if (forward_a2a_outputs.count(j + 1) != 0) {
        for (const auto &forward_a2a_output : forward_a2a_outputs.at(j + 1)) {
          common::AnfAlgo::InsertDepend(backward_a2a_inputs.at(j), forward_a2a_output, manager, kernel_graph,
                                        "1b1f_depend2");
        }
      }
    }
  }
}

void Remove1b1fCallCall(const KernelGraphPtr &kernel_graph) {
  auto order_cnodes = kernel_graph->GetOrderedCnodes();
  CNodePtrList execution_order(order_cnodes.cbegin(), order_cnodes.cend());
  FuncGraphManagerPtr manager = kernel_graph->manager();
  for (const auto &cnode : execution_order) {
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend) && cnode->HasPrimalAttr(kPrimalAttr1b1fCallCall)) {
      auto depend_input = cnode->input(kIndex1);
      (void)manager->Replace(cnode, depend_input);
    }
  }
}
}  // namespace

bool Overlap1b1f::DoOverlap1b1f(const KernelGraphPtr &kernel_graph) {
  Remove1b1fCallCall(kernel_graph);
  auto order_cnodes = kernel_graph->GetOrderedCnodes();
  CNodePtrList execution_order(order_cnodes.cbegin(), order_cnodes.cend());
  auto manager = kernel_graph->manager();
  CNodeMapMap forward_input_points_map;
  CNodeMapMap backward_input_points_map;
  CNodeVectorMapMap forward_output_points_map;
  CNodeVectorMapMap backward_output_points_map;
  std::unordered_map<size_t, std::vector<CNodePtr>> pre_recv_users_map;
  std::unordered_map<size_t, CNodePtr> fp_last_cnodes_map;
  std::unordered_map<size_t, std::vector<CNodePtr>> inter_recv_users_map;
  std::unordered_map<size_t, std::vector<CNodePtr>> inter_recv_map;
  std::unordered_map<size_t, CNodePtr> fp_middle_cnodes_map;
  std::unordered_map<size_t, CNodePtr> backward_begins_map;
  for (const auto &cnode : execution_order) {
    if (cnode->HasAttr(kCNodeAttr1f1bIndexRecv)) {
      auto recv_index = GetValue<size_t>(cnode->GetAttr(kCNodeAttr1f1bIndexRecv));
      auto recv_users = GetRealOutputNodes(cnode);
      for (const auto &recv_user : recv_users) {
        if (!IsPrimitiveCNode(recv_user.first)) {
          continue;
        }
        pre_recv_users_map[recv_index].push_back(recv_user.first->cast<CNodePtr>());
      }
    }
    if (cnode->HasAttr(kCNodeAttr1f1bIndexInterRecv)) {
      auto recv_index = GetValue<size_t>(cnode->GetAttr(kCNodeAttr1f1bIndexInterRecv));
      inter_recv_map[recv_index].push_back(cnode);
      auto recv_users = GetRealOutputNodes(cnode);
      for (const auto &recv_user : recv_users) {
        if (!IsPrimitiveCNode(recv_user.first)) {
          continue;
        }
        inter_recv_users_map[recv_index].push_back(recv_user.first->cast<CNodePtr>());
      }
    }
    if (!cnode->HasAttr(kCNodeAttr1f1bIndexFp) && !cnode->HasAttr(kCNodeAttr1f1bIndexBp)) {
      continue;
    }
    auto index_1f1b = cnode->HasAttr(kCNodeAttr1f1bIndexFp) ? GetValue<size_t>(cnode->GetAttr(kCNodeAttr1f1bIndexFp))
                                                            : GetValue<size_t>(cnode->GetAttr(kCNodeAttr1f1bIndexBp));
    if (cnode->HasAttr(kCNodeAttr1f1bIndexBpBegin)) {
      backward_begins_map[index_1f1b] = cnode;
    }
    if (cnode->HasAttr(kCNodeAttr1f1bLastCNode)) {
      fp_last_cnodes_map[index_1f1b] = cnode;
    }
    if (cnode->HasAttr(kCNodeAttr1f1bMiddleCNode)) {
      fp_middle_cnodes_map[index_1f1b] = cnode;
    }

    if (cnode->HasAttr(kCNodeAttrForwardAll2AllInput)) {
      auto forward_a2a_input_index = GetValue<size_t>(cnode->GetAttr(kCNodeAttrForwardAll2AllInput));
      forward_input_points_map[index_1f1b][forward_a2a_input_index] = cnode;
    }
    if (cnode->HasAttr(kCNodeAttrForwardAll2AllOutput)) {
      auto forward_a2a_output_index = GetValue<size_t>(cnode->GetAttr(kCNodeAttrForwardAll2AllOutput));
      forward_output_points_map[index_1f1b][forward_a2a_output_index].push_back(cnode);
    }
    if (cnode->HasAttr(kCNodeAttrBackwardAll2AllInput)) {
      auto backward_a2a_input_index = GetValue<size_t>(cnode->GetAttr(kCNodeAttrBackwardAll2AllInput));
      backward_input_points_map[index_1f1b][backward_a2a_input_index] = cnode;
    }
    if (cnode->HasAttr(kCNodeAttrBackwardAll2AllOutput)) {
      auto backward_a2a_output_index = GetValue<size_t>(cnode->GetAttr(kCNodeAttrBackwardAll2AllOutput));
      backward_output_points_map[index_1f1b][backward_a2a_output_index].push_back(cnode);
    }
  }
  OverlapAll2All(kernel_graph, forward_input_points_map, backward_input_points_map, forward_output_points_map,
                 backward_output_points_map);
  OverlapReceiveBpBegin(kernel_graph, manager, backward_begins_map, pre_recv_users_map);
  OverlapReceiveLastCNodes(kernel_graph, manager, fp_last_cnodes_map, inter_recv_users_map);
  OverlapReceiveMiddleCNodes(kernel_graph, manager, fp_middle_cnodes_map, inter_recv_map);
  return true;
}

bool Overlap1b1f::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph == nullptr) {
    MS_LOG(DEBUG) << "Failed convert func_graph to kernel_graph, skip it.";
    return false;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto pp_1f1b_value = ms_context->get_param<std::string>(MS_CTX_PP_1F1B_OVERLAP);
  if (pp_1f1b_value.empty()) {
    MS_LOG(DEBUG) << "Pipeline 1f1b overlap option is not AlltoAll, not enable AlltoAll overlap.";
    return false;
  }
  MS_LOG(DEBUG) << "Status record: start overlap 1f1b optimization. graph id: " << kernel_graph->graph_id();
  auto ret = DoOverlap1b1f(kernel_graph);
  MS_LOG(DEBUG) << "Status record: end overlap 1f1b optimization. graph id: " << kernel_graph->graph_id();
  return ret;
}
}  // namespace opt
}  // namespace mindspore
