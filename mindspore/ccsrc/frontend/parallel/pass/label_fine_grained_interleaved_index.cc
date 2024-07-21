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

#include "frontend/parallel/pass/label_fine_grained_interleaved_index.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "include/common/utils/utils.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr size_t kExpectInterleavedNum = 2;
const size_t interleaved_size = 2;

constexpr auto kGradientsFlag = "Gradients";
const size_t node_size_two = 2;
const size_t node_size_three = 3;

bool IsForwardNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  return !(cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId) || cnode->HasAttr(kAttrDuplicated));
}

bool IsBpropNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return node->fullname_with_scope().find(kGradientsFlag) == 0;
}

void SpreadFineGrainedInterleavedIndexForForwardCommNodes(const CNodePtr &cnode, size_t fine_grained_block_index,
                                                          size_t fine_grained_interleaved_index, size_t forward_order) {
  std::queue<CNodePtr> bfs_cnode_queue;
  bfs_cnode_queue.push(cnode);
  auto func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  while (!bfs_cnode_queue.empty()) {
    auto cur_cnode = bfs_cnode_queue.front();
    bfs_cnode_queue.pop();
    auto spread_size = cur_cnode->size();
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      spread_size = node_size_two;
    }
    for (size_t i = 1; i < spread_size; ++i) {
      auto pre_node = cur_cnode->input(i);
      MS_EXCEPTION_IF_NULL(pre_node);
      auto pre_cnode = pre_node->cast<CNodePtr>();
      if (pre_cnode == nullptr) {
        continue;
      }
      // BFS end search condition
      if (IsPrimitiveCNode(pre_cnode, prim::kPrimStridedSlice) &&
          GetCNodePrimitive(pre_cnode)->HasAttr(kAttrFineGrainedInterleavedBlockIndex)) {
        pre_cnode->AddAttr("fine_grained_interleaved_border", MakeValue<size_t>(0));
        pre_cnode->AddAttr(parallel::MICRO_INTERLEAVED_INDEX, MakeValue<size_t>(fine_grained_interleaved_index));
        pre_cnode->AddPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK,
                                 MakeValue<int64_t>(fine_grained_block_index));
        continue;
      }

      if (!IsForwardNode(pre_cnode) || IsPrimitiveCNode(pre_cnode, prim::kPrimUpdateState)) {
        continue;
      }
      if (pre_cnode->HasAttr(FINE_GRAINED_INTERLEAVED_TAG)) {
        continue;
      }
      pre_cnode->AddAttr(FINE_GRAINED_INTERLEAVED_TAG, MakeValue<size_t>(fine_grained_interleaved_index));
      bfs_cnode_queue.push(pre_cnode);
      if (pre_cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId) &&
          (parallel::IsSomePrimitiveList(pre_cnode, {ALL_GATHER, ALL_REDUCE, REDUCE_SCATTER}))) {
        pre_cnode->AddPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK,
                                 MakeValue<int64_t>(fine_grained_block_index));
        pre_cnode->AddAttr(parallel::MICRO_INTERLEAVED_INDEX, MakeValue<size_t>(fine_grained_interleaved_index));
        pre_cnode->AddAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER, MakeValue<size_t>(forward_order));
        MS_LOG(INFO) << "pre_cnode:" << pre_cnode->fullname_with_scope()
                     << " fine_grained_block_index:" << fine_grained_block_index
                     << " add forward_order:" << forward_order;
        forward_order++;
      }
    }
  }
}
void LabelFineGrainedInterleavedBackWardBeginEnd(const std::vector<CNodePtr> &all_nodes) {
  std::vector<CNodePtr> forward_begin_end_nodes;
  for (auto &cnode : all_nodes) {
    if (!cnode->HasAttr("fine_grained_interleaved_border")) {
      continue;
    }
    if (!cnode->HasPrimalAttr(kPrimalAttrUniqueId) || cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }
    forward_begin_end_nodes.push_back(cnode);
  }
  for (auto &cnode : all_nodes) {
    if (!IsBpropNode(cnode)) {
      continue;
    }
    if (!cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      continue;
    }
    auto forward_unique_id = GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId));
    auto bp_node_iter =
      std::find_if(forward_begin_end_nodes.begin(), forward_begin_end_nodes.end(), [&](auto fp_cnode) {
        return GetValue<std::string>(fp_cnode->GetPrimalAttr(kPrimalAttrUniqueId)) == forward_unique_id;
      });
    if (bp_node_iter == forward_begin_end_nodes.end()) {
      continue;
    }
    if ((*bp_node_iter)->HasPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK)) {
      cnode->AddPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK,
                           (*bp_node_iter)->GetPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK));
    }
    if ((*bp_node_iter)->HasAttr(parallel::MICRO_INTERLEAVED_INDEX)) {
      cnode->AddAttr(parallel::MICRO_INTERLEAVED_INDEX, (*bp_node_iter)->GetAttr(parallel::MICRO_INTERLEAVED_INDEX));
    }
    if ((*bp_node_iter)->HasAttr("fine_grained_interleaved_border")) {
      cnode->AddAttr("fine_grained_interleaved_border", (*bp_node_iter)->GetAttr("fine_grained_interleaved_border"));
    }
  }
}

void LabelMicroInterleavedIndexForBackwardCommNodes(const std::vector<CNodePtr> &all_nodes) {
  mindspore::HashMap<std::string, CNodePtr> forward_comm_nodes_map;
  mindspore::HashMap<std::string, CNodePtr> grad_forward_comm_nodes_map;
  for (auto &cnode : all_nodes) {
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    if (!cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
      continue;
    }
    auto forward_node_unique_id = GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrForwardCommNodeUniqueId));
    if (IsBpropNode(cnode)) {
      grad_forward_comm_nodes_map[forward_node_unique_id] = cnode;
      continue;
    }
    if (cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }
    forward_comm_nodes_map[forward_node_unique_id] = cnode;
  }
  for (auto &pair : grad_forward_comm_nodes_map) {
    if (forward_comm_nodes_map.find(pair.first) == forward_comm_nodes_map.end()) {
      continue;
    }
    auto forward_node = forward_comm_nodes_map[pair.first];
    if (!forward_node->HasAttr(parallel::MICRO_INTERLEAVED_INDEX) ||
        !forward_node->HasAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER) ||
        !forward_node->HasPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK)) {
      continue;
    }
    MS_LOG(INFO) << "bp node:" << pair.second->DebugString();
    pair.second->AddAttr(parallel::MICRO_INTERLEAVED_INDEX, forward_node->GetAttr(parallel::MICRO_INTERLEAVED_INDEX));
    pair.second->AddAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER,
                         forward_node->GetAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER));
    pair.second->AddPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK,
                               forward_node->GetPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK));
  }
}

void LabelMicroInterleavedBranchTagForBackwardCommNodes(const std::vector<CNodePtr> &all_nodes) {
  mindspore::HashMap<std::string, CNodePtr> forward_nodes_map;
  mindspore::HashMap<std::string, CNodePtr> grad_forward_nodes_map;
  for (auto &cnode : all_nodes) {
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    if (cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }
    if (cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      auto forward_node_unique_id = GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId));
      grad_forward_nodes_map[forward_node_unique_id] = cnode;
    }
    if (cnode->HasPrimalAttr(kPrimalAttrUniqueId)) {
      auto node_unique_id = GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrUniqueId));
      forward_nodes_map[node_unique_id] = cnode;
    }
  }
  for (auto &pair : grad_forward_nodes_map) {
    if (forward_nodes_map.find(pair.first) == forward_nodes_map.end()) {
      continue;
    }
    auto forward_node = forward_nodes_map[pair.first];
    if (!forward_node->HasAttr(parallel::FINE_GRAINED_INTERLEAVED_TAG)) {
      continue;
    }
    pair.second->AddAttr(parallel::FINE_GRAINED_INTERLEAVED_TAG,
                         forward_node->GetAttr(parallel::FINE_GRAINED_INTERLEAVED_TAG));
  }
}
}  // namespace

static bool IsNeedFineGrainedInterleaved(const FuncGraphManagerPtr &manager) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return false;
  }
  if (!IsTraining(manager)) {
    return false;
  }
  return true;
}

void LabelFineGrainedInterleavedIndex(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  if (!IsNeedFineGrainedInterleaved(manager)) {
    return;
  }

  std::vector<FuncGraphPtr> forward_graphs;
  std::vector<FuncGraphPtr> backward_graphs;
  auto context = MsContext::GetInstance();
  const auto is_cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (!is_cell_reuse) {
    forward_graphs.emplace_back(graph);
    backward_graphs.emplace_back(graph);
  } else {
    for (const auto &each_graph : manager->func_graphs()) {
      if (IsCellReuseForwardGraph(each_graph)) {
        auto forward_graph = each_graph;
        auto backward_graph = GetCellReuseBackwardGraph(forward_graph);
        if (backward_graph == nullptr) {
          MS_LOG(WARNING)
            << "Failed to find backward cell reuse graph, skip pass 'overlap_gradmatmul_and_gradallreduce'.";
          continue;
        }
        forward_graphs.emplace_back(forward_graph);
        backward_graphs.emplace_back(backward_graph);
      }
    }
  }
  for (size_t index = 0; index < forward_graphs.size(); ++index) {
    CNodePtrList forward_interleaved_end_cnode_list;
    auto cur_forward_graph = forward_graphs.at(index);
    auto forward_order_cnodes = cur_forward_graph->GetOrderedCnodes();
    CNodePtrList forward_order_cnode_list(forward_order_cnodes.cbegin(), forward_order_cnodes.cend());
    for (const auto &forward_cnode : forward_order_cnode_list) {
      if (IsPrimitiveCNode(forward_cnode, prim::kPrimConcat) &&
          GetCNodePrimitive(forward_cnode)->HasAttr(kAttrFineGrainedInterleavedBlockIndex)) {
        if (forward_cnode->HasAttr(kAttrDuplicated)) {
          continue;
        }
        forward_interleaved_end_cnode_list.push_back(forward_cnode);
      }
    }

    for (const auto &forward_interleaved_end_cnode : forward_interleaved_end_cnode_list) {
      auto concat_input = forward_interleaved_end_cnode->input(1);
      if (!concat_input->isa<CNode>()) {
        continue;
      }
      auto concat_input_cnode = concat_input->cast<CNodePtr>();
      size_t interleaved_num = concat_input->cast<CNodePtr>()->inputs().size() - 1;
      if (interleaved_num != kExpectInterleavedNum) {
        MS_LOG(WARNING) << "For interleaved end node '" << forward_interleaved_end_cnode->ToString()
                        << "', its interleaved num: " << interleaved_num << " is not equal to " << kExpectInterleavedNum
                        << ", skip it.";
        continue;
      }
      auto block_index = GetValue<int64_t>(
        GetCNodePrimitive(forward_interleaved_end_cnode)->GetAttr(kAttrFineGrainedInterleavedBlockIndex));
      forward_interleaved_end_cnode->AddAttr("fine_grained_interleaved_border", MakeValue<size_t>(1));
      forward_interleaved_end_cnode->AddPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK,
                                                   MakeValue<int64_t>(block_index));
      for (size_t interleaved_index = 0; interleaved_index < interleaved_num; ++interleaved_index) {
        auto input_node = concat_input_cnode->input(interleaved_index + 1);
        auto input_cnode = input_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(input_cnode);
        size_t forward_order = 0;
        if (input_cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId) &&
            (parallel::IsSomePrimitiveList(input_cnode, {ALL_GATHER, ALL_REDUCE, REDUCE_SCATTER}))) {
          input_cnode->AddPrimalAttr(parallel::FINE_GRAINED_INTERLEAVED_BLOCK, MakeValue<int64_t>(block_index));
          input_cnode->AddAttr(parallel::MICRO_INTERLEAVED_INDEX, MakeValue<size_t>(LongToSize(interleaved_index)));
          input_cnode->AddAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER, MakeValue<size_t>(forward_order));
          MS_LOG(INFO) << "pre_cnode:" << input_cnode->fullname_with_scope()
                       << " fine_grained_block_index:" << block_index << " add forward_order:" << forward_order;
          forward_order++;
        }
        SpreadFineGrainedInterleavedIndexForForwardCommNodes(input_node->cast<CNodePtr>(), LongToSize(block_index),
                                                             LongToSize(interleaved_index), forward_order);
        MS_LOG(INFO) << "block_index:" << block_index
                     << ", interleaved_end_cnode:" << forward_interleaved_end_cnode->fullname_with_scope();
      }
      parallel::ParallelContext::GetInstance()->set_enable_fine_grained_micro_interleaved(true);
    }

    auto cur_backward_graph = backward_graphs.at(index);
    auto backward_order_cnodes = cur_backward_graph->GetOrderedCnodes();
    CNodePtrList backward_order_cnode_list(backward_order_cnodes.cbegin(), backward_order_cnodes.cend());
    CNodePtrList all_cnode_list(forward_order_cnode_list);
    all_cnode_list.insert(all_cnode_list.end(), backward_order_cnode_list.begin(), backward_order_cnode_list.end());
    LabelFineGrainedInterleavedBackWardBeginEnd(all_cnode_list);
    LabelMicroInterleavedIndexForBackwardCommNodes(all_cnode_list);
    LabelMicroInterleavedBranchTagForBackwardCommNodes(all_cnode_list);
  }
}
}  // namespace parallel
}  // namespace mindspore
