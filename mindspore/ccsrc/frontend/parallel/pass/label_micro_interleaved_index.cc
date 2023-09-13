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

#include "frontend/parallel/pass/label_micro_interleaved_index.h"
#include <memory>
#include <list>
#include <vector>
#include <string>
#include <algorithm>
#include <queue>
#include <utility>
#include "mindspore/core/ops/framework_ops.h"
#include "include/common/utils/utils.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kGradientsFlag = "Gradients";
const size_t interleaved_size = 2;
const size_t node_size_two = 2;
const size_t node_size_three = 3;
using interleaved_node_pair_vector = std::vector<std::pair<size_t, std::vector<CNodePtr>>>;
bool IsBpropNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  return node->fullname_with_scope().find(kGradientsFlag) == 0;
}

void SpreadMicroInterleavedIndexForForwardCommNodes(const CNodePtr &input_node, size_t micro_interleaved_index,
                                                    int64_t pipeline_micro = -1) {
  std::queue<CNodePtr> node_queue;
  node_queue.push(input_node);
  size_t forward_order = 0;
  while (!node_queue.empty()) {
    auto cnode = node_queue.front();
    node_queue.pop();
    auto cnode_inputs = cnode->inputs();
    auto spread_size = cnode_inputs.size();
    if (IsPrimitiveCNode(cnode, prim::kPrimDepend) || IsPrimitiveCNode(cnode, prim::kPrimLoad)) {
      spread_size = node_size_two;
    }
    for (size_t i = 1; i < spread_size; ++i) {
      auto input = cnode_inputs[i];
      if (!IsPrimitiveCNode(input)) {
        continue;
      }
      if (IsBpropNode(input) || IsPrimitiveCNode(input, prim::kPrimUpdateState)) {
        continue;
      }
      auto input_cnode = input->cast<CNodePtr>();
      if (input_cnode->HasAttr(MICRO_INTERLEAVED_TAG) || input_cnode->HasAttr(INTERLEAVED_NUM)) {
        continue;
      }
      bool is_pipeline = (pipeline_micro >= 0 && input_cnode->HasPrimalAttr(parallel::MICRO));
      if (is_pipeline && GetValue<int64_t>(input_cnode->GetPrimalAttr(parallel::MICRO)) != pipeline_micro) {
        continue;
      }
      input_cnode->AddAttr(MICRO_INTERLEAVED_TAG, MakeValue<size_t>(micro_interleaved_index));
      node_queue.push(input_cnode);
      if (input_cnode->HasPrimalAttr(kPrimalAttrForwardCommNodeUniqueId)) {
        if (pipeline_micro >= 0 && !input_cnode->HasPrimalAttr(parallel::MICRO)) {
          MS_LOG(INFO) << "node :" << input_cnode->DebugString() << " dose not contain micro tag.";
          continue;
        }
        input_cnode->AddAttr(parallel::MICRO_INTERLEAVED_INDEX, MakeValue<size_t>(micro_interleaved_index));
        input_cnode->AddAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER, MakeValue<size_t>(forward_order));
        forward_order++;
      }
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
        !forward_node->HasAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER)) {
      continue;
    }
    pair.second->AddAttr(parallel::MICRO_INTERLEAVED_INDEX, forward_node->GetAttr(parallel::MICRO_INTERLEAVED_INDEX));
    pair.second->AddAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER,
                         forward_node->GetAttr(parallel::MICRO_INTERLEAVED_FORWARD_COMM_ORDER));
  }
}

void LabelMicroInterleavedIndex(const std::vector<CNodePtr> &all_nodes) {
  CNodePtr micro_interleaved_add = nullptr;
  for (auto &cnode : all_nodes) {
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    if (GetCNodePrimitive(cnode)->HasAttr("micro_interleaved_add_flag")) {
      micro_interleaved_add = cnode;
      break;
    }
  }
  if (micro_interleaved_add == nullptr || micro_interleaved_add->size() != node_size_three) {
    return;
  }
  for (size_t i = 1; i < micro_interleaved_add->size(); ++i) {
    auto input_cnode = micro_interleaved_add->input(i)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    SpreadMicroInterleavedIndexForForwardCommNodes(input_cnode, i - 1);
  }
  LabelMicroInterleavedIndexForBackwardCommNodes(all_nodes);
}

void LabelMicroInterleavedIndexLastStage(const std::vector<CNodePtr> &all_nodes) {
  std::vector<CNodePtr> micro_interleaved_add_list;
  for (auto &cnode : all_nodes) {
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    if (GetCNodePrimitive(cnode)->HasAttr("micro_interleaved_add_flag")) {
      micro_interleaved_add_list.push_back(cnode);
    }
  }
  parallel::ParallelContext::GetInstance()->set_pipeline_micro_size(micro_interleaved_add_list.size());
  for (auto &micro_interleaved_add : micro_interleaved_add_list) {
    auto pipeline_micro = GetValue<int64_t>(micro_interleaved_add->GetPrimalAttr(parallel::MICRO));
    for (size_t i = 1; i < micro_interleaved_add->size(); ++i) {
      auto input_cnode = micro_interleaved_add->input(i)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      SpreadMicroInterleavedIndexForForwardCommNodes(input_cnode, i - 1, pipeline_micro);
    }
  }
  LabelMicroInterleavedIndexForBackwardCommNodes(all_nodes);
}

void LabelMicroInterleavedIndexPipelineStage(const std::vector<CNodePtr> &all_nodes) {
  mindspore::HashMap<size_t, std::vector<CNodePtr>> pipeline_end_list_map;
  std::vector<size_t> micro_list;
  for (auto &cnode : all_nodes) {
    if (!IsPrimitiveCNode(cnode)) {
      continue;
    }
    if (IsBpropNode(cnode)) {
      continue;
    }
    if (!cnode->HasPrimalAttr(parallel::PIPELINE_END) || !cnode->HasPrimalAttr(parallel::MICRO)) {
      continue;
    }
    size_t pipeline_end = LongToSize(GetValue<int64_t>(cnode->GetPrimalAttr(parallel::PIPELINE_END)));
    size_t micro = LongToSize(GetValue<int64_t>(cnode->GetPrimalAttr(parallel::MICRO)));
    if (pipeline_end != micro) {
      continue;
    }
    if (pipeline_end_list_map.find(pipeline_end) == pipeline_end_list_map.end()) {
      pipeline_end_list_map[pipeline_end] = {cnode};
      micro_list.push_back(pipeline_end);
    } else {
      pipeline_end_list_map[pipeline_end].push_back(cnode);
    }
  }
  parallel::ParallelContext::GetInstance()->set_pipeline_micro_size(micro_list.size());
  for (size_t i = 0; i < micro_list.size(); ++i) {
    auto pipeline_end_list = pipeline_end_list_map[micro_list[i]];
    if (pipeline_end_list.size() != interleaved_size) {
      continue;
    }
    if (GetCNodePrimitive(pipeline_end_list[0])->HasAttr(parallel::SR_TAG) &&
        GetCNodePrimitive(pipeline_end_list[1])->HasAttr(parallel::SR_TAG)) {
      std::sort(pipeline_end_list.begin(), pipeline_end_list.end(), [](auto cnode1, auto cnode2) {
        return GetValue<int64_t>(GetCNodePrimitive(cnode1)->GetAttr(parallel::SR_TAG)) <
               GetValue<int64_t>(GetCNodePrimitive(cnode2)->GetAttr(parallel::SR_TAG));
      });
    }
    SpreadMicroInterleavedIndexForForwardCommNodes(pipeline_end_list[0], 0, micro_list[i]);
    SpreadMicroInterleavedIndexForForwardCommNodes(pipeline_end_list[1], 1, micro_list[i]);
  }
  LabelMicroInterleavedIndexForBackwardCommNodes(all_nodes);
}
}  // namespace

// Labeling the micro_interleaved_index to all forward nodes,
// and the corresponding backward model parallel communication nodes (by primal_attr 'forward_node_unique_id').
// from the the last add nodes in the final part of forward net for converging the micro_interleaved branch,
// traversing the two branch and tagging the nodes in the same branch.
//      -> A0 -> B0 -> C0 -> D0 ->
// slice                             Add
//      -> A1 -> B1 -> C1 -> D1 ->
void LabelMicroInterleavedIndex(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  if (!parallel::ParallelContext::GetInstance()->enable_micro_interleaved()) {
    return;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (cell_reuse) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> origin_nodes_topological(orders.cbegin(), orders.cend());
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() == 1) {
    LabelMicroInterleavedIndex(origin_nodes_topological);
    return;
  }
  MS_EXCEPTION_IF_NULL(parallel::g_device_manager);
  auto stage_num = parallel::g_device_manager->stage_num();
  auto stage_id = parallel::g_device_manager->stage_id();
  if (stage_id == stage_num - 1) {
    LabelMicroInterleavedIndexLastStage(origin_nodes_topological);
  } else {
    LabelMicroInterleavedIndexPipelineStage(origin_nodes_topological);
  }
}
}  // namespace parallel
}  // namespace mindspore
