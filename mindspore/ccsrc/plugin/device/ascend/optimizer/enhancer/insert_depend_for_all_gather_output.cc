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

#include <algorithm>
#include "plugin/device/ascend/optimizer/enhancer/insert_depend_for_all_gather_output.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "ops/structure_op_name.h"
#include "ops/framework_op_name.h"
#include "ops/framework_ops.h"

namespace mindspore {
namespace opt {

void InsertDependForAllGatherOutput::InsertDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node,
                                                  const FuncGraphPtr &root) const {
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(post_node);
  auto post_cnode = post_node->cast<CNodePtr>();
  auto manager = root->manager();
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), post_cnode->input(1), prior_node};
  auto depend_node = root->NewCNode(depend_input);
  manager->SetEdge(post_node, 1, depend_node);
}

int64_t InsertDependForAllGatherOutput::DealSegment(const std::vector<AnfNodePtr> &node_list) {
  int64_t seg_max = -1;
  for (auto &node : node_list) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    // get forward segment first recv
    if (!cnode->HasPrimalAttr(kPrimalAttrForwardNodeName) && cnode->HasPrimalAttr(kAttrSegment) &&
        cnode->HasPrimalAttr(kAttrMicro) && GetValue<int64_t>(cnode->GetPrimalAttr(kAttrMicro)) == 0 &&
        cnode->HasPrimalAttr("pipeline_begin")) {
      forward_each_seg_first_recv_[GetValue<int64_t>(cnode->GetPrimalAttr(kAttrSegment))].push_back(node);
      MS_LOG(INFO) << "Forward pipeline begin op is: " << node->fullname_with_scope()
                   << ", segment info: " << GetValue<int64_t>(cnode->GetPrimalAttr(kAttrSegment));
    }
    // get max segment
    if (cnode->HasPrimalAttr(kAttrSegment)) {
      auto segment_info = GetValue<int64_t>(cnode->GetPrimalAttr(kAttrSegment));
      seg_max = std::max(seg_max, segment_info);
    }
  }
  return seg_max;
}

bool InsertDependForAllGatherOutput::IsLastSegWithRecv(int64_t seg_max, std::shared_ptr<CNode> cnode) {
  return common::AnfAlgo::GetCNodeName(cnode) == kReceiveOpName && cnode->HasPrimalAttr(kAttrSegment) &&
         GetValue<int64_t>(cnode->GetPrimalAttr(kAttrSegment)) == seg_max &&
         !cnode->HasPrimalAttr(kPrimalAttrForwardNodeName);
}

bool InsertDependForAllGatherOutput::IsGatherNode(std::shared_ptr<CNode> cnode, bool is_recompute) {
  return common::AnfAlgo::GetCNodeName(cnode) == kAllGatherOpName && common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode) &&
         common::AnfAlgo::HasNodeAttr(kAttrSegment, cnode) &&
         common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion) > 0 && !is_recompute;
}

bool InsertDependForAllGatherOutput::IsRedistriuteAllGatherNode(int64_t seg_max, std::shared_ptr<CNode> cnode) {
  return common::AnfAlgo::GetCNodeName(cnode) == kAllGatherOpName &&
         cnode->fullname_with_scope().find("head-PanGuHead") != std::string::npos &&
         cnode->HasPrimalAttr(kAttrSegment) && GetValue<int64_t>(cnode->GetPrimalAttr(kAttrSegment)) == seg_max;
}

void InsertDependForAllGatherOutput::GetEachSegSend(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &node_list,
                                                    int64_t seg_max) {
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (common::AnfAlgo::GetCNodeName(cnode) == kSendOpName && cnode->HasPrimalAttr("pipeline_param") &&
        !cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      pipeline_param_send_ = node;
      MS_LOG(INFO) << "Pipeline_param_send_ is: " << pipeline_param_send_->fullname_with_scope();
    }
    if (IsLastSegWithRecv(seg_max, cnode)) {
      auto micro_info = GetValue<int64_t>(cnode->GetPrimalAttr(kAttrMicro));
      forward_last_seg_each_micro_recv_[micro_info] = node;
    }
    bool is_recompute = cnode->GetAttr(kAttrDuplicated) != nullptr && GetValue<bool>(cnode->GetAttr(kAttrDuplicated));
    if (IsGatherNode(cnode, is_recompute)) {
      all_gather_node_[common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion)] = node;
    }
    if (IsRedistriuteAllGatherNode(seg_max, cnode)) {
      redistribution_all_gather_node_.push_back(cnode->input(1));
    }
    if (common::AnfAlgo::GetCNodeName(cnode) == kGetNextOpName) {
      auto node_users = graph->manager()->node_users()[node];
      auto node_pair = node_users.begin();
      for (size_t j = 0; j < node_users.size(); ++j) {
        auto current_node = node_pair->first;
        auto current_node_users = graph->manager()->node_users()[current_node];
        auto current_node_pair = current_node_users.begin();
        for (size_t k = 0; k < current_node_users.size(); ++k) {
          auto current_node_user_node = current_node_pair->first;
          get_next_tuplegetitem_node_.push_back(current_node_user_node);
        }
        node_pair++;
      }
    }
  }
}

void InsertDependForAllGatherOutput::ReorderGetnext(const FuncGraphPtr &graph, bool *changed) {
  if (pipeline_param_send_ != nullptr) {
    for (size_t i = 0; i < get_next_tuplegetitem_node_.size(); ++i) {
      auto current_node = get_next_tuplegetitem_node_[i];
      MS_LOG(INFO) << "Insert depend for getnext tuplegetitem before first allgather op  "
                   << pipeline_param_send_->fullname_with_scope();
      InsertDepend(current_node, all_gather_node_.begin()->second, graph);
      InsertDepend(current_node, pipeline_param_send_, graph);
      *changed = true;
    }
  }

  auto iter = all_gather_node_.end();
  iter--;
  if (forward_each_seg_first_recv_.find(0) != forward_each_seg_first_recv_.end()) {
    for (auto &node : forward_each_seg_first_recv_[0]) {
      MS_LOG(INFO) << "Insert depend last allgather node before recv op  " << node->fullname_with_scope();
      InsertDepend(iter->second, node, graph);
    }
  }
}

void InsertDependForAllGatherOutput::IsChanged(const FuncGraphPtr &graph, AnfNodePtr node, int64_t segment_info,
                                               bool *changed) {
  if (segment_info != 0) {
    auto node_users = graph->manager()->node_users()[node];
    auto current_node_pair = node_users.begin();
    for (auto &forward_node : forward_each_seg_first_recv_[segment_info]) {
      MS_LOG(INFO) << "Insert depend for tuplegetitem after recv op  " << forward_node->fullname_with_scope();
      InsertDepend(forward_node, current_node_pair->first, graph);
      *changed = true;
    }
    for (size_t j = 0; j < node_users.size() - 1; ++j) {
      auto current_node = current_node_pair->first;
      auto next_node = (++current_node_pair)->first;
      MS_LOG(INFO) << "Current_node " << current_node->fullname_with_scope() << ", next_node "
                   << next_node->fullname_with_scope();
      InsertDepend(current_node, next_node, graph);
      *changed = true;
    }
  }
}

bool InsertDependForAllGatherOutput::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  auto parallel_context = parallel::ParallelContext::GetInstance();
  if (!parallel_context->enable_fold_pipeline()) {
    return changed;
  }
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());

  // find each seg last send, seg_max
  int64_t seg_max = DealSegment(node_list);

  GetEachSegSend(graph, node_list, seg_max);

  if (!forward_each_seg_first_recv_.empty()) {
    for (auto &node_pair : all_gather_node_) {
      auto node = node_pair.second;
      auto segment_info = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrSegment);
      MS_LOG(INFO) << "Node " << node->fullname_with_scope() << ", segment info: " << segment_info;
      IsChanged(graph, node, segment_info, &changed);
    }
  }

  for (size_t i = 0; i < redistribution_all_gather_node_.size(); ++i) {
    auto current_node = redistribution_all_gather_node_[i];
    auto micro_info = GetValue<int64_t>(current_node->cast<CNodePtr>()->GetPrimalAttr(kAttrMicro));
    MS_LOG(INFO) << "Current_node " << current_node->fullname_with_scope() << ", micro_info " << micro_info;
    InsertDepend(forward_last_seg_each_micro_recv_[micro_info], current_node, graph);
    changed = true;
  }

  if (all_gather_node_.empty()) {
    return changed;
  }

  // reorder getnext tensormove to before pipeline param send
  ReorderGetnext(graph, &changed);

  return changed;
}

}  // namespace opt
}  // namespace mindspore
