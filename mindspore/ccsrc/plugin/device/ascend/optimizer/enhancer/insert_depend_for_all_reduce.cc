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
#include "plugin/device/ascend/optimizer/enhancer/insert_depend_for_all_reduce.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "ops/framework_ops.h"

namespace mindspore {
namespace opt {

void InsertDependForAllReduce::InsertDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node,
                                            const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(post_node);
  auto post_cnode = post_node->cast<CNodePtr>();
  auto manager = graph->manager();
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), post_cnode->input(1), prior_node};
  auto depend_node = graph->NewCNode(depend_input);
  manager->SetEdge(post_node, 1, depend_node);
}

void InsertDependForAllReduce::InsertAllReduceOpAfterSendOp(const FuncGraphPtr &graph) {
  for (size_t i = 0; i < all_reduce_node_.size(); ++i) {
    auto prim = GetCNodePrimitive(all_reduce_node_[i]);
    auto segment_info = GetValue<int64_t>(prim->GetAttr(kAttrSegment));
    if (backward_each_seg_last_send_.find(segment_info) != backward_each_seg_last_send_.end() && segment_info != 0) {
      MS_LOG(INFO) << "Backward micro max send is: "
                   << backward_each_seg_last_send_[segment_info]->fullname_with_scope();
      auto before_send_op = backward_each_seg_last_send_[segment_info]->cast<CNodePtr>()->input(1);
      if (IsPrimitiveCNode(before_send_op, prim::kPrimDepend)) {
        before_send_op = before_send_op->cast<CNodePtr>()->input(1);
      }
      MS_LOG(INFO) << "Before send op is:" << before_send_op->fullname_with_scope();
      InsertDepend(all_reduce_node_[i], before_send_op, graph);
    }
  }
}

void InsertDependForAllReduce::HandleAllReduceUsersNode(const FuncGraphPtr &graph) {
  for (size_t i = 0; i < allreduce_users_list_.size(); ++i) {
    for (size_t j = 1; j < allreduce_users_list_[i].size(); ++j) {
      InsertDepend(allreduce_users_list_[i][j - 1], allreduce_users_list_[i][j], graph);
    }
    InsertDepend(last_allreduce_, allreduce_users_list_[i][0], graph);
  }
}

void InsertDependForAllReduce::FindEachSegLastSend() {
  for (auto &node : node_list_) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsPrimitiveCNode(cnode, prim::kPrimSend) && cnode->HasPrimalAttr(kPrimalAttrForwardNodeName) &&
        cnode->HasPrimalAttr(kAttrMicro) && GetValue<int64_t>(cnode->GetPrimalAttr(kAttrMicro)) == micro_max_ &&
        cnode->HasPrimalAttr(kAttrSegment)) {
      backward_each_seg_last_send_[GetValue<int64_t>(cnode->GetPrimalAttr(kAttrSegment))] = node;
    }
  }
}

bool InsertDependForAllReduce::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  auto parallel_context = parallel::ParallelContext::GetInstance();
  if (!parallel_context->enable_fold_pipeline()) {
    return changed;
  }
  node_list_ = TopoSort(graph->get_return());
  for (auto &node : node_list_) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode->HasPrimalAttr(kAttrMicro) && cnode->GetPrimalAttr(kAttrMicro)->isa<Int64Imm>()) {
      int64_t micro = GetValue<int64_t>(cnode->GetPrimalAttr(kAttrMicro));
      micro_max_ = std::max(micro_max_, micro);
    }
    bool is_recompute = cnode->GetAttr(kAttrDuplicated) != nullptr && GetValue<bool>(cnode->GetAttr(kAttrDuplicated));
    if (common::AnfAlgo::GetCNodeName(cnode) == kAllReduceOpName && common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode) &&
        common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion) > 0 && !is_recompute) {
      all_reduce_node_.push_back(node);
      auto segment_info = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrSegment);
      auto fusion_info = common::AnfAlgo::GetNodeAttr<int64_t>(node, kAttrFusion);
      MS_LOG(INFO) << "Find all reduce cnode :" << cnode->fullname_with_scope() << ", segment_info" << segment_info
                   << ", fusion_info" << fusion_info;
      if (fusion_info < min_fusion_) {
        min_fusion_ = fusion_info;
        last_allreduce_ = node;
      }
      if (segment_info == 0) {
        continue;
      }
      auto node_users = graph->manager()->node_users()[node];
      std::vector<AnfNodePtr> node_users_list;
      for (auto &node_user : node_users) {
        MS_LOG(INFO) << "Node_user: " << node_user.first->fullname_with_scope();
        node_users_list.push_back(node_user.first->cast<AnfNodePtr>());
      }
      allreduce_users_list_.push_back(node_users_list);
      changed = true;
    }
  }
  FindEachSegLastSend();
  InsertAllReduceOpAfterSendOp(graph);
  HandleAllReduceUsersNode(graph);
  return changed;
}
}  // namespace opt
}  // namespace mindspore
