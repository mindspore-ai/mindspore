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

#include "backend/common/pass/label_1f1b_overlap_node.h"

#include <string>
#include <memory>
#include <queue>
#include <unordered_set>
#include <vector>
#include "utils/ms_context.h"
#include "include/utils/anfalgo.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "primitive/framework_ops.h"
#include "primitive/array_ops.h"
#include "primitive/other_ops.h"
#include "primitive/auto_generate/gen_ops_primitive_c.h"
#include "primitive/auto_generate/gen_ops_primitive_r.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
void LabelBpBegin(const std::vector<CNodePtr> &begin_cnodes, const std::string &output_tags) {
  if (!begin_cnodes.empty()) {
    size_t middle_cnode_index = begin_cnodes.size() / kSizeTwo;
    auto first_cnode = begin_cnodes[middle_cnode_index];
    first_cnode->AddAttr(output_tags, MakeValue<size_t>(0));
    if (middle_cnode_index >= 1 && output_tags == kCNodeAttrBackwardAll2AllOutput) {
      begin_cnodes[middle_cnode_index - 1]->AddAttr(kCNodeAttr1f1bIndexBpBegin, MakeValue(true));
    }
  }
}

void LabelOutputNodesWithCheck(const AnfNodePtr &node, std::function<bool(const AnfNodePtr &)> check) {
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  const auto &node_users = manager->node_users();
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(node);
  std::unordered_set<AnfNodePtr> visited;
  while (!anf_queue.empty()) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    if (visited.find(queue_end) != visited.end()) {
      continue;
    }
    (void)visited.insert(queue_end);
    if (node_users.count(queue_end) == 0) {
      continue;
    }
    auto user_set = node_users.at(queue_end);
    for (auto &pair : user_set) {
      anf_queue.push(pair.first);
      if (check(pair.first)) {
        pair.first->cast<CNodePtr>()->AddAttr("last_all2all_node_user", MakeValue(true));
        continue;
      }
    }
  }
}

void LabelEndCNode(const CNodePtrList &order_cnode_list, const std::string &input_tags, size_t all2all_input_index,
                   size_t last_a2a_index) {
  std::vector<CNodePtr> end_cnodes;
  auto last_all2all = order_cnode_list.at(last_a2a_index);
  LabelOutputNodesWithCheck(last_all2all, [](auto cnode) {
    return IsPrimitiveCNode(cnode) && AnfUtils::IsRealKernel(cnode) && !common::AnfAlgo::IsNopNode(cnode) &&
           !common::AnfAlgo::IsCommunicationOp(cnode);
  });
  for (size_t idx = last_a2a_index + 1; idx < order_cnode_list.size(); ++idx) {
    auto cnode = order_cnode_list[idx];
    if (cnode->HasAttr("last_all2all_node_user")) {
      end_cnodes.push_back(cnode);
    }
  }
  if (!end_cnodes.empty()) {
    size_t middle_cnode_index = end_cnodes.size() * kSizeThree / kSizeFour;
    auto end_cnode = end_cnodes[middle_cnode_index];
    end_cnode->AddAttr(input_tags, MakeValue<size_t>(all2all_input_index));
    if (end_cnodes.size() > middle_cnode_index + kSizeOne && input_tags == kCNodeAttrForwardAll2AllInput) {
      end_cnodes[middle_cnode_index + kSizeOne]->AddAttr(kCNodeAttr1f1bMiddleCNode, MakeValue(true));
      for (size_t k = middle_cnode_index + kSizeTwo; k < end_cnodes.size(); ++k) {
        end_cnodes[k]->AddAttr(kCNodeAttr1f1bLastCNode, MakeValue(true));
      }
    }
  }
}

FuncGraphManagerPtr GetManager(const FuncGraphPtr &cur_graph) {
  auto mng = cur_graph->manager();
  if (mng == nullptr) {
    auto manager = MakeManager({cur_graph}, false);
    MS_EXCEPTION_IF_NULL(manager);
    manager->AddFuncGraph(cur_graph);
    cur_graph->set_manager(manager);
    mng = manager;
  }
  return mng;
}

bool IsNeededCNode(const CNodePtr &cnode) {
  if (!common::AnfAlgo::IsCommunicationOp(cnode)) {
    return false;
  }
  if (!common::AnfAlgo::IsNeededShape(cnode)) {
    return false;
  }
  auto pp_1f1b_value = MsContext::GetInstance()->get_param<std::string>(MS_CTX_PP_1F1B_OVERLAP);
  return common::AnfAlgo::IsNeededOverlapComm(cnode, pp_1f1b_value);
}
}  // namespace
void LabelAll2AllInputOutput(const FuncGraphPtr &cur_graph, const std::string &input_tags,
                             const std::string &output_tags) {
  auto mng = GetManager(cur_graph);
  auto &node_users = mng->node_users();
  auto order_cnodes = cur_graph->GetOrderedCnodes();
  CNodePtrList order_cnode_list(order_cnodes.cbegin(), order_cnodes.cend());
  std::vector<CNodePtr> begin_cnodes;
  bool push_begin_cnode = true;
  size_t all2all_input_index = 0;
  size_t all2all_output_index = 1;
  size_t last_a2a_index = 0;
  for (size_t idx = 0; idx < order_cnode_list.size(); ++idx) {
    auto cnode = order_cnode_list[idx];
    if (IsPrimitiveCNode(cnode) && AnfUtils::IsRealKernel(cnode) && !common::AnfAlgo::IsNopNode(cnode) &&
        !common::AnfAlgo::IsCommunicationOp(cnode) && cnode->size() > kSizeOne) {
      if (push_begin_cnode) {
        begin_cnodes.push_back(cnode);
      }
    }
    if (!IsNeededCNode(cnode)) {
      continue;
    }
    if (cnode->HasAttr(kAttrDuplicated)) {
      continue;
    }
    last_a2a_index = idx;
    cnode->AddAttr(input_tags, MakeValue<size_t>(all2all_input_index));
    ++all2all_input_index;
    auto all2all_outputs = node_users.at(cnode);
    for (const auto &all2all_output_pair : all2all_outputs) {
      if (IsPrimitiveCNode(all2all_output_pair.first)) {
        all2all_output_pair.first->cast<CNodePtr>()->AddAttr(output_tags, MakeValue<size_t>(all2all_output_index));
      }
    }
    if (!all2all_outputs.empty()) {
      ++all2all_output_index;
      push_begin_cnode = false;
    }
  }
  LabelBpBegin(begin_cnodes, output_tags);
  if (last_a2a_index > 0) {
    LabelEndCNode(order_cnode_list, input_tags, all2all_input_index, last_a2a_index);
  }
}

bool Label1F1BOverlapNode::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto pp_1f1b_value = ms_context->get_param<std::string>(MS_CTX_PP_1F1B_OVERLAP);
  if (pp_1f1b_value.empty()) {
    MS_LOG(DEBUG) << "Pipeline 1f1b overlap option is not AlltoAll, not enable AlltoAll overlap.";
    return false;
  }

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  auto kernel_cnodes = kernel_graph->execution_order();
  for (auto &kernel_cnode : kernel_cnodes) {
    MS_EXCEPTION_IF_NULL(kernel_cnode);
    if (common::AnfAlgo::CheckPrimitiveType(kernel_cnode, prim::kPrimCallInline)) {
      if (!kernel_cnode->HasAttr(kCNodeAttr1f1bIndexFp) && !kernel_cnode->HasAttr(kCNodeAttr1f1bIndexBp)) {
        continue;
      }
      auto inline_subgraph = common::AnfAlgo::GetNodeAttr<KernelGraphPtr>(kernel_cnode, kAttrKernelGraph);
      if (inline_subgraph->has_attr("label_1f1b")) {
        continue;
      }
      if (kernel_cnode->HasAttr(kCNodeAttr1f1bIndexFp)) {
        inline_subgraph->set_attr("forward_graph", kernel_cnode->GetAttr(kCNodeAttr1f1bIndexFp));
      } else {
        inline_subgraph->set_attr("backward_graph", kernel_cnode->GetAttr(kCNodeAttr1f1bIndexBp));
      }
      inline_subgraph->set_attr("label_1f1b", MakeValue(true));
    }
  }
  if (func_graph->has_attr("forward_graph")) {
    LabelAll2AllInputOutput(func_graph, kCNodeAttrForwardAll2AllInput, kCNodeAttrForwardAll2AllOutput);
  }
  if (func_graph->has_attr("backward_graph")) {
    LabelAll2AllInputOutput(func_graph, kCNodeAttrBackwardAll2AllInput, kCNodeAttrBackwardAll2AllOutput);
  }

  return True;
}
}  // namespace opt
}  // namespace mindspore
