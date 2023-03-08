/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <vector>
#include <string>
#include <set>
#include "plugin/device/ascend/hal/device/ascend_label_assign.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace device {
namespace ascend {
static constexpr uint32_t kLabelGotoLabelId = 1;
static constexpr uint32_t kLabelSwitchLabelId = 2;

static void UpdateLabelGoto(NotNull<CNodePtr> node) {
  if (common::AnfAlgo::HasNodeAttr(kAttrLabelIndex, node)) {
    return;
  }
  if (node->size() <= kLabelGotoLabelId) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " has invalid input size " << node->size();
  }

  auto input = node->input(kLabelGotoLabelId);
  uint32_t goto_label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(input, kAttrLabelIndex);
  common::AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue<uint32_t>(goto_label_id), node.get());
  MS_LOG(INFO) << "Node " << node->DebugString() << " goto label id " << goto_label_id;
  node->set_inputs({node->input(0)});
}

static void UpdateLabelSwitch(NotNull<CNodePtr> node) {
  if (common::AnfAlgo::HasNodeAttr(kAttrLabelIndex, node)) {
    return;
  }
  if (node->size() <= kLabelGotoLabelId) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " has invalid input size " << node->size();
  }
  std::vector<uint32_t> label_list;
  for (size_t i = kLabelSwitchLabelId; i < node->size(); ++i) {
    auto input = node->input(i);
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>() || common::AnfAlgo::GetCNodeName(input) != kLabelSetOpName) {
      break;
    }

    uint32_t goto_label_id = common::AnfAlgo::GetNodeAttr<uint32_t>(input, kAttrLabelIndex);
    label_list.push_back(goto_label_id);
    MS_LOG(INFO) << "Switch " << node->DebugString() << " case " << i - kLabelSwitchLabelId << ": id " << goto_label_id;
  }
  common::AnfAlgo::SetNodeAttr(kAttrLabelSwitchList, MakeValue<std::vector<uint32_t>>(label_list), node.get());
  node->set_inputs({node->input(kAnfPrimitiveIndex), node->input(kFirstDataInputIndex)});
}

static void AssignLabelForLabelSet(NotNull<std::shared_ptr<session::KernelGraph>> graph, NotNull<uint32_t *> label_id,
                                   const NotNull<std::set<std::shared_ptr<session::KernelGraph>> *> &memo) {
  if (memo->find(graph.get()) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  MS_LOG(INFO) << "Assign label for " << graph->ToString();
  const auto &nodes = graph->execution_order();

  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    std::string node_name = common::AnfAlgo::GetCNodeName(node);
    if (node_name == kLabelSetOpName && !common::AnfAlgo::HasNodeAttr(kAttrLabelIndex, cnode)) {
      common::AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue<uint32_t>(*label_id), node);
      MS_LOG(INFO) << "Node " << node->DebugString() << " assign label id " << *label_id;
      ++(*label_id);
    }
  }

  for (auto &cg : graph->child_graph_order()) {
    AssignLabelForLabelSet(NOT_NULL(cg.lock()), label_id, memo);
  }
}

static void AssignLabelForGotoSwitch(NotNull<std::shared_ptr<session::KernelGraph>> graph,
                                     const NotNull<std::set<std::shared_ptr<session::KernelGraph>> *> memo) {
  if (memo->find(graph.get()) != memo->end()) {
    return;
  }
  memo->insert(graph.get());

  MS_LOG(INFO) << "Process label goto/switch for " << graph->ToString();

  const auto &nodes = graph->execution_order();
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (IsPrimitiveCNode(cnode, prim::kPrimLabelGoto)) {
      UpdateLabelGoto(NOT_NULL(cnode));
      cnode->set_abstract(nullptr);
    }

    if (IsPrimitiveCNode(cnode, prim::kPrimLabelSwitch)) {
      UpdateLabelSwitch(NOT_NULL(cnode));
    }
  }
  for (auto &cg : graph->child_graph_order()) {
    AssignLabelForGotoSwitch(NOT_NULL(cg.lock()), memo);
  }
}

void AscendLabelAssign::AssignLabel(NotNull<std::shared_ptr<session::KernelGraph>> graph) {
  MS_LOG(INFO) << "Assign label start.";
  std::set<std::shared_ptr<session::KernelGraph>> memo;
  uint32_t label_id = 0;
  AssignLabelForLabelSet(graph, NOT_NULL(&label_id), NOT_NULL(&memo));
  memo.clear();
  {
    std::lock_guard<std::mutex> lock(label_num_mutex_);
    label_num_[graph.get().get()] = label_id;
  }
  AssignLabelForGotoSwitch(graph, NOT_NULL(&memo));
  MS_LOG(INFO) << "Assign label end.";
}

uint32_t AscendLabelAssign::GetLabelNum(NotNull<const session::KernelGraph *> graph) {
  std::lock_guard<std::mutex> lock(label_num_mutex_);
  auto iter = label_num_.find(graph.get());
  if (iter == label_num_.end()) {
    MS_LOG(DEBUG) << "Graph " << graph->ToString() << " has not assigned label, default is 0.";
    return 0;
  }
  return iter->second;
}

uint32_t AscendLabelAssign::GetLabelNum(NotNull<std::shared_ptr<session::KernelGraph>> graph) {
  return GetLabelNum(NOT_NULL(graph.get().get()));
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
