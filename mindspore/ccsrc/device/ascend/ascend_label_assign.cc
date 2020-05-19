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
#include "device/ascend/ascend_label_assign.h"
#include "session/anf_runtime_algorithm.h"

static constexpr uint32_t kLabelGotoLabelId = 1;
static constexpr uint32_t kLabelSwitchLabelId = 2;

namespace mindspore {
namespace device {
namespace ascend {

static void UpdateLabelGoto(NotNull<CNodePtr> node) {
  if (node->size() <= kLabelGotoLabelId) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " has invalid input size " << node->size();
  }
  auto label_set = AnfAlgo::GetCNodePrimitive(node->input(kLabelGotoLabelId));
  MS_EXCEPTION_IF_NULL(label_set);
  auto value = label_set->GetAttr(kAttrLabelIndex);
  MS_EXCEPTION_IF_NULL(value);
  uint32_t goto_label_id = GetValue<uint32_t>(value);
  AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue<uint32_t>(goto_label_id), node.get());
  MS_LOG(INFO) << "Node " << node->DebugString() << " goto label id " << goto_label_id;
}

static void UpdateLabelSwitch(NotNull<CNodePtr> node) {
  if (node->size() <= kLabelGotoLabelId) {
    MS_LOG(EXCEPTION) << "Node " << node->DebugString() << " has invalid input size " << node->size();
  }
  std::vector<uint32_t> label_list;
  for (size_t i = kLabelSwitchLabelId; i < node->size(); ++i) {
    auto input = node->input(i);
    if (!input->isa<CNode>() || AnfAlgo::GetCNodeName(input) != kLabelSetOpName) {
      break;
    }

    auto label_set = AnfAlgo::GetCNodePrimitive(input);
    MS_EXCEPTION_IF_NULL(label_set);
    auto value = label_set->GetAttr(kAttrLabelIndex);
    MS_EXCEPTION_IF_NULL(value);
    uint32_t goto_label_id = GetValue<uint32_t>(value);
    label_list.push_back(goto_label_id);
    MS_LOG(INFO) << "Switch " << node->DebugString() << " case " << i - kLabelSwitchLabelId << ": id " << goto_label_id;
  }
  AnfAlgo::SetNodeAttr(kAttrLabelSwitchList, MakeValue<std::vector<uint32_t>>(label_list), node.get());
}

void AscendLabelAssign::AssignLabel(NotNull<const std::shared_ptr<session::KernelGraph> &> graph) {
  auto cnode_list = graph->execution_order();
  // 1 assign label id to label_set
  uint32_t cur_label_id = 0;
  for (auto &node : cnode_list) {
    if (AnfAlgo::GetCNodeName(node) == kLabelSetOpName) {
      AnfAlgo::SetNodeAttr(kAttrLabelIndex, MakeValue<uint32_t>(cur_label_id), node);
      MS_LOG(INFO) << "Node " << node->DebugString() << " assign label id " << cur_label_id;
      ++cur_label_id;
    }
  }
  // 2 update label_switch / label_goto
  for (auto &node : cnode_list) {
    if (AnfAlgo::GetCNodeName(node) == kLabelGotoOpName) {
      UpdateLabelGoto(NOT_NULL(node));
    }

    if (AnfAlgo::GetCNodeName(node) == kLabelSwitchOpName) {
      UpdateLabelSwitch(NOT_NULL(node));
    }
  }
}

}  // namespace ascend
}  // namespace device
}  // namespace mindspore
