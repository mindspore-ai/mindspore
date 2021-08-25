/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/ascend/ir_fusion/set_fracz_group_attr.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "runtime/device/kernel_info.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kTupleGetItemName = "TupleGetItem";
constexpr auto kDependName = "Depend";
constexpr auto kLoadName = "Load";
constexpr size_t kConvFilterInputIndex = 2;
constexpr size_t kElemFilterInputIndex = 1;

void SetAttrForInputNode(const AnfNodePtr &node, int64_t groups) {
  if (node == nullptr) {
    return;
  }
  if (node->isa<Parameter>()) {
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    param->set_fracz_group(groups);
    MS_LOG(INFO) << "set parameter " << param->fullname_with_scope() << " with fracz_group: " << groups;
  } else if (node->isa<CNode>()) {
    AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), node);
    if (AnfAlgo::GetCNodeName(node) == kTransDataOpName) {
      AnfAlgo::SetNodeAttr(kAttrGroups, MakeValue(groups), node);
    }
    auto cnode = node->cast<CNodePtr>();
    SetAttrForInputNode(cnode->input(kElemFilterInputIndex), groups);
  }
}

void SetAttrForConvInput(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto groups = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrGroups);
  AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), cnode);
  if (groups > 1) {
    SetAttrForInputNode(cnode->input(kConvFilterInputIndex), groups);
  }
}

void SetAttrForOptParamInput(const AnfNodePtr &node, int64_t groups) {
  // For optimizer, there may be other parameters used by opt need to be set.
  // For example, moments param used by FusedMulApplyMomentum.
  MS_EXCEPTION_IF_NULL(node);
  auto opt_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(opt_cnode);
  auto opt_inputs = opt_cnode->inputs();
  for (size_t i = 1; i < opt_inputs.size(); ++i) {
    auto input_node = opt_inputs[i];
    if (input_node->isa<CNode>() && AnfAlgo::GetCNodeName(input_node) == kLoadName) {
      auto input_cnode = input_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      input_node = input_cnode->input(kElemFilterInputIndex);
    }
    if (input_node->isa<Parameter>()) {
      auto param = input_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      param->set_fracz_group(groups);
    }
  }
}

void SetFracZGroupIdxForAllReduce(const AnfNodePtr &node, const int64_t index) {
  // When Allreduce do fusion, there may be several FracZ outputs with groups=1 or groups>1,
  // so we need to record the output index with groups>1
  MS_EXCEPTION_IF_NULL(node);
  auto allreduce = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(allreduce);
  if (AnfAlgo::HasNodeAttr(kAttrFracZGroupIdx, allreduce)) {
    auto fz_group_idx = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(allreduce, kAttrFracZGroupIdx);
    fz_group_idx.push_back(index);
    AnfAlgo::SetNodeAttr(kAttrFracZGroupIdx, MakeValue(fz_group_idx), allreduce);
  } else {
    AnfAlgo::SetNodeAttr(kAttrFracZGroupIdx, MakeValue(std::vector<int64_t>{index}), allreduce);
  }
}

void SetAttrForOutputNode(const FuncGraphManagerPtr &manager, const AnfNodePtr &node, int64_t groups,
                          int64_t getitem_idx = 0) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(node);
  if (AnfAlgo::GetCNodeName(node) != kTupleGetItemName) {
    AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), node);
  }
  for (auto node_index : manager->node_users()[node]) {
    auto output_node = node_index.first;
    auto output_name = AnfAlgo::GetCNodeName(output_node);
    if (kOptOperatorSet.find(output_name) != kOptOperatorSet.end()) {
      SetAttrForOptParamInput(output_node, groups);
      AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), output_node);
    } else if (output_name == kTransDataOpName) {
      // Trans to other format, no need to recurse, but need to set Groups attr for TransData
      AnfAlgo::SetNodeAttr(kAttrGroups, MakeValue(groups), output_node);
    } else if (output_name == kAllReduceOpName) {
      int64_t index = static_cast<int64_t>(node_index.second) - 1;
      SetFracZGroupIdxForAllReduce(output_node, index);
      SetAttrForOutputNode(manager, output_node, groups, index);
    } else if (output_name == kTupleGetItemName) {
      auto getitem = output_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(getitem);
      auto getitem_input2 = getitem->input(kInputNodeOutputIndexInTupleGetItem);
      auto output_idx = GetValue<int64_t>(GetValueNode(getitem_input2));
      if (output_idx == getitem_idx) {
        SetAttrForOutputNode(manager, output_node, groups);
      }
    } else {
      SetAttrForOutputNode(manager, output_node, groups);
    }
  }
}

void SetAttrForConvOutput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto groups = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrGroups);
  if (groups > 1) {
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto manager = kernel_graph->manager();
    SetAttrForOutputNode(manager, cnode, groups);
  }
}
}  // namespace

bool SetFraczGroupAttr::Run(const FuncGraphPtr &func_graph) {
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "func_graph is nullptr.";
    return false;
  }
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  for (auto node : node_list) {
    if (node == nullptr || !node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto node_name = AnfAlgo::GetCNodeName(cnode);
    if (node_name == kConv2DOpName || node_name == kConv2DBackpropInputOpName) {
      SetAttrForConvInput(cnode);
    } else if (node_name == kConv2DBackpropFilterOpName) {
      SetAttrForConvOutput(func_graph, cnode);
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
