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

#include "plugin/device/ascend/optimizer/format_type/set_fracz_group_attr.h"
#include <set>
#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore::opt {
namespace {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
constexpr auto kTupleGetItemName = "TupleGetItem";
constexpr auto kMakeTupleName = "MakeTuple";
constexpr auto kUpdateStateName = "UpdateState";
constexpr auto kDependName = "Depend";
constexpr auto kLoadName = "Load";
constexpr size_t kAvgpoolInputSize = 2;
const std::set<std::string> kInOutOperatorSet = {kAllReduceOpName, kBroadcastOpName, kMakeTupleName};
const std::set<std::string> kNeedSetGroupNodes = {kConv2DOpName, kConv2DBackpropInputOpName,
                                                  kConv2DBackpropFilterOpName, kConv2DBackpropInputDOpName,
                                                  kConv2DBackpropFilterDOpName};

int64_t GetAvgpoolGroups(const AnfNodePtr &node, const std::string &node_name) {
  if (node_name == kAvgPoolOpName && common::AnfAlgo::GetInputTensorNum(node) == kAvgpoolInputSize &&
      AnfAlgo::GetInputFormat(node, 1) == kOpFormat_FRAC_Z) {
    auto filter_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, 1);
    if (!filter_shape.empty() && filter_shape[0] > 0) {
      return filter_shape[0];
    }
  }
  return 1;
}

AnfNodePtr GetOutputItem(const FuncGraphManagerPtr &manager, const CNodePtr &cnode, int64_t groups,
                         const size_t index = 0) {
  if (AnfAlgo::GetOutputTensorNum(cnode) == 1) {
    return cnode;
  }
  std::vector<AnfNodePtr> depend_nodes{cnode};
  while (!depend_nodes.empty()) {
    auto node = depend_nodes.back();
    depend_nodes.pop_back();
    for (const auto &node_index : manager->node_users()[node]) {
      if (common::AnfAlgo::CheckPrimitiveType(node_index.first, prim::kPrimDepend) && node_index.second == 1) {
        (void)depend_nodes.emplace_back(node_index.first);
      } else if (common::AnfAlgo::CheckPrimitiveType(node_index.first, prim::kPrimTupleGetItem)) {
        auto getitem_cnode = node_index.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(getitem_cnode);
        auto out_index = common::AnfAlgo::GetTupleGetItemOutIndex(getitem_cnode);
        if (out_index == index) {
          common::AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), getitem_cnode);
          return getitem_cnode;
        }
      }
    }
  }
  return nullptr;
}

bool HasFraczGroupAttrAndSet(const AnfNodePtr &node, size_t index, int64_t groups) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<Parameter>()) {
    auto param = node->cast<ParameterPtr>();
    if (param->fracz_group() != 1) {
      return true;
    }
    param->set_fracz_group(groups);
    return false;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto node_name = common::AnfAlgo::GetCNodeName(cnode);
    if (node_name == kDependName && index != 0) {
      return true;
    }
    bool has_group_attr = false;
    if (kInOutOperatorSet.find(node_name) != kInOutOperatorSet.end()) {
      auto input_num = common::AnfAlgo::GetInputTensorNum(node);
      if (index >= input_num) {
        MS_LOG(EXCEPTION) << "Index out of range, node[" << node->fullname_with_scope() << "] only have " << input_num
                          << " inputs, but get index " << index;
      }
      std::vector<int64_t> fz_group_idx(input_num, 1);
      if (common::AnfAlgo::HasNodeAttr(kAttrFracZGroupIdx, cnode)) {
        fz_group_idx = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrFracZGroupIdx);
        if (input_num > fz_group_idx.size()) {
          (void)fz_group_idx.insert(fz_group_idx.cbegin(), input_num - fz_group_idx.size(), 1);
        }
        if (fz_group_idx[index] != 1) {
          has_group_attr = true;
        }
      }
      fz_group_idx[index] = groups;
      common::AnfAlgo::SetNodeAttr(kAttrFracZGroupIdx, MakeValue(fz_group_idx), cnode);
      return has_group_attr;
    }
    has_group_attr = common::AnfAlgo::HasNodeAttr(kAttrFracZGroup, cnode);
    common::AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), cnode);
    if (node_name == kTransDataOpName) {
      common::AnfAlgo::SetNodeAttr(kAttrGroups, MakeValue(groups), cnode);
    }
    return has_group_attr;
  }
  return true;
}

std::vector<KernelWithIndex> GetCNodeNeighborFraczNodes(const FuncGraphManagerPtr &manager, const CNodePtr &cnode,
                                                        size_t index, int64_t groups) {
  auto node_name = common::AnfAlgo::GetCNodeName(cnode);
  auto input_num = common::AnfAlgo::GetInputTensorNum(cnode);
  auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
  auto &node_user = manager->node_users();
  std::vector<KernelWithIndex> ret;
  if (node_name == kDependName || node_name == kLoadName) {
    if (index != 0) {
      return ret;
    }
    input_num = 1;
    output_num = 1;
  }
  // traverse input
  for (size_t i = 0; i < input_num; ++i) {
    if (AnfAlgo::GetInputFormat(cnode, i) == kOpFormat_FRAC_Z) {
      auto input = cnode->input(i + 1);
      MS_EXCEPTION_IF_NULL(input);
      if (node_name == kTupleGetItemName) {
        auto item_index = common::AnfAlgo::GetTupleGetItemOutIndex(cnode);
        while (input->isa<CNode>() && common::AnfAlgo::GetCNodeName(input) == kDependName) {
          common::AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), input);
          input = input->cast<CNodePtr>()->input(1);
          MS_EXCEPTION_IF_NULL(input);
        }
        (void)ret.emplace_back(input, item_index);
      } else {
        (void)ret.emplace_back(input, 0);
      }
    }
  }
  // traverse output
  for (size_t i = 0; i < output_num; ++i) {
    if (AnfAlgo::GetOutputFormat(cnode, i) == kOpFormat_FRAC_Z) {
      auto output = GetOutputItem(manager, cnode, groups, i);
      if (output != nullptr) {
        (void)std::transform(node_user[output].begin(), node_user[output].end(), std::back_inserter(ret),
                             [](const KernelWithIndex &node_index) {
                               return KernelWithIndex{node_index.first, node_index.second - 1};
                             });
      }
    }
  }
  return ret;
}

std::vector<KernelWithIndex> GetNeighborFraczNodes(const FuncGraphManagerPtr &manager, const AnfNodePtr &node,
                                                   size_t index, int64_t groups) {
  std::vector<KernelWithIndex> ret;
  auto &node_user = manager->node_users();
  if (node->isa<Parameter>()) {
    std::transform(node_user[node].begin(), node_user[node].end(), std::back_inserter(ret),
                   [](const KernelWithIndex &node_index) {
                     return KernelWithIndex{node_index.first, node_index.second - 1};
                   });
  }
  if (!node->isa<CNode>()) {
    return ret;
  }
  auto cnode = node->cast<CNodePtr>();
  auto node_name = common::AnfAlgo::GetCNodeName(cnode);
  if (node_name == kUpdateStateName || node_name == kTransDataOpName) {
    return ret;
  } else if (kInOutOperatorSet.find(node_name) != kInOutOperatorSet.end()) {
    (void)ret.emplace_back(cnode->input(index + 1), index);
    auto output = GetOutputItem(manager, cnode, groups, index);
    if (output != nullptr) {
      (void)std::transform(node_user[output].begin(), node_user[output].end(), std::back_inserter(ret),
                           [](const KernelWithIndex &node_index) {
                             return KernelWithIndex{node_index.first, node_index.second - 1};
                           });
    }
  } else {
    ret = GetCNodeNeighborFraczNodes(manager, cnode, index, groups);
  }
  return ret;
}

bool SetAttrFraczGroup(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto groups = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrGroups);
  if (groups == 1) {
    return false;
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<KernelWithIndex> todo{KernelWithIndex{cnode, 0}};
  while (!todo.empty()) {
    KernelWithIndex node_index = todo.back();
    if (HasFraczGroupAttrAndSet(node_index.first, node_index.second, groups)) {
      todo.pop_back();
      continue;
    }
    auto next_nodes = GetNeighborFraczNodes(manager, node_index.first, node_index.second, groups);
    (void)std::copy(next_nodes.begin(), next_nodes.end(), std::back_inserter(todo));
  }
  return true;
}

bool SetAttrFraczGroup(const FuncGraphPtr &func_graph, const ParameterPtr &param) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(param);
  auto groups = param->fracz_group();
  if (groups == 1) {
    return false;
  }
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<KernelWithIndex> todo{};
  auto used_cnodes = GetNeighborFraczNodes(manager, param, 0, groups);
  std::copy(used_cnodes.begin(), used_cnodes.end(), std::back_inserter(todo));
  while (!todo.empty()) {
    KernelWithIndex node_index = todo.back();
    if (HasFraczGroupAttrAndSet(node_index.first, node_index.second, groups)) {
      todo.pop_back();
      continue;
    }
    auto next_nodes = GetNeighborFraczNodes(manager, node_index.first, node_index.second, groups);
    (void)std::copy(next_nodes.begin(), next_nodes.end(), std::back_inserter(todo));
  }
  return true;
}
}  // namespace

bool SetFraczGroupAttr::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  // clear cnode fracz_group first, since the fracz_group info may be out-of-date in later graph of multi-graph scene
  for (auto &node : node_list) {
    if (node != nullptr && node->isa<CNode>() &&
        common::AnfAlgo::HasNodeAttr(kAttrFracZGroup, node->cast<CNodePtr>()) &&
        common::AnfAlgo::GetCNodeName(node) != kTransDataOpName) {
      common::AnfAlgo::EraseNodeAttr(kAttrFracZGroup, node);
    }
  }
  // set fracz_group attr
  for (auto &node : node_list) {
    if (node == nullptr) {
      continue;
    }
    if (node->isa<Parameter>()) {
      // transmit fracz_group attr through multi graph by parameter
      auto param = node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      changed = SetAttrFraczGroup(func_graph, param) || changed;
    }
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode == nullptr) {
        continue;
      }
      auto node_name = common::AnfAlgo::GetCNodeName(cnode);
      if (kNeedSetGroupNodes.count(node_name) != 0) {
        changed = SetAttrFraczGroup(func_graph, cnode) || changed;
      }
      if (int64_t avgpool_group = GetAvgpoolGroups(node, node_name); avgpool_group != 1) {
        common::AnfAlgo::SetNodeAttr(kAttrGroups, MakeValue(avgpool_group), node);
        changed = SetAttrFraczGroup(func_graph, cnode) || changed;
      }
    }
  }
  return changed;
}
}  // namespace mindspore::opt
