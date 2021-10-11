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

#include "backend/optimizer/ascend/format_type/set_fracz_group_attr.h"
#include <set>
#include <string>
#include <algorithm>
#include "backend/session/anf_runtime_algorithm.h"
#include "utils/utils.h"
#include "runtime/device/kernel_info.h"
#include "backend/optimizer/common/helper.h"

namespace mindspore {
namespace opt {
namespace {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
constexpr auto kTupleGetItemName = "TupleGetItem";
constexpr auto kMakeTupleName = "MakeTuple";
constexpr auto kUpdateStateName = "UpdateState";
constexpr auto kDependName = "Depend";
constexpr auto kLoadName = "Load";
const std::set<std::string> kInOutOperatorSet = {kAllReduceOpName, kBroadcastOpName, kMakeTupleName};

AnfNodePtr GetOutputItem(const FuncGraphManagerPtr &manager, const CNodePtr &cnode, int64_t groups,
                         const size_t index = 0) {
  if (AnfAlgo::GetOutputTensorNum(cnode) == 1) {
    return cnode;
  }
  std::vector<AnfNodePtr> depend_nodes{cnode};
  while (!depend_nodes.empty()) {
    auto node = depend_nodes.back();
    depend_nodes.pop_back();
    for (auto node_index : manager->node_users()[node]) {
      if (AnfAlgo::CheckPrimitiveType(node_index.first, prim::kPrimDepend) && node_index.second == 1) {
        (void)depend_nodes.emplace_back(node_index.first);
      } else if (AnfAlgo::CheckPrimitiveType(node_index.first, prim::kPrimTupleGetItem)) {
        auto getitem_cnode = node_index.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(getitem_cnode);
        auto out_index = AnfAlgo::GetTupleGetItemOutIndex(getitem_cnode);
        if (out_index == index) {
          AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), getitem_cnode);
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
    auto node_name = AnfAlgo::GetCNodeName(cnode);
    if (node_name == kDependName && index != 0) {
      return true;
    }
    if (kInOutOperatorSet.find(node_name) != kInOutOperatorSet.end()) {
      auto index_l = SizeToLong(index);
      if (AnfAlgo::HasNodeAttr(kAttrFracZGroupIdx, cnode)) {
        auto fz_group_idx = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrFracZGroupIdx);
        if (std::find(fz_group_idx.begin(), fz_group_idx.end(), index_l) != fz_group_idx.end()) {
          return true;
        }
        fz_group_idx.push_back(index_l);
        AnfAlgo::SetNodeAttr(kAttrFracZGroupIdx, MakeValue(fz_group_idx), cnode);
        return false;
      } else {
        AnfAlgo::SetNodeAttr(kAttrFracZGroupIdx, MakeValue(std::vector<int64_t>{index_l}), cnode);
      }
    }
    if (AnfAlgo::HasNodeAttr(kAttrFracZGroup, cnode)) {
      return true;
    }
    AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), cnode);
    if (node_name == kTransDataOpName) {
      AnfAlgo::SetNodeAttr(kAttrGroups, MakeValue(groups), cnode);
    }
    return false;
  }
  return true;
}

std::vector<KernelWithIndex> GetCNodeNeighborFraczNodes(const FuncGraphManagerPtr &manager, const CNodePtr &cnode,
                                                        size_t index, int64_t groups) {
  auto node_name = AnfAlgo::GetCNodeName(cnode);
  auto input_num = AnfAlgo::GetInputTensorNum(cnode);
  auto output_num = AnfAlgo::GetOutputTensorNum(cnode);
  auto node_user = manager->node_users();
  std::vector<KernelWithIndex> ret;
  if (node_name == kDependName || node_name == kLoadName) {
    if (index != 0) {
      return ret;
    }
    input_num = 1;
    output_num = 1;
  }
  for (size_t i = 0; i < input_num; ++i) {
    if (AnfAlgo::GetInputFormat(cnode, i) == kOpFormat_FRAC_Z) {
      auto input = cnode->input(i + 1);
      if (node_name == kTupleGetItemName) {
        auto item_index = AnfAlgo::GetTupleGetItemOutIndex(cnode);
        while (input->isa<CNode>() && AnfAlgo::GetCNodeName(input) == kDependName) {
          AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(groups), input);
          input = input->cast<CNodePtr>()->input(1);
        }
        (void)ret.emplace_back(input, item_index);
      } else {
        (void)ret.emplace_back(input, 0);
      }
    }
  }
  if (kOptOperatorSet.find(node_name) == kOptOperatorSet.end()) {
    for (size_t i = 0; i < output_num; ++i) {
      if (AnfAlgo::GetOutputFormat(cnode, i) == kOpFormat_FRAC_Z) {
        auto output = GetOutputItem(manager, cnode, groups, i);
        if (output != nullptr) {
          std::transform(node_user[output].begin(), node_user[output].end(), std::back_inserter(ret),
                         [](KernelWithIndex node_index) {
                           return KernelWithIndex{node_index.first, node_index.second - 1};
                         });
        }
      }
    }
  }
  return ret;
}

std::vector<KernelWithIndex> GetNeighborFraczNodes(const FuncGraphManagerPtr &manager, const AnfNodePtr &node,
                                                   size_t index, int64_t groups) {
  std::vector<KernelWithIndex> ret;
  auto node_user = manager->node_users();
  if (node->isa<Parameter>()) {
    std::transform(node_user[node].begin(), node_user[node].end(), std::back_inserter(ret),
                   [](KernelWithIndex node_index) {
                     return KernelWithIndex{node_index.first, node_index.second - 1};
                   });
  }
  if (!node->isa<CNode>()) {
    return ret;
  }
  auto cnode = node->cast<CNodePtr>();
  auto node_name = AnfAlgo::GetCNodeName(cnode);
  if (node_name == kUpdateStateName || node_name == kTransDataOpName) {
    return ret;
  } else if (kInOutOperatorSet.find(node_name) != kInOutOperatorSet.end()) {
    (void)ret.emplace_back(cnode->input(index + 1), index);
    auto output = GetOutputItem(manager, cnode, groups, index);
    if (output != nullptr) {
      std::transform(node_user[output].begin(), node_user[output].end(), std::back_inserter(ret),
                     [](KernelWithIndex node_index) {
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
  auto groups = AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrGroups);
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
    std::copy(next_nodes.begin(), next_nodes.end(), std::back_inserter(todo));
  }
  return true;
}
}  // namespace

bool SetFraczGroupAttr::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool changed = false;
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
    if (node_name == kConv2DOpName || node_name == kConv2DBackpropInputOpName ||
        node_name == kConv2DBackpropFilterOpName) {
      changed = SetAttrFraczGroup(func_graph, cnode) || changed;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
