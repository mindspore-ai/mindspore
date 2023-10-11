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

#include "plugin/device/ascend/optimizer/ge/hcom/add_depend_for_all_gather.h"
#include <map>
#include <vector>
#include <utility>
#include "ops/other_op_name.h"
#include "ops/sequence_ops.h"
#include "ops/framework_ops.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr GetFirstNextUsers(const FuncGraphPtr &graph, const AnfNodePtr &input,
                             const std::map<AnfNodePtr, size_t> &node_index_map,
                             const std::vector<AnfNodePtr> &all_nodes) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(input);
  if (iter == node_users.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "node has no output in manager." << trace::DumpSourceLines(input);
  }
  auto user_items = iter->second;
  auto min_index = all_nodes.size();
  for (const auto &node_pair : user_items) {
    auto node = node_pair.first;
    MS_EXCEPTION_IF_NULL(node);
    auto cnode = node->cast<CNodePtr>();
    if (IsPrimitiveCNode(node, prim::kPrimMakeTuple) || common::AnfAlgo::GetCNodeName(cnode) == kDependOpName) {
      continue;
    }
    if (node_index_map.find(node) == node_index_map.end()) {
      MS_LOG(EXCEPTION) << "Can not find node in node_index_map, node: " << node->fullname_with_scope();
    }
    if (min_index > node_index_map.at(node)) {
      min_index = node_index_map.at(node);
    }
  }

  if (min_index == all_nodes.size()) {
    MS_LOG(INFO) << "Node has no successor node, node: " << input->fullname_with_scope();
    return nullptr;
  }

  auto succ_node = all_nodes[min_index];
  MS_LOG(DEBUG) << "Input node: " << input->fullname_with_scope()
                << ", successor node: " << succ_node->fullname_with_scope();
  return succ_node;
}

void AddDependCtrl(AnfNodePtr first_node, AnfNodePtr second_node, const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(first_node);
  auto first_cnode = first_node->cast<CNodePtr>();
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                    common::AnfAlgo::GetInputNode(first_cnode, 0), second_node};
  MS_LOG(DEBUG) << "Add Depend between node: " << second_node->fullname_with_scope()
                << " and node: " << first_node->fullname_with_scope();
  auto new_input = graph->NewCNode(inputs);
  new_input->set_abstract(common::AnfAlgo::GetInputNode(first_cnode, 0)->abstract());
  common::AnfAlgo::SetNodeInput(first_cnode, new_input, 0);
}

bool InsertDependForAllGatherParallelNoFusion(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &all_gather_node,
                                              const std::vector<AnfNodePtr> &allgather_succ_nodes,
                                              const std::vector<AnfNodePtr> &allgather_second_succ_nodes) {
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    MS_LOG(DEBUG) << "AllGather parallel optimization is not required in pipeline parallel mode.";
    return false;
  }
  bool changed = false;
  for (size_t i = 1; i < all_gather_node.size(); i++) {
    MS_EXCEPTION_IF_NULL(all_gather_node[i]);
    if (allgather_second_succ_nodes[i] == nullptr || allgather_succ_nodes[i] == nullptr) {
      MS_LOG(DEBUG) << "AllGather has no successor node or second successor node, AllGather name: "
                    << all_gather_node[i]->fullname_with_scope();
      continue;
    }
    if (!IsPrimitiveCNode(allgather_succ_nodes[i], prim::kPrimLoad)) {
      MS_LOG(DEBUG) << "AllGather successor node it not Load, but is: "
                    << allgather_succ_nodes[i]->fullname_with_scope()
                    << ", AllGather node:" << all_gather_node[i]->fullname_with_scope();
      continue;
    }
    AnfNodePtr another_input = nullptr;
    auto second_succ_cnode = allgather_second_succ_nodes[i]->cast<CNodePtr>();
    auto succ_cnode = allgather_succ_nodes[i]->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(second_succ_cnode);
    MS_EXCEPTION_IF_NULL(succ_cnode);
    for (size_t j = 1; j < second_succ_cnode->inputs().size(); j++) {
      auto succ_input = second_succ_cnode->input(j);
      if (succ_input != allgather_succ_nodes[i]) {
        another_input = succ_input;
        break;
      }
    }
    if (another_input == nullptr) {
      MS_LOG(DEBUG) << "AllGather second successor node has no other input, AllGather name: "
                    << all_gather_node[i]->fullname_with_scope()
                    << ", second successor node: " << second_succ_cnode->fullname_with_scope();
      continue;
    }
    AddDependCtrl(succ_cnode, another_input, graph);
    changed = true;
  }
  return changed;
}

bool InsertDependForAllGatherParallel(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &node_list,
                                      const std::map<int64_t, std::vector<AnfNodePtr>> &fusion_allgather_nodes,
                                      const std::vector<AnfNodePtr> &no_fusion_allgather_nodes,
                                      const std::vector<AnfNodePtr> &no_fusion_allgather_succ_nodes,
                                      const std::map<AnfNodePtr, size_t> &node_index_map, const bool is_need_fusion) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  if (!is_need_fusion) {
    std::vector<AnfNodePtr> no_fusion_allgather_second_succ_nodes;
    for (auto &succ_node : no_fusion_allgather_succ_nodes) {
      if (succ_node == nullptr) {
        continue;
      }
      auto allgather_second_succ = GetFirstNextUsers(graph, succ_node, node_index_map, node_list);
      no_fusion_allgather_second_succ_nodes.push_back(allgather_second_succ);
    }
    changed = InsertDependForAllGatherParallelNoFusion(graph, no_fusion_allgather_nodes, no_fusion_allgather_succ_nodes,
                                                       no_fusion_allgather_second_succ_nodes);
    return changed;
  }
  std::map<int64_t, std::vector<AnfNodePtr>> fusion_allgather_outputs;
  auto iter = fusion_allgather_nodes.begin();
  for (size_t i = 1; i < fusion_allgather_nodes.size(); i++) {
    auto pre_fusion_id = (iter)->first;
    auto nodes = (iter++)->second;
    for (size_t j = 0; j < nodes.size(); ++j) {
      auto node = j < nodes.size() - 1 ? nodes[j + 1] : (iter->second).at(0);
      MS_EXCEPTION_IF_NULL(node);
      auto allgather_first_succ = GetFirstNextUsers(graph, node, node_index_map, node_list);
      auto allgather_second_succ = GetFirstNextUsers(graph, allgather_first_succ, node_index_map, node_list);
      if (allgather_first_succ == nullptr || allgather_second_succ == nullptr) {
        MS_LOG(DEBUG) << "AllGather has no successor node or second successor node, AllGather name: "
                      << node->fullname_with_scope();
        continue;
      }
      if (!IsPrimitiveCNode(allgather_first_succ, prim::kPrimLoad)) {
        MS_LOG(DEBUG) << "AllGather successor node is not Load, but is: " << node->fullname_with_scope()
                      << ", AllGather node:" << node->fullname_with_scope();
        continue;
      }
      AnfNodePtr another_input = nullptr;
      auto second_succ_cnode = allgather_second_succ->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(second_succ_cnode);
      for (size_t k = 1; k < second_succ_cnode->inputs().size(); k++) {
        auto succ_input = second_succ_cnode->input(k);
        if (succ_input != nullptr && succ_input != allgather_first_succ) {
          another_input = succ_input;
          break;
        }
      }
      if (another_input != nullptr) {
        fusion_allgather_outputs[pre_fusion_id].push_back(another_input);
      }
    }
  }
  auto node_iter = fusion_allgather_nodes.begin();
  for (size_t i = 1; i < fusion_allgather_nodes.size(); i++) {
    auto pre_fusion_id = node_iter->first;
    auto nodes = (++node_iter)->second;
    auto node = nodes.at(0);
    MS_EXCEPTION_IF_NULL(node);
    auto allgather_first_succ = GetFirstNextUsers(graph, node, node_index_map, node_list);
    if (allgather_first_succ == nullptr) {
      MS_LOG(DEBUG) << "AllGather has no successor node or second successor node, AllGather name: "
                    << node->fullname_with_scope();
      continue;
    }
    if (!IsPrimitiveCNode(allgather_first_succ, prim::kPrimLoad)) {
      MS_LOG(DEBUG) << "AllGather successor node it not Load, but is: " << node->fullname_with_scope()
                    << ", AllGather node:" << node->fullname_with_scope();
      continue;
    }
    auto &pre_fusion_outputs = fusion_allgather_outputs[pre_fusion_id];
    auto &pre_fusion_last_output = pre_fusion_outputs.back();
    AddDependCtrl(allgather_first_succ, pre_fusion_last_output, graph);
    changed = true;
  }
  return changed;
}

bool AddDependCtrlForMemReuse(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &node_list,
                              const std::map<int64_t, std::vector<AnfNodePtr>> &fusion_allgather_nodes,
                              const std::map<int64_t, AnfNodePtr> &all_gather_node,
                              const std::vector<AnfNodePtr> &no_fusion_allgather_nodes,
                              const std::vector<AnfNodePtr> &no_fusion_allgather_succ_nodes,
                              const std::map<AnfNodePtr, size_t> &node_index_map, const bool is_need_fusion) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  if (is_need_fusion) {
    auto iter = all_gather_node.begin();
    auto fusion_nodes_iter = fusion_allgather_nodes.begin();
    auto fusion_size = all_gather_node.size();
    for (size_t i = 1; i < fusion_size; ++i) {
      auto fusion_nodes = (fusion_nodes_iter++)->second;
      auto next_fusion_nodes = fusion_nodes_iter->second;
      auto fusion_first_node = fusion_nodes.at(0);
      auto fusion_last_node = (iter++)->second;
      MS_EXCEPTION_IF_NULL(fusion_first_node);
      MS_EXCEPTION_IF_NULL(fusion_last_node);
      auto fusion_first_succ = GetFirstNextUsers(graph, fusion_first_node, node_index_map, node_list);
      auto fusion_second_succ = GetFirstNextUsers(graph, fusion_first_succ, node_index_map, node_list);
      auto fusion_last_succ = GetFirstNextUsers(graph, fusion_last_node, node_index_map, node_list);
      for (auto &next_fusion_node : next_fusion_nodes) {
        AddDependCtrl(next_fusion_node, fusion_last_succ, graph);
        AddDependCtrl(next_fusion_node, fusion_second_succ, graph);
      }
    }
    changed = true;
  } else {
    for (int64_t i = 0; i < SizeToInt(no_fusion_allgather_nodes.size()) - 1; ++i) {
      if (no_fusion_allgather_succ_nodes[i] == nullptr) {
        continue;
      }
      auto next_node = no_fusion_allgather_nodes[i + 1];
      MS_EXCEPTION_IF_NULL(next_node);
      auto next_cnode = next_node->cast<CNodePtr>();
      AddDependCtrl(next_cnode, no_fusion_allgather_succ_nodes[i], graph);
    }
    changed = true;
  }
  return changed;
}
}  // namespace

bool AddDependForAllGather::Run(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    MS_LOG(DEBUG) << "AllGather parallel optimization is not required in pipeline parallel mode.";
    return false;
  }
  MS_EXCEPTION_IF_NULL(graph);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (cell_reuse) {
    return false;
  }
  bool changed = false;
  const auto &node_list = TopoSort(graph->get_return());
  std::map<int64_t, AnfNodePtr> all_gather_node;
  std::map<int64_t, std::vector<AnfNodePtr>> fusion_allgather_nodes;
  std::map<AnfNodePtr, size_t> node_index_map;
  std::vector<AnfNodePtr> no_fusion_allgather_nodes;
  std::vector<AnfNodePtr> no_fusion_allgather_succ_nodes;
  for (size_t i = 0; i < node_list.size(); i++) {
    node_index_map.insert(std::pair<AnfNodePtr, int64_t>(node_list[i], i));
  }
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    bool is_recompute = cnode->GetAttr(kAttrDuplicated) != nullptr && GetValue<bool>(cnode->GetAttr(kAttrDuplicated));
    if (common::AnfAlgo::GetCNodeName(cnode) == kAllGatherOpName && common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode) &&
        common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion) > 0 && !is_recompute) {
      fusion_allgather_nodes[common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion)].push_back(node);
      auto succ_node = GetFirstNextUsers(graph, node, node_index_map, node_list);
      if (succ_node == nullptr) {
        MS_LOG(DEBUG) << "allgather node " << node->fullname_with_scope() << "has no outputs.";
        continue;
      }
      no_fusion_allgather_nodes.push_back(node);
      no_fusion_allgather_succ_nodes.push_back(succ_node);
      all_gather_node[common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion)] = node;
    }
  }
  // Fusion of Hcom Nodes may cause OOM, so now adapt no-fusion as well.
  changed = AddDependCtrlForMemReuse(graph, node_list, fusion_allgather_nodes, all_gather_node,
                                     no_fusion_allgather_nodes, no_fusion_allgather_succ_nodes, node_index_map, false);

  return InsertDependForAllGatherParallel(graph, node_list, fusion_allgather_nodes, no_fusion_allgather_nodes,
                                          no_fusion_allgather_succ_nodes, node_index_map, false) ||
         changed;
}
}  // namespace opt
}  // namespace mindspore
