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

#include "plugin/device/ascend/optimizer/ge/add_depend_for_all_gather.h"
#include <map>
#include <vector>
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace opt {
AnfNodePtr GetFirstNextUsers(const FuncGraphPtr &graph, const AnfNodePtr &input,
                             std::map<AnfNodePtr, size_t> node_index_map, std::vector<AnfNodePtr> all_nodes) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(input);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager." << trace::DumpSourceLines(input);
  }
  auto user_items = iter->second;
  auto min_index = all_nodes.size();
  for (const auto &node_pair : user_items) {
    auto node = node_pair.first;
    MS_EXCEPTION_IF_NULL(node);
    if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      continue;
    }
    if (node_index_map.find(node) == node_index_map.end()) {
      MS_LOG(EXCEPTION) << "Can not find node in node_index_map, node: " << node->fullname_with_scope();
    }
    if (min_index > node_index_map[node]) {
      min_index = node_index_map[node];
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

bool AddDependForAllGather::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  const auto &node_list = TopoSort(graph->get_return());
  std::vector<AnfNodePtr> all_gather_node;
  std::vector<AnfNodePtr> allgather_succ_nodes;
  std::vector<AnfNodePtr> allgather_second_succ_nodes;
  std::map<AnfNodePtr, size_t> node_index_map;
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
      all_gather_node.push_back(node);
      auto allgather_first_succ = GetFirstNextUsers(graph, node, node_index_map, node_list);
      allgather_succ_nodes.push_back(allgather_first_succ);
      allgather_second_succ_nodes.push_back(GetFirstNextUsers(graph, allgather_first_succ, node_index_map, node_list));
    }
  }
  for (int64_t i = 0; i < SizeToInt(all_gather_node.size()) - 1; ++i) {
    if (allgather_succ_nodes[i] == nullptr) {
      continue;
    }
    auto next_node = all_gather_node[i + 1];
    MS_EXCEPTION_IF_NULL(next_node);
    auto next_cnode = next_node->cast<CNodePtr>();
    std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                      common::AnfAlgo::GetInputNode(next_cnode, 0), allgather_succ_nodes[i]};
    auto new_input = graph->NewCNode(inputs);
    new_input->set_abstract(common::AnfAlgo::GetInputNode(next_cnode, 0)->abstract());
    common::AnfAlgo::SetNodeInput(next_cnode, new_input, 0);
    changed = true;
  }

  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    MS_LOG(DEBUG) << "AllGather parallel optimization is not required in pipeline parallel mode.";
    return changed;
  }

  for (size_t i = 1; i < all_gather_node.size(); i++) {
    MS_EXCEPTION_IF_NULL(all_gather_node[i]);
    if (allgather_second_succ_nodes[i] == nullptr || allgather_succ_nodes[i] == nullptr) {
      MS_LOG(DEBUG) << "AllGather has no successor node or second successor node, AllGather name: "
                    << all_gather_node[i]->fullname_with_scope();
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

    std::vector<AnfNodePtr> new_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          common::AnfAlgo::GetInputNode(succ_cnode, 0), another_input};
    auto new_input_depend = graph->NewCNode(new_inputs);
    new_input_depend->set_abstract(succ_cnode->abstract());
    common::AnfAlgo::SetNodeInput(succ_cnode, new_input_depend, 0);
    changed = true;
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
