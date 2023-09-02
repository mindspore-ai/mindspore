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
}  // namespace

bool InsertDependForAllGatherParallel(const FuncGraphPtr &graph, const std::map<int64_t, AnfNodePtr> &all_gather_nodes,
                                      const std::map<int64_t, std::vector<AnfNodePtr>> &fusion_allgather_nodes,
                                      const std::map<AnfNodePtr, size_t> &node_index_map,
                                      const std::vector<AnfNodePtr> &node_list) {
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    MS_LOG(DEBUG) << "AllGather parallel optimization is not required in pipeline parallel mode.";
    return false;
  }
  bool changed = false;
  std::map<int64_t, AnfNodePtr> fusion_allgather_outputs;
  auto iter = fusion_allgather_nodes.begin();
  for (size_t i = 1; i < fusion_allgather_nodes.size(); i++) {
    auto pre_fusion_id = iter->first;
    auto nodes = (++iter)->second;
    for (auto &node : nodes) {
      MS_EXCEPTION_IF_NULL(node);
      auto allgather_first_succ = GetFirstNextUsers(graph, node, node_index_map, node_list);
      auto allgather_second_succ = GetFirstNextUsers(graph, allgather_first_succ, node_index_map, node_list);
      if (allgather_first_succ == nullptr || allgather_second_succ == nullptr) {
        MS_LOG(DEBUG) << "AllGather has no successor node or second successor node, AllGather name: "
                      << node->fullname_with_scope();
        continue;
      }
      if (!IsPrimitiveCNode(allgather_first_succ, prim::kPrimLoad)) {
        MS_LOG(DEBUG) << "AllGather successor node it not Load, but is: " << node->fullname_with_scope()
                      << ", AllGather node:" << node->fullname_with_scope();
        continue;
      }
      AnfNodePtr another_input = nullptr;
      auto second_succ_cnode = allgather_second_succ->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(second_succ_cnode);
      for (size_t j = 1; j < second_succ_cnode->inputs().size(); j++) {
        auto succ_input = second_succ_cnode->input(j);
        if (succ_input != allgather_first_succ) {
          another_input = succ_input;
          break;
        }
      }
      fusion_allgather_outputs[pre_fusion_id] = another_input;
      break;
    }
  }
  auto node_iter = all_gather_nodes.begin();
  for (size_t i = 1; i < all_gather_nodes.size(); i++) {
    auto pre_fusion_id = node_iter->first;
    auto node = (++node_iter)->second;
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
    auto succ_cnode = allgather_first_succ->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(succ_cnode);
    auto &pre_fusion_output = fusion_allgather_outputs[pre_fusion_id];
    std::vector<AnfNodePtr> new_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          common::AnfAlgo::GetInputNode(succ_cnode, 0), pre_fusion_output};
    auto new_input_depend = graph->NewCNode(new_inputs);
    new_input_depend->set_abstract(succ_cnode->abstract());
    common::AnfAlgo::SetNodeInput(succ_cnode, new_input_depend, 0);
    changed = true;
  }
  return changed;
}

bool AddDependForAllGather::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  static const auto cell_reuse = context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (cell_reuse) {
    return false;
  }
  bool changed = false;
  const auto &node_list = TopoSort(graph->get_return());
  std::map<int64_t, AnfNodePtr> all_gather_node;
  std::map<int64_t, std::vector<AnfNodePtr>> fusion_allgather_nodes;
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
      fusion_allgather_nodes[common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion)].push_back(node);
      all_gather_node[common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion)] = node;
    }
  }
  auto next_iter = all_gather_node.begin();
  auto iter = fusion_allgather_nodes.begin();
  auto fusion_size = all_gather_node.size();
  for (size_t i = 1; i < fusion_size; ++i) {
    auto next_node = (++next_iter)->second;
    MS_EXCEPTION_IF_NULL(next_node);
    auto next_cnode = next_node->cast<CNodePtr>();
    auto fusion_nodes = (iter++)->second;
    for (auto &node : fusion_nodes) {
      MS_EXCEPTION_IF_NULL(node);
      auto succ_node = GetFirstNextUsers(graph, node, node_index_map, node_list);
      if (succ_node == nullptr) {
        MS_LOG(DEBUG) << "allgather node " << node->fullname_with_scope() << "has no outputs.";
        continue;
      }
      std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                        common::AnfAlgo::GetInputNode(next_cnode, 0), succ_node};
      MS_LOG(DEBUG) << "Add Depend between node: " << succ_node->fullname_with_scope()
                    << " and node: " << next_node->fullname_with_scope();
      auto new_input = graph->NewCNode(inputs);
      new_input->set_abstract(common::AnfAlgo::GetInputNode(next_cnode, 0)->abstract());
      common::AnfAlgo::SetNodeInput(next_cnode, new_input, 0);
    }
    changed = true;
  }
  changed =
    InsertDependForAllGatherParallel(graph, all_gather_node, fusion_allgather_nodes, node_index_map, node_list) ||
    changed;
  return changed;
}
}  // namespace opt
}  // namespace mindspore
