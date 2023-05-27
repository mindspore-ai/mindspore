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

#include <map>
#include <vector>
#include "plugin/device/ascend/optimizer/ge/add_depend_for_all_gather.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
AnfNodePtr GetNextUsers(const FuncGraphPtr &graph, const AnfNodePtr &input) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(input);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager." << trace::DumpSourceLines(input);
  }
  auto user_items = iter->second;
  for (const auto &node_pair : user_items) {
    auto node = node_pair.first;
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(INFO) << "This node used by other kernel: " << node->fullname_with_scope();
    return node;
  }
  return nullptr;
}

bool AddDependForAllGather::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::vector<AnfNodePtr> all_gather_node;
  std::vector<AnfNodePtr> allgather_succ_nodes;
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
      auto allgather_first_succ = GetNextUsers(graph, node);
      allgather_succ_nodes.push_back(allgather_first_succ);
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
  return changed;
}
}  // namespace opt
}  // namespace mindspore
