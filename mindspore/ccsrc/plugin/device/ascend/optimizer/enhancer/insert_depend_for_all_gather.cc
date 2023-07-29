/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/enhancer/insert_depend_for_all_gather.h"
#include "ops/other_op_name.h"
#include "ops/framework_ops.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace opt {
bool InsertDependForAllGather::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::map<int64_t, AnfNodePtr> all_gather_node;
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    bool is_recompute = cnode->GetAttr(kAttrDuplicated) != nullptr && GetValue<bool>(cnode->GetAttr(kAttrDuplicated));
    if (common::AnfAlgo::GetCNodeName(cnode) == kAllGatherOpName && common::AnfAlgo::HasNodeAttr(kAttrFusion, cnode) &&
        common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion) > 0 && !is_recompute) {
      all_gather_node[common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrFusion)] = node;
    }
  }
  auto iter = all_gather_node.begin();
  for (int64_t i = 0; i < SizeToInt(all_gather_node.size()) - 1; ++i) {
    auto current_node = iter->second;
    MS_EXCEPTION_IF_NULL(current_node);
    auto next_node = (++iter)->second;
    MS_EXCEPTION_IF_NULL(next_node);
    auto next_cnode = next_node->cast<CNodePtr>();
    std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                      common::AnfAlgo::GetInputNode(next_cnode, 0), current_node};
    auto new_input = graph->NewCNode(inputs);
    new_input->set_abstract(common::AnfAlgo::GetInputNode(next_cnode, 0)->abstract());
    common::AnfAlgo::SetNodeInput(next_cnode, new_input, 0);
    changed = true;
    if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
      continue;
    }
    std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
    auto next_cnode_inputs = next_cnode->inputs();
    std::copy(next_cnode_inputs.begin() + 1, next_cnode_inputs.end(), std::back_inserter(make_tuple_inputs));
    std::vector<AnfNodePtr> next_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                           current_node, graph->NewCNode(make_tuple_inputs)};
    auto cur_new_input = graph->NewCNode(next_inputs);
    cur_new_input->AddAttr("opt_shard_depend", MakeValue(true));
    if (current_node->isa<CNode>()) {
      auto manager = graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      auto cur_node_users = manager->node_users()[current_node];
      for (const auto &allgather_node_user : cur_node_users) {
        if (!IsPrimitiveCNode(allgather_node_user.first) ||
            IsPrimitiveCNode(allgather_node_user.first, prim::kPrimDepend)) {
          continue;
        }
        auto allgather_node_user_cnode = allgather_node_user.first->cast<CNodePtr>();
        common::AnfAlgo::SetNodeInput(allgather_node_user_cnode, cur_new_input, 0);
      }
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
