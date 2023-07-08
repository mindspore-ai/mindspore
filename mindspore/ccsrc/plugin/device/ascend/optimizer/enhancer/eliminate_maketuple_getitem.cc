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

#include <vector>
#include "plugin/device/ascend/optimizer/enhancer/eliminate_maketuple_getitem.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace opt {
bool EliminateMaketupleGetitem::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  bool changed = false;
  auto manager = graph->manager();
  auto node_users_map = manager->node_users();
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  for (const auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      continue;
    }
    auto make_tuple_cnode = node->cast<CNodePtr>();
    for (const auto &node_pair : node_users_map[make_tuple_cnode]) {
      if (!IsPrimitiveCNode(node_pair.first, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto tuple_getitem_cnode = node_pair.first->cast<CNodePtr>();
      ValuePtr tuple_index_value = GetValueNode(tuple_getitem_cnode->input(2));
      MS_EXCEPTION_IF_NULL(tuple_index_value);
      if (!tuple_index_value->isa<Int64Imm>()) {
        MS_LOG(EXCEPTION) << "The index of tuple getitem is not int64";
      }
      auto tupleget_item_index = tuple_index_value->cast<Int64ImmPtr>()->value();
      auto make_tuple_input_index = tupleget_item_index + 1;
      auto real_input_node = make_tuple_cnode->input(make_tuple_input_index);
      manager->Replace(tuple_getitem_cnode, real_input_node);
      changed = true;
    }
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
