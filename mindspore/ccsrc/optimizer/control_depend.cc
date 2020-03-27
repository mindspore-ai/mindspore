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

#include "optimizer/control_depend.h"

#include <vector>
#include <list>
#include <utility>
#include <memory>
#include <algorithm>

#include "optimizer/optimizer.h"

namespace mindspore {
namespace opt {
std::vector<AnfNodePtr> DoControlDepend(const FuncGraphPtr &graph, const CNodePtr &return_node,
                                        const std::vector<size_t> &effect_index, const std::vector<CNodePtr> &cnodes) {
  std::vector<AnfNodePtr> depend_nodes{NewValueNode(prim::kPrimDepend), return_node->input(1)};
  std::vector<AnfNodePtr> make_tuple{NewValueNode(prim::kPrimMakeTuple)};
  size_t effect_size = effect_index.size();
  for (size_t i = 0; i < effect_size; i++) {
    size_t pre_index = 0;
    if (i > 0) {
      pre_index = effect_index[i - 1] + 1;
    }
    size_t this_index = effect_index[i];
    size_t last_index = cnodes.size() - 2;
    if (i < effect_size - 1) {
      last_index = effect_index[i + 1];
    }

    if (this_index > pre_index) {
      std::vector<CNodePtr> pre_segment;
      for (size_t k = pre_index; k < this_index; k++) {
        // Skip depend, make_tuple, and tuple_get_item, because these primitives are not real operator in GE.
        if (IsPrimitiveCNode(cnodes[k], prim::kPrimDepend) || IsPrimitiveCNode(cnodes[k], prim::kPrimMakeTuple) ||
            IsPrimitiveCNode(cnodes[k], prim::kPrimTupleGetItem)) {
          continue;
        }
        pre_segment.push_back(cnodes[k]);
      }
      auto roots = FindRoots(pre_segment);
      for (auto iter = roots->begin(); iter != roots->end(); (void)iter++) {
        AnfNodePtr control_depend =
          graph->NewCNode({NewValueNode(prim::kPrimControlDepend), *iter, cnodes[this_index]});
        make_tuple.push_back(control_depend);
      }
    }
    if (last_index > this_index) {
      std::vector<CNodePtr> last_segment;
      for (size_t k = this_index + 1; k <= last_index; k++) {
        // Skip depend, make_tuple, and tuple_get_item, because these primitives are not real operator in GE.
        if (IsPrimitiveCNode(cnodes[k], prim::kPrimDepend) || IsPrimitiveCNode(cnodes[k], prim::kPrimMakeTuple) ||
            IsPrimitiveCNode(cnodes[k], prim::kPrimTupleGetItem)) {
          continue;
        }
        last_segment.push_back(cnodes[k]);
      }
      auto leaves = FindLeaves(last_segment);
      for (auto iter = leaves->begin(); iter != leaves->end(); (void)iter++) {
        AnfNodePtr control_depend =
          graph->NewCNode({NewValueNode(prim::kPrimControlDepend), cnodes[this_index], *iter});
        make_tuple.push_back(control_depend);
      }
    }
  }
  depend_nodes.push_back(graph->NewCNode(make_tuple));
  return depend_nodes;
}

void AddControlDepend(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::list<CNodePtr> orders = graph->GetOrderedCnodes();
  std::vector<CNodePtr> cnodes(orders.begin(), orders.end());
  size_t cnodes_size = cnodes.size();
  // get effect index of cnodes
  std::vector<size_t> effect_index{};
  for (size_t i = 0; i < cnodes_size; i++) {
    if (graph->HasEffect(cnodes[i])) {
      effect_index.push_back(i);
    }
  }
  if (effect_index.empty()) {
    return;
  }
  AnfNodePtr last_node = cnodes[cnodes_size - 1];
  CNodePtr return_node;
  if (last_node->isa<CNode>()) {
    return_node = last_node->cast<CNodePtr>();
  }
  MS_EXCEPTION_IF_NULL(return_node);
  if (!IsPrimitiveCNode(return_node, prim::kPrimReturn)) {
    MS_LOG(EXCEPTION) << "The last cnode after sorting, not return cnode.";
  }
  if (return_node->inputs().size() < 2) {
    MS_LOG(EXCEPTION) << "Number of return node inputs should be great than or equal to 2.";
  }

  auto depend_node_inputs = DoControlDepend(graph, return_node, effect_index, cnodes);
  auto depend_cnode = graph->NewCNode(depend_node_inputs);
  depend_cnode->set_abstract(depend_cnode->input(1)->abstract());
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (!manager->Replace(return_node->input(1), depend_cnode)) {
    MS_LOG(EXCEPTION) << "Depend replace node failed";
  }
}
}  // namespace opt
}  // namespace mindspore
